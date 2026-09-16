"""The domain-independent ReAct loop shared by both agents.

Part 1 completes the generic loop here; the two subclasses in this package
supply only their own tools and tool executors.
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
import yaml

from assignment.env import Environment
from assignment.agent.tools import INVOKE_SKILL_TOOL

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_KEEP_RECENT_STEPS = 1
DEFAULT_COMPACTION_MAX_TOKENS = 1_200
MAX_OBSERVATION_CHARS = 10_000

# A skill file opens with a YAML frontmatter block fenced by `---` lines. The
# regex only carves out that block; PyYAML parses what is inside it, so a
# description containing a colon or quotes still reads correctly.
SKILL_FRONTMATTER_PATTERN = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL
)

# TODO(Part 2): Write instructions that make the model produce concise working
# memory for a software agent. The prompt should preserve concrete progress,
# failures, test results, constraints, and next steps without copying raw output.
COMPACTION_SYSTEM_PROMPT = """\
You are compacting the working memory of a software engineering agent. You are \
given the agent's objective and the older part of its transcript. The recent \
steps are kept verbatim and are not shown to you.

Write the notes the agent needs to keep working as if it remembered this \
stretch. Address the agent, and write plain prose under short headings, not a \
narration of the transcript.

Preserve, when the transcript establishes them:
- The objective, and any constraint or instruction it must still obey.
- Files read or changed, by path, and what each change was.
- Commands run that produced a decisive result, and the result itself: tests \
passing or failing with counts and names, error types and messages, exit codes.
- Approaches already tried that did not work, and why, so they are not retried.
- What is verified versus assumed, anything still blocking, and the next action.

Do not copy raw output: no file contents, no directory listings, no full \
tracebacks or logs. Reduce them to the fact they established. A command whose \
result changed nothing is not worth a line. Never invent a result the \
transcript does not show, and never claim work is finished unless it says so.\
"""


class StepLimitError(Exception):
    """Raised when an agent exhausts its model-call budget."""


def format_tool_output(output: dict[str, Any]) -> str:
    """Format a terminal result as a compact, tagged model observation."""

    elements: list[str] = []
    for key in sorted(output):
        value = output[key]
        if isinstance(value, str) and len(value) > MAX_OBSERVATION_CHARS:
            # Leave room for the elision notice so the formatted value itself,
            # not just its retained source slices, stays below the limit.
            retained_at_each_end = 4_900
            omitted = len(value) - (2 * retained_at_each_end)
            value = (
                f"{value[:retained_at_each_end]}\n"
                f"[{omitted} characters elided; read a narrower range]\n"
                f"{value[-retained_at_each_end:]}"
            )
        elements.append(f"<{key}>{value}</{key}>")
    return "\n".join(elements)


def render_transcript(messages: list[dict[str, Any]]) -> str:
    """Flatten messages into plain text for the summarizer.

    Actions and their results become labelled lines, so the summarizer is never
    handed a `tool_call_id` it would be expected to answer.
    """

    lines: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content") or ""
        if role == "assistant":
            if content:
                lines.append(f"[assistant] {content}")
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                lines.append(
                    f"[action {call.get('id', '')}] "
                    f"{function.get('name', 'unknown')} {function.get('arguments', '')}"
                )
        elif role == "tool":
            lines.append(f"[result of {message.get('tool_call_id', '')}]\n{content}")
        else:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def rough_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens without a provider-specific tokenizer."""

    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / 4))


class Agent:
    """Base class for a ReAct agent with pluggable tools."""

    def __init__(
        self,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        self.env = environment
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("OPENAI_MODEL is not set.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set.")
        try:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        except ValueError as exc:
            raise RuntimeError("OPENAI_MAX_RETRIES must be an integer.") from exc
        if max_retries < 0:
            raise RuntimeError("OPENAI_MAX_RETRIES must be non-negative.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )

        self.logs_save_path = logs_save_path
        self.step_limit = step_limit
        self.auto_stop_environment = auto_stop_environment
        if compact_threshold_tokens is not None and compact_threshold_tokens <= 0:
            raise ValueError("compact_threshold_tokens must be positive or None")
        if (
            compaction_keep_recent_steps is not None
            and compaction_keep_recent_steps < 1
        ):
            raise ValueError("compaction_keep_recent_steps must be at least 1")
        if compaction_max_tokens is not None and compaction_max_tokens < 1:
            raise ValueError("compaction_max_tokens must be positive")
        # A None threshold turns compaction off. The other two settings then
        # describe a compaction that never happens, so fall back to the
        # defaults rather than leaving a None for later code to trip over.
        self.compact_threshold_tokens = compact_threshold_tokens
        self.compaction_keep_recent_steps = (
            DEFAULT_COMPACTION_KEEP_RECENT_STEPS
            if compaction_keep_recent_steps is None
            else compaction_keep_recent_steps
        )
        self.compaction_max_tokens = (
            DEFAULT_COMPACTION_MAX_TOKENS
            if compaction_max_tokens is None
            else compaction_max_tokens
        )

        # Each agent supplies its own opening messages: the standing
        # instructions, and the task statement that starts the run.
        self.system_prompt: str = """
        <system_information>
        {
            "machine": <machine>,
            "release": <release>,
            "system": <system>,
            "version": <version>
        }
        </system_information>
        """
        self.task_prompt: str = "You are a helpful agent. Make sure to obey what the user says and follow directions to the best of your ability."

        self.api_prompts: list[list[dict[str, Any]]] = []
        self.api_responses: list[dict[str, Any]] = []
        self.compaction_events: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.finished = False
        self.steps_taken = 0

        self.skills_path = Path(skills_path) if skills_path is not None else None
        self.skills: dict[str, dict[str, str]] = (
            self.load_skills(self.skills_path) if self.skills_path is not None else {}
        )

        if self.skills:
            self.tools.append(INVOKE_SKILL_TOOL)

        # TODO(1.1.a): Add machinery to maintain agent state as it takes actions
        # and observes the results.
        self.working_memory: list[dict[str, Any]] = []


    def load_skills(self, skills_path: Path) -> dict[str, dict[str, str]]:
        """Load the skill folders exposed to this agent.

        ``skills_path`` is a catalog directory whose children are skill
        directories, each holding a ``SKILL.md``. Returns a mapping keyed by
        the frontmatter ``name``, where each value carries a one-line
        ``metadata`` entry for the prompt's skill catalog and the file's whole
        ``content`` for ``invoke_skill`` to hand back on demand.
        """

        if not skills_path.exists():
            raise ValueError(f"Skills path does not exist: {skills_path}")
        if not skills_path.is_dir():
            raise ValueError(f"Skills path is not a directory: {skills_path}")

        skills: dict[str, dict[str, str]] = {}
        # Remember where each name came from so a duplicate can name both
        # offenders rather than just the second one.
        sources: dict[str, Path] = {}

        for child in sorted(skills_path.iterdir()):
            skill_file = child / "SKILL.md"
            # A child that is not a skill directory is not an error: the
            # catalog may also hold a README, or a stray __pycache__.
            if not child.is_dir() or not skill_file.is_file():
                continue

            content = skill_file.read_text(encoding="utf-8")
            match = SKILL_FRONTMATTER_PATTERN.match(content)
            if match is None:
                raise ValueError(
                    f"{skill_file} has no YAML frontmatter: the file must open "
                    "with a `---` line and close the block with another."
                )
            try:
                frontmatter = yaml.safe_load(match.group(1))
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"{skill_file} has malformed YAML frontmatter: {exc}"
                ) from exc
            if not isinstance(frontmatter, dict):
                raise ValueError(
                    f"{skill_file} frontmatter must be a YAML mapping, got "
                    f"{type(frontmatter).__name__}."
                )

            name = frontmatter.get("name")
            description = frontmatter.get("description")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"{skill_file} frontmatter is missing a non-empty `name`."
                )
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"{skill_file} frontmatter is missing a non-empty `description`."
                )

            name = name.strip()
            if name in skills:
                raise ValueError(
                    f"Duplicate skill name {name!r}: defined in both "
                    f"{sources[name]} and {skill_file}."
                )

            # The catalog goes in the system prompt, so collapse the
            # description to a single line and keep the body out of it.
            summary = " ".join(description.split())
            skills[name] = {
                "metadata": f"- {name}: {summary}",
                "content": content,
            }
            sources[name] = skill_file

        return skills

    def skill_catalog_prompt(self) -> str:
        """Render the loaded skills as a prompt section, or "" when there are none.

        Only each skill's one-line ``metadata`` goes in. The bodies stay out
        until the model calls ``invoke_skill``, which is what makes the catalog
        cheap enough to carry on every request.
        """

        if not self.skills:
            return ""

        catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
        return (
            "\n\nReusable skills are available. Call `invoke_skill` with a "
            "skill's name to load its instructions, and follow them in place "
            f"of your default approach.\n\n<skills>\n{catalog}\n</skills>\n"
        )

    def query_language_model(self) -> dict[str, Any]:
        """Send one tool-enabled Chat Completions request and normalize it."""

        messages = self.build_prompt()
        self.api_prompts.append(deepcopy(messages))
        step_number = self.steps_taken + 1
        print(
            f"[agent] step {step_number}/{self.step_limit}: requesting action",
            flush=True,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                reasoning_effort="medium",
                max_completion_tokens=4096,
            )
        except Exception as exc:
            print(
                f"[agent] step {step_number}: model request failed after retries "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            raise
        self.api_responses.append(response.model_dump(mode="json"))
        self.steps_taken += 1
        message = self.process_response(response)
        tool_names = [
            call.get("function", {}).get("name", "unknown")
            for call in message.get("tool_calls", [])
            if isinstance(call, dict)
        ]
        if tool_names:
            print(
                f"[agent] step {step_number}: tool call(s): {', '.join(tool_names)}",
                flush=True,
            )
        else:
            print(
                f"[agent] step {step_number}: response contained no parsed tool call; "
                "the loop should preserve the response and continue",
                flush=True,
            )
        return message

    def process_response(self, response: Any) -> dict[str, Any]:
        """Return relevant parts of the language model's response."""

        message = response.choices[0].message.model_dump(exclude_none=True)

        # A reasoning model returns its scratchpad beside the answer. It is not
        # valid input on the next request, and echoing it back grew to 38% of
        # the prompt on a chess run, crowding out the board it was reasoning
        # about. `api_responses` still records the untouched response, so the
        # trajectory keeps every reasoning token for analysis.
        message.pop("reasoning_content", None)

        # Providers restart tool-call ids at `call_0` on every response, so a
        # long transcript accumulates many calls and many results all sharing
        # one id, with nothing tying a result to the call it answers. Qualify
        # each id by the step that produced it. The observations built from
        # these calls copy the id, so both sides stay consistent.
        for index, call in enumerate(message.get("tool_calls") or []):
            if isinstance(call, dict):
                call["id"] = f"{call.get('id') or f'call_{index}'}_s{self.steps_taken}"

        return message

    def build_prompt(self) -> list[dict[str, Any]]:
        # TODO(1.1.a): Construct a sequence of messages that form the language
        # model prompt. This should include standing instructions, task
        # specification, prior interaction including observations, reasoning,
        # and actions from previous turns. Note that this method should be
        # domain-agnostic and construct the prompt in a way that would apply
        # to any of the inheriting domain-specific agents.

        # You want to be careful about which attributes of the class you modify
        # here as they may also be handled by the subclasses.
        # Read, never write: subclasses assign `task_prompt` after
        # `super().__init__()`, and this runs once per step plus several times
        # per compaction, so anything appended here would accumulate.
        prompt: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task_prompt},
        ]
        prompt.extend(self.working_memory)

        return prompt

    def estimate_active_prompt_tokens(self) -> int:
        """Estimate the next prompt, calibrated by the provider's latest usage."""

        current_prompt = self.build_prompt()
        rough_current = rough_message_tokens(current_prompt)
        if not self.api_prompts or not self.api_responses:
            return rough_current

        usage = self.api_responses[-1].get("usage") or {}
        actual_previous = usage.get("prompt_tokens")
        if not isinstance(actual_previous, int):
            return rough_current

        rough_previous = rough_message_tokens(self.api_prompts[-1])
        added_since_previous_request = max(0, rough_current - rough_previous)
        return actual_previous + added_since_previous_request

    @property
    def compaction_enabled(self) -> bool:
        """Whether this agent compacts its context at all."""

        return self.compact_threshold_tokens is not None

    def compact_context(self):
        """Replace parts of prompt with model-generated working memory. Changes the
        content that `build_prompt` emits."""

        # TODO(2.1): Prompt the model to compact the context. The system
        # prompt should ask for concise factual working memory and preserve
        # the objective, constraints, files, commands, edits, concrete
        # results, failed approaches, tests, blockers, and next action.
        # Summarize only an old prefix; retain the original system/task
        # messages verbatim and at least the latest complete assistant action
        # with all linked tool observations. The resulting summary should change
        # what `build_prompt` emits, and reduce the length of the prompt.

        # An assistant message and the tool messages answering it are one
        # action. Splitting between them would orphan a `tool_call_id` and the
        # provider would reject the next request, so the cut always lands on an
        # assistant message: keep the last `compaction_keep_recent_steps` of
        # them plus everything that follows, and summarize the prefix.
        actions = [
            index
            for index, message in enumerate(self.working_memory)
            if message.get("role") == "assistant"
        ]
        keep = self.compaction_keep_recent_steps
        split = actions[-keep] if len(actions) > keep else 0
        older, recent = self.working_memory[:split], self.working_memory[split:]

        # The system and task messages are not in `working_memory`; they are
        # re-rendered by `build_prompt` and so survive compaction untouched.
        # The objective still goes to the summarizer, which cannot judge what
        # matters without it.
        compaction_prompt = [
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"<objective>\n{self.task_prompt}\n</objective>\n\n"
                    f"<transcript>\n{render_transcript(older)}\n</transcript>"
                ),
            },
        ]

        ### Do not modify this section ###
        compaction_response = self.client.chat.completions.create(
            model=self.model,
            messages=compaction_prompt,
            reasoning_effort="medium",
            max_completion_tokens=self.compaction_max_tokens,
        )
        ##################################

        # Use `compaction_response` to update what `build_prompt` emits, but
        # DO NOT modify the object itself. Let the method return it unchanged.
        #
        # Read only: the summary is taken from the response, and the prefix it
        # replaces is dropped from `working_memory`. An empty summary would
        # erase that prefix and put nothing in its place, so keep the raw
        # context rather than lose it.
        summary = (compaction_response.choices[0].message.content or "").strip()
        if summary:
            self.working_memory = [
                {
                    "role": "user",
                    "content": f"<working_memory>\n{summary}\n</working_memory>",
                },
                *recent,
            ]
        else:
            logger.warning("Compaction returned an empty summary; context kept.")

        ### Do not modify this section ###
        return compaction_prompt, compaction_response.model_dump(mode="json")
        ##################################

    def maybe_compact_context(self) -> bool:
        """Compact before the next action request when the threshold is reached."""

        if not self.compaction_enabled:
            return False

        # Context too short to compact yet
        if self.estimate_active_prompt_tokens() < self.compact_threshold_tokens:
            return False

        prompt_before = deepcopy(self.build_prompt())

        # Not enough steps (each assistant turn corresponds to a step) to force
        # compaction yet
        if (
            len([m for m in prompt_before if m.get("role") == "assistant"])
            <= self.compaction_keep_recent_steps
        ):
            return False

        compaction_prompt, compaction_response = self.compact_context()
        prompt_after = deepcopy(self.build_prompt())
        self.compaction_events.append(
            {
                "step": self.steps_taken,
                "estimated_tokens_before": rough_message_tokens(prompt_before),
                "estimated_tokens_after": rough_message_tokens(prompt_after),
                "active_prompt_before": deepcopy(prompt_before),
                "compaction_prompt": compaction_prompt,
                "compaction_response": compaction_response,
            }
        )
        return True

    def run(self) -> None:
        """Run ReAct steps, always saving the trajectory and stopping Modal."""

        try:
            # TODO(1.2) Run the ReAct loop. Orchestrate the sequence of
            # prompting the language model to produce reasoning and actions,
            # extracting the tool calls produced by the model, and executing
            # the tool calls to obtain the agent's observation for the next
            # step. Ensure you identify when the agent has completed the task
            # by setting `Agent.finished`. If the agent exceeds the
            # `step_limit`, raise `StepLimitError`.
        

            # TODO(2.2) Call `maybe_compact_context()` before each new action
            # request in your shared loop. It already estimates active tokens
            # and handles the threshold, and tracks compaction events for
            # logging.
            while not self.finished and self.steps_taken < self.step_limit:
                # Compact before the request, never after: the threshold is
                # about the prompt we are about to send, and shrinking it
                # afterwards would still have paid for the oversized one.
                self.maybe_compact_context()

                response = self.query_language_model()
                self.working_memory.append(response)

                tool_calls = response.get("tool_calls") or []
                if tool_calls:
                    self.working_memory.extend(self.execute_tool_calls(tool_calls))
                else:
                    # Nothing ran, so nothing would be appended, and the next
                    # request would repeat this context almost verbatim -- which
                    # is how one empty response becomes a run of them. Give the
                    # model something new to react to instead.
                    self.working_memory.append(
                        {
                            "role": "user",
                            "content": (
                                "Your last response contained no tool call, so "
                                "nothing was executed and the state is "
                                "unchanged. Reply with a tool call."
                            ),
                        }
                    )

            if not self.finished:
                raise StepLimitError()
                

            
        finally:
            # This block is provided infrastructure. Do not modify it: a
            # trajectory is required even when a run fails.
            if self.logs_save_path:
                path = Path(self.logs_save_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "prompts": self.api_prompts,
                            "responses": self.api_responses,
                            "compactions": self.compaction_events,
                        },
                        indent=2,
                    )
                )
            if self.auto_stop_environment:
                stop = getattr(self.env, "stop", None)
                if callable(stop):
                    stop()

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute domain-specific calls and return linked tool observations."""

        # You do not need to implement anything here. This method is
        # domain-specific and implemented by the relevant subclasses
        raise NotImplementedError
