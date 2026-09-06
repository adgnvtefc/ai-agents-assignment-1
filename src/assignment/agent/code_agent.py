"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
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
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        self.tools.append(EXECUTE_TOOL)
        self.tools.append(SEND_MESSAGE_TOOL)

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        system_information = json.dumps(
            {
                "machine": self.env.machine,
                "release": self.env.release,
                "system": self.env.system,
                "version": self.env.version,
            },
            indent=2,
        )

        self.system_prompt = f"""You are a software engineering agent working \
directly inside a sandboxed container. You fix real defects in the repository \
you are given, and you do it by running commands, not by describing what you \
would run.

<system_information>
{system_information}
</system_information>

Use the `execute` tool to inspect the repository, run code, and edit files. \
Every command runs in a fresh subshell, so a `cd` does not carry over to the \
next call; pass `cwd` instead. Files you write do persist.

Work in this order, and do not skip ahead:
1. Explore enough of the repository to understand the code the report refers to.
2. Reproduce the reported failure and confirm you have seen it fail. A fix you \
cannot show was needed is a guess.
3. Make the smallest change that addresses the root cause. Do not paper over \
the symptom, and do not rewrite unrelated code.
4. Re-run your reproduction to confirm it now passes, then run the repository's \
own tests to confirm you broke nothing else.

When the work is done, call `send_message` with a short summary of what was \
wrong and what you changed. Do not call it before the fix is verified."""

        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.
        self.system_prompt += self.skill_catalog_prompt()

        self.task_prompt = f"""Fix the following issue in the repository at \
`{self.env.cwd}`.

<issue>
{self.task}
</issue>

Begin by reproducing the failure described above."""

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        observations: list[dict[str, str]] = []
        for call in tool_calls:
            call_id = call.get("id", "")
            function = call.get("function") or {}
            name = function.get("name")

            # `invoke_skill` is registered by the base class, but only when
            # this agent was given skills, so recognize it on the same
            # condition rather than unconditionally.
            recognized = {"execute", "send_message"}
            if self.skills:
                recognized.add("invoke_skill")
            if name not in recognized:
                available = ", ".join(f"`{tool}`" for tool in sorted(recognized))
                observations.append(
                    self.tool_message(
                        call_id,
                        f"Unknown tool: {name!r}. Available tools: {available}.",
                    )
                )
                continue

            # A model can emit arguments that are not valid JSON, or omit them
            # entirely. Either way the agent should be told, not crash.
            try:
                arguments = json.loads(function.get("arguments") or "")
            except (json.JSONDecodeError, TypeError) as exc:
                observations.append(
                    self.tool_message(
                        call_id,
                        f"Could not parse the arguments to `{name}` as JSON: "
                        f"{exc}. Send the arguments again as a valid JSON object.",
                    )
                )
                continue
            if not isinstance(arguments, dict):
                observations.append(
                    self.tool_message(
                        call_id,
                        f"The arguments to `{name}` must be a JSON object, "
                        f"got {type(arguments).__name__}.",
                    )
                )
                continue

            if name == "execute":
                observations.append(self.run_execute(call_id, arguments))
            elif name == "invoke_skill":
                observations.append(self.run_invoke_skill(call_id, arguments))
            else:
                observations.append(self.run_send_message(call_id, arguments))

        return observations

    def run_invoke_skill(self, call_id: str, arguments: dict[str, Any]) -> dict[str, str]:
        """Return one skill's full instructions, the half the catalog withheld."""

        name = arguments.get("name")
        if not isinstance(name, str) or not name.strip():
            return self.tool_message(
                call_id, "`invoke_skill` requires a `name` argument."
            )

        skill = self.skills.get(name.strip())
        if skill is None:
            available = ", ".join(sorted(self.skills)) or "none"
            return self.tool_message(
                call_id,
                f"Unknown skill {name!r}. Available skills: {available}.",
            )
        return self.tool_message(call_id, skill["content"])

    def run_execute(self, call_id: str, arguments: dict[str, Any]) -> dict[str, str]:
        """Run one `execute` call in the sandbox and format what it produced."""

        command = arguments.get("command")
        if command is None:
            return self.tool_message(
                call_id, "`execute` requires a `command` argument."
            )

        # The optional arguments keep the environment's defaults unless the
        # model set them; `Environment.execute` treats None as "use the default".
        output = self.env.execute(
            command,
            timeout=arguments.get("timeout"),
            cwd=arguments.get("cwd"),
            env=arguments.get("env"),
            shell=arguments.get("shell"),
        )
        return self.tool_message(call_id, format_tool_output(output))

    def run_send_message(
        self, call_id: str, arguments: dict[str, Any]
    ) -> dict[str, str]:
        """Record the agent's closing message and end the run."""

        summary = arguments.get("summary")
        if summary is None:
            return self.tool_message(
                call_id, "`send_message` requires a `summary` argument."
            )

        print(f"[agent] send_message: {summary}", flush=True)
        self.finished = True
        return self.tool_message(call_id, "Message delivered to the user.")

    @staticmethod
    def tool_message(call_id: str, content: str) -> dict[str, str]:
        """Build the observation message that answers one tool call."""

        return {"role": "tool", "tool_call_id": call_id, "content": content}
