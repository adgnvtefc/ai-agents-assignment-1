# Token Usage: Full Context vs. Compaction

Instance: `django__django-15368` | Model: `deepseek/deepseek-v4-flash-0731` | Threshold: 6,000 tokens

| Trajectory | Condition |
|---|---|
| `artifacts/django__django-15368-baseline-trajectory.json` | no compaction |
| `artifacts/django__django-15368-6000token_trajectory.json` | compaction at 6,000 |

## Measurements

| | No compaction | Compaction @6,000 |
|---|---|---|
| Steps | 30 | 47 |
| Compaction events | 0 | 8 (2 produced no reduction) |
| **Total prompt tokens** | **285,219** | **207,205** (−27%) |
| Total completion tokens | 4,189 | 6,250 (+49%) |
| Mean prompt tokens/step | 9,507 | 4,408 (−54%) |
| First prompt | 1,716 | 1,716 |
| Peak prompt | 16,704 | 6,313 (−62%) |
| Final prompt | 16,704 | 5,050 |
| Outcome | `RESOLVED: yes` (1/1 FAIL_TO_PASS, 29/29 PASS_TO_PASS) | `RESOLVED: yes` (1/1 FAIL_TO_PASS, 29/29 PASS_TO_PASS) |

Compaction events, as recorded in the trajectory (rough character-based estimate
before and after each event):

| Step | Before | After |
|---|---|---|
| 6 | 5,215 | 5,215 (no reduction) |
| 7 | 5,516 | 5,516 (no reduction) |
| 8 | 6,471 | 2,273 |
| 14 | 5,644 | 2,002 |
| 22 | 6,048 | 2,307 |
| 30 | 5,864 | 2,431 |
| 35 | 5,384 | 1,594 |
| 43 | 5,893 | 2,642 |

## Trends

The two conditions differ in the *shape* of context growth, not only in totals.

Without compaction the prompt grows monotonically: all 29 step-transitions
increase it, at a mean of +516 tokens per step, and it never once decreases.
Each step appends an assistant action and its tool observation to a history that
is resent in full on the next request, so cost per step rises linearly and
without bound. The run ends sending 16,704 tokens per request, roughly ten times
its opening prompt.

With compaction the prompt follows a sawtooth: 6 of 46 transitions decrease it,
each corresponding to an event above. Every successful event returns the prompt
to roughly 2,000–2,600 tokens, after which growth resumes at a similar slope
until the threshold is crossed again. The prompt stays under 6,313 tokens for all
47 steps. The baseline first exceeds that same figure at step 8 and never
returns below it.

The headline number is counterintuitive: the compacted run spent 27% fewer prompt
tokens *while taking 57% more steps*. Per-step prompt cost more than halved, which
is enough to lower the total despite the longer run.

## Tradeoffs

Compaction trades more model calls for a bounded prompt.

What it costs: 49% more completion tokens, 17 additional steps, and one extra
model call per event (8 here, each capped at `compaction_max_tokens`). Summarised
context is also lossy by construction — early exploration survives only as the
summary's claims, so any detail the summariser omitted is unrecoverable.

What it buys: a ceiling. Peak prompt fell 62%, and per-step cost fell 54%. The
correctness of the result did not suffer: the compacted run produced the same
canonical fix and passed `check-swebench` with `RESOLVED: yes`.

On this instance the baseline never approached a context limit, so compaction
reads as a modest token saving bought with more wall-clock. The case for it is
the baseline's +516 tokens/step slope, which has no ceiling: on a task long
enough to exhaust the context window, compaction is the difference between
finishing and failing, and the relative saving grows with run length.

## Anomalies

The events at steps 6 and 7 reduced nothing. Both compaction responses came back
with `finish_reason: length`, `reasoning_tokens: 1200`, and zero content: the
model spent the entire `compaction_max_tokens` budget reasoning and had nothing
left to emit a summary with. The empty-summary guard retained the raw context
rather than replacing it with nothing, so the cost was 2,400 completion tokens
and two round-trips for no reduction, and the context stayed oversized until
step 8. Step 8 succeeded because it reasoned more briefly (649 tokens), leaving
room to write. The lesson is that `compaction_max_tokens` is a budget shared
between reasoning and output, and 1,200 is too tight for a reasoning model at
`reasoning_effort="medium"`.

## Limitations

One run per condition. Step counts vary substantially between runs of the same
configuration, so the 30-vs-47 difference is not attributable to compaction on
this evidence. The per-step and peak figures are structural consequences of the
mechanism and are the defensible comparisons; the totals depend on run length and
should be read alongside it. Only the compacted patch was replayed through
`check-swebench`, so the baseline's resolution status is unverified.
