# Observation A/B: Board Only vs. Board + Legal Moves

Task: `chess-terminal-move` | Step limit: 200 | Two models × two observation conditions

The only variable is `--omit-legal-moves`, which removes the `legal_moves` field
from every observation and selects the matching system prompt. No source changes
separate the arms.

| Arm | Trajectory | Result |
|---|---|---|
| deepseek, board only | `part3-no-legal-moves-deepseek.json` | `part3-no-legal-moves-deepseek-result.json` |
| deepseek, board+legal | `part3-legal-moves-deepseek.json` | `part3-legal-moves-deepseek-result.json` |
| gpt-oss, board only | `part3-no-legal-moves-gpt-oss.json` | `part3-no-legal-moves-gpt-oss-result.json` |
| gpt-oss, board+legal | `part3-legal-moves-gpt-oss.json` | `part3-legal-moves-gpt-oss-result.json` |

## Required metrics

| Model | Condition | Total `play_move` calls | Rejected as illegal | Invalid-move rate | Reached `game_over: true` |
|---|---|---|---|---|---|
| deepseek | board only | 41 | 4 | 10% | yes — Black wins |
| deepseek | board + legal | 37 | 0 | 0% | yes — Black wins |
| gpt-oss | board only | 102 | 12 | 12% | yes — Black wins |
| gpt-oss | board + legal | 17 | 0 | 0% | yes — Black wins |

Invalid-move rate is rejections divided by tool observations returned, so it
normalises for the differing number of calls per arm.

## Comparison

**Legality.** Including the legal-move list drove rejected moves to zero in both
models, from 10% and 12% respectively. This is the clearest effect in the
experiment and it is consistent across two models that otherwise behave very
differently.

**Cost.** The larger effect is on how much work each arm took. gpt-oss spent
3,340,042 prompt tokens board-only against 84,869 with the list — a 39× difference
for the same model on the same task, driven by taking 162 steps instead of 19.
deepseek showed the same direction far more weakly: 279,799 against 301,902, which
is effectively flat. The interface penalty is therefore severe for one model and
negligible for the other, so it is a property of the model–interface pair rather
than of the interface alone.

**Failure modes differ.** gpt-oss board-only produced 60 steps with no tool call
at all. Only one was a token truncation; the other 59 finished with
`finish_reason: "stop"` — the model chose to answer in prose rather than act,
announcing moves in algebraic notation it never submitted ("I will continue with a
strategic move. **Move:** **Rd5**"). The same model with the legal-move list
produced 2 such steps in 19. Removing the list did not merely make the task
harder; it made this model stop playing and start narrating.

**What did not vary.** All four arms reached a terminal state, and all four lost
to Black. Both observation formats produced legal, plausible opening play followed
by aimless middlegame drift — rooks touring the board, kings oscillating between
adjacent squares. The observation format changed how legally and how efficiently
the agent moved; it changed nothing about the quality of the moves it chose. That
is the gap the `select-move` skill addresses in the following section, and it is
evidence that richer observations are not a substitute for an explicit strategy.

## Limitations

One run per cell, with no repeats, against a deterministic opponent. Run-to-run
variance is large: an earlier deepseek board-only run under identical settings ran
86 steps and never reached `game_over`, against 41 steps here.

Rejection counts are small in absolute terms (4, 0, 12, 0), so the per-arm
differences rest on a handful of events. The token and step differences for
gpt-oss are large enough to be robust to that concern; the legality differences
are not.
