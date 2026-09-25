# Imitation samples: timing, inputs and the rules that keep them honest

Status 2026-09-25: the replay-side layer (`il/timeline.py`) and its audit (`il/audit.py`) exist.
Board state from the engine is the next stage; the model's full input list is at the end.

## Timing

Everything is in game ticks (20 per second).

| quantity | value | where it comes from |
|---|---|---|
| command age: issue -> executes | 21 ticks, both players | live queue, every play (NOTES "Delay") |
| bot: decision frame -> issue | median 5 (3-7) | 908 bot plays, 2026-09-25 friendlies |
| opponent command visible before it executes | median 14 (p10 10, p90 19) | 1131 commands, 34 sessions |
| decision grid | every 5 ticks from tick 90 | FirstLight's policy grid |

A replay tick `L` is when a play executes (its unit is on the board from `L + 1`). The human
decided around `L - 21`. For the bot, a play that lands at `L` must be decided on the grid tick
at or before `L - delay`, where `delay = 21 + overhead` (24-28, drawn per actor per replay from
the measured overhead, and given to the model as an input). The label's `offset` (0-4) is how far
into that 5-tick window the play is sent.

FirstLight labelled each play 0-4 ticks before it lands, so its models learned to act on a board
they would only see a second later. This is the fix.

## What the actor knows at decision tick t

| input | rule | known at |
|---|---|---|
| own hand | the deal minus every own play decided in an earlier window, each replaced by its draw (what the screen shows) | own decision ticks |
| own pending | own plays decided before t that land at or after t | own decision ticks |
| opponent pending | opponent commands with `L - lead <= t <= L` | `L - lead` |
| opponent history | opponent plays landed or visible by t, in order | same |
| opponent hand | exact once 4 plays are known: the deck minus the last 4 played; before that, candidates | same |
| opponent deck | from the API (recorded in each replay) | 0 |
| tower troops | both, from the replay | 0 |
| command delay | this actor's delay | 0 |

Not from the replay: board state (units with attack/charge/deploy timers, projectiles, towers),
both elixirs, evolution progress, hero abilities. These come from the engine at tick t, which
is causal by construction.

`hand_certain`: after 4 own plays the hand is exact. Before that, the deal is one random choice
that fits every play, so the other unplayed cards in hand are a guess. Training can down-weight
or drop those samples; they are about the first 20-40 seconds of each side.

## Rules, and the check that enforces each

| rule | check (`il/audit.py`) |
|---|---|
| nothing from after t, and nothing hidden from the actor | **leak**: build each sample from the full replay and again from the replay cut down to what the actor could know at t; every input must be identical |
| every input stamped no later than t | **stamp** |
| a chosen card is in hand when chosen (two plays in one window: the second is drawn by the first) | **in_hand** |
| a card already sent is not also in hand | **pending** |
| labels land 0-4 ticks into their window | **offset** |
| the deduced opponent hand is right | **opp_hand**: equals the true hand from the deal whenever it claims to be exact and nothing issued is still invisible |

The audit was tested by planting three mistakes; each fails it (every sample for the opponent's
true hand; 5,656 samples for "opponent commands visible from the moment they are issued";
3,599 + 3,442 for "the chosen play shown as already sent").

## Still to verify

- Opponent command leads above 21 ticks (about 10% of those measured, up to 48) would mean a
  command was visible before it was issued. Held out of the lead table until explained, most
  likely a command that executed later than issue + 21.
- The live console must build these same inputs with this same code (parity check on recorded
  matches), including the conversion from the reader's hand (updates when a play executes) to
  the on-screen hand (updates when it is sent).
