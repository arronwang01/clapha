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

## Board state: engine conversion, then the live code (2026-09-25 night)

`il/engine_convert.py` replays each recording in Null's engine and saves a snapshot every
decision tick (`il/frames.py`). `il/samples.py` turns each snapshot into the frame the memory
reader would give, and runs it through the console's own code: `firstlight_obs.build` and
`FirstLightRunner` in teacher mode (no policy; the expert's action is recorded as the previous
action, where the policy's own is live). FirstLight's tensorizer then makes the model input and
its `build_expert_action_batch` aligns the label, so a label is kept only if the live mask would
offer that play.

| rule | why |
|---|---|
| a play dated L executes in the step after L | the engine's hand still holds the card in the snapshot at L and not at L + 5; `plays` (executed) holds it from L + 1 |
| own hand and elixir are the **screen's**: plays sent and not executed are taken out of the hand (next card in) and their cost off the elixir | the client does that at the tap; the game state ~1 s later. Humans play the card that just cycled in (Hog, then the Ice Spirit behind it); on the reader's hand the bot cannot. **The console must apply the same conversion to its in-flight taps before a model trained on it plays live** (and tap by the screen hand) |
| the mask counts elixir as of the actor's latest tap in the window (`elixir_lead_ticks` = overhead + 4) | the game checks elixir at the tap, 3-11 ticks after the decision. Labels aligned: 87% with elixir at the decision tick, 94% with elixir at the tap |
| hero controllers are named by FirstLight's ability id (the engine's `actionDataName`) | two-hero decks exist (Hero Musketeer + Hero Ice Golem); champions have no hero form and are skipped, as live |
| labels still illegal at the decision tick are masked out of the loss (FirstLight's own handling), not turned into WAIT | ~5% of labels: cards short of elixir at the tap (below), abilities not ready yet |

Label alignment on 12 exact replays: 720 of 772 labels (93%); 42 short of elixir at the tap, 10
abilities not ready.

## Still to verify

- **Plays that look tapped short of elixir.** Of 17,599 plays (300 exact replays), the elixir
  spare when the play executes peaks at 23 ticks of regeneration at 1x and 2x, which is "tapped
  the moment it was affordable, executed 21 ticks later": L is the execution tick. But ~9% have
  less than 21 ticks spare (4% at 1x, 9% at 2x, 17% at 3x, with a bump at 12-16 ticks), which a
  fixed 21-tick age does not allow. Live recordings (1,830 commands, both sides) show every
  command issued with enough elixir and 20-21 ticks in the queue, so it is not early placement.
  Engine elixir is right (starts at 6.0, 178.6 raw per tick at 1x as live, charges exactly each
  card's cost). Unexplained; those labels are masked.
- Opponent command leads above 21 ticks: the live queue shows no command staying more than 20
  ticks, so the tail is a measurement artifact of the earlier lead table, not waiting commands.
- The live console must build these same inputs with this same code (parity check on recorded
  matches): screen hand and elixir from in-flight taps, `elixir_lead_ticks`, and the new inputs.
