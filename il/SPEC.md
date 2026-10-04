# Imitation samples: timing, inputs and the rules that keep them honest

Status 2026-09-25 night: replay side (`il/timeline.py`, audited by `il/audit.py`), engine
conversion (`il/engine_convert.py`, `il/frames.py`), samples through the live code
(`il/samples.py`), trainer (`il/train.py`, runs on the campus 4080) and engine duels with the real
delay (`il/duel.py`). The new inputs (pending commands etc.) are the next stage.

## Timing

Everything is in game ticks (20 per second).

| quantity | value | where it comes from |
|---|---|---|
| command age: issue -> executes | 21 ticks, both players | live queue, every play (NOTES "Delay") |
| bot: decision frame -> issue | median 5 (3-7) | 908 bot plays, 2026-09-25 friendlies |
| opponent command visible before it executes | median 14 (p10 10, p90 19) | 1131 commands, 34 sessions |
| decision grid | every 5 ticks from tick 90 | FirstLight's policy grid |

One command: issued at `I` (the tap registers), executes at `X = I + 21` (hand, elixir and the
viewer's executed plays change at `X`; its unit is on the board from `X + 1`). A replay tick `L`
is `X - 1`: FirstLight queues it for native execution at `L + 1`, and the engine's hand still
holds the card in the snapshot at `L` (corrected 2026-09-25 night; it was taken as `X`, putting
decisions one tick early). So the human tapped around `L - 20`. For the bot, a play at replay
tick `L` must be decided on the grid tick at or before `L - delay`, where `delay = 20 + overhead`
(23-27, drawn per actor per replay from the measured overhead, and given to the model as an
input). The label's `offset` (0-4) is how far into that 5-tick window the play is sent.
Executed plays are dated `L + 1`, as the viewer dates them (issue + 21), and an opponent command
is visible from `X - lead`.

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
| a play dated L executes at L + 1 | the engine's hand still holds the card in the snapshot at L and not at L + 5; a command queued for execute tick X is out of the hand in the snapshot at X; `plays` (executed) holds it from L + 1, dated L + 1 |
| a hand slot the engine leaves empty after a play (the next card is still being drawn: a five-card cycle) is filled with the head of the cycle | the engine shows the hole for a few ticks; the screen shows the card, and a policy may already have chosen it. The reader shows the same hole as -1, so the console fills it the same way |
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
  card's cost). Also ruled out: regen lost while a card waits at 10 elixir. Live, a tap at 10
  keeps elixir at exactly 10.0 for 21 ticks and then drops by the cost (67 of 71 such taps in the
  recorded friendlies), as in the engine. Unexplained; those labels are masked.
- Opponent command leads above 21 ticks: the live queue shows no command staying more than 20
  ticks, so the tail is a measurement artifact of the earlier lead table, not waiting commands.
- The live console must build these same inputs with this same code (parity check on recorded
  matches): screen hand and elixir from in-flight taps, `elixir_lead_ticks`, and the new inputs.

## What the checkpoints make of these samples (2026-09-25 night)

Held-out Hog 2.6 game sides (16), inputs built by the live code, loss per head (lower is better;
guessing among ~6 cards is 1.79, among the ~500 legal tiles ~6.3):

| checkpoint | label timing | labels usable | card | target tile | timing |
|---|---|---|---|---|---|
| fl:il (FirstLight's imitation model) | FirstLight's (execute at decision) | 99.9% | 0.82 | 2.80 | 1.56 |
| fl:hog2 (Hog specialist, self-play) | FirstLight's | 99.9% | 1.60 | 6.27 | 5.34 |
| fl:hog2 | the bot's (this spec) | 93% | 1.79 | 7.61 | 5.06 |

fl:il predicting humans well from our inputs is the parity check: a model trained on FirstLight's
own engine pipeline reads the live code's observations. fl:hog2 predicts human cards at chance and
tiles worse than uniform: self-play moved it far from human play, so imitation from fl:hog2
changes more than timing. Both starts are trained and the duel decides.

## Training and evaluation

- `il/train.py`: FirstLight's IL recipe (32-turn chunks, recurrent state carried, their
  imitation_loss with act x8, AdamW 3e-5), one sequence per Hog 2.6 side of an exact replay, 2%
  held out by tag, samples made on the fly by DataLoader workers (no tensor cache). Value head
  left out (FirstLight's reward; RL later). Windows: `tools/windows/` (D:\crtrain\py312 has torch).
- `il/duel.py`: two checkpoints in the engine, both fed by the live code; `live` plays tap
  overhead + offset after the decision, wait for elixir at the tap, execute 21 ticks after it;
  `none` is FirstLight's sandbox (execute max(1, offset) after the decision). Hog 2.6 mirrors from
  converted decks, sides swapped each match. Abilities are not played yet (counted).
