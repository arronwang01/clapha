# Training: settings, design, runs

The reference for how our models are trained: what a training game is and how it matches the live
game, how RL is built and why, what is recorded, and a journal of every run. NOTES.md keeps the
history of findings; this file is the current state. Keep it true when anything changes.

## 1. The training game vs the live game

A training game is the real Clash Royale engine (Null's offline build) stepped as fast as the
collector can go, with every input the model gets built by the same code the live console uses.

| | Live console (MuMu, the user's accounts) | Training game (engine) |
|---|---|---|
| Game code | official client, build 160402012 | Null's offline build: libg SHA 110aa2b5..., content 15.535.86 |
| Cards | 152 cards | same 152 ids, names and elixir costs; MinionGiant (new in 160402012) absent. **Unit stats (HP, damage, speeds) between 15.535.86 and 160402012 not yet compared** (open item) |
| Levels | whatever the two accounts have | the replay's tags: mostly level 16 cards and King (55/60 sampled deals), some 11 |
| Tower troops | per account | the replay's: mostly Princess, some Cannoneer / Dagger Duchess |
| Decks and deal | as played | real games' decks and starting hands (runs/conv-hog26: games with a Hog 2.6 side) |
| Decision cadence | every 5 ticks (4 per second) | the same |
| Observation | memory reader -> firstlight_obs.build | engine snapshot -> il/samples.reader_frame (the reader's frame) -> the same firstlight_obs.build |
| Own hand / elixir | as the screen shows them | the same rule (a played card leaves the hand at its tap; elixir as of the tap) |
| Pending cards | ours from our taps; theirs once in the command queue | ours from our commands; theirs once they would be visible (lead drawn from the measured queue-lead table) |
| Arrival of a pending card | il/flight.py table | the same table |
| Opponent elixir | exact (memory) | exact (engine) |
| Hero / champion states | both players' controllers (memory) | both players' controllers (engine), same encoding |
| When a play lands | held so it lands TARGET_DELAY = 26 ticks after its moment (moment = decision tick + the model's 0-4 tick offset); later only if the phone is slower | the same: tap at moment + 6, executes 21 ticks later (replay tick = moment + 26), later only if the drawn pipeline time is longer |
| Taps that fail | possible (adb) | never |
| Opponent | a person (or the other console) | a model (see the league, section 3) |
| Network | online, friendlies / Training Camp only | offline: the VM's firewall is up before the game starts; no account |

Open fidelity items, in order of expected effect: (1) the content gap 15.535.86 -> 160402012;
(2) the opponents are models, not people; (3) live taps can fail or lag more than the held 26 ticks.

**The content gap, measured 2026-09-27** (il/content_diff.py: every named entry of both versions'
logic tables, gameplay numbers only -- schema changes such as Damage -> { BaseDamage = ... } or
HitsAir -> Filter are not counted): of ~14,600 shared entries, 34 numbers differ -- about one
balance update. For the Hog 2.6 deck: **Ice Golem HP 514 -> 480 (level 1), its death slow 2.0 ->
2.5 s; Fireball's tower damage -75% -> -77%**; everything else in the deck is identical. Elsewhere:
Freeze 4.0 -> 3.5 s, Vines 2.0 -> 1.4 s, Lightning 460 -> 500 ms, Poison tower -77 -> -78%;
Electro Giant 64 -> 72 dmg, Wizard load 1.0 -> 0.9 s and 110 -> 119, Ronin faster, Archer Queen 88 ->
91, Merge Maiden 121 -> 125; Bomber 88 -> 83, Goblinstein 932 -> 875 HP, Mini Zap Machine slower;
evo Electro Giant 7 -> 4 cycles, evo Zap Machine 6 -> 4; evo Barbarians nerfed; hero abilities
tweaked. Missing from the engine: Minion Giant, evo Ice Spirit, evo Skeleton Balloon, the Skeleton
King rework. A newer engine cannot be built within our rules: the current libg is encrypted in the
x86_64 build too (5 section headers, loader segments, code entropy 8.0 from MB 7 on, 85 strings in
25 MB), like the ARM64 one. The data is not encrypted; the engine's own update files use the same
TOML shape, so the few values could be carried over if the engine accepts edited content (it
checks file fingerprints -- untested).

## 2. Throughput: where a training game's time goes

Measured 2026-09-27 on the 4080 PC (i7-12700KF 12 cores / 20 threads, 64 GB, RTX 4080 SUPER,
which another user's YOLOv8 job was using at ~79% and 13 GB) with one collector:

| per game (108 s) | seconds |
|---|---|
| model decisions (2,418 forwards, one decision each) | ~80 |
| building observations (firstlight_obs.build, tensorize, extras) | ~25 |
| the engine itself | ~6 |

One forward of our model on the 4080 costs about the same whatever the batch (il/bench_forward.py):

| decisions per forward | 1 | 4 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| forward + copy (ms) | 33 | 32 | 34 | 35 | 40 | 47 |
| per decision (ms) | 33 | 8.0 | 2.1 | 1.1 | 0.62 | 0.36 |

(The PC's CPU with 2 threads: 33 ms per decision at batch 1, 9.6 at batch 16. The Mac's CPU:
22 / 8.7 ms at batch 1 / 64.)

So the engine is not the bottleneck, deciding one position at a time is. The collector design
that follows (section 3) batches every running game's decisions into one forward, which moves
the limit to building observations: ~12 ms of Python per decision per side, ~30 s of CPU per game.

First server test (8 engines, 8 collectors, eager forwards): ~95 decisions/s, only ~3.8 decisions
per forward at ~35 ms each -- the server itself became the limit (one forward at a time, ~28 per
second, and 8 games whose two sides decide one after the other never fill a batch): ~2.4
games/min, worse than 8 independent collectors. Hence CUDA graphs in the server (a forward is
launch-bound; a captured replay is one launch) at fixed shapes: every observation padded to
effects 16 / relations 64 / candidates 8 (measured maxima in our games: 0 / 46 / 6; anything larger
runs eagerly), batches padded to 8 / 16 / 32 / 64. Checked on the Mac (eager): the learner
reproduces the server's log-probs and values exactly (il/rl_check.py: AGREE, differences 0).

| collection setup (PC, 2026-09-27) | decisions/s | per forward | ms/forward | games/min |
|---|---|---|---|---|
| 1 collector, model in-process (batch 1) | ~22 | 1 | 33 | 0.55 |
| 8 collectors, models in-process | | 1 | 33 | ~4.4 (est.) |
| 8 collectors + 1 server, eager | 95 | 3.8 | 35 | 2.4 |
| 8 collectors + 1 server, CUDA graphs | 180 | 2.8 | 9.7 | 4.5 |
| 16 engines, both sides in one request, 1 server | 285 | 7.8 | 13.5 | 7.1 |
| 16 engines, 3 servers (YOLO busy on the GPU) | ~215 | ~3.3 | 35-40 | ~5.3 |
| 16 engines, 1 server, flat-buffer graphs (YOLO at 81%) | 505 | ~5 | 8-10 | **12.5** |

(games/min = decisions/s / 2,420 decisions per game.) With 16 engines the PC's CPU was at 35%: the
single server's own Python (unpacking, padding and stacking ~200 small tensors per request, copying
them into the graph's inputs, splitting the outputs) was the limit, not the collectors or the GPU
(25% busy). Fix: a collector pads its input to the graph's fixed sizes and sends it as one flat
byte buffer (il/pack layout); the server copies a batch's buffers into one pinned block, one copy
to the GPU, and the captured graph cuts the fields out and casts them on the GPU. Checked on the
PC: the learner reproduces the graph-collected log-probs within 0.0015 (ratio 0.9985-1.0009,
GPU vs CPU rounding) and the values exactly. At 12.5 games/min the CPU is at ~67%: the collectors'
own work (building observations, ~12 ms per decision) is the next limit. Engines: VM `crengine12` at 16 vCPUs (stage7.ps1; MuMu refused 20 GB memory, error
-106, it stays 12 GB: 16 engines x ~0.59 GB fit).

## 3. RL design

### Collection
- **Engines**: the MuMu Android 12 VM `crengine12` on the PC runs N engine processes (FirstLight's
  cluster APK, our run-lean probe), started one at a time with Android's cached-app freezer off
  (D:\crtrain\engine\stage6c.ps1). An engine needs its own virtual CPU (the probe pins it), so
  N engines = N vCPUs; 8 today, the APK allows 24.
- **Workers** (CPU): each plays games on its engine(s) through il.duel's live path (the console's
  rules above) and asks for decisions instead of running the model itself.
- **Inference server** (GPU): holds every model in the league once, keeps each game's recurrent
  state, and answers all waiting decisions in one forward per model. Target: ~20-25 games/min on
  the PC (from 4.4 today), CPU-bound on observation building.
- Every decision of a trained side is kept: the model input, the sampled action, its log-prob,
  the value (one lane per side per game, il/rl.py). Checked: at the collection weights the
  learner reproduces the recorded log-probs to 1e-5 (ratio 1.000) and the values to 3e-5.

### Sharing the PC's GPU
The 4080 is shared with another user's jobs (a YOLOv8 training holding ~3-12 GB and up to ~95% of
the GPU while it runs). First rule (the user's call, 2026-09-27: "when it's not training it's our
turn"): il/rl_turns.py reads Windows' per-process GPU memory every minute, starts our run once other
processes have held under 1.5 GB for 3 minutes, and stops it within a minute of someone else taking
memory; the run resumes from latest.pt. That evening the YOLO job had trained nonstop for 14 h and
pilot1 had not played one game, so the user chose to **share at full speed**: our run goes on next
to it and backs off only while other processes hold more than 7 GB (`--others-gb 7 --calm 2`).
Their job cycles ~18 min at ~3 GB (training, ~67% of the GPU) and ~5 min at ~12 GB (validation,
presumably): 73% of 199 readings left room for us. The log (runs/rl/<run>/turns.log) is the
record of when the GPU was ours. First try next to it (16 engines, 64 lanes per update, 8 per
minibatch): the learner's first update ran the GPU out of memory (~9.5 GB free), and the PC's RAM
ran out (their job holds 15 GB plus 11 loader workers of ~0.8 GB; Windows does not overcommit,
~20 GB of commit was left for us) -- collectors died of MemoryError. Now: 12 engines, 32 lanes per
update, 2 lanes per minibatch, one pass (epoch) per update, collectors pause at 64 waiting lanes;
the learner waits and retries when the GPU is full, a collector drops a game it cannot save instead
of dying. Next to their job the learner is the slow part: its first 32-lane update did not finish
in the 5 min before their 12 GB stretch stopped the run. A bigger page file on the PC (the user's to change)
would give more room. Inside a run: the learner updates only with 5 GB free (--min-free-gb) and
gives its cached memory back after every update; collectors pause while 192 lanes wait
(--max-backlog). Our engine VM itself holds ~1.6 GB of GPU memory (MuMu's renderer). Several
servers at once made things worse while the GPU was busy (35-40 ms per forward, 15.7 of 16.4 GB
used): one server.

### Learning (il/rl_learn.py)
PPO: clipped ratio 0.2, clipped value 0.2, entropy 0.001, GAE (gamma 0.999 per decision, lambda
0.95), AdamW, grad-norm 1.0, anchor to the starting model 0.3 x 0.5 (log ratio)^2. Since
2026-09-29 (pilot1's settings in brackets):
- **one optimizer step per update** (32 lanes, ~38,000 decisions), the gradient averaged over every
  decision [a step per 32-decision chunk of 2 lanes: ~600 steps per 16 games]; lr 1e-5 per step;
- advantages normalized over the whole update [per 2-lane minibatch];
- Adam's state kept across restarts (optimizer.pt) [a fresh Adam at every restart, every ~4 updates];
- the first 16 updates train the value only, a step per 2 lanes [4]: FirstLight's value head
  learned their shaped reward, not ours;
- **grad agree** (learn.jsonl `grad_cos`, status.txt): cosine between the gradients of the two
  halves of an update's games. It is the signal-to-noise check: about 0 = the update is noise, and
  more games per step are needed before more steps can help; clearly > 0 = the games agree.
32 lanes per update, 2 per minibatch (memory, next to the other job); lanes older than 2 updates are
dropped. The collectors reload the latest weights after every update. Resumable (latest.pt,
optimizer.pt).

Checked on the PC (value-only updates, where the policy is exactly the one that played): clip
fraction 0.0001-0.0004, approx KL 9e-5 -- the learner (bf16, train mode) and the collectors
(fp32) agree; the mismatch is not why pilot1 got worse.

### Reward (per decision of the trained side; the user's, 2026-09-26)
- terminal: win +1, loss -1, draw 0
- princess towers: +0.3 x (share of a princess tower's health dealt) - 0.3 x (share taken)
- King Tower: nothing for damage; -0.2 when **our spell** is the first thing to damage their King
  Tower while both their princess towers stand (activation). Dated by the spell's landing cell and
  flight time (il/flight.py) or a spell object on it. An activation the opponent engineers is not
  charged; it shows in the result.
- Every component is logged per game, so its effect on behaviour can be read off (section 4).

Measured before any RL (20 engine games, v2 vs no-delay hog2): **both models activate the enemy
King with their own Fireball in 80-85% of games** while both princess towers stand -- FirstLight's
reward never charged it. The activation term targets exactly this.

### The league (who the trained model plays)
| opponent | why | deck |
|---|---|---|
| itself (latest) | self-play, both sides trained | Hog 2.6 mirror |
| past snapshots | against forgetting and cycling | Hog 2.6 mirror |
| v2 (fixed) | the running score against where we started | Hog 2.6 mirror |
| fl:hog2 with no delay | the strongest 2.6 player we have (the distillation teacher) | Hog 2.6 mirror |
| fl:general, no delay | real games' opponent decks: air units, other win conditions, a mirror never shows them | real meta decks |

Next for "all sorts of players": a **field model** -- FirstLight's General trained by imitation on
the real opponents' sides of our replays (people playing meta decks, through our live path with
the delay), then kept current by RL in the league. Our trained model stays Hog 2.6.

### Evidence before scale
GCP only after the PC shows that more games make the model better: a pilot of a few thousand
games, judged by (1) the running score against fixed v2 (sampled play; 400 games at 58% is
p < 0.001), (2) milestone benchmarks against no-delay hog2 and General on real decks, (3) the
behaviour metrics moving the way the reward intends (activations down, no new bad habits).

### Unattended (the keeper, il/rl_turns.py)
Days without anyone watching: the keeper runs the turns above and also
- restarts the run when a part of it ended (the server, the learner, a collector -- a collector
  ends itself after 5 failed games in a row) or nothing happened for 20 min (no game, no update);
- before every start asks each engine port for its status: fewer than 3 in 4 answering -> restarts
  the engines (D:\crtrain\engine\stage6c.ps1); the VM not answering (a reboot) -> restarts the VM,
  puts the offline firewall back and checks it, then the engines (stage7.ps1); starts on the
  engines that answer;
- at most one restart per 10 min; after a reboot, clears the old pids first (a stale id could be
  the other user's process; rl.ps1 -Stop also only kills processes whose command line is this run's);
- disk: under 30 GB free, deletes recordings beyond the newest 200 and policy files other than every
  50th and the league's snapshots; under 8 GB stops the run;
- writes **status.txt** every 10 min: GPU hours ours, games, updates, restarts, disk, the report.
On the PC (clapha-train folder): `start-training.cmd` (safe twice: one keeper per run, a lock file),
`stop-training.cmd` (keeper and run). The Startup folder has clapha-training.cmd (the user's OK,
2026-09-27), so after a reboot and login everything comes back by itself. The learner keeps
policy-NNNN.pt every 10 updates (~50 MB each), no longer every update.

## 4. What is recorded, and where

Per run folder `runs/rl/<run>/` on the PC:
- `games.jsonl`: one row per game: opponent, deals, result, crowns, tower health, each side's reward
  components (princess damage dealt / taken, King activation tick), steps, seconds, weights version.
- `learn.jsonl`: one row per update: losses, entropy, KL to the start, clip fraction, value error,
  mean return, wins in the batch.
- `policy-NNNN.pt` every 10 updates, `latest.pt`; logs per process (rewritten at each start).
- `status.txt` (every 10 min), `turns.log` (a line per minute, every keeper action), `launch.log`,
  `engines.log` (what the start and engine scripts printed).
- Behaviour metrics per checkpoint (il/habits.py and the activation check): Log targets, Ice Golem
  use, King activations, princess damage dealt / taken. To add: elixir leaked at 10, spell value,
  card use, placements.

## 5. Journal

- **2026-09-29** **Why pilot1 got worse: steps, not reward.** The learner/collector agreement holds
  on the PC (clip fraction ~0.0002 in value-only updates). The first policy update alone moved the
  policy to 6.5% clipped / KL 0.02: pilot1 took ~600 Adam steps per 16 games, each on 2 lanes x 32
  decisions. With one +-1 result spread over ~1,200 decisions, such a step is almost pure noise, and
  Adam turns noise into full-size moves; the model random-walked away from v2 (entropy up, every
  fixed score down) while the reward's easy terms (King activations) still moved the intended way --
  so the reward reached the model, the steps drowned it. Fix: one step per update over all its
  games, advantages normalized over the update, Adam kept across restarts, and grad agree logged to
  measure whether an update's games carry a direction at all. pilot2 at that point: 88 games, 2
  updates, 27 restarts in 6.7 h (the learner ending; learn.err to read).

- **2026-09-28** **pilot1 got worse than its start.** 88 updates, 2,131 games. Learner vs v2: 46%
  in updates 0-9, then 32% +-5 (97-206) from update 10 on. vs hog2 35% -> ~16%; vs General
  42% -> ~18%. Newer versions lost to their own snapshots. It drifted far from v2 (sampled KL
  0.05 at update 10 -> 0.2-6, one estimate of 550) and got more random (entropy 0.20 -> 0.36).
  The easy reward terms moved first: King activations 59% -> 30%, damage dealt 1.24 -> 1.1.
  Likely causes: steps too big and noisy for the signal (a step per 64 decisions at lr 1e-5,
  after the memory cuts), a value head with 4 warm-up updates, and an anchor estimate that
  explodes. **pilot2** (from v2 again):
  - lr 3e-6, one step per 4 chunks;
  - anchor 0.3 x 0.5 (log ratio)^2;
  - 16 value-only updates;
  - the keeper halts the run once 150 recent games against v2 score under 42%;
  - sharing line 9.5 GB (their job now sits at 7.6 GB);
  - running scores in status.txt.

- **2026-09-27 evening** pilot1 had not played a game: the YOLO job trained all day (see sharing).
  The user chose to share at full speed and to have the run come back after reboots. The keeper
  (il/rl_turns.py) now also restarts broken parts, engines and the VM, guards the disk, writes
  status.txt; start/stop scripts; collectors get a fresh seed per start (the run restarts every
  ~23 min with their job's cycle, and a fixed seed replayed the same first deals). The user is
  away ~4 days: this is the first long unattended run.

- **2026-09-27 afternoon** Collection rebuilt around a GPU inference server: 0.55 -> 12.5 games/min
  on the PC (section 2), every step checked for learner agreement. 16 engines (VM 16 vCPUs,
  stage7.ps1; a MuMu shutdown hung on its window process and was cleared by hand). Pilot `pilot1`
  set up under the turn-taking supervisor (league self 4 / snapshots 2 / v2 anchor 2 / hog2 no-delay
  1 / General real decks 2); it waits for the other GPU job to finish.

- **2026-09-27** Best model so far: v2 (clapha:v2 in the console). vs the previous clapha model
  15-5; vs no-delay hog2 8-12 at lead 9 and 8-12 at the console's exact timing (target 26). RL
  pipeline checked end to end on the Mac engine (collect -> PPO update -> resume). PC: 8 engines up
  (freezer off, one at a time); one self-play game 108 s; decision cost measured (section 2).
  Habits (engine games): Log never on air-only targets (0 of 175 in mirrors, 0 of 56 vs real decks);
  Ice Golem on lone building-targeters 10% (teacher 12%); King activations 80% of games (teacher 85%).
