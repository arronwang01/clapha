# clapha — live reader for Training Camp (build 160402012)

Scope agreed: Training Camp vs. the Royal Trainer only. Read-only memory + later `input tap`.

## Setup facts
- MuMu Pro on M2, adb at `127.0.0.1:5555`, adb shell is already root (no `su` needed).
- Screen 1440x2560. ABI arm64-v8a.
- Game versionCode `160402012`, libg.so SHA-256 `aec6cc5e473edc809f10c157ae553dbbb8724d816f34148344b7f94a67e5a97c`
  (copy: `libg-160402012.so`). Repo offsets are for `160402002` — do not apply as-is.
- NDK: `~/Library/Android/sdk/ndk/28.2.13676358`, compiler `aarch64-linux-android31-clang`.
- libg layout (live): rw segment at `+0x197d000..+0x1a28000` (file off), `[anon:.bss]` right after
  (~`+0x1a30000`, 0x85000 bytes), rwx segments at file off `0x1ab0000` and `0x1adc000` (likely packed/self-modifying code).

## Tools (build/ -> pushed to /data/local/tmp/)
- `find_manager PID [CTX_MIN CTX_MAX]` — scans libg rw + .bss for slot->root->+ctx->+0x90->+0x60 tick chains.
- `tick_scan` — author's `mumu_arm64_tick_scan.c` unchanged (delta 150–400).
- `tick_scan_20hz` — same, delta window 15–25 per 1 s (`src/tick_scan_20hz.c`).
- `live_sampler` — author's `mumu_live_private_sampler.c` unchanged; args `PID INTERVAL_MS MANAGER_RVA ROOT_CONTEXT_OFFSET [--unified N]`.

## Findings so far
- Author's chain (global -> +0x28 -> +0x90 -> +0x60) does NOT hold in 012: 770 chain-shaped pointers in libg rw/.bss,
  none with an advancing tick for ctx offsets 0x08..0x100.
- Author's 150–400 tick scan: 646 hits, none inside an object with a libg vtable — noise.
- 20 Hz scan in a live battle (PID 12264): 106 hits, none with libg vtable at object start (scanner only looks back 0x300).
  Best tick candidates (value ~1080 = ~54 s into the match at 20 Hz):
  - `0x76d3075784` 1079 -> 1096
  - `0x76d3076b1c` 1081 -> 1100
  Heap addresses change every launch; re-scan each session.
- MuMu is unstable: repeated SIGSEGV SEGV_ACCERR (fault addr ending `...874`) across many system processes and the game.
  Not caused by our reads. Restarting MuMu may help.

## Pointer tagging (found after session 1)
- Android heap pointers carry a top-byte tag (e.g. `0xb4...`). Session-1 `find_manager` rejected any pointer
  >= 0x800000000000, so tagged roots in .bss were silently discarded — likely why 0 of 770 chains ticked.
  Fixed: `untag()` masks with `0x00FFFFFFFFFFFFFF` before validating/reading. Rebuilt, not yet re-run.
- The author's `live_sampler` (`read_exact`) does not untag either. If chains are tagged on this MuMu,
  patch it the same way before running.

## RESULT (session 1, late): manager RVA for 160402012 = 0x1a57e88
- Tag-aware `find_manager PID 0x8 0x100` hit: `rva 0x1a57e88`, ctx `+0x28`, root `0xb4000078355dbd30`,
  battle `0xb40000786562bef0`, tick 2983 -> 3005 in 1 s. (Author's 002 RVA was 0x1a569a8; shift +0x14e0.)
  Second hit `rva 0x1a918b8` ctx `+0xc8` reaches the same battle object (alternate path, untagged root).
- Chain offsets +0x28 / +0x90 / +0x60 unchanged from the repo.
- `live_sampler_tbi` (= author's sampler + untag in `read_exact`) with `0x1a57e88 0x28` resolves root/context/battle,
  but `players_unresolved` — screenshot showed the lobby, so no live player state. Needs a Training Camp battle to verify.
- Author's MuMu (Windows) ran ARM64 via translation with libg at low addresses (0x301c000) and no pointer tags;
  Mac MuMu Pro is native ARM64 with tagged heap pointers — that's the environment difference.

## Mac/012 profile (faithful port of the author's pipeline)
- `mac012/mac_profile.py` — imports `native_core.mumu_live_protocol` unchanged and rebinds only:
  VERSION_CODE 012, new SHA, MANAGER_RVA 0x1a57e88, serial :5555, reader path, and
  `root_command` -> identity (adbd is already root here; `su` is absent).
- `mac012/run_probe.py` — screenshot-first Training Camp gate, then calls the author's
  `mumu_live_probe.main()` untouched. Without `--confirmed-training-camp` it writes the
  screenshot to `artifacts/mode-checks/` and exits 3.
- `bindings-mumu-live-160402012-arm64.json` — binding for this build; marks which offsets are
  verified (manager/ctx/battle/tick) vs. still unverified (player fields, entities, calibration).
- Verified against the lobby: `verify_runtime` passes (version, ABI, libg SHA all match) and
  `install_reader` pushes `build/live_sampler_tbi` to `/data/local/tmp/mumu-live-reader-v2`.
  Resource version here is 16.402.13 (author saw 16.402.7).

## READING VERIFIED (Training Camp, 20 s, 200 frames @100 ms)
- `artifacts/probes/20260924T004748196819Z/` — 200/200 frames coherent, 198 `live_candidate`, 2 `warming_up`.
- Tick 622 -> 1021 in 19.96 s = **19.99 ticks/s**. Read latency p50 280 us, p95 466 us.
- Chain path `[0xA8]` on all 200 frames; no fallback discovery. All upstream offsets hold on 012.
- Cross-checked against screenshots: own king tower hp 108 == on-screen 108; princess 1890 == 1890;
  destroyed bottom-left princess absent from entity list; elixir raw/10000 == on-screen elixir;
  hand deck indices resolve through deck_card_ids to the right cards (28000000 = Fireball cost 4,
  26000002 = Goblins cost 2).
- Local player is **side 1** (high y). Entity count 9-19 during play.
- **Opponent (side 0) hand is hidden — all -1 in every frame.** No hidden-information advantage on this build/mode.

## Cross-check against the user's own prior notes (cr-engine-extraction, Null's 15.535.13)
Source: `~/Documents/GitHub/cr-engine-extraction` (`CR-DATA-NOTES.md`, `handoff/NULLS-LIVE-READER.md`).
Different client/version, but the *semantics* are client-independent and every one matched our reads:
- x 0..18000, y 0..32000, 1 tile = 1000; tick 20/s; elixir int32 x10000 (max 100000); hp with max_hp. OK
- Tower coords: king (9000,3000)/(9000,29000), princess (3500,6500)/(14500,6500)/(3500,25500)/(14500,25500). OK
- card_id = table*1e6 + row; 26 troop, 27 building, 28 spell. Towers card_id -1, native ids 5000000-5000005. OK
- Hand/cycle hold deck slot indices 0..7, not card ids; next = cycle[0]. OK
- Destroyed tower disappears from the object list (keep as rubble, hp 0). OK (our bottom-left princess)
- Opponent's *selected* card is never in the simulation, so it cannot be read. Matches side 0 hand = all -1.
Copied `live_card_catalog.json` (152 cards, names/elixir/type) into clapha; verified our read hand
[7,5,4,2] -> Skeletons(1), Valkyrie(4), Fireball(4), Goblins(2), matching the on-screen costs 1/4/4/2.
Their deployment note: release ~45 px lower than target at 1080x2400, because the card lands about one
tile above the finger. Upstream ScreenLayout encodes a one-tile shift too (`.685/32`) but in the other
direction — do NOT stack both blindly; verify the arena box first, then test one stationary building.

## Calibration (no taps sent) — arena box confirmed
`mac012/calibrate.py` maps memory coords onto a screenshot with the upstream ScreenLayout.
At 1440x2560 (exactly 9:16, same as the author's 1080x1920) the fractions scale with no edit:
  arena box (79, 214) -> (1361, 1968); hand slots x = 446/720/994/1267, y = 2278
  tile size on screen: 71 px wide, 54.8 px tall
Predicted tower centres: enemy king (720,378), my king (720,1803), princess (328,570)/(1112,570)/
(328,1611)/(1112,1611). Zoomed crops: all six markers land inside the tower footprint, consistently
a little low and right of the sprite centre (< 1 tile) — expected, since sprites are drawn in
perspective while the logical position is the footprint. Good enough to test a real placement.
Unit markers also tracked correctly (Skeletons, Tombstone, Bomber all on their sprites).

## TAP TEST PASSED (Training Camp, 2026-09-24)
`artifacts/action-smoke/20260924T005958031112Z/` — sent 3, confirmed 3, 1467 frames,
stop_reason `battle_ended_or_inactive`, `passed: true`, `game_memory_written: false`.
- Cannon, hand slot 3, requested cell 170 (canonical 8500,9500). Native read-back (9500,22500);
  derotated = **(8500, 9500), exact — zero error**. Upstream's own best was 1 native unit off.
- Skeletons, cell 155: first skeleton derotated (11501, 9308) — x exact, y 0.8 tile off because
  three skeletons spawn spread around the point and had moved by the time of the receipt (25 ticks).
- Fireball, cell 150: no new entity, as expected for a spell.
- Receipt latency 1.21-1.36 s (deploy + animation), matching upstream's 1.2-1.35 s.
- Screen calibration therefore needs **no change** at 1440x2560, and the user's "+45 px lower"
  note from the 1080x2400 device must NOT be added on top — upstream's `.685/32` shift already
  covers it here. Binding's `screen_calibration` can be marked verified for this resolution.

## Command queue + server lag (from the user's prior notes) — present on 160402012
Key mapping: **their "manager" is our `battle` object** (their world at manager+0xa8 == our
player_state at battle+0xA8). So their manager+0x38/+0x60/+0x64 are our battle+0x38/+0x60/+0x64.
`src/queue_probe.c` (read-only, untagged) confirms on this build:
- `battle+0x60` simulated tick 2040, `battle+0x64` = 2059 -> **gap 19 ticks (0.95 s)**, inside their
  stated 11..32 (median 17). This is the client's deliberate jitter buffer: the hand we read is ~0.85 s old.
- `battle+0x38` -> object with count at +0x14; count 0 while idle (battle had ended, tick frozen).
  Entry layout to verify with a live play: +0x10 issue tick, +0x18 account lo32, +0x28/+0x2c x/y,
  +0x38 LogicData (card id at +0x40), +0x50 sequence. Do NOT filter on +0x04 (their note: 111 vs 121).
- Other header fields seen: +0x30 = 5, +0x68 = 1000, +0x70 mirrors the tick, +0x74 = 544.
Use: confirm *our own* plays from the queue (exact x/y, works for spells and multi-unit cards, ~0.65 s
sooner than the board receipt) and gate re-decisions while our own command is still pending.

### Command queue decoded on 160402012 (live, read-only)
`build/queue.jsonl` — 120 samples @100 ms during a real battle. Entry layout from the user's notes
holds exactly: +0x10 issue tick, +0x18 account lo32, +0x28/+0x2c x/y, +0x38 LogicData (+0x40 card id),
+0x50 sequence. Observed commands:
  Bomber  (5499,19499) issue 365 account 79227807 seq 5  visible ticks 372..384
  Minions (3500,14500) issue 401 account -1       seq -1 visible ticks 412..420
  Knight  (2499,20499) issue 503 account 79227807 seq 6  visible ticks 511..523
  Bomber  (3499,14500) issue 521 account -1       seq -1 visible ticks 531..541
- **Own vs opponent is trivially separable**: our account_lo is 79227807 with an incrementing seq;
  the trainer's commands carry account -1 / seq -1.
- Commands sit ~7-12 samples (~0.6-1.0 s) in the queue before the unit exists.
- Client lag `0x64 - 0x60`: min 6, median 13.5, max 19 ticks (median 675 ms).
- **Tap collision**: the smoke sent Cannon cell 170 at tick 503, but the queue shows a Knight from
  our account at tick 503 — the user was hand-playing at the same time. Tap tests need hands off.

### Viewer
`mac012/viewer.py` — read-only local HTTP viewer on http://127.0.0.1:8777. Spawns the sampler,
serves `/state` (to_state-shaped) and draws the board in a canvas: local player at the bottom,
towers/units with hp bars and card names, own hand + next card + elixir, trainer's elixir only
(their hand is hidden and stays hidden). Geometry checked: my king (198,638), enemy king (198,66)
on a 396x704 canvas. Restart with `python3 mac012/viewer.py`.

### UPSTREAM BUG FOUND: side-0 x mirroring (measured, then fixed)
Upstream `ScreenLayout.deployment_point` mirrors the canonical x fraction only when `side == 1`;
the author only ever verified side 1. The user's prior project had found the opposite for owner 0.
Settled by measurement in a friendly battle with local side 0 (`mac012/mirror_test.py`):
  requested canonical (3500, 9500) -> landed native (14500, 9500)   # y exact, x mirrored
After `mac012/layout_fix.py` (mirror for BOTH sides), same cell and side:
  requested canonical (3500, 9500) -> landed native (3499, 10307)   # x 1 unit off; y drift = unit walked
Reasoning in canonical space, both sides need `x_fraction = 1 - canonical_x/18000`:
  side 1: canonical_x = 18000 - native_x and the camera keeps native x
  side 0: canonical_x = native_x and the camera mirrors it
Equivalent to the user's native-space rule (x' = 18000 - x for owner 0, x' = x for owner 1).
Fix is applied by importing `mac012/layout_fix.py`; wired into mirror_test and run_action_smoke.
The viewer's `place()` now uses `fx = 1 - cxn/NW` for both sides too.

### Friendly battle (local side 0) — what it proved
- Opponent verified as the user's own second account: 0 vs ~600 trophies (unmatchable), both sides'
  towers level-normalised (king 4824, princess 3052 on BOTH sides), and neither player placed a card
  for ~2 minutes. Training-Camp-only scope treated as satisfied by "no third party involved".
- **Opponent deck is NOT readable even against a real account**: side 1 `deck_card_ids` stayed empty
  and hand stayed [-1,-1,-1,-1] for the whole match. Same as the trainer. So no opponent-deck feature
  is available to feed a model, whoever the opponent is.
- `mirror_test.py` now picks the Cannon if it is in hand, else any affordable troop/building —
  with nobody playing, the cycle never rotates, so waiting for a specific card times out.

### CORRECTION: revealed cards ARE readable for BOTH players (player+0x288)
The user was right that deck data is available; my earlier "not readable" was wrong because every
sample until then came from matches where nothing had been played.
`player+0x288 .. +0x2a4` = card ids in first-play order, -1 unused. Measured live:
  player0 (us):       MiniPekka, Archer, Bomber, Goblins, Knight, Skeletons
  player1 (opponent): Archer, Giant, Minions, Knight, MiniPekka, Musketeer, Arrows   <- 7 of 8
This is *revealed* cards, i.e. cards you watched them play, so it is not hidden information;
it is ordinary deck tracking. Added to `queue_probe` so it streams continuously.
Still NOT available for the opponent on this build: the live hand (array exists, ptr non-null,
cap 4, len 4, but reads [-1,-1,-1,-1] in every sample) and the cycle (null pointer).
Opponent **elixir** IS exact and readable (+0x2f8) — 200/200 frames showed the two sides differing
(e.g. side0 2.07 vs side1 10.0), so this one is genuinely hidden info in a normal match.
The user's Null's 15.535.13 offsets (hand +0x220, cycle +0x230) are shifted 0x10 from this build's
(+0x210/+0x220) — different client version. Null's is a private server and likely sends full state.

### Gentle collection: `src/snapshot.c`
One chain walk, then whole regions copied in single preads (battle, world, both players, player_root,
owners, owner entries, avatars, hand/cycle payloads) — ~15 reads total instead of hundreds per second.
Everything else is analysed offline on the copy. Output is one JSON with hex blocks + a `notes`
field describing the chain and confirmed offsets; `share/snapshot-160402012.json` was sent to the
user for others to analyse.

### Bot console (`mac012/console.py`) — http://127.0.0.1:8777
One page: live board + bot panel (model radio, arm checkbox, start/stop, status, scope, log).
Reuses the viewer's frame/queue pumps, so it also records to `artifacts/viewer-sessions/`.
- `mac012/model_adapter.py` bridges our frames to the user's `sim_engine.Adapter`
  (`~/Documents/GitHub/cr-sim-upstream/replay`). Models: 14k/40k/100k; 100k carries a calibrated
  threshold 0.3629. Verified offline on recorded frames: 56 decisions -> 15 plays (~27%, close to
  the calibrated human rate), e.g. Cannon at tile (8.5,9.5), Skeletons at (3.5,14.5).
- **Important:** `sim_engine.tensors()` reads `them['hand_deck_indices']` — these models were
  trained **with the opponent's hand as an input feature**. Unreadable on this build, so we feed
  the vocabulary's unknown token (0) for all five opponent slots and an all-unknown opponent deck.
  The models therefore run somewhat out of distribution here.
- Bot loop mirrors the user's `bots.py`: decide every `STEP_TICKS` (10 ticks), one play in flight,
  never decide while our own command is still in the queue (the ~0.85 s lag trap), full 4-card hand
  required, deck slot -> hand position via the live hand order.

### Scope gate (`mac012/scope_gate.py`) — a check, not a promise
The bot reads BOTH players' account ids live (queue_probe now emits them from
player+0x10 -> context+0x98 -> root, root+0x30 + side*8 -> avatar, +0x00/+0x04 = hi/lo) and
refuses to tap unless the opponent is either a bot/trainer (account <= 0) or listed in
`allowed_opponents.json`. Re-checked on every decision, not just at start; a block forces dry run
and logs `SCOPE BLOCK`. Adding an opponent is a file edit on purpose, not a one-click button.
Tested: own second account -> allow; trainer (-2) -> allow; unknown account -> refuse;
unreadable -> refuse.

### Not doing (decided 2026-09-24)
- No code injection into the official client: no touch interceptor via `GameApp.nOnTouchEvent`,
  no `eglSwapBuffers` GOT hook for an in-game HUD, no patched-APK hooks. Ordinary `input tap` only
  (measured 1.2-1.36 s end to end, Cannon exact). Upstream's contract is the same: reads only.
- No opponent look-ahead from the queue (the 0.6-0.8 s landing warning). Useless against the
  Royal Trainer and only meaningful against a human.

## Tap test tooling
`mac012/run_action_smoke.py` = Training Camp gate + the author's `mumu_live_action_smoke` unchanged:
bounded plays, each needing a receipt (chosen slot rotated AND elixir dropped) before the next,
no blind retries, auto-stop at battle end. `--execute` additionally requires `--confirmed-training-camp`.
Author's test-card set is Cannon/Hog/IceSpirits/Skeletons/Musketeer/IceGolem/Fireball/Log.
Current deck overlaps only on **Skeletons and Fireball**; both are non-stationary, so coordinate
read-back is imprecise. **Put a Cannon in the deck** for the placement check — that is the stationary
building the author used to verify landing cells.

## Next steps
(old) In a Training Camp battle: `live_sampler_tbi PID 500 0x1a57e88 0x28 --unified 20`; if players still unresolved,
   check battle+0xA8 and player fields (+0xE0/+0xE8, +0x2F8 elixir, +0x210 hand) by hand.
(old) Push the rebuilt `find_manager` and re-run it first (`find_manager PID 0x8 0x100`) — cheapest test of the tag theory.
1. In a fresh battle, run `tick_scan_20hz` and keep hits whose value ≈ elapsed_seconds × 20.
2. For those, widen the look-back for the owning object (vtable in libg), or search libg .bss / heap for pointers
   into that region to rebuild the chain back to a static global -> new manager RVA + offsets.
3. Verify player fields (+0x2F8 elixir, +0x210 hand, +0x220 cycle) against the screen; then run `live_sampler`.
4. Only after reading is verified: calibrate taps for 1440x2560.

### OPPONENT DECK FOUND (Training Camp, 2026-09-24) — method + caveat

**The bug that hid it:** every scanner (mine *and* the repo's `mumu_*_scan.c`) uses
`#define MAX_MAPS 2048`. This device's `/proc/<pid>/maps` has **4841** entries, so the
mappings holding the deck structures were past the cutoff and never scanned. Raising it to
16384 turned 0 hits into 331. Any repo scanner run on this host needs the same change.

**Tool:** `src/deck_vector_scan.c`, written in `mumu_cycle_vector_scan.c`'s idiom — scudo:
mappings only, 4 KB chunks, look for a vector-shaped header, validate the payload. Accepts
`{data, cap, size}` and `{data, ..., count}` (the owner layout: entries +0x20, count +0x2c),
and two payload forms: eight flat card ids, or eight entry pointers resolved
`entry+0x10 -> data -> +0x40`.

**Identifying which deck is whose:** the scan returns ~320 8-card groups, mostly UI lists
(shop offers, suggested decks, card categories). The opponent's real deck must contain every
card they have revealed (`player+0x288`). That filter left exactly ONE deck per side:

  side 1 (us)      0x75ba6a6e90  = owner1+0x20, the structure we already knew
  side 0 (trainer) 0x7609f4eab8  Knight, Archer, Goblins, Minions, Skeletons, Bomber,
                                 BattleRam, **Tombstone** <- not yet revealed

**Where it lives:** NOT in the six battle owner slots (owner0's entries are all -1).
`ptr_find --range ... --all` found the referrers: three sibling objects at
0x7609f4e810 / 0x7609f4e930 / 0x7609f4ea50, spaced exactly 0x120 apart (an array of records).
The deck header sits at **object base + 0x68**, count at +0x74. Five distinct pointers
reference the base. Region 0x7609... is separate from the battle owners at 0x75ba...,
consistent with the battle-start/session layer rather than the battle graph.

**Not yet done:** walking from those five referrers up to a stable root, so this becomes a
fixed chain instead of a scan. The game process exited before that step.

**CAVEAT that must be closed first:** this was Training Camp against a scripted bot, whose
deck the client may simply know locally. It does NOT yet show that a *human* opponent's deck
is present. Repeat in a friendly against a real account before believing it.

**Tools added:** `src/deck_vector_scan.c`, `src/ptr_find.c` (exact + `--range` + `--all`),
`src/peek.c` (dump N bytes at an address). `src/heap_deck_scan.c` (my whole-heap version)
was deleted in favour of the repo-idiom scanner.

### CONFIRMED vs a REAL ACCOUNT (friendly, 2026-09-24): deck YES, hand NO

Friendly on two MuMu instances, same build 160402012:
  device 0 (adb 26624 / 5555) rooted, our read target, account 79227807, side 1
  device 1 (adb 26656) not rooted, opponent account 48410163, side 0
MuMu supports this natively: `mumu-cli` has create/clone/open; VM 0 and VM 1 already existed.

**Opponent DECK: present, before a single card was played.**
`deck_vector_scan` found 314 distinct 8-card groups. The opponent's real deck was identified by
matching cards the user named: Valkyrie, Bomber, MegaMinion, **AngryBarbarians** (= Elite
Barbarians, 26000043, 6 elixir), Berserker, GoblinHut, Tombstone, GoblinCurse.
  deck header 0x75ba8aa8f0  -> object base 0x75ba8aa8d0 (entries at +0x20, count at +0x2c)
  NOT one of the six battle owner slots (owner0 base 0x75ba69b8e0, entries still all -1)
  exactly ONE referrer: 0x75ba8970a0  <- next step is to walk up from there to a stable root
This reproduces the Training Camp result against a human account, so it is not a
bot-only artefact. It arrives before any play, consistent with the battle-start message.

**Opponent HAND / card order: NOT present.** Three independent methods agree:
  1. direct offsets: opponent +0x210 = [-1,-1,-1,-1], +0x220 (cycle) null pointer
  2. blind one-level pointer walk from the player object: 32 leaves, no 0..7 quadruple
  3. the repo's own `mumu_cycle_vector_scan`: 47 four-value vectors, exactly ONE complementary
     hand+cycle pair (ours, 0x765a03eaa0/0x765a03eab0, 16 bytes apart = +0x210/+0x220)

**Second bug in the repo's scanners (besides MAX_MAPS):** `rd()` does not untag pointers, so
every payload read fails on this host and the scanner silently reports zero. Patched copy:
`src/cycle_vector_scan_fixed.c` (MAX_MAPS 16384 + untag). Both fixes are needed for any
repo scanner to work here.

### Deck order vs hand order (ground-truth check, friendly 2026-09-24)
Found deck array order: 0 Ebarb, 1 Curse, 2 MegaMinion, 3 Bomber, 4 GoblinHut, 5 Valkyrie,
6 Berserker, 7 Tombstone. User's stated opponent hand order (Hut, Minion, Ebarb, Curse)
= deck slots [4,2,0,1]. So the stored array IS the canonical slot order that hand indices
reference — the mapping is correct, we simply do not have their indices.
Our own side is the same relationship: hand [6,0,4,7] against our 8-card array.

The shuffled order is NOT stored for the opponent:
- their hand array is allocated (cap 4, len 4, live pointer) but reads [-1,-1,-1,-1]
- their cycle pointer is null
- the session object holding their deck has 3 other pointers: entries array (8 card entries),
  a pointer table and a string table — no second 0..7 index array
Consistent with the shuffle being applied server-side and the client never being told the
result; it learns their cards as played, which is what +0x288 accumulates.

### CAVEAT on the deck finding: it may be friendly/Training-Camp specific
The object holding the opponent deck also contains asset filename fragments (".tch", ".png",
"_aca" at +0x54/+0x58/+0xb4/+0xc8) and a vtable at +0x00 pointing just past libg's exec range.
That looks like a DISPLAY record. Friendly battles show both decks in the clan-chat result
card, and Training Camp's trainer deck is local data — so neither case shows that a LADDER
opponent's deck is present. Not tested, and out of scope.

Targeted parent check (18 known objects: player_state, battle, both players, both contexts,
player_root, 6 avatars, 6 owner slots) found NO reference to the deck object, confirming it
sits outside the battle graph.

### Opponent hand: CONCLUSIVE NEGATIVE on 160402012 (needle test, friendly 2026-09-24)
User supplied ground truth mid-battle: opponent hand = Bomber, Curse, Berserker, GoblinHut,
in that order = deck slots [3,1,6,4] against the scanned deck order
(0 Ebarb, 1 Curse, 2 MegaMinion, 3 Bomber, 4 GoblinHut, 5 Valkyrie, 6 Berserker, 7 Tombstone).
Searched for that known needle five ways — all negative:
 1. user's Null's offsets: +0x220 is our CYCLE here, +0x230 is null for BOTH players
 2. opponent +0x210: allocated (cap 4, len 4, live ptr) but reads [-1,-1,-1,-1]
 3. blind one-level pointer walk from the opponent player object: 32 leaves, nothing
 4. repo's mumu_cycle_vector_scan (patched): 47 four-value vectors, none is [3,1,6,4];
    only ONE complementary hand+cycle pair exists and it is ours (0x765a0c8200/0x765a0c8210)
 5. `src/perm_scan.c` inline 0..7 permutations: 102 found, all identity/reverse/rotations
Reconciliation with the user's late friend's demo: the user's own note says that capability was
on **Null's 15.535.13**, a private server, which broadcasts full state (and whose offsets sit
0x10 from this build's). The repo author independently documented the same boundary: opponent
hand hidden live, BOTH hands visible in replays.

### `mac012/cycle_tracker.py` — deduce the hand instead of reading it
The cycle is deterministic (play -> card to back of queue, queue front enters hand), so given
the opponent deck (readable) plus their observed plays (command queue, public), brute-forcing
all 8! = 40320 initial orderings and filtering by consistency collapses to a unique answer.
Reports current hand and next card, flagged certain vs likely with percentages.
Cannot give the STARTING hand before any play — that needs the seed, which the client does not
appear to hold (a blank allocated array is what you get when the input never arrives).

### DEFINITIVE: opponent hand appears AFTER the battle, not during (2026-09-24)
Two rooted MuMu instances, same build, friendly battle, each reading its own memory:
  device 0 = acct 79227807 (adb 26624), device 1 = acct 48410163 (adb 26656)
  `mumu-cli config 1 -s '{"vmRootEnable":true}'` + restart enables root on the 2nd instance.

DURING the battle (both clients, mirrored):
  device 0: side 0 hand [3,7,1,5] (own)  | side 1 hand [-1,-1,-1,-1], cycle NULL
  device 1: side 0 hand [-1,-1,-1,-1]    | side 1 hand [3,2,0,4] (own), cycle populated
AFTER the battle ended (DRAW screen, tick frozen at 6150), device 1 shows BOTH:
  side 0 [3,7,1,5]  <- the opponent's real hand, same +0x210 that was blank during play
  side 1 [3,2,0,4]
So the client is sent the full battle record at settlement (result screen + replay), which
is where the data comes from. Matches the repo author's evidence table
("实战结算后...双方手牌显示") — now reproduced directly rather than taken on trust.

Searches for the hand DURING play, all negative, with ground truth verified from device 1:
  raw slot indices, adjacent int32, all writable maps  -> 0 (exact seq), noise-only (set)
  card-id form {26000013,26000039,26000043,27000001}   -> 0 hits over 1.3 GB, all maps
  differential (T1 hand -> T2 hand, intersect addresses) -> 228 and 16 candidates, 0 overlap
  repo's cycle_vector_scan (patched)                    -> only ONE hand+cycle pair, ours
  inline 0..7 permutation scan, all maps                -> 102, all identity/reverse/rotation
Caveat on several of these: MuMu restarted the game mid-experiment more than once (pid
3909 -> 4608), which stales the needle. The differential is the one that controls for this.

Tools: `src/cardset_scan.c` (four adjacent int32 matching a card-id set),
`src/perm_scan.c` (--all, --set, --seq), `build/diff_hunt.py` (two-client differential).

### SPECTATE: neither hand (2026-09-24) — UI confirms it visually
Spectating on device 0 (acct 79227807) a match device 1 (acct 48410163) was playing:
  spectator:  side 0 hand [-1,-1,-1,-1]  side 1 hand [-1,-1,-1,-1]  both cycles NULL
  player:     side 0 hand [-1,-1,-1,-1]  side 1 hand [5,7,2,4] own, cycle [1,3,0,6]
Spectate is MORE restricted than playing: the spectator sees no hands at all, only both
players' elixir (2.04/9.04 vs the player's 2.64/9.64 — the spectate delay) and the board.
The spectator UI itself renders the "Cards Played" bars as revealed cards plus **"?"
placeholders** for everything unplayed — i.e. the UI is built from the same +0x288 reveal
list we read, and shows "?" because the client genuinely lacks the rest.

Final mode table for build 160402012:
  playing      -> own hand only; opponent [-1,-1,-1,-1], cycle NULL (mirrored on both clients)
  spectating   -> NEITHER hand; UI shows "?"
  after settle -> BOTH hands populated at the same +0x210 (verified, tick frozen 6150)
  replay       -> both hands per the repo author's evidence table (not re-tested here)

### Single-word encoding differential — negative (mid-battle, verified ground truth)
`build/encdiff.py`: device 1 gives its own hand, device 0 is scanned for every single-word
encoding of it, twice, and the address sets are intersected. Both samples mid-battle and the
post-match state explicitly excluded (rejects frames where BOTH hands are visible).
  T1 tick 1019 hand [7,0,6,4] -> bitmask 209: 557 addrs, packed_be 0, packed_le 2,
                                 nibbles_be 7, nibbles_le 4, octal 26
  T2 tick 1157 hand [7,1,6,4]  (one card changed = one real play)
  addresses holding encode(H1) then encode(H2): 0 in ALL SIX encodings
A single-word field cannot hide from this: it would have to change address between samples,
which a per-player field does not do. `src/value_scan.c` added.

Forms now tested against verified ground truth, all negative during play:
  4 adjacent slot indices (exact + any order, scudo and all maps)
  4 adjacent card ids
  bitmask / packed BE / packed LE / nibbles BE / nibbles LE / octal
  differential across a real hand change (catches any of the above, wherever located)
Untested: byte or int16 width, obfuscated storage, or computed-on-demand.

### FirstLight V4 RUNS on this client (2026-09-24) — `mac012/firstlight_obs.py`
Their live path needs an injected probe; it is not required. Working chain, no their-engine:
  EpisodeConfigV1 (hand-built, FAIR, decision_hz 4.0 = every 5 ticks)
    -> build_episode_tensorizer_v4(ep, actor_owner=side)
    -> PolicySessionV4(model, tensorizer, device='cpu')     # bypasses build_policy_session_v4,
                                                            # which is what needs BattleEnvV1
    -> start_episode(obs, initial_elixir=...) -> decide(obs) -> PolicyDecisionV4  OK
Checkpoint General loads in ~5 s as LoadedPolicyV4 / UniversalCardPolicyV4.

Why it fits: tensorizer.py:364 rejects ORACLE and accepts FAIR actor observations ONLY, so
these models never expected the opponent's hand — exactly what this client withholds.
OpponentBeliefV1 (deck probabilities, hand hypotheses, next-card probabilities, 11-bucket
elixir) matches what cycle_tracker produces.

Contract requirements discovered while building ObservationV1:
  - side towers need a tower_troop_id from table 159 (we assume Tower Princess 159000000)
  - king tower only accepts Royal Chef (159000004) or None
  - tower kinds are king / princess_left / princess_right
  - private_state_visible=False FORBIDS exact elixir/hand/deck/cycle for the opponent:
    FirstLight is STRICTER than this environment (we can read their exact elixir; it won't take it)
  - actor player needs metadata['hand_slot_by_card'] and ['hand_runtime_by_slot'][slot]['form_code']
  - action_mask.reasons['effective_elixir'] must be exact and finite

REMAINING GAP (the one that matters): entities resolve archetypes by `native_data_global_id`
(a LogicData global id), which our reader does not expose. Running required
`tz.reject_unknown_public_semantics = False`, so EVERY unit reaches the model as UNKNOWN
archetype — positions/HP/ownership survive, unit identity does not. It runs; it will not play
well until that field is found. Lead: the command queue proves entities carry a LogicData
pointer (entry+0x38 -> data, card id at +0x40), so the entity object likely has one too.
Also missing: `events=()` (no in-process combat telemetry; only the last decision window is
read, and empty is structurally valid).

### ARCHETYPE GAP CLOSED — offline, from FirstLight's own catalogs (2026-09-24)
The tensorizer resolves entity archetypes from `native_data_global_id`, which an external
reader cannot see. But the mapping is static, so it can be rebuilt offline:
    card_id -> CardSpecV1.summoned_forms name -> catalog.form_vocab_id(name)
            -> invert catalog._runtime_vocab_by_global_id -> native_data_global_id
Verified: Knight->34000000, Archer->34000001, Bomber->34000013, EliteBarbs->34000042,
MegaMinion->34000037, GoblinHut->1959820449, Berserker->932239209, Skeletons->34000008 (8/8).
116/134 summoned forms resolve; spells legitimately have none.
**The tensorizer now runs in STRICT mode** (no reject_unknown_public_semantics relaxation) and
spatial non-zeros went 12 -> 18. `mac012/firstlight_obs.py`.

### FirstLight in the console — `mac012/firstlight_bot.py` + console branch
Console now offers 8 models: 14k / 40k / 100k (sim) + fl:general / fl:hog1 / fl:hog2 /
fl:il / fl:active-il. FirstLight path uses no BattleEnv and no injected probe:
  EpisodeConfigV1 (hand-built) -> build_episode_tensorizer_v4 -> PolicySessionV4
  observations from firstlight_obs.build(); decisions every 5 ticks (4 Hz, their training rate)
  ActionV1(play_card, hand_slot, target_grid) -> send_card_taps via the existing layout
Loads and waits correctly: "fl:general loaded (FirstLight V4, FAIR tier, 4 Hz)".

STILL OPEN for FirstLight:
  - never run through a live battle end to end (needs a friendly)
  - `target_grid` assumed (row, col); must be verified against one real placement
  - events=() — no in-process combat telemetry
  - tower troop assumed Tower Princess (159000000), not read
  - abilities: ActionKind.activate_ability exists but no tap mapping yet
  - opponent deck: friendly only; outside one the episode config uses a stand-in deck
  - ACTION LATENCY NOT YET MEASURED (see below)

### Tap latency measured (MuMu, device 0)
`input tap` here costs **9-20 ms**, not the 150-400 ms seen on the user's Null's setup.
`src/fast_tap.c` writes MT protocol B events straight to /dev/input/event1 (touchscreen,
coords already in screen pixels 1440x2560): pure injection overhead ~2 ms for a full
two-tap placement. So the input path is NOT the bottleneck; the ~1.2 s receipt latency is
dominated by the server round trip plus the client's ~0.85 s jitter buffer, which is
inherent to online play and not fixable locally. fast_tap is still worth using (no process
spawn per tap, precise hold control) but it buys ~5-8%, not the 1.2 s.

## Why the FirstLight models played terribly (2026-09-24)

Five faults, all on our side of the boundary. Found by reading their tensorizer and comparing
field by field against what we were sending; all fixed and checked offline against all five
checkpoints on both sides.

1. **Placements were transposed.** `decode_action_sequence_v4` returns `perspective.action_to_native(...)`,
   so `target_grid` is a **native `[x, y]`** point -- x is the column, y is the row. The console
   read it as `(row, column)`. Every placement the model asked for went somewhere else, and
   because both values are in range for much of the board it never looked like an error.
2. **As side 1, the legal-placement grid pointed at the enemy half.** `placement_rows()` always
   marked rows 0..15 legal. Native y runs from side 0's back line upward, so side 1's own half
   is rows 16..31. Side 1 was told it could only deploy in the opponent's half. The mask must be
   native/unmirrored: the tensorizer perspective-flips it itself.
3. **Any spell on the board killed every decision.** `build_episode_tensorizer_v4` sets
   `reject_unknown_public_semantics=True`. A spell's board entity is an AreaEffectData object,
   but our reader reports the *card* id at `+0xAC`, and spell cards have no entity archetype ->
   `ValueError: public entity resolved to UNKNOWN archetype`. 2.6 Hog runs Log and Fireball, so
   that model raised constantly -- the reported "2.6 model keeps sending errors". Unresolvable
   entities are now left out and named once in the console log.
   `entity_kind` also came from the card type, and their child-kind vocabulary has no `spell`.
   It now comes from their catalog's `child_kind`, with `character` handed over as `troop`:
   `_entity_child_kind` keeps our word for a character, so `character` would have filed every
   troop under a different child type than training used.
4. **The clock read as a finished match.** `remaining_ms` was unset -> `_finite(None)` = 0.0, so
   every frame said zero time left; `phase` was `'battle'`, which is in neither of their phase
   tests, so overtime never registered; `elixir_multiplier` stayed 1.0 through double and triple.
   Now taken from `load_game_mode_timeline(STANDARD_GAME_MODE)` -- their own CSV assets, loaded
   offline: Normal 3600 ticks (1x, then 2x from 2400), Overtime 2400 (2x, 3x from 4800),
   `NATIVE_GAMEPLAY_END_TICK` 6000.
   Corroborates the tick reading: a draw froze our `game_tick` at 6150, just past their 6000
   gameplay end, and ~1080 read as 54 s. So `game_tick` is already battle-relative and is used
   as-is -- subtracting an episode-start offset would restart the clock whenever the bot is
   switched on mid-battle.
5. **Nothing moved and no tower ever fell.** `velocity` was never set (all zeros -> a frozen
   board); `entity_id` was `category`, which collides, so group/child tracking and previous-action
   retention were corrupted; `crowns` stayed 0/0; `TowerStateV1.active` defaults True, so a
   destroyed tower still read as standing. A per-battle `Battle` object now carries stable ids
   (keyed on the heap address), `age_ms`, velocity in native units per tick (their unit: position
   delta / tick delta), crowns from towers seen alive and then absent, and `active`.
   Building footprints were a 1x1 guess; they now come from `building_placement_profile`, whose
   default is 3x3, and a `blocked` profile marks the slot illegal instead of inventing an anchor.

Offline result, 60 decisions per run with a spell on the board and a tower destroyed partway:
all five checkpoints, both sides, 0 errors, 5-10 plays each, 0 placements in the wrong half.

Still stand-ins, not reads: `events=()` (no in-process telemetry), the tower troop identity,
shields/attack phases/statuses, and evolution form codes.

## Second console / second device -- the actual cause (2026-09-24)

`start_reader` executes `/data/local/tmp/mumu-live-reader-v2` (`REMOTE_READER`), which is not
one of the names `start-consoles.sh` pushes. Device 0 had it from earlier manual work; device 1
never did, so the shell ran a nonexistent binary, stdout stayed empty, no frame ever arrived,
and the console reported "no battle" forever -- indistinguishable from an idle game.

`viewer.pump` now calls the repo's own `install_reader(ADB, SERIAL, READER)` before
`start_reader`. It checks the SHA, skips the push when the binary already matches, and refuses
to overwrite a reader another observer is running. Both devices are now self-provisioning.
Verified: device 1 (`GC3VE`, CR pid 5562) emits frames with its own battle pointer
`0xb40000726ce4dc70`, distinct from device 0's `0xb4000078147fa160`.

Also found: neither device was attached to adb at all (`adb devices` showed only
`emulator-5554`, which is instance 0 under its emulator name -- instance 0 answers as
`emulator-5554`, `127.0.0.1:5555` and `127.0.0.1:26624`; instance 1 only as `127.0.0.1:26656`).
A TCP adb device drops on sleep or emulator restart and nothing brought it back, so `pump`
now retries `adb connect` on its error path.

`mac012/preflight.py` reports per device whether the reader can attach, and checks for
`mumu-live-reader-v2` rather than `live_sampler_tbi`, which is the name that actually matters.
Its first version forgot to call `mac_profile.apply()` and so rejected both devices as an
unverified build -- the profile must be applied before `verify_runtime`.

## Second console / second device

The console could not tell a broken device from an idle one: `viewer.pump` records the real
reason in `STATE['error']` (unverified build, game not running, no root, no unique libg
mapping), but the bot loop printed "waiting for a battle" regardless. It now reports
`reader not attached - <reason>`, and `/state` carries `reader_error`.

`start-consoles.sh` skipped pushing the native tools when `adb shell "[ -f ... ]"` appeared to
succeed, which it does unreliably here -- so a device that had never received the tools never
got them and its reader could never start. Pushes are now unconditional and verified against
`ls /data/local/tmp`, and `mac012/preflight.py` reports per device whether the reader can
attach, with the reason when it cannot.

### NameError in the velocity tracker (2026-09-24)

`Battle.forget` did `del self.previous[a]` where `a` was a list-comprehension variable, which
does not leak into the enclosing scope in Python 3 -- `decide failed: name 'a' is not defined`.
It only fires once an entity disappears, so the first offline test never reached it: that test
kept every entity alive for the whole run. The replacement stress test churns entities through a
small pool of reused heap addresses (spawn, damage, death, address reuse) and a random spell
effect, which is what the real frame stream does. All five checkpoints, both sides, 120
decisions each: 0 errors.

## Following FirstLight's own inference contract (2026-09-24)

Read against `native_runner/training/v4/{serve_policy,policy_session,expert,factory}.py`. Our
loop was driving the policy in a way it was never trained under. Seven differences, all fixed.

1. **No warm-up.** `PolicyService` requires `observe` (tensorize only, no decision) for every
   tick below `FIRST_POLICY_DECISION_TICK = 90`, and refuses `act` there. We decided from the
   first frame, so the policy's first ~4.5 s of context never existed.
2. **Skipped decision turns.** `PolicySessionV4.decide` advances a recurrent state and calls
   `record_action` internally -- their comment: *"Training feeds the selected semantic action
   into the next decision. Offline inference must do exactly the same on every five-tick turn."*
   Our loop skipped turns while a tap was in flight or a command was queued, up to 2.5 s, i.e.
   ten consecutive missed turns. Turns now run on a fixed five-tick grid and are never skipped;
   the tap bookkeeping suppresses a *tap*, never a turn, and a missed turn is logged.
3. **Only the first action was played.** `ModelConfigV4.max_micro_actions = 2`: a turn can carry
   two plays, each with its own `execute_offset_ticks`. The second was discarded. All decoded
   actions are now played, in offset order.
4. **Revealed cards were never passed.** The console read `STATE['revealed']` but only used it
   to guess the opponent's deck; `FLO.build` was called without it, so `revealed_cards` was
   empty for both players all match. The model's whole public view of the opponent's deck was
   blank. Now passed, filtered through `known_cards()` and capped at eight -- the tensorizer
   raises both on a revealed card its catalog cannot name and on a ninth reveal, either of
   which kills a decision.
5. **Evolutions were denied.** `build_episode_tensorizer_v4` reads
   `deck<n>_form_availability` to derive deck roles and seed the tracker's evolution cycles;
   absent, it defaults to all zeros, i.e. "no evolutions, no hero". Our sampler already reads
   `deck_form_flags` per slot (validated 0..2) and we were discarding it. Now passed as that
   tag. Their mask is bit 0x1 = evolution, 0x2 = hero; our flag range matches exactly, but that
   bit identification is inferred from the range and their mask, not proven -- equip an
   evolution and check the flag to confirm.
6. **Initial elixir was hardcoded 5.0/5.0.** The tensorizer demands exact initial elixir for
   both owners on a tracker-backed episode. Now read from the frame, which also makes a
   mid-battle start honest.
7. **No causal groups.** `build_causal_groups` falls back to `singleton:<entity_id>` when
   `causal_group` is None, so every unit was its own group and a Skeletons or Minions
   deployment read as three or four unrelated individuals instead of the swarm the policy was
   trained on. Reconstructed from the spawn cohort -- same owner, same card, same birth tick,
   which is exactly a deployment. Verified: 3 same-card units born together -> 1 group, 3
   children.

Also added: `reasons['reserved_elixir']`, their own field for elixir committed to a play the
server has not acknowledged. The client does not debit for ~20 ticks, and now that turns are
never skipped, a second play could otherwise be chosen against elixir already spent.

`sample` stays False, matching `PolicySessionV4`'s default and `evaluate.py`.

**Operational consequence: start the model before the battle starts.** The warm-up and the
recurrent state now begin at tick 0. Arming mid-battle starts the episode cold at whatever tick
is current, which is a state the policy never saw in training.

Still not reads, and the honest remaining gap:
  * `events=()` -- a 32-slot event channel (`max_recent_events`) left empty. Their probe
    collects it inside the process; damage attribution, shields, targeting and projectiles
    cannot be derived from 100 ms position/hp diffs without inventing them.
  * spell entities, dropped because our reader reports the spell's card id at `+0xAC` and spell
    cards have no entity archetype (21 of 122 standard cards; all 101 troops/buildings resolve).
  * the tower troop identity, shields, attack phases, statuses, and per-instance evolution form.

Offline these changes are verified structurally only: correct cadence, correct warm-up, no
errors, correct half, groups, clock, revealed cards, reserved elixir. Play *quality* cannot be
judged on a synthetic board -- the hand never cycles there, so the policy sees an input it would
never meet. That needs a real friendly.

### What play rate is correct (2026-09-24)

Worth writing down, because "the model does nothing" is partly an expectation problem. A 2.6
cycle deck is bounded by elixir, not by willingness:

  * 1x elixir (0:00-2:00) gives 42.9 elixir -> ~16 plays, one every **7.3 s**
  * 2x (2:00-3:00) gives another 42.9 -> ~16 plays, one every 3.6 s
  * a whole three-minute match: ~91 elixir -> **~35 plays**
  * against 702 decision turns (ticks 90..3600 at five-tick windows) that is an act rate of
    about **5% of turns**

So a correct bot WAITs on ~95% of its turns and looks idle for seconds at a stretch. Measuring
"it did nothing for 10 seconds" does not distinguish a broken policy from a correct one.

Our offline runs sit at 0.6-2.5% of turns, i.e. roughly 15-50% of the available elixir budget --
under-using it, but the same order, and on a synthetic board that is not evidence either way.
The placements themselves look deliberate rather than random: side 0 clusters on row 14 and
side 1 on row 17, which is just behind the bridge on each own side.

## The real reason the models did nothing: greedy decoding (2026-09-24)

Every shipped checkpoint is a **stochastic-rollout** policy. From the checkpoint payloads:

    fl:general    ppo-league-snapshot  step   460  gateT=0.2  actT=1.0  contT=2.5
    fl:hog1       ppo-league-snapshot  step   192  gateT=0.2  actT=1.0  contT=5.0
    fl:hog2       ppo-league-snapshot  step   616  gateT=0.2  actT=1.0  contT=5.0
    fl:il         imitation            step 29396  gateT=1.0  actT=1.0  contT=1.0
    fl:active-il  ppo-league-snapshot  step    30  gateT=0.2  actT=1.0  contT=2.5

`load_policy_v4` restores those temperatures onto the model (checkpoint.py:106), and **only the
sampling path reads them** -- model.py: `gate_temperature=self.ppo_gate_temperature if sample
else 1.0`. `PolicySessionV4` defaults to `sample=False`, i.e. `model.act()`, i.e. argmax, and we
took that default.

The act/wait gate is a distribution, and a correct policy waits on ~95% of turns (~35 plays over
~700 turns is the entire elixir budget of a three-minute match). **The argmax of a head that puts
95% on WAIT is WAIT on every single turn** except the rare state where acting is outright more
likely than waiting. That is not a degraded policy, it is a policy that structurally cannot act.

Measured directly on a fixed board, reading `out.actions.gate` while varying one input at a time:
the greedy gate stayed shut at 2, 5 and 10 elixir with no threat, shut at 10 elixir against four
attackers, and only opened at **ten elixir against eight attackers with our own towers at 20%
health** -- and then only for three of the five checkpoints. hog1 and hog2 never opened at all.

For `fl:il` argmax is worse still: imitation learns the expert's action *distribution*, and the
expert waits most turns, so argmax reproduces "wait" and nothing else.

`FirstLightRunner` now defaults to `sample=True`. This is also a useful diagnostic result on its
own: the gate responding to threat and tower damage at all proves the policy is reading our
observation and reacting to it, so the pipeline is sound.

### Correction: greedy decoding was not the on/off switch (2026-09-24)

Measured properly -- full 3600-tick battle, real card cycle, real elixir accrual with the 2x
switch at tick 2400, 702 decision turns:

    model         mode    plays  act rate  elixir spent  distinct cells
    fl:hog1       greedy     15      2.1%            56               1
    fl:hog1       sample     15      2.1%            56               3
    fl:il         greedy     23      3.3%            85               3
    fl:il         sample     25      3.6%            85              19
    fl:general    greedy     11      1.6%            43               2
    fl:general    sample     13      1.9%            53               3

So sampling barely changes the **act rate**; what it changes is **placement diversity** --
fl:il goes from 3 distinct cells to 19. Greedy collapses placement onto one or two spots, which
is predictable and wrong for most situations, but it does not stop the policy acting.

The earlier "the greedy gate never opens" reading was measured on a static board with no card
cycle at six probe points, which is not representative, and the zero-play runs came from a
harness whose hand never cycled. sample=True remains correct because it is what
`run_offline_match` does by default, but it is a quality fix, not the switch.

**Elixir utilisation is the metric, not play count**, because play count depends on average card
cost. Against the ~91 elixir a three-minute match provides:

    fl:il       25 plays,  85 elixir =  94% utilisation, avg cost 3.4   <- essentially at ceiling
    fl:hog1     15 plays,  56 elixir =  62%,             avg cost 3.7
    fl:general  13 plays,  53 elixir =  58%,             avg cost 4.1

fl:il is spending nearly all the elixir available to it, which is what a healthy policy does. It
also has by far the most training behind it (29,396 imitation steps, versus 192 and 616 for the
hog specialists and 460 for general). It is the model to judge the pipeline on.

## Input coverage, measured (2026-09-24 evening)

`mac012/input_coverage.py [SESSION]` replays a recorded match through the live pipeline and
reports, for every feature slot of every model input tensor (labelled from FirstLight's own
tensorizer source), how often it is non-zero, and whether an empty slot is data we never supply.
Run it after any match; it is the honest answer to "what is the model missing".

Found by it, all fixed:
- **Placement masks** were a half-board approximation: offered tiles under our own towers and
  gave buildings no anchor (Cannon raised every turn). Now `arena.card_placement_mask`, called as
  `BattleEnvV1._cached_placement_entry` calls it. 244/244 card-side checks pass
  (`mac012/test_all_cards.py`); Mirror is offered as illegal.
- **Attack state was entirely empty.** Located on this build with `src/comp_probe.c`: attack
  component vtable libg+0x193ad50 (+0x10 target, +0x20 stage, +0x24 timeline ms, +0x28 load ms),
  movement libg+0x193aeb8 (+0x1e0 charge progress, -1 unavailable), object +0x15c deploy
  remaining. Object layout otherwise identical to FirstLight's Null's 15.535.13 headers.
  Target pointed at a live object 2326/2410 samples. Fed through their own
  `phase_runtime.resolve_attack/movement/deployment_runtime`; WINDUP is inferred from the
  snapshot (their hooks provide the start edge), RELEASE/INTERRUPTED stay unknown. Towers too.
- **The opponent tracker was dead.** `DeterministicPublicTracker` learns plays only from
  `action_executed` events; we sent none, so it believed the opponent never spent elixir and
  knew nothing of their cycle. Now built from the command queue: a play executes at
  issue_tick + 21 (measured 22 ticks queue->unit on every play; COMMAND_CONSUMPTION_STEPS = 21).
  Replayed on the recorded 2.6 match: opponent elixir ceiling never below the truth (0/641
  turns), median 0.32 above. Remaining drift is champion ability spend (not yet evented).
- **The opponent was modelled with our own deck.** Their deck is not readable mid-battle; each
  console now publishes its own account's deck to build/decks/<account>.json and the other
  reads it (friendlies between the user's two accounts). Unknown deck -> opponent plays are
  withheld from the tracker rather than crashing it.
- Hero and evolution units were dropped (form ids 203xxxxxx / 13xxxxxx); the reader also
  filtered every evolution unit out entirely (id below 20,000,000). Both fixed.
- Opponent private state: the builder no longer marks the opponent's hand visible even when
  this client exposes it (post-settlement).
- Champion ability activations come through the queue as card id 65535; separated from plays.

Still not supplied (slots empty because the domain is never filled, from the report):
ability runtime (14 slots: champion / hero abilities, own and opponent), tower-troop runtime (6,
irrelevant for Tower Princess), projectiles in flight (5), evolution counters (4), visibility (4),
relocation (6), capture (2), shields (2), resource (1), periodic modifier (1), and every
combat event other than card plays. Plus ~300 ms of our own latency per play.

## Comparison against FirstLight's own play loop (2026-09-25)

FirstLight is public (github.com/Jaasssoooonnnnn/FirstLight_CR, commit 9f622d5), so this was
read against their source rather than inferred. How "the one in Null's" plays
(`native_runner/training/v4/offline_agent.py`, `evaluate.py`):

  * **Lockstep.** `environment.step(commands)` advances exactly five ticks per decision; the
    game waits for the model. Rendered play backdates the command age so a play executes on
    the **next tick** (`_rendered_action`: `execute_offset_ticks=1`), and `PolicySessionV4`
    hard-codes `base_latency_ticks = 1`. The model has no latency input: every checkpoint
    was trained, evaluated and hand-tested with ~50 ms from decision to execution.
  * Their probe supplies per-card forms (`card_parameter & 0xF`), the full combat event
    ring (`public_card_play_events_from_combat_ring`), spell/area entities, abilities.
  * Engine: Null's Royale 15.535.13 with content 15.535.86. This client is 160402012, a
    later balance.

What that means here, and what changed:

1. **~1.2 s from decision to unit, which the policy never saw.** Tried and REMOVED
   (user decision, 2026-09-25): showing the policy an extrapolated future board. Rejected
   because it feeds the model invented state. What is actually in that 1.2 s:
   * **The game's own command age: 20 ticks + 1, for everyone.** A live command carries
     `t` (issue) and `t2 = t + 20` (FirstLight `cr_native_env.py`: "Live commands normally
     carry a 20-tick t/t2 age"; `LIVE_COMMAND_AGE_TICKS = 20`). A human's tap waits exactly
     as long. Offline, FirstLight backdates it (`play_immediate`, `_rendered_action`), which
     is why their models never learned it. This part cannot be made faster by any input
     method; it has to be *trained for*.
   * **Our pipeline** (all ours to cut): frame staleness (reader polled every 100 ms -> now
     50 ms, `CR_READER_MS`/`CR_QUEUE_MS`), waiting for the five-tick decision grid,
     inference, and the tap itself. The tap was `adb shell "input tap; sleep 0.05; input
     tap"` per play: a new adb client, a shell, two Java `input` launches, a fixed 50 ms sleep,
     blocking the loop. Now `mac012/tapper.py`: one resident `fast_tap` over a persistent
     adb shell, a play is one stdin line, the loop does not wait. Gesture shape
     (`CR_TAP_MODE` place/drag, `CR_TAP_GAP_MS`, `CR_TAP_HOLD_MS`) is to be chosen by
     `mac012/tap_bench.py` in Training Camp, not assumed.
   * The console logs one line per play: frame age, turn wait, inference, time to tap sent,
     gesture duration, tap -> issue ticks. That is the budget to cut.
2. **Every card was sent as its normal form (fixed).** `hand_runtime_by_slot` and the
   placement entries carried `form_code: 0`. The Hog specialists' deck is Hero Musketeer,
   Evo Cannon, Evo Skeletons -- three of eight cards the policy saw as a different card.
   `FirstLightRunner.hand_forms`: hero flag (0x2) -> form 2 always (their BattleEnv does
   the same for hero form ids); evolution flag (0x1) -> form 1 exactly when *their tracker*,
   fed our executed plays, has counted the evolution cycles down. Derived, not read: verify
   on the device that the tracker's "ready" matches the in-game evo glow (see below).
3. **The second play of a turn was always dropped (fixed).** The in-flight guard skipped any
   tap while the previous card was still in hand (~1.2 s), and a queued own command blocked
   all taps. So `max_micro_actions = 2` never happened and Hog + Ice Spirit, one decision,
   was always half a play. Now every play in a turn taps; only a re-tap of a card already
   in flight is dropped; reserved elixir is the sum of all in-flight plays until
   issue+21; a play chosen a moment before the client credits the elixir waits (≤0.5 s)
   until the client can place it.
4. **Taps go by card identity, not slot.** The screen position is looked up from the card
   the policy chose in the hand as memory holds it, so a hand that changed since the frame, or any slot
   disagreement, cannot tap the wrong card.
5. **Deck check.** Starting fl:hog1/fl:hog2 now logs whether the deck and forms match the
   specialist's training deck. Out of it they are a different, weaker model (their table:
   specialist 2 vs General 88% on its deck, 59% on others).

Offline, `mac012/test_forms.py` (synthetic 3-minute match, Hog deck with forms,
tap -> queue 3 ticks -> execute +21, reader hand/elixir changing only on execution):
all five checkpoints, both sides: 0 decide errors, forms always equal their tracker's
readiness, hero always offered, 88-99% of the match's elixir spent.

### Roadblocks that remain (not fixable from outside the process, or need the device)

* **The policy was trained with no command age.** Every live play lands 21 ticks after
  issue and the model expects 1. Not fixable at inference time without inventing state;
  the fix is training with the real age (FirstLight's env already has the mechanism:
  `execute_tick` / `LIVE_COMMAND_AGE_TICKS`, backdated only for the offline viewer).
  Related: their IL anchors each human action at `native_observable_tick`, when the unit
  appeared, not when the human committed ~21 ticks earlier.
* **Combat events are empty** (`events` holds only card plays). Their probe fills a 32-slot
  channel from inside the game: damage, projectiles, shields, ability casts. Needs an
  in-process hook, i.e. what their APK patch does.
* **Spell entities are dropped** (Log/Fireball in flight): our reader reports the card id,
  not the AreaEffect/projectile data id.
* **Different game version.** 15.535 vs 160402012: any balance change since then is a
  change the policy never trained on.
* **Tower troop assumed Tower Princess.**
* Things to confirm on the device, in one friendly:
  1. `python3 mac012/tap_bench.py` in Training Camp: which gesture is accepted, fastest;
     then the per-play "latency ..." lines in a real match;
  2. evo readiness: when the tracker offers Evo Skeletons/Cannon as evolved, does the card
     glow in hand? If evolutions start the match charged on this build, the tracker's
     initial state is wrong and the fix is to seed it;
  3. whether the hand in memory changes at tap or at execution (both are handled; the
     answer tells which path runs);

## Side-1 placements were point-mirrored (found 2026-09-25)

`ScreenLayout.deployment_point` takes a CANONICAL cell (local player's view, row 0 = own back
line; the Training Camp tap test above is the proof: canonical 170 -> native (9500, 22500) as
side 1). FirstLight decodes to NATIVE tiles. The console passed the native cell straight
through, so as side 1 every play went to (17 - col, 31 - row): troops aimed into the enemy half
and snapped by the client to the nearest legal tile, spells on the mirrored lane. As side 0
native == canonical and placements were right. This is the "madman" play the user saw; it
predates every change in this session. Fix: `console.screen_cell`. The console now also logs
`placement <card>: asked native (x, y), game got (x, y), off N tiles` from the queue entry,
so a mapping error can never again go unseen.

Tap benchmark (Training Camp, 4 trials each): old adb path 3/4 accepted, 148 ms gesture;
fast_tap place gap 50/hold 34 119 ms; 30/20 71 ms; 16/16 49 ms; 8/16 41 ms; 0/16 33 ms (all 4/4);
drag 3x10 41 ms 4/4, 2x5 16 ms 3/4, 1x17 34 ms 4/4. Default now place gap 8 hold 16.
First live latency lines (gap 30/hold 20): frame age 5-30 ms, inference 88-134 ms, tap sent
<=1 ms after the decision, gesture 71 ms, decision frame -> issued 4-6 ticks; then the game's
21. Inference is now the largest part we own.

## Opponent plays silently lost -- fixed (2026-09-25)

Three independent ways an opponent play never reached FirstLight's tracker:
1. **Unknown opponent deck** (Training Camp always; a friendly whose other console is not
   running). The episode used our deck as a stand-in and filtered opponent plays by
   `decks[opp] = ()` -> *every* opponent play was dropped, reported once per card, so after
   the first few it looked silent. The tracker then believed the opponent never spent elixir.
   A stale published file (fallback to any old publication) did the same for most cards.
2. **Form ids** (evo 13xxxxxx, hero 203xxxxxx): `executed_plays` filed anything outside
   25M..30M as an ability, and `play_events` skips abilities without a word.
3. A queue entry whose side could not be resolved was dropped silently.

Now: `viewer.card_identity` normalises any command id to (base card, form code, kind) and marks
ids outside this build's catalog `unknown`; `FirstLightRunner` keeps the opponent's side of the
tracker as a *registry that grows*: empty when the deck is unknown (published deck when fresh),
every card registered the first time it is played or revealed, only a ninth distinct card
refused. The stale-file fallback is gone. Every refused play, unknown id, unattributed play and
ability activation is logged, once per play. Status line: `opponent deck N/8 seen (learned|
published)`. `mac012/test_opponent_registry.py`: unknown / stale / exact deck, both sides, an
evolved play arriving as a form id, a ninth card: 8/8 recorded, ninth refused loudly, tracker's
opponent elixir ceiling falls on every spend, 0 decide errors (fl:hog2, fl:il, fl:general).

Also: **evolution units were invisible when unit and card names differ.** Our catalog names the
evolution *card* form (Skeletons_EV1); FirstLight's archetypes are keyed by the *unit* form
(Skeleton_EV1), which its own card spec names (evolution Transform `summoned_form`). Evo
Skeletons, Barbarians, Bats, Recruits, Royal Hogs, Wall Breakers now resolve; forms FirstLight
never had are shown as the base unit and logged (`mac012/drift_check.py` lists them: Goblin
Barrel, Zap, Snowball evolutions).

## Delay: what FirstLight trained with (verified in source, 2026-09-25)

* PPO self-play: a play executes `base_latency_ticks (1) + policy delay offset (0..4)` ticks
  after the decision (`decoding._timing_metadata`, `resident_batch_vector.queue_hand_action_at`).
  No 20-tick command age.
* Imitation: each human play is keyed by its source command tick (the execution boundary, +1 for
  the observable state) and paired with the decision turn 0..4 ticks before it
  (`expert.decision_tick_for_action`). The human committed ~20 ticks earlier; the model learned
  to act on a board it could only have seen at execution.
* Pending commands: `ObservationV1.pending_actions` is own-side only, empty in policy
  observations (`policy_features_only`), and never read by the tensorizer. The model has no
  input for a queued command of either side.
* Live: decision frame -> issued 4-6 ticks (ours) + 21 (game) = the play lands ~1.3 s after the
  board it was chosen on. Opponent commands are in our queue before they land;
  `mac012/queue_lead.py` measures how early from any recorded session.

So for a model that fits the real game: train with the 21-tick command age (plus measured
round trip) and give it pending commands of both sides as an input, with their execute ticks.
Neither fits inside the current checkpoints.

## Real-game inputs vs cr-native-sandbox vs FirstLight (2026-09-25)

cr-native-sandbox's live path (`mumu_live_private_sampler.c` + `mumu_live_controller.py`):
entities (category, side, x, y, card id, level, hp, max hp, behaviour), players (elixir, hand,
cycle, next, deck ids, form flags, refill timer). Enemy cards learned from board units only (no
spells). No command queue, no reveal list, no attack state; its controller has fields for
per-unit ability state but its reader never fills them. **Our reader is a strict superset**
(+ attack stage/timers, charge, deploy remaining, target; + queue_probe: both sides' commands,
accounts, reveal lists, received-tick gap).

Still not supplied to the model (FirstLight fields left empty), in order of likely value:
1. **Hero / champion ability runtime** (player `ability_runtime_states`, mask `ability_sources`)
   -> the bot *cannot* use an ability: the mask never offers it and the console dropped
   non-card actions. Training covered it (the Hog specialist's deck has Hero Musketeer).
   Source on 15.535: controllers at player+0x3a0 (+0x78 cooldown, +0x80 charges, +0x98 button,
   back-pointer +0x20). `src/runtime_probe.c` finds them on this build.
2. **Evolution progress** (player `evolution_runtime_states`): derived from their tracker today;
   our catalog and FirstLight disagree on cycles for 29 cards (Skeletons 3 vs 2, Ice Spirits
   1 vs 2, Cannon 3 vs 2). 15.535 source: player+0x2e8 progress vector; the probe finds it.
3. Pending commands (both sides) -- needs a new model input (see Delay).
4. Combat events, spell/projectile objects, shields, statuses, tower troop identity.
Deliberately not supplied (FAIR tier): opponent exact elixir, although the client holds it.

## Maintenance after a game update

What breaks, and where it shows:
* libg SHA / offsets (manager RVA, component vtables, player struct, queue entry): the reader
  refuses the build ("reader not attached - unverified build"); re-run the scans.
* New cards / forms: live_card_catalog.json must be regenerated from the build; until then the
  console logs `queue: card id N is not in this build's catalog`.
* FirstLight's catalog is frozen at 15.535: `drift_check.py` lists cards/forms it cannot
  represent; the console logs refusals and unit fallbacks per match.
* Balance changes: invisible to the frozen checkpoints; only retraining fixes them.

## Device results, 2026-09-25 (runtime_probe, screenshot, queue_lead)

**Ability controllers: same offsets as 15.535.** player+0x3a0 and +0x3a8 (slots 1, 2), found
by the back-pointer (+0x20 == player). Selected character (+0x90 -> +0x40) matched FirstLight's
archetype ids exactly: 130283371 = Hero Musketeer, 2979504115 (printed signed, -1315463181) =
Hero Ice Golem. Observed: charges (+0x80) -1 (no hero bound) / 1 (unused) / 0 (used); configured
cooldown 0 (hero abilities are single-use, per the user: usable once, when the hero is on the
board and elixir suffices); button state (+0x98) takes 0, 1, 2, 6, 9 -- meaning still to be
mapped (log it next to "hero on board / enough elixir / used"). action data id 1171272209.
**Evolution progress: player+0x2e8, unchanged.** One int per deck slot. Evo Skeletons went
0 -> 1 -> 2 -> 0: two normal plays, then the evolved play resets it. So the requirement is 2 --
FirstLight's number is right for this build, and our catalog's `evolution_cycles` (3) counts
the evolved play too. Keep the tracker on FirstLight's counts; better, read this vector.
**Ability buttons** (screenshot, 1440x2560 device): one hero -> the button is at the LEFT;
two heroes -> left and right. Measured centres ~(140, 1955) and ~(1300, 1955), just above the
hand. (User setting decides placement with two heroes; default is both sides.)
*Superseded the same morning:* on the account in the 07:55 friendly a single hero's button was on
the RIGHT (centre ~(1295, 1965)), and both left-side taps did nothing. The side is per account;
the console now learns it (see "Friendly 2026-09-25 07:55").
**Command lead** (queue_lead, 107 commands): opponent commands are in our queue a median
14 ticks (700 ms, p10 11, p90 18) before they execute; ours 13. Issue -> our queue ~7-8 ticks.

## HANDOFF (cloud session -> local session, 2026-09-25)

Branch `main-8x1wxo`, everything pushed. FirstLight source is at
~/Documents/GitHub/FirstLight_CR (override: FIRSTLIGHT_ROOT); native_core comes from the
cr-native-sandbox checkout the profile already uses.

Offline checks (all passing at handoff):
    python3 mac012/test_forms.py fl:hog2 fl:il
    python3 mac012/test_opponent_registry.py fl:hog2
    python3 mac012/test_firstlight_churn.py
    python3 mac012/drift_check.py --deck <ids>

Done this session: side-1 placement mirroring fixed (verified live: 0 tiles off), resident
fast_tap (place gap 8 / hold 16), 50 ms reader/queue polling, per-play latency + placement
log, both plays of a turn tapped, evo/hero hand forms, opponent deck learned card by card
(no silent drops), evolution units visible, form ids normalised, drift/queue-lead/runtime
probes. The latency extrapolation was tried and REMOVED at the user's request -- do not
reintroduce predicted board state.

Next, in order:
1. **Hero ability (implementation, not training).** Add to live_sampler_tbi.c per player: the
   two controllers at +0x3a0/+0x3a8 (charges +0x80, button +0x98, character +0x90->+0x40) and
   the progress vector at +0x2e8. Map button states. Then in firstlight_obs: own
   `ability_runtime_states` (ability_id from the catalog, elixir_cost, charges, available,
   attributes.controller_slot, attributes.source_card_id, source_entity = the hero unit's
   entity id) and `action_mask.ability_sources` + kinds[activate_ability]; see FirstLight
   training/v4/native_actions.ability_runtime_contract for exactly what is required. Console:
   stop skipping ACTIVATE_ABILITY; tap the left button (right = second hero's slot 2).
2. **Evolution state from memory** (+0x2e8) instead of the tracker derivation; fill
   `evolution_runtime_states`.
3. **Opponent deck memory** (user's list): remember revealed cards per opponent account
   across matches; optional RoyaleAPI lookup by player tag and popular-deck completion as a
   *prior* (not ground truth); evolution cycles tracked for the opponent too.
4. **Pending commands** (both sides, ~700 ms of warning): a separate input type from real
   troops, per the user. Needs a model input -> part of retraining, not the current
   checkpoints.
5. Retraining plan: command age 21 + measured round trip in the engine, IL aligned to the
   human's decision (execute - 21), CLIENT-tier inputs (pending commands, exact own ability
   and evolution state), oracle-tier critic.
Reader already has attack wind-up (atk_stage / atk_timeline / atk_load) and movement charge
progress (+0x1e0, Prince / Dark Prince charge) -- Sparky's charge is its attack load.

## Overnight session, 2026-09-25 (local): hero/evo into the model, every turn reaches the policy

**Hero abilities + evolution progress (read from memory).** Reader: per player, controllers at
+0x3a0/+0x3a8 (back-pointer checked; +0x78 cooldown, +0x7c configured, +0x80 charges, +0x98
button, +0x90->+0x40 selected character) and the +0x2e8 progress vector. From the saved device
run (build/runtime_probe.jsonl): controllers present for both players in 513/513 samples; the
controller's action data id (1171272209) is identical for EVERY controller, so the selected
character is what identifies the hero (130283371 Hero Musketeer, 2979504115 Hero Ice Golem =
FirstLight's archetype ids). Button enum = FirstLight's ABILITY_BUTTON_STATE_LABELS exactly
(1 ChampionAbsent, 2 Ready, 6 AllChargesConsumed, 9 NotEnoughElixir seen); the full lifecycle
absent -> Ready -> used -> absent -> Ready (replayed) is in the run. Evolution vector index =
our deck order (Cannon slot 2 read 2; Skeletons slot 0 counted 1 -> 2 -> 0). Ability joined
character -> hero form -> base card -> the card's single FirstLight ability (Musketeer_hero_
Ability 3 elixir, IceGolemiteHero_Ability 2). Offered under BattleEnv's legality rule.
FirstLight's V4 contract: <= 2 heroes, <= 2 evolutions, <= 3 special cards per deck -- every
recorded deck fits (the game allows 3 special slots). Hog specialists' training deck: Hero
Musketeer, Evo Cannon, Evo Skeletons, plain Ice Golem (user's deck, per user, is that one;
the Hero Ice Golem was a temporary test). Training Camp gives a preset deck with no heroes or
evolutions, so these reads are verified offline (test_hero_evo) and against the device run,
not yet live -- first friendly with the hero deck is the live check.

**Missed decision turns: two causes, both fixed, verified live.** The log now breaks a missed
turn down (previous turn prep/decide/after, time away, frame age). Findings:
1. After every play the played slot reads -1 for 50-550 ms while the next card is drawn (the
   card is already at the end of the cycle, so hand + cycle still partition the deck). The
   console skipped the turn; the two longest draw gaps were exactly the two missed turns.
   FirstLight's env keeps deciding with the slot unplayable -- now so do we; the builder also
   treats our hand as ours when slot 0 is the empty one.
2. First decision of a match: 461 ms (torch cold start) -> `FirstLightRunner.warm_up()` at load.
   Late-match 638 ms observation build that never reproduces offline (<= 18 ms) -> Python
   full-GC pass over the model + catalogs -> `gc.freeze()` after load.
Live Training Camp (fl:hog2, side 1), after both fixes: 0 missed turns, 0 errors, 35/35
placements exact (0.0 tiles), inference 72-93 ms. `mac012/replay_decide.py` replays any
recording through build -> decide: 4 recorded matches, 2,587 decisions, 0 errors, draw-gap
turns included.

Also: input coverage on a live match with the new reader -- unit target 83%, attack cooldown
25%, windup 10%, deploy 12%, tower target 87% (all 0% before). Remaining zeros for units and
towers are projectile-only slots, shields, visibility, abilities-on-entity: the known gaps.

## The console is a native Mac app now (2026-09-25 overnight)

`Clapha.app` (SwiftUI, built by `app/build.sh`, no Xcode project; `Clapha Consoles.command` opens
it). One window for everything: both devices side by side (live board drawn from our side,
hand, both elixir bars, battle clock, reader health), one **Off · Watch · Play** switch per
device (Watch = the model decides every turn and logs what it would play, no taps; Play = it
taps; the old start/stop + "armed" checkbox is gone), the model as a short list with the
running one marked, the bot log, and a Tasks panel (input report for the last match, replay
the last match through the model, hero/evolution checks, all-cards sweep, tap benchmark,
bot log). Top bar: Start everything (emulators, tool rebuild/push, both engines -- no browser
tabs), Stop engines, Check devices. A banner says so when the engines are not running.

The Python engines stay as the tested backend (the decision loop is unchanged); the app talks to
them over localhost. New engine endpoint `/api/mode?mode=off|watch|play&model=...`: watch <-> play
on the same model only flips taps, so the episode and recurrent state carry on; a different
model restarts cleanly (joins the old loop first).

Performance: 3-7% of a core while polling at 4 Hz in battle / 1 Hz otherwise. It started at 38%:
SwiftUI's @Published fires on every assignment, even an unchanged one, so setting
`reachable = true` each poll re-rendered the whole window 8x a second; plus board / summary /
log are now separate observable objects so only what changed is redrawn.

Verification without a display: `app/snapshot.sh` renders the window off-screen from recorded
engine states (app/fixture_*.json) to build/app_snapshot.png; `build/app.log` records startup
and polling. Native AppKit controls cannot be drawn off-screen, which is one reason the mode
switch and model list are plain SwiftUI (the other: they are clearer).

## Objects resolved by their own data id: spawned units fixed, projectiles visible (2026-09-25)

`src/ent_probe.c` listed every object in a live battle, unfiltered, with object+0x48 -> +0x40 (the
object's own data record, FirstLight's kObjectDataOffset). Findings:
- For every unit, that data id IS FirstLight's archetype global id (Knight 34000000 = form:Knight,
  Goblins 3565953160 = form:Goblin_Stab, ...).
- **Spawned units were mislabelled by our card-based mapping.** A Battle Ram's Barbarians carry
  the Battle Ram's card (+0xAC 26000036) but data 34000009 (Barbarian); a Tombstone's Skeletons
  carry the Tombstone's card (27000009, a building archetype) but data 34000008 (Skeleton). Every
  spawner card (Tombstone, Witch, Battle Ram, Goblin Hut, Furnace, Graveyard...) was affected.
- **Projectiles were in the entity list all along, dropped by our reader's kind filter**: kind 0,
  category 4xxxxxx, data = ProjectileData (Fireball 10000000, Archer arrow 10000002, Musketeer
  10000014, Bomber 10000010, tower shots 10000003 with card -1).
- Tower shots are card -1 like towers and spawn on the tower's tile -- anything recognising towers
  by card -1 must also require a tower object kind (12/13). Fixed in firstlight_obs, viewer,
  model_adapter, calibrate.
Reader now emits `data_id` and keeps kind-0 objects whose data is ProjectileData (10xxxxxx) or
AreaEffectData (22xxxxxx). Builder resolves every object by data id first (card mapping only as the
fallback for older recordings); projectiles get FirstLight's ProjectileStateV1 (in flight,
velocity, source card, data id; damage/radius from their archetype as their probe leaves them).
Spawned units use their own unit's hit speed. `mac012/test_effects.py` covers it. Live Training
Camp: 39 plays, 0 errors, 0 missed turns, 37/37 placements exact; over 729 decisions 13.9% of
unit rows were projectiles in flight; unit features filled 17 -> 22 of 72; slots empty for lack
of data 47 -> 44. Still missing for projectiles: destination and homing.

## Staying current (2026-09-25)

- `mac012/update_check.py`: installed build per device vs the reader's certified build (version +
  libg SHA from the device's own install), card catalog stamp, and whether FirstLight /
  cr-native-sandbox upstream moved. `--pull` copies a new build from the device into
  runtime/<version>/ (git-ignored; nothing downloaded), decodes its tables and diffs cards; the
  report lists every pinned value that must be re-derived and the probe that does it.
- `mac012/game_data.py`: reads the build's LZMA tables (Supercell's short header) from the pulled
  APK. `mac012/update_catalog.py` brought live_card_catalog.json from 15.535.29 to 160402012:
  152/152 ids and names unchanged, no elixir changes, one new card (MinionGiant 26000107, 4).
  `PrestigeCount` is where the old catalog's evolution cycles came from; it is NOT the play count
  (Skeletons 3, measured 2) and no single rule maps it (PrestigeCount-1 fits 19/42 of FirstLight's
  numbers) -- so the console measures the requirement from the game instead: a deck slot's
  progress counter falling from k to 0 means the card needs k (build/evo_cycles.json).
- FirstLight: the user's download predates the whole Sept 23-24 series. **Models identical**
  (SHA-256 = upstream manifest); code behind in 14 files incl. policy_session.py and
  offline_agent.py. Upstream now clones into ref-firstlight/ (git-ignored) and is the default
  root; every test passes on it. Upstream changes that matter: their human-vs-AI loop still
  samples; their new AI-vs-AI duel decodes greedily (force_act is only a manual GUI button); their
  offline engine executes plays on the next tick (backdated command age) -- the policies trained
  with ~1 tick, the live game has 21. Console decoding is now Auto (sampled vs a human/bot,
  greedy when the other device runs a model against us), Sampled or Greedy.
- GPU (MPS) inference is slower than CPU for this batch-1 recurrent model (137 vs 106 ms).

## Friendly 2026-09-25 07:55 (fl:hog2, device 1 vs the main account) -- what memory shows

Recording artifacts/viewer-sessions/20260925T115451Z (one battle, ticks 0..5040).
- Input/timing healthy: every placement checked landed exactly (0.0 tiles), inference 83-123 ms,
  tap -> issued 5-6 ticks, and each play shows in the hand ~1.3 s (26 ticks) after the decision.
- **Evolution live:** "evolution requirement measured: Skeletons needs 2 plays" from +0x2e8.
- **Hero ability taps missed.** The policy chose Musketeer_hero_Ability at t=106.5 and 109.8 s;
  the console tapped (140, 1955) because the rule said "one hero -> left". Memory: controller 1
  button stayed 2 (Ready) with 1 charge and elixir never dropped by 3; screenshot at t=203 shows
  the button on the right. Fix: `_ability_side()` -- one hero defaults to RIGHT, two heroes left
  for slot 1 / right for slot 2, and a tap that leaves the button Ready with the same charges for
  2.5 s flips the side for that account (build/ability_buttons.json), logged once.
- **The "repeated" Cannon/Log were taps after the battle was over.** Opponent's left princess tower
  fell between t=235 and t=248 in overtime (sudden death -> the bot won). From t=246.1 no command
  executed (hand frozen, elixir only rising) while the client's clock ran to 5040 and accepted
  taps. Fix: `FLO.battle_result()` (king tower; unequal crowns at/after full time; tiebreak at
  6000) + `OutcomeLatch` in the console: acted on only after holding 0.5 s, withdrawn if a tower
  reappears; then no taps, deferred plays dropped, one line in the log and in build/results.jsonl
  (model, sampled, side, won, reason, tick, plays, opponent account).
  `mac012/test_battle_result.py` replays every recorded battle through it: ~75 battles, no
  result ever withdrawn, none decided more than ~6 s before its recording's clock stopped. It also
  found the one wrong verdict: a finished battle still on screen when the viewer restarted (frozen
  at tick 5700) lost tower objects one by one as the client tore it down, reading a win as a loss.
  No verdict now until the clock has moved while we watch (the console never acts on a frozen
  clock anyway; this keeps results.jsonl honest when a console starts on a finished battle).
- **Unreadable health crashed whole frames.** Session 20260925T010215Z tick 2863: a Tombstone read
  hp -1 / max -1 for one frame -> EntityStateV1 contract error -> "decide failed", turn lost.
  Units with health < 0 are now left out for that frame (`battle.unreadable_hp` counts them);
  a tower with unreadable health keeps its last reading (a -1 would have counted as destroyed --
  a crown, and in overtime the end of the battle).
- **Evolution requirement mis-measured at every battle start.** `_measure_evolutions` compared
  against the last snapshot from the previous battle; every counter starts a battle at 0, so a
  battle that ended with Skeletons/Cannon at 1 logged "needs 1 plays (was 2)" at 09:34:29 and fed
  the model a one-play evolution until the real evolved plays (09:36) measured 2 again. Now only
  a fall inside one battle (same battle pointer, clock moved forward <= 100 ticks) counts; replayed
  on the session that holds both battles, it keeps the two genuine measurements and drops the two
  boundary ones. (The viewer session file keeps growing across battles while the viewer runs:
  20260925T115451Z holds the 07:55 and the 09:34 battles.)
- Device 2's console said "Clash Royale is not running" -- correct: the game was not open there.
- **"load failed: No module named 'torch'" after Start everything (10:38).** The app runs tasks in
  a login zsh, which does not read ~/.zshrc, where conda puts miniforge on PATH; `python3` there is
  python.org 3.14 without torch (the 07:54 consoles had been started from a terminal). `./py` now
  runs the first Python that has torch ($CLAPHA_PYTHON, cached choice in build/python_path,
  PATH, miniforge/conda/Homebrew locations); start-consoles.sh and every app button use it.

## Sandbox from the latest libg: assessment (2026-09-25)

What it takes (cr-native-sandbox docs/SANDBOX_RUNTIME_TECHNICAL #27): freeze hashes; relocate
JNI_OnLoad, key functions and struct offsets; re-verify DataTables/map resources; regenerate card
and form catalogs; re-verify Replay, six towers, tower HP, RNG, public hashes; deploy, abilities,
grid, tick; reset/time/elixir/terminal certificates. Their engine host (android_probe: Java JniHost
+ 7,859-line jni_bridge.cpp) loads the game's own libg into its own process and drives battles by
calling internal functions -- every address pinned to the x86_64 15.535.29 build.
- Their route runs x86_64 libg in an x86_64 AVD (Windows/Hyper-V) or x86_64 Linux. Not on this Mac.
- Their ARM64 bring-up (IMAX9D/cr-native-linux-bionic, branch port/arm64-160402002-bringup,
  2026-09-08, 160402002): ARM64 Bionic entry, real ART, FMOD, Sentry load; **libscid_sdk
  SIGSEGVs during init** (indirect call at RVA 0x17b170 jumps to unmapped 0x681780 right after
  reading /proc/self/maps); libg JNI_OnLoad fails the same way. Tested only under QEMU (user and
  system emulation); they could not tell emulation artefact from a real environment requirement.
  No battle ever created.
- FirstLight's engine patches a Null's (private server) APK with their probe. Doing that to the
  official current client is modifying the official app -- not done.
- This Mac is native ARM64 and MuMu is a real ARM64 Android, the one environment their research
  never had. Tools present: build-tools 36 (d8, aapt2, apksigner), platform 37, JDK 21/25, NDK 28.
  The current build's APK splits are in runtime/160402012/ (SHA256SUMS), libg SHA = profile.
Plan, in order, each a go/no-go gate:
  1. Load gate: a minimal ARM64 host APK in a fresh, separate MuMu instance (not a game device)
     loads libc++_shared, FMOD, libscid_sdk, then libg with JNI_OnLoad -- does SCID init pass on
     real ARM64 Android? (Their blocker; decides whether the rest is possible.)
  2. Relocate the host's entry points on 160402012 ARM64: resource/DataTables init, GameMain,
     Replay/Scene creation, tick, card command, ability command, reset -- from exported JNI
     symbols, the JNI registration table and string xrefs. No reuse of x86_64 RVAs.
  3. Certify against the live client: same deck, same commands -> same towers/HP/elixir, using the
     reader's live frames as the reference (we can read both).
  4. Throughput and multi-battle stability; then training with the real 21-tick command age.
Realistically weeks, not a night; gate 1 is a day and tells whether it is feasible at all.

## Sandbox gate 1: result -- closed on ARM64 (2026-09-25 morning)

Test app `sandbox/loadprobe/` (Java ProbeActivity + build.sh; aapt2/d8/apksigner, no INTERNET
permission, extractNativeLibs) bundling the build's own ARM64 libs from
runtime/160402012/apks/split_config.arm64_v8a.apk (build output git-ignored: it contains the
game's libraries). Ran in a fresh MuMu instance created for it (never signed in, no game
installed), then deleted; the game devices were not touched.
- Order c++_shared, fmod, fmodstudio, sentry, sentry-android, scid_sdk, g: **libscid_sdk SIGSEGV**
  (SEGV_MAPERR, pc 0x68d9c0 unmapped, lr libscid_sdk+0x173644) -- the same failure the
  cr-native-sandbox authors saw under QEMU (RVA 0x17b170 -> 0x681780 on 160402002). Not an
  emulation artefact. (FMOD's JNI_OnLoad returns a bad version when loaded standalone; harmless.)
- Why: the ARM64 libscid_sdk.so (2.0 MB; x86_64 15.535 copy is 14.7 MB) and libg.so are packed:
  5 section headers, extra loader LOAD segments (libscid 0x178000/0x1a0000, libg 0x1ab8000/
  0x1ae4000), rwx mappings in the live game. libg's code is **encrypted on disk**: live code pages
  differ from the file (0.4% equal bytes), entropy 7.99 bits/byte on disk vs 6.69 in memory.
  The game loads libsupercell_clashroyale.so first (dzmxszrox.aI, from the Application) --
  its protection runtime; 0x68d9c0 lies inside that library's size, i.e. the unpacker calls into
  it and jumps to base 0 when it is absent.
- Loading libsupercell_clashroyale first: loads in 95 ms, then the process vanishes (no tombstone,
  no ApplicationExitInfo) -- consistent with it refusing to run inside anything but the genuine
  app. Going further means spoofing the game's identity, patching the protection or dumping the
  decrypted code: circumventing anti-tamper. **Not done; the ARM64 route is closed.**
- Remaining in-bounds route = cr-native-sandbox's: x86_64 libraries (their 15.535 x86_64 libg
  loads without the protection library) in an x86_64 Android. Needs the current build's
  split_config.x86_64.apk from an official install on an Intel/AMD device (the user's own Play
  install; nothing from mirrors) and an x86_64 host to run it. Unverified whether the current
  x86_64 build is packed too.
- Consequence for staying current: offsets cannot be re-derived from the file offline.
  `mac012/rederive.py` does it from game *data* during one battle (find_manager -> reader checks
  on hand/elixir/deck/towers/units -> comp_probe classification of the attack and movement
  vtables), writes build/rederive_<version>.json, and with --apply updates mac_profile.py and the
  reader's #defines. Checks tested on recordings; not yet run live (`--force` on the current build
  must re-find 0x1A57E88 / 0x28 / 0x193ad50 / 0x193aeb8).

## Opponent hero abilities charged to their elixir (2026-09-25, cloud session)

The queue shows an opponent activation only as card id 65535; it was logged and dropped, so
FirstLight's tracker believed the opponent kept that elixir. Now
`FirstLightRunner.attribute_opponent_abilities` joins each one to the hero that cast it, using
the opponent's controllers the reader already emits (selected character -> hero card): the only
hero, or with two heroes the one whose charges are spent and not yet attributed. The hero card
gets FirstLight's ability contract in the tracker (cost, charges, cooldown), and play_events
sends a public `ability_activation` with `fair_ability_activation_exact`, so the tracker takes
the exact cost off the opponent's elixir ceiling. Unjoinable activations are logged, never
guessed. `mac012/test_opponent_ability.py`: one hero and two heroes, both sides: ceiling falls
by the ability's 3 elixir (2.82 after 10 ticks of regen), 0 decide errors. Suites re-run:
opponent_registry, hero_evo, effects -- OK. Not yet seen live: needs a friendly where the
opponent account uses a hero ability (log line "opponent used <hero> hero ability").

## TRAINING BRIEF (start here for the training session, 2026-09-25)

Facts the new model must be built around:
- **Command delay is part of the game:** every command executes at issue + 21 ticks (1.05 s),
  for humans too. Our own overhead on top: ~4-6 ticks decision -> issued (inference 70-130 ms,
  tap ~40 ms). FirstLight trained with ~1-5 ticks (self-play) and IL aligned to execution, so
  its models never learned the delay. Plan: train with the real age (FirstLight's engine
  supports scheduled execution), ideally with the delay as an input so a change needs no retrain.
- **Opponent commands are visible ~0.7 s (median 14 ticks) before they land** (queue_lead.py).
  The model should get them as "pending", distinct from real units.
- **Readable live, per 50 ms frame:** units (position, hp, attack stage/timers, charge, deploy
  remaining, target, own data id), towers, projectiles/area effects in flight, own hand / next /
  cycle / elixir, both elixirs (opponent's exact, excluded by FAIR), evolution progress
  (player+0x2e8), hero ability controllers both sides (+0x3a0/+0x3a8), command queue (both
  sides), reveal lists. Opponent hand/cycle: NOT available mid-battle (proven).
- **Opponent deck:** API deck by player tag (opponent_intel.py), trusted until contradicted;
  otherwise learned card by card. Recorded per match.
- **Not read yet:** combat events (damage, stun/freeze, rage, shields), tower troop identity,
  per-turn decision logging. Decide per the chosen input set.
- **Data on disk per match** (artifacts/viewer-sessions/<time>/): frames.jsonl (20 Hz board),
  queue.jsonl (every play/ability, both sides), opponent_intel.jsonl; results in
  build/results.jsonl. Human matches recorded too -> imitation data aligned to *decision* time.
- **Tiers:** deployable = what this client holds; an oracle (opponent hand) cannot run live ->
  use only as critic/teacher.
- **Robustness:** prefer inputs a screen reader could also supply; memory-only extras are a
  bonus (root/memory access could be closed by a game update).
- **Compute / sandbox:** the ARM64 game library is encrypted and runs only inside the real app
  (gate 1 closed). An x86_64 sandbox needs an Intel/AMD PC (campus Windows + 4080, or a
  friend's) -- first test: is the current global x86_64 library readable.
- **Baseline to beat:** FirstLight fl:hog2 via our pipeline, 3-1 vs the user's main account in
  friendlies (all overtime); results logged per model.

## Null's engine setup (CR_4k), checked 2026-09-25 evening

- **Evo Elite Barbarians is in the engine's data.** CR_4k runs FirstLight's offline engine: Null's
  libg SHA 110aa2b5... (same file as cr-engine-extraction/binary) with content 15.535.86
  (fingerprint.json SHA-256 e0cb2fb9..., 986 update files incl. angry_barbarian_evo.toml). The
  old converter lost it because it used the 15.535.29 sandbox's card list, not the engine's data.
- **The "four lazily-fetched files" are not missing.** ghost_ev1, goblin_hut_rework,
  hero_form/balloon_hero_spell and skeleton_balloon_ev1 are inside the Null's APK itself
  (assets/csv_logic/...), byte-identical (SHA-1) to the fingerprint's entries. No fidelity issue;
  the old MACOS-PORT note was wrong.
- **Null's "broken" = full disk.** files/crlive (the probe's per-battle recordings, ~106 MB per
  match) had filled the 10 GB AVD; an interrupted content update left fingerprint "15.535.3" with
  most update files gone. Recordings (129 files, 4.3 GB, md5-verified) moved to
  cr-engine-extraction/macos-port/recordings/crlive_emulator_2026-09-25/; official Clash Royale,
  APKPure and their downloads removed (user's instruction: only Null's on that device). The
  client then re-downloaded 15.535.86 and reached the menu. 6+ GB free = ~60 recorded battles:
  offload crlive periodically.
- The Sep 10 backup /data/local/tmp/gamedata.tgz holds the same 15.535.86 content (same
  fingerprint) if the update directory is ever damaged again.

## Replay conversion: from 126-256 s to ~8 s per replay (2026-09-25 evening)

- **FirstLight's cache builder was the bottleneck, not the engine.** Profile of one replay: 245 of
  256 s waiting on the engine, ~0.6 s per five-tick step, and the run died at ~256 s. The engine
  itself simulates ~38,000 ticks/s (step(1000) = 0.03 s); a plain observe is ~4-7 ms. The cost was
  its rich/atomic snapshot: **7.8 MB and ~0.3-0.7 s per call on a 12-unit board**, almost all event
  histories (combat, phase, movement, state, visibility rings) appended to every snapshot.
- **`observe-lean`** (probe command, `il/probe_observe_lean.patch` against FirstLight_CR's
  probe, deployed to CR_4k): the rich snapshot's per-unit runtime state and players without event
  histories or component inventories. 4.6 ms / 42 KB; every per-unit field and both players
  identical to observe-rich. Training inputs are limited to what a snapshot shows, as live.
  Deployed library md5 05247d1a...; the original (0883d753..., byte-identical to a clean build of
  the unpatched source) is saved on CR_4k at /data/local/tmp/libcrprobe.installed-2026-09-25.so
  and the source backup in runs/probe_backup_2026-09-25/ (git-ignored).
- **`il/engine_convert.py`**: prepare + calibrate (0.1 s), create the match headless, queue every
  recorded card play at its exact execute tick (queue_hand_action_at, FirstLight's tile +
  sub-cell conversion), hero abilities by FirstLight's rule (newest ready unit of the named card),
  a lean snapshot every decision tick, then the end state against the recording (winner, crowns,
  all six towers' final HP). 9 test replays: 4.8-13.5 s each (~8 s), 8/9 winner and crowns right,
  4 exact to the tower HP; the one failure had two abilities no unit accepted.
- **CR_4k boot snapshot.** Every boot restored a snapshot saved 2026-09-24 17:14, an hour after an
  interrupted content update, with a full disk: that is why Null's stayed broken and why fixes
  vanished after a restart (the console boots with -no-snapshot-save). Redone (recordings
  md5-verified against the Mac copy and removed; official CR, APKPure and their downloads removed;
  content re-downloaded, 15.535.86 / 986 files) and saved as the new default_boot at the Null's
  menu with network, 7 GB free and the lean probe installed.
- **Why FirstLight looks stronger in its video:** in its sandbox the AI's plays are queued with
  execute_in_ticks = max(1, offset 0-4) and the engine backdates the command 20 ticks
  (battle_env.py, cr_native_env.queue_hand_action_at): the AI's card lands 1-4 ticks after it
  decides, while a human's tap takes the normal ~21. No bot can do that in the live game.

