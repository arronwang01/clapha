"""Watch a training game in Null's Royale itself: the recorded game played again in the real game, on
the Mac's engine emulator (CR_4k), with FirstLight's direct scheduling (each card registered in the
probe for the tick it went in) in the stock renderer at 0.25-4x, and seeking both ways (watch()).

    ./py -m il.watch_nulls RECORDING [--speed 1] [--check]

The Clapha app's Training games window runs this. Controls on stdin, one per line: pause, resume,
speed X, seek TICK, stop. Progress on stdout, one JSON object per line. --check builds the replay
and prints its setup without touching the emulator.

A game is played again exactly when the battle is set up as it was. Every training deal comes from
runs/conv-hog26, whose battles all use FirstLight's fixed deal seed, where the decks' slot order
IS the deal; the recording keeps both decks in that order, with their forms. Newer recordings also
keep the rest (played.config); for older ones, levels and tower troops are read off the first
snapshot -- the King's and the princess towers' full health give them (TOWERS, from every battle
in runs/conv-hog26). Before the first card the engine's deal is checked against the recording, so
a game that would not replay exactly is refused, not shown wrong.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
FIRSTLIGHT = Path(os.environ.get('FIRSTLIGHT_ROOT') or Path.home() / 'Documents/GitHub/FirstLight_CR')
PORT_DIR = Path.home() / 'Documents/GitHub/cr-engine-extraction/macos-port'
ADB = str(Path.home() / 'Library/Android/sdk/platform-tools/adb')
PACKAGE = 'nullsroyale.rel.free'
PORT = 26789
# FirstLight's attested probe (tools/play_firstlight.sh installs it; its sandbox and player expect it)
PROBE_SHA = '2257d2c7051c5b4dc33d381bc0be2d40ecff3d71dbda517b18bbfba3bb88429d'
DEAL_SEED = 1_784_463_263          # royaleapi_replay.NATIVE_DEAL_SEED: every converted battle's seed
KINGS = {7728: 16, 7032: 15, 6408: 14, 4824: 11}
# (level, a princess tower's full health) -> the tower troop (159000000 Tower Princess, 001 Cannoneer,
# 002 Dagger Duchess, 004 Royal Chef). All 27,660 sides of runs/conv-hog26, one troop per pair.
TOWERS = {(16, 4858): 159000000, (16, 4164): 159000001, (16, 4406): 159000002, (16, 4302): 159000004,
          (15, 4424): 159000000,
          (14, 4032): 159000000, (14, 3657): 159000002, (14, 3571): 159000004,
          (11, 3052): 159000000, (11, 2616): 159000001, (11, 2768): 159000002, (11, 2703): 159000004}
if str(FIRSTLIGHT) not in sys.path:
    sys.path.insert(0, str(FIRSTLIGHT))


def say(**fields) -> None:
    print(json.dumps(fields), flush=True)


def gameclock(tick: int) -> str:
    seconds = tick // 20
    return f'{seconds // 60}:{seconds % 60:02d}'



def _load(path: Path) -> tuple[dict, dict]:
    """The header and the first snapshot (the rest is not needed)."""
    import zstandard
    with path.open('rb') as handle, zstandard.ZstdDecompressor().stream_reader(handle) as reader:
        lines = io.BufferedReader(reader, buffer_size=1 << 20)
        header = json.loads(lines.readline())
        first = json.loads(lines.readline())
    header.pop('expert_actions_pickle', None)
    return header, first


def setup_of(header: dict, first: dict) -> dict:
    """The battle's setup: the recording's own (played.config), or read off its first snapshot."""
    timeline = header['timeline']
    setup = {'deck0': list(timeline['decks'][0]), 'deck1': list(timeline['decks'][1]),
             'deck0_form_availability': list(timeline['form_availability'][0]),
             'deck1_form_availability': list(timeline['form_availability'][1]), 'seed': DEAL_SEED}
    recorded = (header.get('played') or {}).get('config')
    if recorded:
        return {**setup, **{key: value for key, value in recorded.items() if value is not None}}
    full = {int(o['nativeObjectId']): int(o.get('maxHp') or 0) for o in first.get('objects') or ()}
    levels = set()
    for owner, (king, left, right) in ((0, (5000000, 5000001, 5000002)), (1, (5000003, 5000004, 5000005))):
        level = KINGS.get(full.get(king, 0))
        troop = TOWERS.get((level, full.get(left, 0)))
        if level is None or troop is None or full.get(left) != full.get(right):
            raise ValueError(f'unknown tower health for owner {owner}: King {full.get(king)}, princess '
                             f'{full.get(left)}/{full.get(right)} (not in il/watch_nulls.py TOWERS)')
        levels.add(level)
        setup[f'tower_troop{owner}_id'] = troop
    if len(levels) != 1:
        raise ValueError(f'the two sides have different levels {sorted(levels)}')
    level = levels.pop()
    setup.update(level_cap=level, minimum_card_level=level, king_tower_level=level)
    return setup


def names(header: dict) -> tuple[str, str]:
    """Owner 0's and owner 1's names on the battle screen: the learner and its opponent."""
    from il.game_viewer import _opponent, _version
    played = header.get('played') or {}
    learner = f'Learner u{_version(played)}'
    opponent = _opponent(played).replace(' (no delay)', '').replace("frozen pilot3 (ex1's target)", 'pilot3 frozen')
    side = int(played.get('a_side', 0))
    return (learner, opponent[:20]) if side == 0 else (opponent[:20], learner)


def build_replay(path: Path):
    """(FirstLight TrainingReplayV1, the recording's header) for a recorded training game."""
    from native_runner.contracts import ActionKind, ActionV1, EpisodeConfigV1, TargetKind
    from native_runner.match_factory import NATIVE_MATCH_END_TICK
    from native_runner.snapshot import SnapshotOperationV1
    from native_runner.training.replay_archive import TrainingReplayV1
    header, first = _load(path)
    setup = setup_of(header, first)
    timeline = header['timeline']
    owner0, owner1 = names(header)
    groups: dict[int, list] = {}
    for index, play in enumerate(sorted(timeline['plays'], key=lambda p: (int(p['lands']), int(p.get('index', 0))))):
        owner, start = int(play['owner']), int(play['lands'])     # lands = the tick before it executes
        if play.get('kind', 'card') == 'card':
            action = ActionV1(owner=owner, kind=ActionKind.PLAY_CARD, hand_slot=0, card_id=int(play['card_id']),
                              target_kind=TargetKind.GRID, target_grid=tuple(play['grid']), subcell_offset=(0.0, 0.0),
                              execute_offset_ticks=1, action_id=f'{timeline["replay_tag"]}-{index}')
        else:
            action = ActionV1(owner=owner, kind=ActionKind.ACTIVATE_ABILITY, source_entity=0, execute_offset_ticks=1,
                              action_id=f'{timeline["replay_tag"]}-{index}',
                              metadata={'ability_runtime_hints': list(play.get('ability_keys') or ())})
        groups.setdefault(start, []).append(action)
    starts = sorted(groups)
    end_tick = int(header.get('end_tick') or timeline.get('end_tick') or 0)
    operations = tuple(SnapshotOperationV1(actions=tuple(groups[start]), advance_ticks=stop - start,
                                           requested_advance_ticks=stop - start, start_native_tick=start,
                                           end_native_tick=stop)
                       for start, stop in zip(starts, starts[1:] + [max(end_tick, starts[-1] + 1)]))
    tags = {key: setup[key] for key in ('deck0_form_availability', 'deck1_form_availability', 'tower_troop0_id',
                                        'tower_troop1_id', 'level_cap', 'minimum_card_level', 'king_tower_level')
            if setup.get(key) is not None}
    tags.update(location=int(setup.get('location', 15000199)), owner0_name=owner0, owner1_name=owner1,
                end_tick=max(int(NATIVE_MATCH_END_TICK), end_tick + 200))
    episode = EpisodeConfigV1(ruleset_id='clapha-training-recording.v1', deck0=tuple(setup['deck0']),
                              deck1=tuple(setup['deck1']), seed=int(setup['seed']),
                              game_mode=int(setup.get('game_mode', 72000006)), arena=int(setup.get('arena', 54000001)),
                              decision_hz=20.0, event_driven_decisions=False, render_mode='native-render', tags=tags)
    winner = timeline.get('winner')
    replay = TrainingReplayV1(completed_match_index=1, ruleset_id=episode.ruleset_id, episode_config=episode,
                              episode_id=str(timeline['replay_tag']), operations=operations,
                              terminal={'winner': winner if winner in (0, 1) else None, 'native_tick': end_tick},
                              source={'recording': path.name})
    return replay, header


# --- the engine on the Mac --------------------------------------------------------------------

def cr4k() -> str | None:
    """CR_4k's adb serial, by the AVD name the device itself reports (ro.boot.qemu.avd_name).
    Never assume "emulator-5554": MuMu Pro's adbd listens on port 5555 and adb lists anything there
    as emulator-5554 -- the live MuMu device. tools/play_firstlight.sh boots CR_4k on its own ports
    (emulator-5580)."""
    try:
        listed = subprocess.run((ADB, 'devices'), capture_output=True, text=True, timeout=10).stdout
        for line in listed.splitlines()[1:]:
            serial, _, state = line.partition('\t')
            if state.strip() == 'device' and _adb(serial, 'shell', 'getprop', 'ro.boot.qemu.avd_name') == 'CR_4k':
                return serial
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _adb(serial: str, *args: str, timeout: float = 20.0) -> str:
    return subprocess.run((ADB, '-s', serial, *args), capture_output=True, text=True, timeout=timeout).stdout.strip()


def _ready(serial: str | None) -> bool:
    """CR_4k booted, with FirstLight's attested probe inside Null's."""
    if serial is None or _adb(serial, 'shell', 'getprop', 'sys.boot_completed') != '1':
        return False
    lib = _adb(serial, 'shell', f'ls -d /data/app/*/{PACKAGE}*/lib/arm64')
    return bool(lib) and _adb(serial, 'shell', f'sha256sum {lib}/libcrprobe.so').split(' ')[0] == PROBE_SHA


def _request(command: str, timeout: float = 1.5) -> dict | None:
    import socket
    try:
        with socket.create_connection(('127.0.0.1', PORT), timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.sendall(command.encode() + b'\n')
            line = connection.makefile('r').readline()
        return json.loads(line) if line.strip() else None
    except (OSError, ValueError):
        return None


def _cold_ready() -> bool:
    status = _request('status')
    manager = str((status or {}).get('manager', '')).strip().casefold()
    return bool(status and status.get('ok') and status.get('coldReady') and not status.get('configured')
                and int(status.get('generation', 0)) == 0 and manager in {'', '0', '0x0', '(nil)', 'none'})


def _engine_busy() -> str | None:
    """Something else of ours driving the engine: refuse rather than restart it under them."""
    out = subprocess.run(('ps', '-eo', 'args='), capture_output=True, text=True).stdout
    for line in out.splitlines():
        if 'interface_mac.py' in line:
            return "FirstLight's console is open (it owns the engine): close it first"
        if any(f'-m {name}' in line for name in ('il.duel', 'il.rl ', 'il.engine_convert')):
            return 'an engine job is running on the Mac (il.duel / il.rl / il.engine_convert)'
    return None


def prepare_engine() -> None:
    """CR_4k up with FirstLight's attested probe, and a fresh (cold-ready) engine on port 26789."""
    busy = _engine_busy()
    if busy:
        raise RuntimeError(busy)
    serial = cr4k()
    if not _ready(serial):
        say(state='preparing', message='Booting CR_4k and installing FirstLight\'s probe (about a minute the first time)…'
            if serial is None else 'Installing FirstLight\'s probe…')
        result = subprocess.run(('/bin/bash', str(CLAPHA / 'tools/play_firstlight.sh'), '--install-only'),
                                capture_output=True, text=True, timeout=420)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip()[-400:] or 'play_firstlight.sh failed')
        serial = cr4k()
        if serial is None:
            raise RuntimeError('CR_4k did not come up')
    if _cold_ready():
        return
    say(state='preparing', message='Starting Null\'s offline engine (about 30 s)…')
    env = dict(os.environ, CR_ADB=ADB, CR_ADB_SERIAL=serial, CR_CONTROL_PORT=str(PORT))
    result = subprocess.run(('/bin/bash', str(PORT_DIR / 'start_offline.sh')), cwd=str(PORT_DIR), env=env,
                            capture_output=True, text=True, timeout=420)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip()[-400:] or 'start_offline.sh failed')


def shutdown() -> None:
    """Close CR_4k (it idles at ~85% CPU). Only CR_4k: found by name, never the MuMu device."""
    serial = cr4k()
    if serial is not None:
        subprocess.run((ADB, '-s', serial, 'emu', 'kill'), capture_output=True, timeout=20)


# --- playing it -------------------------------------------------------------------------------

SNAPSHOT_EVERY = 120      # ticks between kept snapshots (6 s): a whole game fits the probe's 64 handles
FASTEST = 4.0             # the stock renderer's fastest speed (the probe allows 0.25-4x)


def _deal_mismatch(observation: dict, deals: dict) -> str:
    """'' when the engine dealt both sides what the recording dealt (paused at tick 0)."""
    problems = []
    for player in observation.get('players') or ():
        owner = int(player['owner'])
        hand = [h['cardId'] for h in sorted(player['hand'], key=lambda h: h['handIndex'])]
        cycle = [c['cardId'] for c in sorted(player['cycle'], key=lambda c: c['cycleIndex'])]
        want = deals.get(str(owner)) or deals.get(owner)
        if want and (hand != list(want[0]) or cycle != list(want[1])):
            problems.append(f'owner {owner} was dealt {hand} / {cycle}, the recording {want[0]} / {want[1]}')
    return '; '.join(problems)


def watch(path: Path, speed: float) -> int:
    """Play the game in the stock renderer; seek anywhere, both ways.

    Every card is registered in the probe for its tick up front (FirstLight's direct scheduling:
    the probe resolves it from the live hand at the boundary). A snapshot is kept every
    SNAPSHOT_EVERY ticks for the whole game, so a seek back -- or forward to anywhere already
    played -- restores the nearest one at or before the target and runs the rest exactly
    (at most 2 x 120 ticks at 4x); a seek further ahead runs there at 4x, keeping snapshots on
    the way. After a restore the cards still to come are registered again.

    It holds one tick before the recorded end: once the stock battle ends, its HUD (hands, names,
    clock) is gone for good and a restore brings back only the board, so the last tick is not
    played and every position stays watchable."""
    import bisect
    import queue
    from native_runner.cr_native_env import NativeClashEnv
    from native_runner.training.replay_viewer import (_direct_render_commands, _match_config_from_replay,
                                                       _schedule_direct_render_commands)
    replay, header = build_replay(path)
    end_tick = replay.end_native_tick
    commands = _direct_render_commands(replay)
    targets = [command.target_tick for command in commands]
    prepare_engine()
    say(state='starting', message='Setting up the battle…')
    native = NativeClashEnv('127.0.0.1', PORT, timeout=30.0)
    native.wait_ready(timeout=30.0)
    native.create_native_match(_match_config_from_replay(replay))
    native.pause()
    mismatch = _deal_mismatch(native.observe(), header['timeline']['deals'])
    if mismatch:
        say(state='failed', message='This game cannot be played again exactly: ' + mismatch)
        return 2

    controls: queue.Queue = queue.Queue()

    def read_controls() -> None:
        for line in sys.stdin:
            controls.put(line.strip())
        controls.put('stop')                       # the app went away
    threading.Thread(target=read_controls, daemon=True).start()

    snapshots: dict[int, dict] = {}                # tick -> handle, kept for the whole game
    scheduled: dict[int, int] = {}                 # command index -> the probe's sequence
    failed: set[int] = set()
    checked = 0                                    # cards before this index have been accounted for

    def schedule_after(tick: int) -> None:
        nonlocal checked
        start = bisect.bisect_right(targets, tick)
        scheduled.clear()
        scheduled.update(_schedule_direct_render_commands(native, commands, start_index=start))
        checked = start

    def keep_snapshot() -> None:
        try:
            handle = native.create_snapshot()
            snapshots[int(handle['tick'])] = dict(handle)
        except Exception:  # noqa: BLE001  (mid-step, or the probe is full: the next bucket)
            pass

    schedule_after(0)
    keep_snapshot()
    native.set_speed(speed)
    native.resume()
    paused, finished, heading_to, reached, last_report = False, False, None, 0, 0.0
    say(state='playing', message='Playing in Null\'s (the CR_4k window).', tick=0, end=end_tick, speed=speed,
        of=len(commands))

    def run_to(target: int, tick: int) -> None:
        """From a paused `tick` <= target: exactly to target at 4x, then the chosen speed."""
        if target > tick:
            native.set_speed(FASTEST)
            native.advance_native_render(target - tick)
        native.set_speed(speed)
        if not paused:
            native.resume()

    last = end_tick - 1                            # held here: see the docstring

    def seek(target: int) -> None:
        nonlocal heading_to, finished
        target = max(0, min(int(target), last))
        finished = False                           # reaching the end again reports it again
        native.pause()
        tick = int(native.status()['tick'])
        base = max((t for t in snapshots if t <= target), default=None)
        if base is not None and (target < tick or base > tick):
            tick = int(native.restore(snapshots[base])['tick'])
            schedule_after(tick)
        if target - tick <= 2 * SNAPSHOT_EVERY:
            heading_to = None
            run_to(target, tick)
        else:
            heading_to = target                    # far ahead: run there at 4x, snapshots on the way
            native.set_speed(FASTEST)
            native.resume()

    def step(name: str, value: str) -> None:
        """One control (if any), then one look at the battle."""
        nonlocal paused, finished, heading_to, reached, last_report, speed, checked
        if name == 'pause':
            paused, heading_to = True, None
            native.pause()
            native.set_speed(speed)
            say(state='paused')
        elif name == 'resume':
            paused = False
            native.resume()
            say(state='playing')
        elif name == 'speed' and value:
            speed = float(value)
            if heading_to is None:
                native.set_speed(speed)
            say(speed=speed)
        elif name == 'seek' and value:
            seek(int(float(value)))
            say(state='paused' if paused else 'playing', seeking=heading_to)
        status = native.status()
        tick = int(status['tick'])
        reached = max(reached, tick)
        if tick // SNAPSHOT_EVERY not in {t // SNAPSHOT_EVERY for t in snapshots}:
            keep_snapshot()
        while checked < len(commands) and targets[checked] <= tick:
            receipt = native.replay_schedule_status(scheduled[checked]) if checked in scheduled else {'state': 'failed'}
            if receipt.get('state') == 'pending':
                break
            if receipt.get('state') == 'failed':
                failed.add(checked)
            checked += 1
        if heading_to is not None and tick >= heading_to - 2 * SNAPSHOT_EVERY:
            native.pause()
            target, heading_to = heading_to, None
            run_to(target, int(native.status()['tick']))
            say(state='paused' if paused else 'playing', seeking=None)
        if not finished and heading_to is None and tick >= last - (int(10 * speed) + 3) and not status.get('ended'):
            # the end: exactly to the last tick before it, paused there
            native.pause()
            tick = int(native.status()['tick'])
            if tick < last:
                native.advance_native_render(last - tick)
                tick = last
            finished, paused = True, True
            say(state='done', tick=tick, message=f'End of the game ({gameclock(end_tick)}), as in training'
                + (f', but {len(failed)} card(s) could not be played: it drifted.' if failed else
                   ': every card went in at its tick.') + ' Drag back to watch any part again.')
        if status.get('ended') and not finished:
            finished = True
            say(state='done', tick=tick, message=(
                f'The game ended early, at {gameclock(tick)} instead of {gameclock(end_tick)}: it drifted from training.'
                if tick < end_tick else 'End of the game, as in training (the battle screen closed; dragging back '
                'shows the board without hands).'))
        if time.monotonic() - last_report >= 0.2:
            last_report = time.monotonic()
            say(tick=tick, plays=checked, skipped=len(failed), explored=reached)

    errors = 0
    while True:
        try:
            command = controls.get(timeout=0.05)
        except queue.Empty:
            command = ''
        name, _, value = command.partition(' ')
        if name == 'stop':
            break
        try:
            step(name, value)
            errors = 0
        except Exception as error:  # noqa: BLE001  (one refused command must not end the session)
            errors += 1
            say(message=f'{type(error).__name__}: {error}')
            if errors >= 40:
                say(state='failed', message=f'The engine stopped answering: {error}')
                return 1
            time.sleep(0.05)
    native.pause()
    for handle in snapshots.values():
        try:
            native.release_snapshot(handle)
        except Exception:  # noqa: BLE001
            pass
    say(state='stopped', message='Stopped.')
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('recording', nargs='?', type=Path)
    parser.add_argument('--speed', type=float, default=1.0, choices=(0.25, 0.5, 1.0, 2.0, 4.0))
    parser.add_argument('--check', action='store_true', help='build the replay and print its setup only')
    parser.add_argument('--shutdown', action='store_true', help='close the CR_4k emulator')
    args = parser.parse_args(argv)
    if args.shutdown:
        shutdown()
        say(state='closed', message='CR_4k is shut down.')
        return 0
    if args.recording is None:
        parser.error('a recording is needed')
    try:
        if args.check:
            replay, header = build_replay(args.recording)
            config = replay.episode_config
            print(json.dumps({'played': header.get('played'), 'seed': config.seed, 'deck0': config.deck0,
                              'deck1': config.deck1, 'tags': dict(config.tags), 'cards': len(replay.operations),
                              'ticks': [replay.start_native_tick, replay.end_native_tick]}, default=list))
            return 0
        return watch(args.recording, args.speed)
    except Exception as error:  # noqa: BLE001
        say(state='failed', message=f'{type(error).__name__}: {error}')
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
