"""Watch a training game in Null's Royale itself: the recorded game played again in the real game, on
the Mac's engine emulator (CR_4k), by FirstLight's own replay player
(native_runner/training/replay_viewer.py: each card is scheduled in the probe for the tick it went
in, and the stock renderer runs at 0.25-4x with pause and 5 s back).

    ./py -m il.watch_nulls RECORDING [--speed 1] [--check]

The Clapha app's Training games window runs this. Controls on stdin, one per line: pause, resume,
speed X, back, stop. Progress on stdout, one JSON object per line. --check builds the replay and
prints its setup without touching the emulator.

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
import hashlib
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
SERIAL = 'emulator-5554'
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

def _adb(*args: str, timeout: float = 20.0) -> str:
    return subprocess.run((ADB, '-s', SERIAL, *args), capture_output=True, text=True, timeout=timeout).stdout.strip()


def _booted() -> bool:
    try:
        listed = subprocess.run((ADB, 'devices'), capture_output=True, text=True, timeout=10).stdout
        return any(line.startswith(SERIAL + '\t') for line in listed.splitlines()) and _adb('shell', 'getprop',
                                                                                            'sys.boot_completed') == '1'
    except (OSError, subprocess.TimeoutExpired):
        return False


def _probe_installed() -> bool:
    lib = _adb('shell', f'ls -d /data/app/*/{PACKAGE}*/lib/arm64')
    return bool(lib) and _adb('shell', f'sha256sum {lib}/libcrprobe.so').split(' ')[0] == PROBE_SHA


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
    if not _booted() or not _probe_installed():
        say(state='preparing', message='Booting CR_4k and installing FirstLight\'s probe (about a minute the first time)…'
            if not _booted() else 'Installing FirstLight\'s probe…')
        result = subprocess.run(('/bin/bash', str(CLAPHA / 'tools/play_firstlight.sh'), '--install-only'),
                                capture_output=True, text=True, timeout=420)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip()[-400:] or 'play_firstlight.sh failed')
    if _cold_ready():
        return
    say(state='preparing', message='Starting Null\'s offline engine (about 30 s)…')
    env = dict(os.environ, CR_ADB=ADB, CR_ADB_SERIAL=SERIAL, CR_CONTROL_PORT=str(PORT))
    result = subprocess.run(('/bin/bash', str(PORT_DIR / 'start_offline.sh')), cwd=str(PORT_DIR), env=env,
                            capture_output=True, text=True, timeout=420)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip()[-400:] or 'start_offline.sh failed')


def shutdown() -> None:
    """Close the emulator (it idles at ~85% CPU)."""
    subprocess.run((ADB, '-s', SERIAL, 'emu', 'kill'), capture_output=True, timeout=20)


# --- playing it -------------------------------------------------------------------------------

def watch(path: Path, speed: float) -> int:
    from native_runner.training.replay_viewer import ReplayRewindControl, play_training_replay
    replay, header = build_replay(path)
    deals = header['timeline']['deals']
    prepare_engine()
    stop, pause, rewind = threading.Event(), threading.Event(), ReplayRewindControl()
    native_box: list = []
    mismatch: list[str] = []
    skipped = [0]

    def ready(native) -> None:
        # paused at tick 0: the engine must deal what the recording dealt, or the game is not this one
        observed = native.observe()
        for player in observed.get('players') or ():
            owner = int(player['owner'])
            hand = [h['cardId'] for h in sorted(player['hand'], key=lambda h: h['handIndex'])]
            cycle = [c['cardId'] for c in sorted(player['cycle'], key=lambda c: c['cycleIndex'])]
            want = deals.get(str(owner)) or deals.get(owner)
            if want and (hand != list(want[0]) or cycle != list(want[1])):
                mismatch.append(f'owner {owner} was dealt {hand} / {cycle}, the recording {want[0]} / {want[1]}')
        if mismatch:
            stop.set()
            return
        native_box.append(native)
        say(state='playing', message='Playing in Null\'s (the CR_4k window).', tick=0, speed=speed)

    def progress(value) -> None:
        if str(value.phase).startswith('skipped'):
            skipped[0] += 1
        say(state='playing', tick=int(value.native_tick), plays=int(value.operation_index),
            of=int(value.operation_count), phase=str(value.phase), skipped=skipped[0])

    def controls() -> None:
        for line in sys.stdin:
            command, _, value = line.strip().partition(' ')
            if command == 'pause':
                pause.set()
                say(state='paused')
            elif command == 'resume':
                pause.clear()
                say(state='playing')
            elif command == 'speed' and native_box:
                try:
                    native_box[0].set_speed(float(value))
                    say(speed=float(value))
                except Exception as error:  # noqa: BLE001
                    say(message=f'speed: {error}')
            elif command == 'back':
                say(message='Back 5 s…' if rewind.request_rewind() else rewind.message)
            elif command == 'stop':
                pause.clear()
                stop.set()
        stop.set()                                  # the app went away

    def clock() -> None:
        # the tick for the app's board, four times a second (FirstLight reports only at each card)
        while not stop.is_set():
            time.sleep(0.25)
            if native_box and not pause.is_set():
                try:
                    status = native_box[0].status()
                    say(tick=int(status['tick']), ended=bool(status.get('ended')))
                except Exception:  # noqa: BLE001  (the session is closing)
                    pass
    threading.Thread(target=controls, daemon=True).start()
    threading.Thread(target=clock, daemon=True).start()
    say(state='starting', message='Setting up the battle…')
    result = play_training_replay(replay, port=PORT, speed=speed, stop_event=stop, pause_event=pause,
                                  on_native_ready=ready, on_progress=progress, rewind_control=rewind)
    stop.set()
    if mismatch:
        say(state='failed', message='This game cannot be played again exactly: ' + '; '.join(mismatch))
        return 2
    same = result.expected_winner == result.actual_winner
    say(state='stopped' if result.stopped else 'done', tick=int(result.final_native_tick),
        message=('Stopped.' if result.stopped else
                 ('Finished: same result as in training.' if same else
                  f'Finished, but the winner differs from training (owner {result.actual_winner} won, '
                  f'owner {result.expected_winner} in training): the replay drifted.'))
        + (f' {skipped[0]} card(s) could not be played.' if skipped[0] else ''))
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
