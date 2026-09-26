"""Replay conversion on the Null's engine: a board snapshot at every decision tick, saved to disk.

FirstLight's cache builder spends ~0.6 s per five-tick step in its resident multi-battle layer
(measured 2026-09-25: 245 of 256 s waiting on the engine); the engine itself simulates ~38,000
ticks/s in single-battle headless mode. This drives that mode directly with FirstLight's own
pieces: prepare the replay, calibrate the deal (0.1 s), create the match headless, queue each
recorded play at its exact execute tick (queue_hand_action_at), and take a lean snapshot at
every decision tick. At the end the simulated result is compared with the recorded one.

Speed (one replay, 612 snapshots): 8.3 s with a new connection per request and a step plus an
observe-lean per snapshot; 4.3 s on FirstLight's persistent control session; the default mode
here asks the probe's run-lean command (il/probe_observe_lean.patch) for every snapshot up to
the next recorded play in one request, and leaves the provenance/capability tables (the same in
every snapshot, over half the bytes) to meta.json.

Output (--out DIR), resumable, one engine per process (split machines with --shard i/n):
  DIR/frames/<tag[:2]>/<tag>.jsonl.zst   il/frames.py (load_replay); line 1: header (timeline,
                                         calibrated replay, fidelity,
                                         failures); then one bare lean snapshot per line, ticks
                                         5, 10, 15, ... (state after that tick, before any play
                                         queued at it)
  DIR/index.jsonl                        one line per replay tried, ok or not
  DIR/meta.json                          probe attestation and one full observe-lean
  DIR/selection.json                     the replays chosen from the dataset, in order

    ./py -m il.engine_convert --out runs/conv-hog26 [--deck hog26|all] [--shard 0/2] [--serial emulator-5554]
    ./py -m il.engine_convert --test 9 [--mode run|step]      the il-mini replays, printed, not saved
Uses ~/Documents/GitHub/FirstLight_CR (the copy matching the installed engine).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from il.frames import save_replay

FIRSTLIGHT = Path.home() / 'Documents/GitHub/FirstLight_CR'
sys.path.insert(0, str(FIRSTLIGHT))

DECISION_TICKS = 5
IL_REPLAY = Path.home() / 'Documents/GitHub/cr-engine-extraction/IL_Replay'
HOG26 = frozenset({'hog-rider', 'musketeer', 'cannon', 'ice-golem', 'ice-spirit', 'skeletons', 'fireball',
                   'the-log'})


def replay_commands(replay) -> list[dict]:
    """Card plays and abilities with their exact execute tick and native target (FirstLight's
    replay_viewer conversion: tile centre + sub-cell offset * 1000, clamped to the arena)."""
    from native_runner.arena import cell_to_world
    from native_runner.contracts import ActionKind
    commands = []
    for operation in replay.operations:
        for action in operation.actions:
            tick = operation.start_native_tick + max(1, action.execute_offset_ticks or 1)
            if action.kind is ActionKind.PLAY_CARD:
                x, y = cell_to_world(action.target_grid)
                if action.subcell_offset is not None:
                    x += int(round(action.subcell_offset[0] * 1000))
                    y += int(round(action.subcell_offset[1] * 1000))
                commands.append({'tick': tick, 'kind': 'card', 'owner': action.owner, 'card_id': action.card_id,
                                 'hand_slot': action.hand_slot, 'x': min(max(x, 0), 17_999),
                                 'y': min(max(y, 0), 31_999)})
            elif action.kind is ActionKind.ACTIVATE_ABILITY:
                commands.append({'tick': tick, 'kind': 'ability', 'owner': action.owner,
                                 'hints': tuple(action.metadata.get('ability_runtime_hints', ())),
                                 'source_keys': tuple(action.metadata.get('ability_source_keys', ()))})
    return sorted(commands, key=lambda c: c['tick'])


def _request_bytes(native, command: str) -> bytes:
    """One request on the persistent control session, reply unparsed (run-lean replies are ~MBs)."""
    with native._lock:
        try:
            native._ensure_persistent_transport_locked()
            native._persistent_connection.sendall(command.encode() + b'\n')
            reply = native._persistent_reader.readline(64 << 20)
        except OSError:
            native._close_persistent_transport_locked()
            raise
        if not reply.endswith(b'\n'):
            native._close_persistent_transport_locked()
            raise ConnectionError(f'{command!r}: the engine closed the connection mid-reply')
    return reply


def run_lean(native, until: int) -> tuple[list[dict], int, bool]:
    """Advance to tick `until`; bare lean snapshots at every decision tick on the way."""
    import orjson
    frames = []
    while True:
        reply = orjson.loads(_request_bytes(native, f'run-lean {until} {DECISION_TICKS}'))
        if not reply.get('ok'):
            raise RuntimeError(f"run-lean {until}: {reply.get('error', reply)}")
        frames += reply['frames']
        if reply['complete'] or reply['ended']:
            return frames, int(reply['tick']), bool(reply['ended'])


def convert(native, payload: dict, mode: str = 'run') -> dict:
    """mode 'run': run-lean; 'step': a step and an observe-lean per snapshot (the reference)."""
    from native_runner.cr_native_env import HandAction
    from native_runner.royaleapi_replay import (_episode_match_config, calibrate_collected_replay_deal,
                                                prepare_collected_replay)
    from native_runner.training.v4.cache_builder import _headless_calibration_probe

    timing = {}
    t0 = time.perf_counter()
    prepared = prepare_collected_replay(payload)
    calibrated = calibrate_collected_replay_deal(prepared, _headless_calibration_probe(native))
    timing['prepare'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    native.create_match(_episode_match_config(calibrated.replay.episode_config))
    commands = replay_commands(calibrated.replay)
    terminal = calibrated.replay.terminal
    end_tick = int(terminal.get('native_tick') or terminal.get('source_end_tick'))
    frames, failures, queued = [], [], 0
    tick, ended_at = 0, None

    def advance(to: int) -> None:
        nonlocal tick, ended_at
        if to <= tick or ended_at is not None:
            return
        if mode == 'run':
            got, tick, ended = run_lean(native, to)
            frames.extend(got)
        else:
            ended = False
            while tick < to and not ended:
                target = min(tick + DECISION_TICKS - tick % DECISION_TICKS, to)
                result = native.step(target - tick)
                tick, ended = int(result['tick']), bool(result.get('ended'))
                if tick % DECISION_TICKS == 0:
                    state = {k: v for k, v in native.observe().items() if k not in ('objects', 'returned', 'truncated')}
                    frames.append({**native._request('observe-lean'), 'state': state})
        if ended and tick < to:
            ended_at = tick

    for command in commands:
        # a play goes in the tick before it executes, after that tick's snapshot
        advance(command['tick'] - 1)
        if ended_at is not None:
            break
        try:
            if command['kind'] == 'card':
                native.queue_hand_action_at(HandAction(command['owner'], command['hand_slot'], command['x'],
                                                       command['y']), execute_tick=command['tick'])
                queued += 1
            elif queue_ability(native, command):
                queued += 1
            else:
                failures.append((command['tick'], 'ability', 'no ready source unit accepted it'))
        except Exception as error:  # noqa: BLE001
            failures.append((command['tick'], command['kind'], str(error)[:120]))
    advance(end_tick)
    timing['simulate'] = time.perf_counter() - t0

    final = native.observe()
    return {'replay_tag': calibrated.replay_tag, 'calibrated': calibrated, 'frames': frames,
            'end_tick': end_tick, 'ended_at': ended_at, 'commands': len(commands), 'queued': queued,
            'failures': failures, 'timing': timing,
            'fidelity': fidelity(final, terminal, payload, calibrated.replay.episode_config.tags)}


_SOURCE_IDS: dict[str, set[int]] = {}


def ability_source_ids(key: str) -> set[int]:
    """Unit card ids a RoyaleAPI ability source key can appear as on the board: the card and
    its hero form (a hero unit carries the form id, e.g. 203000014 for Hero Musketeer)."""
    if key not in _SOURCE_IDS:
        from native_runner.royaleapi_replay import resolve_native_card
        ids: set[int] = set()
        try:
            card_id = int(resolve_native_card(key).card_id)
            ids.add(card_id)
            catalog = json.loads((Path(__file__).resolve().parents[1] / 'live_card_catalog.json').read_text())
            for card in catalog['cards']:
                if int(card['card_id']) == card_id and card.get('hero_form_id'):
                    ids.add(int(card['hero_form_id']))
        except Exception:  # noqa: BLE001
            pass
        _SOURCE_IDS[key] = ids
    return _SOURCE_IDS[key]


def queue_ability(native, command: dict) -> bool:
    """FirstLight's rule for an ability whose caster the replay does not name: the newest ready
    source. Candidates are the owner's units of the named card, newest first; the engine refuses
    one that is not ready, so the next is tried."""
    from native_runner.cr_native_env import NATIVE_OBJECT_ID_ENTITY_KEY_TAG, AbilityAction
    ids = set().union(*(ability_source_ids(k) for k in command['source_keys'])) if command['source_keys'] else set()
    objects = [o for o in native.observe().get('objects', []) if o.get('owner') == command['owner']
               and (not ids or o.get('cardId') in ids) and o.get('nativeObjectId')]
    for obj in sorted(objects, key=lambda o: -int(o['nativeObjectId'])):
        try:
            native.queue_ability_action_at(AbilityAction(command['owner'], NATIVE_OBJECT_ID_ENTITY_KEY_TAG,
                                                         int(obj['nativeObjectId'])), execute_in_ticks=1)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


# Crown-tower positions (native units), as the reader and FirstLight place them.
TOWERS = {(0, 'king'): (9000, 3000), (0, 'left'): (3500, 6500), (0, 'right'): (14500, 6500),
          (1, 'king'): (9000, 29000), (1, 'left'): (3500, 25500), (1, 'right'): (14500, 25500)}


def fidelity(final: dict, terminal: dict, payload: dict, tags) -> dict:
    """Simulated end state against the recorded one: winner, crowns, every tower's final HP."""
    expected_crowns = {int(p['owner']): int(p['crowns']) for p in terminal.get('players', ())}
    crowns = list(final.get('crownsRaw') or [])
    simulated_crowns = {owner: int(crowns[owner]) for owner in (0, 1) if owner < len(crowns)}
    alive = {}
    for obj in final.get('objects', []):
        for (owner, name), (x, y) in TOWERS.items():
            if obj.get('owner') == owner and abs(obj['x'] - x) <= 600 and abs(obj['y'] - y) <= 600:
                alive[(owner, name)] = int(obj.get('hp') or 0)
    team_owner = int(dict(tags).get('source_team_owner', 0))
    recorded = {}
    for side, owner in (('team', team_owner), ('opponent', 1 - team_owner)):
        hp = payload['battle'][side]['players'][0].get('final_tower_hitpoints') or {}
        recorded[owner] = {'king': hp.get('king'), 'left': hp.get('princess_left'), 'right': hp.get('princess_right')}
    tower_error = 0
    towers = {}
    for owner in (0, 1):
        sim = [alive.get((owner, n), 0) for n in ('king', 'left', 'right')]
        rec = [recorded[owner][n] for n in ('king', 'left', 'right')]
        if rec[0] == 0:
            # three crowns: the recording lists every tower of the loser as 0, whatever the
            # princess towers had left when the king fell; only the king can be compared
            tower_error += sim[0]
            towers[owner] = {'simulated': sim, 'recorded': rec}
            continue
        # left/right can be named from either player's view; take the closer pairing
        straight = sum(abs(a - b) for a, b in zip(sim, rec) if b is not None)
        swapped = abs(sim[0] - (rec[0] or 0)) + abs(sim[1] - (rec[2] or 0)) + abs(sim[2] - (rec[1] or 0))
        tower_error += min(straight, swapped)
        towers[owner] = {'simulated': sim, 'recorded': rec}
    return {'winner_match': final.get('winner') == terminal.get('winner'),
            'crowns_match': simulated_crowns == expected_crowns,
            'crowns': (simulated_crowns, expected_crowns), 'tower_hp_error': tower_error, 'towers': towers}


# ---- dataset -------------------------------------------------------------------------------

def is_hog26(payload: dict) -> bool:
    battle = payload['battle']
    for side in ('team', 'opponent'):
        deck = {c['card_key'].replace('-ev1', '').replace('-hero', '')
                for p in (battle.get(side) or {}).get('players', []) for c in p.get('deck', [])}
        if HOG26 <= deck:
            return True
    return False


def select(dataset: Path, deck: str, out: Path) -> list[tuple[str, int, str]]:
    """(part file, row, replay tag) of every replay to convert, cached in out/selection.json."""
    cache = out / 'selection.json'
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved['dataset'] == str(dataset) and saved['deck'] == deck:
            return [tuple(item) for item in saved['replays']]
    import pyarrow.parquet as pq
    chosen = []
    for part in sorted(glob.glob(str(dataset / 'replays' / '*.parquet'))):
        table = pq.read_table(part, columns=['replay_tag', 'payload_json'])
        for row, (tag, payload) in enumerate(zip(table['replay_tag'].to_pylist(), table['payload_json'].to_pylist())):
            if deck == 'all' or is_hog26(json.loads(payload)):
                chosen.append((Path(part).name, row, tag))
        print(f'  selecting: {Path(part).name}, {len(chosen)} so far', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({'dataset': str(dataset), 'deck': deck, 'replays': chosen}))
    return chosen


# ---- engine --------------------------------------------------------------------------------

def adb_path() -> str:
    for candidate in (os.environ.get('ADB'), str(Path.home() / 'Library/Android/sdk/platform-tools/adb')):
        if candidate and Path(candidate).exists():
            return candidate
    return 'adb'


def restart_engine(serial: str, native, timeout: float = 180.0) -> None:
    """Force-stop and relaunch Null's (the probe comes back headless), then wait until ready."""
    adb = adb_path()
    subprocess.run([adb, '-s', serial, 'shell', 'am force-stop nullsroyale.rel.free; '
                    'am start -n nullsroyale.rel.free/com.supercell.clashroyale.GameApp'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    native.close_transport()
    deadline = time.monotonic() + timeout
    while True:
        try:
            native.wait_ready(timeout=5.0)
            return
        except Exception:  # noqa: BLE001
            if time.monotonic() > deadline:
                raise
            time.sleep(2.0)


def connect(port: int):
    from native_runner.cr_native_env import NativeClashEnv
    native = NativeClashEnv(host='127.0.0.1', port=port, timeout=120.0)
    native._persistent_transport_requested = True   # one TCP session instead of one per request
    return native


def check_probe(native) -> None:
    """The installed probe must know run-lean (an unknown command is answered with the command list)."""
    import orjson
    reply = orjson.loads(_request_bytes(native, 'run-lean 0 5'))
    if not reply.get('ok') and str(reply.get('error', '')).startswith('commands:'):
        raise SystemExit('the installed probe has no run-lean: install runs/libcrprobe_run.so')


def engine_answers(native) -> bool:
    try:
        return bool(native._request('status').get('ok'))
    except Exception:  # noqa: BLE001
        return False


# ---- main ----------------------------------------------------------------------------------

def ended_early(result: dict) -> bool:
    """The engine's battle ended well before the recorded one (the recording's last tick is 80-110
    ticks after the battle ends, so every replay "ends" a little early)."""
    return result['ended_at'] is not None and result['end_tick'] - result['ended_at'] > 150


def _summary(result: dict, seconds: float) -> str:
    f = result['fidelity']
    return (f"{result['replay_tag'][:8]} {seconds:4.1f}s {len(result['frames'])} frames, plays "
            f"{result['queued']}/{result['commands']}, winner {'ok' if f['winner_match'] else 'WRONG'}, "
            f"crowns {'ok' if f['crowns_match'] else 'WRONG'}, tower HP off {f['tower_hp_error']}"
            + (f", ENGINE ENDED AT {result['ended_at']} (recording {result['end_tick']})"
               if ended_early(result) else ''))


def test(native, count: int, mode: str) -> int:
    import pyarrow.parquet as pq
    rows = []
    for part in sorted(glob.glob(str(Path(__file__).resolve().parents[1] / 'runs/il-mini/dataset/replays/*.parquet'))):
        rows += [json.loads(r['payload_json']) for r in pq.read_table(part, columns=['payload_json']).to_pylist()]
    for payload in rows[:count]:
        started = time.perf_counter()
        result = convert(native, payload, mode)
        print(_summary(result, time.perf_counter() - started), flush=True)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--dataset', type=Path, default=IL_REPLAY)
    parser.add_argument('--deck', default='hog26', choices=('hog26', 'all'))
    parser.add_argument('--shard', default='0/1', help='i/n: this process takes the replays whose tag hash is i mod n')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--port', type=int, default=26789)
    parser.add_argument('--serial', default=None, help='adb serial: relaunch the engine if it stops answering')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--test', type=int, default=0, help='convert N il-mini replays, print, save nothing')
    parser.add_argument('--mode', default='run', choices=('run', 'step'))
    args = parser.parse_args(argv)

    native = connect(args.port)
    native.wait_ready(timeout=60.0)
    if args.test:
        return test(native, args.test, args.mode)
    if args.out is None:
        parser.error('--out is required')

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    check_probe(native)
    meta_path = out / 'meta.json'

    shard, shards = (int(v) for v in args.shard.split('/'))
    chosen = [item for item in select(args.dataset, args.deck, out)
              if int(hashlib.sha256(item[2].encode()).hexdigest()[:8], 16) % shards == shard]
    index_path = out / 'index.jsonl'
    done = set()
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            record = json.loads(line)
            if record['ok'] or not args.retry_failed:
                done.add(record['tag'])
    todo = [item for item in chosen if item[2] not in done]
    print(f'{len(chosen)} replays in shard {args.shard}, {len(chosen) - len(todo)} already done, '
          f'{len(todo)} to convert' + (f' (this run: {args.limit})' if args.limit else ''), flush=True)
    if args.limit:
        todo = todo[:args.limit]

    import pyarrow.parquet as pq
    started_all, converted, byte_total = time.perf_counter(), 0, 0
    by_part: dict[str, list[tuple[int, str]]] = {}
    for part, row, tag in todo:
        by_part.setdefault(part, []).append((row, tag))
    with index_path.open('a') as index:
        for part, rows in by_part.items():
            payloads = pq.read_table(args.dataset / 'replays' / part, columns=['payload_json'])['payload_json']
            for row, tag in rows:
                started = time.perf_counter()
                record = {'tag': tag, 'part': part, 'row': row}
                for attempt in (1, 2):
                    try:
                        result = convert(native, json.loads(payloads[row].as_py()), args.mode)
                        size = save_replay(out, result, DECISION_TICKS)
                        if not meta_path.exists():
                            # the tables bare snapshots leave out, and what produced them
                            meta_path.write_text(json.dumps({'attest': native._request('attest'),
                                                             'observe_lean': native._request('observe-lean')}))
                        f = result['fidelity']
                        record.update(ok=True, frames=len(result['frames']), end_tick=result['end_tick'],
                                      ended_at=result['ended_at'], commands=result['commands'],
                                      queued=result['queued'], failures=len(result['failures']),
                                      winner_match=f['winner_match'], crowns_match=f['crowns_match'],
                                      tower_hp_error=f['tower_hp_error'], bytes=size,
                                      seconds=round(time.perf_counter() - started, 2))
                        record.pop('error', None)
                        byte_total += size
                        print(_summary(result, time.perf_counter() - started), flush=True)
                        break
                    except Exception as error:  # noqa: BLE001
                        record.update(ok=False, error=f'{type(error).__name__}: {error}'[:300])
                        if engine_answers(native):
                            print(f'{tag[:8]}: failed: {record["error"]}', flush=True)
                            break
                        # the engine stopped answering: relaunch it and try this replay once more
                        print(f'{tag[:8]}: engine lost ({record["error"][:100]})', flush=True)
                        if args.serial is None:
                            raise SystemExit('engine lost and no --serial to relaunch it') from error
                        restart_engine(args.serial, native)
                        record['relaunched'] = attempt
                index.write(json.dumps(record) + '\n')
                index.flush()
                converted += 1
                if converted % 50 == 0:
                    rate = (time.perf_counter() - started_all) / converted
                    print(f'-- {converted}/{len(todo)} in {time.perf_counter() - started_all:.0f}s, '
                          f'{rate:.2f} s per replay, ~{rate * (len(todo) - converted) / 3600:.1f} h left, '
                          f'{byte_total / 1e6:.0f} MB written', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
