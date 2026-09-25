"""Lean replay conversion on the Null's engine: board snapshots at every decision tick.

FirstLight's cache builder spends ~0.6 s per five-tick step in its resident multi-battle layer
(measured 2026-09-25: 245 of 256 s waiting on the engine); the engine itself simulates ~38,000
ticks/s in single-battle headless mode. This drives that mode directly with FirstLight's own
pieces: prepare the replay, calibrate the deal (0.1 s), create the match headless, queue each
recorded play at its exact execute tick (queue_hand_action_at), and observe at every decision
tick. At the end the simulated result is compared with the recorded one.

    ./py -m il.engine_convert [--replays 1] [--observe lean|plain|rich|atomic]
Uses ~/Documents/GitHub/FirstLight_CR (the copy matching the installed engine).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

FIRSTLIGHT = Path.home() / 'Documents/GitHub/FirstLight_CR'
sys.path.insert(0, str(FIRSTLIGHT))

DECISION_TICKS = 5


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


def convert(native, payload: dict, observe: str = 'lean') -> dict:
    from native_runner.cr_native_env import HandAction
    from native_runner.royaleapi_replay import (_episode_match_config, calibrate_collected_replay_deal,
                                                prepare_collected_replay)
    from native_runner.training.v4.cache_builder import _headless_calibration_probe

    timing = {}
    t0 = time.perf_counter()
    prepared = prepare_collected_replay(payload)
    calibrated = calibrate_collected_replay_deal(prepared, _headless_calibration_probe(native))
    timing['prepare+calibrate'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    native.create_match(_episode_match_config(calibrated.replay.episode_config))
    commands = replay_commands(calibrated.replay)
    end_tick = int(calibrated.replay.terminal.get('native_tick') or calibrated.replay.terminal.get('source_end_tick'))
    # 'lean' = observe-rich's per-unit runtime state without event histories (our probe command,
    # 2026-09-25): 4.6 ms / 42 KB against 301 ms / 7.8 MB on a 12-unit board, fields identical.
    snapshot = {'plain': native.observe, 'lean': lambda: native._request('observe-lean'),
                'rich': native.observe_rich, 'atomic': native.observe_atomic}[observe]
    frames, failures, queued = [], [], 0
    tick, next_command = 0, 0
    while tick < end_tick:
        target = min(tick + DECISION_TICKS - tick % DECISION_TICKS, end_tick)
        # queue every play whose card must go in during this stretch, from the tick before it executes
        while next_command < len(commands) and commands[next_command]['tick'] - 1 <= target:
            command = commands[next_command]
            wanted = command['tick'] - 1
            if wanted > tick:
                native.step(wanted - tick)
                tick = wanted
            try:
                if command['kind'] == 'card':
                    native.queue_hand_action_at(HandAction(command['owner'], command['hand_slot'], command['x'],
                                                           command['y']), execute_tick=command['tick'])
                    queued += 1
                else:
                    if queue_ability(native, command):
                        queued += 1
                    else:
                        failures.append((command['tick'], 'ability', 'no ready source unit accepted it'))
            except Exception as error:  # noqa: BLE001
                failures.append((command['tick'], command['kind'], str(error)[:120]))
            next_command += 1
        if target > tick:
            native.step(target - tick)
            tick = target
        if tick % DECISION_TICKS == 0:
            frames.append(snapshot())
    timing['simulate+observe'] = time.perf_counter() - t0

    final = native.observe()
    terminal = calibrated.replay.terminal
    return {'replay_tag': calibrated.replay_tag, 'frames': frames, 'end_tick': end_tick,
            'commands': len(commands), 'queued': queued, 'failures': failures, 'timing': timing,
            'fidelity': fidelity(final, terminal, payload, calibrated.replay.episode_config.tags),
            'frame_bytes': sum(len(json.dumps(f)) for f in frames[:50]) / max(1, min(50, len(frames)))}


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
        # left/right can be named from either player's view; take the closer pairing
        straight = sum(abs(a - b) for a, b in zip(sim, rec) if b is not None)
        swapped = abs(sim[0] - (rec[0] or 0)) + abs(sim[1] - (rec[2] or 0)) + abs(sim[2] - (rec[1] or 0))
        tower_error += min(straight, swapped)
        towers[owner] = {'simulated': sim, 'recorded': rec}
    return {'winner_match': final.get('winner') == terminal.get('winner'),
            'crowns_match': simulated_crowns == expected_crowns,
            'crowns': (simulated_crowns, expected_crowns), 'tower_hp_error': tower_error, 'towers': towers}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--replays', type=int, default=1)
    parser.add_argument('--observe', default='lean', choices=('plain', 'lean', 'rich', 'atomic'))
    parser.add_argument('--dataset', type=Path, default=Path('runs/il-mini/dataset'))
    parser.add_argument('--port', type=int, default=26789)
    args = parser.parse_args(argv)
    import pyarrow.parquet as pq
    from native_runner.cr_native_env import NativeClashEnv
    rows = []
    for part in sorted(glob.glob(str(args.dataset / 'replays' / '*.parquet'))):
        rows += [json.loads(r['payload_json']) for r in pq.read_table(part, columns=['payload_json']).to_pylist()]
    native = NativeClashEnv(host='127.0.0.1', port=args.port, timeout=120.0)
    for payload in rows[:args.replays]:
        started = time.perf_counter()
        result = convert(native, payload, args.observe)
        total = time.perf_counter() - started
        print(f"{result['replay_tag'][:8]}: {total:.1f} s ({', '.join(f'{k} {v:.1f}s' for k, v in result['timing'].items())}); "
              f"{len(result['frames'])} frames ~{result['frame_bytes'] / 1000:.0f} KB each; "
              f"plays {result['queued']}/{result['commands']} queued ({len(result['failures'])} skipped); "
              f"winner {'OK' if result['fidelity']['winner_match'] else 'WRONG'}, crowns "
              f"{'OK' if result['fidelity']['crowns_match'] else 'WRONG'} {result['fidelity']['crowns'][0]}, "
              f"tower HP off by {result['fidelity']['tower_hp_error']}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
