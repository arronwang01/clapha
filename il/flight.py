"""Ticks from a command's execution to its arrival on the target tile, for cards that travel there.

A troop or building is on the board the tick after its command executes. A thrown spell (Fireball,
Rocket, Arrows, Goblin Barrel, Snowball, ...) first flies from its owner's King Tower, and a Miner
or Goblin Drill tunnels from there, so its effect lands later, by an amount that grows with the
distance. The pending-card input (il/extras.py) gives the model both times: a pending Goblin
Barrel says when its goblins land, not only when it was cast.

FLIGHT is measured in Null's engine (the game's own code) with `python -m il.flight --measure`:
every listed card is cast by side 0 at targets at several distances; arrival is the first tick
something of that play is on the target (a unit, an area effect, the goblins of a barrel) or its
projectile is gone (impact). Each card gets base + per_tile * (tiles from its King Tower).

    ./py -m il.flight --measure [--port 26789]      measure and print the table (engine up)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

KING_TOWER = {0: (9000, 3000), 1: (9000, 29000)}     # native units (firstlight_obs.TOWERS)
UNIT_APPEARS_AFTER = 1                               # a troop is on the board at execute + 1

# card id -> (base ticks, ticks per tile from the King Tower); cards not listed arrive at
# execute + UNIT_APPEARS_AFTER. Filled from --measure (NOTES "Spell flight times").
FLIGHT: dict[int, tuple[float, float]] = {}

TRAVELLERS = {
    28000000: 'Fireball', 28000001: 'Arrows', 28000003: 'Rocket', 28000004: 'GoblinBarrel',
    28000005: 'Freeze', 28000007: 'Lightning', 28000008: 'Zap', 28000009: 'Poison', 28000010: 'Graveyard',
    28000011: 'Log', 28000012: 'Tornado', 28000014: 'Earthquake', 28000015: 'BarbLog', 28000017: 'Snowball',
    28000018: 'RoyalDelivery', 28000023: 'DarkMagic', 28000024: 'GoblinCurse', 28000026: 'Vines',
    26000032: 'Miner', 27000013: 'GoblinDrill',
}
# side 0 casts at these native points (tile centres): its own half, the bridge, the river,
# the opponent's towers and their pocket
TARGETS = ((9500, 8500), (3500, 14500), (14500, 17500), (9500, 20500), (3500, 24500), (14500, 24500),
           (9500, 26500))


def distance_tiles(owner: int, tile) -> float:
    """Tiles from the owner's King Tower to a tile (col, row) centre."""
    kx, ky = KING_TOWER[int(owner)]
    return math.hypot(tile[0] * 1000 + 500 - kx, tile[1] * 1000 + 500 - ky) / 1000.0


def flight_ticks(card_id: int, owner: int, tile) -> int:
    """Ticks from the command's execution until it arrives on its target tile (col, row)."""
    fit = FLIGHT.get(int(card_id))
    if fit is None or tile is None:
        return UNIT_APPEARS_AFTER
    base, per_tile = fit
    return max(0, round(base + per_tile * distance_tiles(owner, tile)))


def _snapshots(native, until: int) -> tuple[list[dict], int, bool]:
    """Every tick's lean snapshot up to `until` (il.engine_convert.run_lean at interval 1)."""
    import orjson
    from il.engine_convert import _request_bytes
    frames = []
    while True:
        reply = orjson.loads(_request_bytes(native, f'run-lean {until} 1'))
        if not reply.get('ok'):
            raise RuntimeError(f"run-lean {until}: {reply.get('error', reply)}")
        frames += reply['frames']
        if reply['complete'] or reply['ended']:
            return frames, int(reply['tick']), bool(reply['ended'])


def measure(port: int, frames_dir: Path) -> list[dict]:
    """Cast every TRAVELLERS card at every target for side 0 (side 1 idle); one row per cast."""
    from dataclasses import replace
    from il.engine_convert import connect
    from il.frames import load_replay
    from native_runner.cr_native_env import HandAction
    from native_runner.royaleapi_replay import _episode_match_config
    import il.samples as S

    native = connect(port)
    native.wait_ready(timeout=60.0)
    header, _frames = load_replay(sorted((frames_dir / 'frames').glob('*/*.jsonl.zst'))[0])
    episode = header['calibrated'].replay.episode_config
    cards = list(TRAVELLERS)
    todo = [(card, target) for card in cards for target in TARGETS]
    rows: list[dict] = []
    opponent_deck = tuple(sorted(S.HOG26_CARDS))
    while todo:
        # eight of the remaining cards at a time, normal forms; the opponent never plays
        batch = list(dict.fromkeys(card for card, _target in todo))[:8]
        filler = [card for card in cards if card not in batch]
        deck = tuple((batch + filler)[:8])
        tags = dict(episode.tags)
        tags['deck0_form_availability'] = (0,) * 8
        tags['deck1_form_availability'] = (0,) * 8
        native.create_match(_episode_match_config(replace(episode, deck0=deck, deck1=opponent_deck, tags=tags)))
        tick, ended, stalled = 0, False, 0
        while not ended and todo and stalled < 3:
            state = native.observe()
            me = next(p for p in state['players'] if p['owner'] == 0)
            hand = {int(h['cardId']): int(h['handIndex']) for h in me['hand']}
            playable = [(card, target) for card, target in todo if card in hand]
            if not playable:
                # cycle: play the hand card with the most casts left anywhere cheap, at the first target
                stalled += 1
                if not hand:
                    break
                card = next(iter(hand))
                playable = [(card, TARGETS[0])]
            card, target = playable[0]
            cost = S._card_cost(card) or 0.0
            waited = 0
            while me['elixirRaw'] / 10000.0 < cost and waited < 400 and not ended:
                _got, tick, ended = _snapshots(native, tick + 10)
                waited += 10
                state = native.observe()
                me = next(p for p in state['players'] if p['owner'] == 0)
                hand = {int(h['cardId']): int(h['handIndex']) for h in me['hand']}
            if ended or card not in hand:
                continue
            execute = tick + 2
            before = {int(o['nativeObjectId']) for o in state.get('objects') or ()}
            try:
                native.queue_hand_action_at(HandAction(0, hand[card], target[0], target[1]), execute_tick=execute)
            except Exception as error:  # noqa: BLE001  (a placement the rules refuse)
                rows.append({'card': card, 'target': target, 'error': str(error)[:120]})
                todo.remove((card, target))
                continue
            frames, tick, ended = _snapshots(native, execute + 120)
            arrival, how, tracked = None, None, {}
            for frame in frames:
                t = int(frame['tick'])
                if t < execute:
                    continue
                seen_now = set()
                for o in frame.get('objects') or ():
                    oid = int(o['nativeObjectId'])
                    if oid in before or int(o['owner']) != 0:
                        continue
                    seen_now.add(oid)
                    near = math.hypot(int(o['x']) - target[0], int(o['y']) - target[1]) <= 1500
                    tracked.setdefault(oid, {'first': t, 'card': int(o['cardId']), 'unit': o.get('hp') is not None})
                    tracked[oid]['last'] = t
                    tracked[oid]['near'] = near
                    if near and arrival is None:
                        arrival, how = t, ('unit' if o.get('hp') is not None else 'effect')
                for oid, info in tracked.items():
                    if (arrival is None and oid not in seen_now and not info.get('gone')
                            and not info['unit']):
                        info['gone'] = t
                        arrival, how = t, 'impact'
                if arrival is not None:
                    break
            if (card, target) in todo:
                todo.remove((card, target))
            rows.append({'card': card, 'name': TRAVELLERS.get(card, str(card)), 'target': target,
                         'tiles': round(math.hypot(target[0] - 9000, target[1] - 3000) / 1000.0, 2),
                         'execute': execute, 'flight': None if arrival is None else arrival - execute, 'how': how})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def fit(rows: list[dict]) -> dict[int, tuple[float, float]]:
    """Least squares base + per_tile per card over its measured casts."""
    table = {}
    for card in sorted({r['card'] for r in rows if r.get('flight') is not None}):
        points = [(r['tiles'], r['flight']) for r in rows if r['card'] == card and r.get('flight') is not None]
        n = len(points)
        mean_x = sum(x for x, _y in points) / n
        mean_y = sum(y for _x, y in points) / n
        spread = sum((x - mean_x) ** 2 for x, _y in points)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread if spread else 0.0
        table[card] = (round(mean_y - slope * mean_x, 2), round(slope, 3))
    return table


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--measure', action='store_true')
    parser.add_argument('--port', type=int, default=26789)
    parser.add_argument('--frames', type=Path, default=Path(__file__).resolve().parents[1] / 'runs/conv-hog26')
    parser.add_argument('--out', type=Path, help='also write the rows here (jsonl)')
    args = parser.parse_args(argv)
    if not args.measure:
        print(json.dumps({str(k): v for k, v in FLIGHT.items()}, indent=1))
        return 0
    rows = measure(args.port, args.frames)
    if args.out:
        args.out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    for card, (base, slope) in fit(rows).items():
        print(f'    {card}: ({base}, {slope}),   # {TRAVELLERS.get(card, card)}')
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mac012'))
    raise SystemExit(main(sys.argv[1:]))
