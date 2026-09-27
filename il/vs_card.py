"""How the Hog 2.6 side answers one opponent card, in recorded engine games (il.duel --record; engine
games only, never the user's battles).

For every play of the card by the other side (default Bandit, 26000046): the unit is followed from
its landing until it dies, and the Hog side's plays from shortly before it lands until then are
scored -- which cards, when (ticks after the unit landed), how far from the unit they landed, the
elixir spent, and the tower health the Hog side lost while the unit lived (+1 s). The Ice Golem is
singled out: its landing time and its distance to the unit at that moment. Per model (the
recording's `a` label):

    ./py -m il.vs_card DIR [DIR ...] [--card 26000046]
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))

ICE_GOLEM = 26000038
TOWERS = range(5000000, 5000006)


def _objects(frame: dict, owner: int, card: int) -> list[dict]:
    return [o for o in frame.get('objects') or () if int(o.get('owner', -1)) == owner
            and int(o.get('cardId') or 0) == card and o.get('hp') is not None]


def _tower_hp(frame: dict, owner: int) -> int:
    return sum(int(o['hp']) for o in frame.get('objects') or ()
               if int(o['nativeObjectId']) in TOWERS and int(o['owner']) == owner and o.get('hp') is not None)


def score(path: Path, card: int) -> list[dict]:
    import il.samples as S
    from il.frames import load_replay
    header, frames = load_replay(path, with_replay=False)
    played = header.get('played') or {}
    ours = played.get('a_side')
    if ours not in (0, 1):
        return []
    theirs = 1 - ours
    by_tick = sorted(frames, key=lambda f: int(f['tick']))
    ticks = [int(f['tick']) for f in by_tick]

    def frame_at(tick: int) -> dict | None:
        for index, value in enumerate(ticks):
            if value >= tick:
                return by_tick[index]
        return None
    plays = header['timeline']['plays']
    rows = []
    for play in plays:
        if play['owner'] != theirs or play.get('card_id') != card or play.get('grid') is None:
            continue
        landed = int(play['lands']) + 1
        x0, y0 = play['grid'][0] * 1000 + 500, play['grid'][1] * 1000 + 500
        start = frame_at(landed + 2)
        if start is None:
            continue
        candidates = _objects(start, theirs, card)
        if not candidates:
            continue
        unit = min(candidates, key=lambda o: math.hypot(int(o['x']) - x0, int(o['y']) - y0))
        uid = unit['nativeObjectId']
        track = {}
        for frame in by_tick:
            tick = int(frame['tick'])
            if tick < landed:
                continue
            match = next((o for o in frame.get('objects') or () if o['nativeObjectId'] == uid), None)
            if match is None and track:
                break
            if match is not None:
                track[tick] = (int(match['x']), int(match['y']))
        if not track:
            continue
        died = max(track)
        before = frame_at(landed)
        after = frame_at(died + 20) or by_tick[-1]
        answers = []
        for other in plays:
            if other['owner'] != ours or other.get('kind') != 'card' or other.get('grid') is None:
                continue
            executed = int(other['lands']) + 1
            if not landed - 10 <= executed <= died:
                continue
            near = min(track, key=lambda t: abs(t - executed))
            ux, uy = track[near]
            gx, gy = other['grid'][0] * 1000 + 500, other['grid'][1] * 1000 + 500
            answers.append({'card': int(other['card_id']), 'dt': executed - landed,
                            'distance': math.hypot(gx - ux, gy - uy) / 1000.0,
                            'cost': S._card_cost(int(other['card_id'])) or 0.0})
        golem = next((a for a in answers if a['card'] == ICE_GOLEM), None)
        rows.append({'model': played.get('a'), 'lifetime': died - landed, 'answers': answers,
                     'elixir': sum(a['cost'] for a in answers),
                     'tower_lost': max(0, _tower_hp(before, ours) - _tower_hp(after, ours)) if before else 0,
                     'golem_dt': golem['dt'] if golem else None,
                     'golem_distance': golem['distance'] if golem else None})
    return rows


def main(argv: list[str]) -> int:
    import firstlight_bot  # noqa: F401  (puts FirstLight's native_runner on the path)
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', nargs='+', type=Path)
    parser.add_argument('--card', type=int, default=26000046)
    args = parser.parse_args(argv)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for directory in args.dirs:
        for path in sorted(directory.glob('frames/*/*.jsonl.zst')):
            for row in score(path, args.card):
                by_model[row['model']].append(row)
    for model, rows in sorted(by_model.items()):
        golems = [r for r in rows if r['golem_dt'] is not None]
        print(f'{model}: {len(rows)} plays of {args.card} answered')
        print(f'  unit lived: median {statistics.median(r["lifetime"] for r in rows) / 20:.1f} s; '
              f'elixir spent against it: mean {statistics.mean(r["elixir"] for r in rows):.1f}; '
              f'tower health lost meanwhile: mean {statistics.mean(r["tower_lost"] for r in rows):.0f}')
        if golems:
            print(f'  Ice Golem used on {len(golems)} of {len(rows)}: landed {statistics.median(r["golem_dt"] for r in golems) / 20:+.2f} s '
                  f'after the unit (median; spread {min(r["golem_dt"] for r in golems) / 20:+.1f} .. '
                  f'{max(r["golem_dt"] for r in golems) / 20:+.1f}), '
                  f'{statistics.median(r["golem_distance"] for r in golems):.1f} tiles from it (median)')
        multi = [r for r in rows if len(r['answers']) >= 3]
        print(f'  answered with 3+ cards: {len(multi)} of {len(rows)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
