"""How our side's Cannon does against the opponent's Hog Riders in engine games, where every play lands exactly
when the model was told it would (no pipeline lateness): the model's own timing, apart from the live pipeline.

For every Hog Rider the opponent played (engine recordings: frames every 5 ticks, each object's position and
target), what it ended up attacking first -- our Cannon (pulled) or a tower -- and where our Cannon's landing
fell: already standing, landed during the Hog's run, landed after the Hog had reached the tower, or never
played. For Cannons that landed during the run and pulled, the slack: how many ticks later the Hog would have
reached its tower (distance left to the tower's attack point at the Cannon's landing / the Hog's speed). A
Cannon that pulls with 2 ticks of slack is lost by a play that lands 2 ticks late.

    ./py -m il.hog_defence [--games runs/rl/check-full] [--max 30]
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))

HOG, CANNON = 26000021, 27000000
CANNON_UNITS = (27000000, 13000096)          # the Cannon and its evolution as they stand on the board
TOWERS = range(5000000, 5000006)


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def game(path: Path) -> list[dict]:
    from il.frames import load_replay
    header, frames = load_replay(path, with_replay=False)
    played = header.get('played') or {}
    ours = played.get('a_side')
    if ours not in (0, 1):
        return []
    theirs = 1 - ours
    # our Cannons: the tick each is on the board (replay tick + 1) and its entity
    landings = sorted(p['lands'] + 1 for p in header['timeline']['plays']
                      if p['owner'] == ours and p['card_id'] == CANNON and p.get('kind', 'card') == 'card')
    hogs: dict[tuple, list] = {}
    cannons: dict[tuple, list] = {}
    towers: dict[tuple, tuple] = {}
    for frame in frames:
        tick = int(frame['tick'])
        for item in frame['objects']:
            key = tuple(item['entityKey'])
            if item['nativeObjectId'] in TOWERS:
                towers[key] = (item['x'], item['y'])
            if item.get('hp') is None:
                continue                         # a projectile
            target = tuple(item['targetEntityKey']) if item.get('targetEntityKey') else None
            if item.get('cardId') == HOG and item['owner'] == theirs:
                hogs.setdefault(key, []).append((tick, item['x'], item['y'], target))
            elif item.get('cardId') in CANNON_UNITS and item['owner'] == ours:
                cannons.setdefault(key, []).append(tick)
    rows = []
    for key, track in hogs.items():
        spawn = track[0][0]
        moving = [i for i in range(1, len(track)) if (track[i][1], track[i][2]) != (track[i - 1][1], track[i - 1][2])]
        if not moving:
            continue                              # killed while deploying
        arrived = next((i for i in range(moving[0] + 1, len(track))
                        if (track[i][1], track[i][2]) == (track[i - 1][1], track[i - 1][2]) and track[i][3]), None)
        died = track[-1][0]
        row = {'game': path.stem[:12], 'spawn': spawn, 'version': played.get('a')}
        standing = [c for c, ticks in cannons.items() if ticks[0] <= spawn <= ticks[-1]]
        during = [c for c, ticks in cannons.items() if spawn < ticks[0] <= died]
        # the exact landing tick of a Cannon seen first in frame F: the landing in (F - 5, F]
        def landed(cannon) -> int:
            first = cannons[cannon][0]
            return next((tick for tick in landings if first - 5 < tick <= first), first)
        if arrived is None:
            row['outcome'] = 'died on the way'
        else:
            at, x, y, target = track[arrived]
            stop_tick = track[arrived - 1][0]     # it stood still from the frame before
            row['arrived'] = stop_tick
            if target in cannons:
                row['outcome'] = 'pulled'
                # slack: when the Cannon landed, how far the Hog still was from attacking its tower
                if target in during:
                    cannon_tick = landed(target)
                    before = [p for p in track if p[0] <= cannon_tick and p[3] in towers]
                    run = [track[i] for i in moving if track[i][0] <= cannon_tick]
                    if before and len(run) >= 2:
                        tower = towers[before[-1][3]]
                        # speed from the run so far (units per tick), position at the landing by interpolation
                        first, last = track[moving[0] - 1], run[-1]
                        speed = distance(first[1:3], last[1:3]) / max(1, last[0] - first[0])
                        after = next((p for p in track if p[0] >= cannon_tick), last)
                        prior = max((p for p in track if p[0] <= cannon_tick), key=lambda p: p[0])
                        span = max(1, after[0] - prior[0])
                        w = (cannon_tick - prior[0]) / span
                        here = (prior[1] + (after[1] - prior[1]) * w, prior[2] + (after[2] - prior[2]) * w)
                        row.update(cannon='landed during the run', cannon_tick=cannon_tick, speed=speed,
                                   left=distance(here, tower), after_spawn=cannon_tick - spawn)
                    else:
                        row['cannon'] = 'landed during the run'
                else:
                    row['cannon'] = 'already standing'
            elif target in towers:
                row['outcome'] = 'reached the tower'
                row['stop_distance'] = distance((x, y), towers[target])
                late = [landed(c) for c in during]
                if standing:
                    row['cannon'] = 'standing, did not pull'
                elif late:
                    row['cannon'] = 'landed after it arrived' if min(late) > stop_tick else 'landed during the run, did not pull'
                    row['cannon_after_arrival'] = min(late) - stop_tick
                else:
                    row['cannon'] = 'none played'
            else:
                row['outcome'] = 'other target'
        rows.append(row)
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', action='append')
    parser.add_argument('--max', type=int, default=30)
    args = parser.parse_args(argv)
    folders = [Path(f) for f in (args.games or [str(CLAPHA / 'runs/rl/check-full')])]
    paths = sorted({p for folder in folders for p in folder.rglob('*.jsonl.zst')}, key=lambda p: p.stat().st_mtime)
    rows = [row for path in paths[-args.max:] for row in game(path)]
    if not rows:
        print('no opponent Hog Riders found')
        return 0
    stops = [r['stop_distance'] for r in rows if 'stop_distance' in r]
    reach = statistics.median(stops) if stops else 2500.0
    print(f'{len(rows)} opponent Hog Riders in {len({r["game"] for r in rows})} games; a Hog attacks a tower from '
          f'{reach / 1000:.1f} tiles (median of {len(stops)})')
    outcomes = Counter((r['outcome'], r.get('cannon', '')) for r in rows)
    for (outcome, cannon), count in outcomes.most_common():
        print(f'  {count:4} ({100 * count / len(rows):3.0f}%)  {outcome}' + (f' -- Cannon {cannon}' if cannon else ''))
    slack = sorted((r['left'] - reach) / r['speed'] for r in rows if r.get('left') is not None and r.get('speed'))
    if slack:
        print(f'\nCannons that landed during the run and pulled ({len(slack)}): ticks the Hog was still away from '
              f'attacking its tower')
        print(f'  least {slack[0]:.0f}, 10% {slack[len(slack) // 10]:.0f}, median {statistics.median(slack):.0f}, '
              f'most {slack[-1]:.0f}')
        for limit in (1, 2, 3, 5, 10):
            close = sum(1 for s in slack if s < limit)
            print(f'  under {limit:2} ticks to spare: {close} ({100 * close / len(slack):.0f}%)')
        waits = sorted(r['after_spawn'] for r in rows if r.get('after_spawn') is not None)
        print(f'  the Cannon landed {statistics.median(waits):.0f} ticks after the Hog (median; 10% {waits[len(waits) // 10]}, '
              f'90% {waits[9 * len(waits) // 10]})')
    late = sorted(r['cannon_after_arrival'] for r in rows if r.get('cannon_after_arrival') is not None)
    if late:
        print(f'\nCannons played against a Hog that reached the tower anyway ({len(late)}): landed this many ticks '
              f'after the Hog arrived (negative: before): {late}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
