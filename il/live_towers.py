"""What damaged the bot's towers in live matches the user marked, unit by unit, and what the bot did about each
attacker. Same recordings as il/live_hogs.py (artifacts/viewer-sessions/<session>), only the named battles.

Every drop of a tower's hitpoints is given to the opponent unit that had that tower as its target at that tick
(two attackers at once: the one whose usual hit is closest to the drop); drops with no unit on the tower are
spells or death damage. Per attacker that got through: how long it shot, the bot's elixir and hand when it
started, every card the bot played while it lived and how far from it, and the bot's taps the game never
registered in that time.

    ./py -m il.live_towers artifacts/viewer-sessions/<session> [--port 8778] [--mirror-only] [--last N]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))

from il.live_hogs import (CANNON, HOG, HOG26, NAMES, clock, name, read_battles, read_commands,   # noqa: E402
                          read_timing, tiles)

TROOPS = (26000021, 26000014, 26000038, 26000030, 26000010)     # Hog, Musketeer, Ice Golem, Ice Spirit, Skeletons
MUSKETEER = 26000014
# one hit on a tower at the level friendlies are played at (measured where one unit was alone on a tower;
# a battle's own readings replace these when it has them)
HIT = {26000021: 317, 26000014: 217, 26000038: 84, 26000030: 110, 26000010: 81}


def analyse(battle: dict, commands: list[dict], timing: list[dict]) -> tuple[dict, list[dict]]:
    side = battle['side']
    frames = battle['frames']
    tracks: dict[tuple, list] = defaultdict(list)
    for f in frames:
        for address, kind, owner, x, y, hp, target in f['units']:
            tracks[(address, kind, owner)].append((f['tick'], x, y, hp, target))
    towers = {key[0]: key for key in tracks if key[1] == 'tower' and key[2] == side}
    attackers = {key: track for key, track in tracks.items() if key[2] == 1 - side and key[1] in TROOPS}
    by_tick: dict[int, list] = defaultdict(list)          # tick -> (attacker, tower address) aiming at our tower
    for key, track in attackers.items():
        for tick, x, y, hp, target in track:
            if target in towers:
                by_tick[tick].append((key, target))
    # every drop of a tower: who was on it
    drops = []
    for address, key in towers.items():
        track = tracks[key]
        for i in range(1, len(track)):
            lost = track[i - 1][3] - track[i][3]
            if lost > 0:
                on_it = [k for k, tower in by_tick.get(track[i][0], []) + by_tick.get(track[i - 1][0], []) if tower == address]
                drops.append({'tick': track[i][0], 'tower': address, 'lost': lost, 'on_it': sorted(set(on_it))})
    usual = dict(HIT)
    for kind in TROOPS:
        alone = [d['lost'] for d in drops if len(d['on_it']) == 1 and d['on_it'][0][1] == kind]
        if len(alone) >= 2:
            usual[kind] = statistics.mode(alone)
    sources: Counter = Counter()
    per_unit: dict[tuple, list] = defaultdict(list)
    for d in drops:
        if not d['on_it']:
            sources['spells and other'] += d['lost']
            continue
        # the units on the tower whose usual hits add up closest to the drop (one, or two landing together)
        units = d['on_it']
        options = [(u,) for u in units] + [(a, b) for i, a in enumerate(units) for b in units[i + 1:]]
        best = min(options, key=lambda group: abs(sum(usual[u[1]] for u in group) - d['lost']))
        if abs(sum(usual[u[1]] for u in best) - d['lost']) > 60:
            sources['spells and other'] += d['lost']          # a spell landed while units stood there
            continue
        for unit in best:
            sources[name(unit[1])] += usual[unit[1]]
            per_unit[unit].append({'tick': d['tick'], 'lost': usual[unit[1]]})
    ours = [c for c in commands if c['side'] == side and c['kind'] == 'card']
    by_frame = {f['tick']: f for f in frames}
    ticks = sorted(by_frame)
    rows = []
    for key, hits in sorted(per_unit.items(), key=lambda item: item[1][0]['tick']):
        if key[1] != MUSKETEER:
            continue
        track = attackers[key]
        spawn, died, first = track[0][0], track[-1][0], hits[0]['tick']
        f = by_frame[max(t for t in ticks if t <= first)]
        position = {t: (x, y) for t, x, y, _hp, _g in track}

        def near(tick: int) -> tuple:
            return position[min(position, key=lambda t: abs(t - tick))]
        answer = [(name(c['card']), c['lands'] + 1 - spawn, round(tiles((c['x'], c['y']), near(c['lands'] + 1)), 1))
                  for c in ours if spawn - 20 <= c['issue'] <= died]
        hogs = [k for k, t in attackers.items() if k[1] == HOG and t[0][0] <= died and t[-1][0] >= spawn]
        lost = [(t['card'], t.get('tap_tick', 0) - spawn) for t in timing
                if t['kind'] == 'lost' and spawn <= (t.get('tap_tick') or -1) <= died]
        dropped = [(t['card'], t.get('wait')) for t in timing
                   if t['kind'] == 'dropped' and spawn <= (t.get('turn') or -1) <= died]
        rows.append({'clock': clock(spawn), 'shots': len(hits), 'damage': sum(h['lost'] for h in hits),
                     'lived': died - spawn, 'first_shot_after': first - spawn, 'last_hp': track[-1][3],
                     'with_hog': bool(hogs), 'elixir': round(f['elixir'], 1), 'hand': [name(c) for c in f['hand']],
                     'answer': answer, 'lost_taps': lost, 'dropped': dropped})
    return dict(sources), rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=Path)
    parser.add_argument('--port', type=int)
    parser.add_argument('--mirror-only', action='store_true')
    parser.add_argument('--last', type=int)
    parser.add_argument('--first', type=int, help='only the first N battles')
    args = parser.parse_args(argv)
    battles = read_battles(args.session, cards=(*TROOPS, CANNON))
    commands = read_commands(args.session)
    timing = read_timing(args.port)
    if args.first:
        battles = battles[:args.first]
    if args.last:
        battles = battles[-args.last:]
    total: Counter = Counter()
    everything = []
    for number, battle in enumerate(battles, 1):
        plays = commands.get(str(battle['battle']), [])
        theirs = {c['card'] for c in plays if c['side'] == 1 - battle['side'] and c['kind'] == 'card'}
        if args.mirror_only and not (set(battle['deck']) == HOG26 and theirs and theirs <= HOG26):
            continue
        sources, rows = analyse(battle, plays, timing.get(str(battle['battle']), []))
        total.update(sources)
        print(f'\n=== battle {number}: tower damage taken ' + ', '.join(f'{k} {v}' for k, v in
                                                                        sorted(sources.items(), key=lambda kv: -kv[1])))
        for r in rows:
            print(f'  Musketeer at {r["clock"]}{" (with a Hog)" if r["with_hog"] else ""}: {r["shots"]} shots, {r["damage"]} '
                  f'damage; lived {r["lived"] / 20:.0f} s, first shot {r["first_shot_after"] / 20:.1f} s after landing, '
                  f'{"killed" if r["last_hp"] < 300 else "still alive at the end / walked off"}')
            print(f'       bot then: {r["elixir"]} elixir, hand {r["hand"]}')
            print(f'       bot played (card, ticks after she landed, tiles from her): {r["answer"]}')
            if r['lost_taps'] or r['dropped']:
                print(f'       taps the game never registered: {r["lost_taps"]}; decisions dropped: {r["dropped"]}')
        everything += rows
    if total:
        print('\nAll battles: ' + ', '.join(f'{k} {v}' for k, v in sorted(total.items(), key=lambda kv: -kv[1])))
        shots = [r['shots'] for r in everything]
        if shots:
            print(f'{len(everything)} Musketeers reached a tower: {sum(shots)} shots; {sum(1 for s in shots if s >= 5)} of them '
                  f'got 5 or more')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
