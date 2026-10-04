"""Where a play's ticks go: the console's timing record (build/timing_<port>.jsonl) for the games the user
marks -- a window of time, nothing outside it is read.

A play is decided on turn T with the model's offset k; its moment is M = T + k. The model is told it lands
TARGET_DELAY (26) ticks after M, i.e. its command must be issued at M + 6 (the game adds 20). The tap is held
until the frame shows not_before = M + 6 - lag (lag: the recent median of tap -> issue ticks). So exactly:

    late = issue - (M + 6) = (tap tick - not_before)  +  (tap -> issue - lag)
                              tapped late                 the device/game slower than the median

and "tapped late" is either the decision finishing after the tap's moment (frame arrival, preparing,
deciding) or the hold itself only looking at a new frame every 50 ms.

    ./py -m il.timing --since 21:30 [--until 22:15] [--port 8777] [--day 2026-10-03]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
TICK_MS = 50.0


def window(day: str | None, since: str, until: str | None) -> tuple[float, float]:
    day = day or time.strftime('%Y-%m-%d')
    start = time.mktime(time.strptime(f'{day} {since}', '%Y-%m-%d %H:%M'))
    end = time.mktime(time.strptime(f'{day} {until}', '%Y-%m-%d %H:%M')) if until else time.time()
    return start, end


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else float('nan')


def spread(values: list[float]) -> str:
    if not values:
        return '-'
    return (f'median {statistics.median(values):.0f}, 90% {pct(values, 0.9):.0f}, 99% {pct(values, 0.99):.0f}, '
            f'max {max(values):.0f}')


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--since', required=True, help='HH:MM, local time (the first marked game)')
    parser.add_argument('--until', help='HH:MM (default: now)')
    parser.add_argument('--day', help='YYYY-MM-DD (default: today)')
    parser.add_argument('--port', type=int, action='append', help='console port (default: both)')
    args = parser.parse_args(argv)
    start, end = window(args.day, args.since, args.until)
    rows = []
    for port in args.port or (8777, 8778):
        path = CLAPHA / 'build' / f'timing_{port}.jsonl'
        if path.exists():
            rows += [row for row in map(json.loads, path.read_text().splitlines()) if start <= row['t'] <= end]
    turns = [r for r in rows if r['kind'] == 'turn']
    plays = [r for r in rows if r['kind'] == 'play']
    lost = [r for r in rows if r['kind'] == 'lost']
    dropped = [r for r in rows if r['kind'] == 'dropped']
    battles = {r['battle'] for r in rows}
    print(f'{len(battles)} battles, {len(turns)} decision turns, {len(plays)} plays reached the game, '
          f'{len(lost)} taps lost, {len(dropped)} decisions dropped before a tap')
    if not plays:
        return 0

    # Frame freshness: the device's sample clock against the Mac's; the smallest gap seen is the
    # transport's best case, so what is above it is extra age (the best case itself: half the tap ack)
    offset = min(r['received'] - r['frame_us'] / 1e6 for r in turns if r.get('frame_us'))
    one_way = min(((p['ack_ms'] - p['gesture_ms']) / 2 for p in plays if p.get('ack_ms') and p.get('gesture_ms')),
                  default=None)
    extra_age = [(r['received'] - r['frame_us'] / 1e6 - offset) * 1000 for r in turns if r.get('frame_us')]
    print('\nDecision turns')
    print(f'  frame age on arrival above the best case (ms): {spread(extra_age)}'
          + (f'; best case ~{one_way:.0f} ms (half the tap round trip)' if one_way is not None else ''))
    print(f'  frame tick past the turn tick (ticks): {dict(sorted(Counter(r["frame_tick"] - r["turn"] for r in turns).items()))}')
    print(f'  preparing (ms): {spread([r["prep_ms"] for r in turns])}')
    print(f'  deciding (ms): {spread([r["decide_ms"] for r in turns])}')
    print(f'  turns missed: {sum(r.get("skipped") or 0 for r in turns)}')

    def report(name: str, items: list[dict]) -> None:
        if not items:
            return
        lates = Counter(max(-1, min(3, p['late'])) for p in items if p.get('late') is not None)
        n = sum(lates.values())
        on_time = lates.get(-1, 0) + lates.get(0, 0)
        print(f'\n{name}: {len(items)} plays -- on time {on_time} ({100 * on_time / max(1, n):.0f}%), '
              f'1 tick late {lates.get(1, 0)}, 2 late {lates.get(2, 0)}, 3+ late {lates.get(3, 0)}; '
              f'waited for elixir {sum(p["elixir_wait"] for p in items)}')
        late = [p for p in items if (p.get('late') or 0) > 0 and p.get('not_before') is not None]
        if not late:
            return
        tapped_late = [p['tap_tick'] - p['not_before'] for p in late]
        slower = [(p['issue_tick'] - p['tap_tick']) - p['lag_used'] for p in late]
        print(f'  late plays: tapped after their tick by {spread(tapped_late)} ticks; the game took '
              f'{spread(slower)} ticks more than the tap lag used')
        causes = Counter()
        for p in late:
            # the wall time the not_before tick came, from the decision frame (fresh frame assumed)
            due = p['received'] + (p['not_before'] - p['frame_tick']) * TICK_MS / 1000
            if p['elixir_wait']:
                causes['waited for elixir'] += 1
            elif p['tap_tick'] > p['not_before'] and p['decided'] > due:
                causes['decision ready after its tap moment'] += 1
            elif p['tap_tick'] > p['not_before']:
                causes['hold saw the tick late (frames every 50 ms)'] += 1
            else:
                causes['tapped on time, game slower than the lag used'] += 1
        print('  causes: ' + ', '.join(f'{k} {v}' for k, v in causes.most_common()))
        slow = [p for p in late if p['decided'] > p['received'] + (p['not_before'] - p['frame_tick']) * TICK_MS / 1000]
        if slow:
            print(f'  of those, frame age at decision (ms): {spread([p["frame_age_ms"] for p in slow])}; deciding '
                  f'(ms): {spread([p["inference_ms"] for p in slow])}; turn already {spread([p["frame_tick"] - p["turn"] for p in slow])} ticks old')

    report('All plays', plays)
    report('Cannon', [p for p in plays if p['card'] == 'Cannon'])
    if dropped:
        print('\nDropped before a tap: ' + ', '.join(f'{k} {v}' for k, v in
                                                       Counter((d['card'], d.get('wait')) for d in dropped).most_common()))
    if lost:
        print('Lost taps (never in the game\'s queue): ' + ', '.join(f'{k} {v}' for k, v in
                                                                   Counter(p['card'] for p in lost).most_common()))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
