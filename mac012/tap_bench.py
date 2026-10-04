"""Find the fastest touch gesture the client accepts. Run in a Training Camp battle.

For each gesture below, places the cheapest card in hand on our own back row, several times,
and watches the command queue for it. Reports, per gesture: how many placements the game
accepted, how long the gesture itself took on the device, and ticks from the tap to the
command's issue tick. The winner goes in CR_TAP_MODE / CR_TAP_GAP_MS / CR_TAP_HOLD_MS.

    python3 mac012/tap_bench.py [TRIALS]        (default 4 per gesture)

Taps only our own side of a Training Camp board; it will refuse to run against a real account
(scope_gate). Results are also written to build/tap_bench.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viewer as V  # noqa: E402
import scope_gate  # noqa: E402
import tapper as TAP  # noqa: E402
from mac_profile import ADB, SERIAL  # type: ignore  # noqa: E402
from native_core.mumu_live_actions import ScreenLayout, send_card_taps  # noqa: E402
from native_core.mumu_live_protocol import adb_run  # noqa: E402

X_TILES = 18
# (label, mode, gap, hold). 'adb' is the old path: input tap; sleep 0.05; input tap.
GESTURES = [('adb input tap (old)', 'adb', 0, 0),
            ('place gap 50 hold 34', 'place', 50, 34),
            ('place gap 30 hold 20', 'place', 30, 20),
            ('place gap 16 hold 16', 'place', 16, 16),
            ('place gap 8 hold 16', 'place', 8, 16),
            ('place gap 0 hold 16', 'place', 0, 16),
            ('drag 3 steps x 10 ms', 'drag', 3, 10),
            ('drag 2 steps x 5 ms', 'drag', 2, 5),
            ('drag 1 step x 17 ms', 'drag', 1, 17)]


def snapshot():
    with V.LOCK:
        return (V.STATE['frame'], V.STATE['health'], list(V.STATE['queue']),
                V.STATE['accounts'])


def main() -> int:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    threading.Thread(target=V.pump, daemon=True).start()
    threading.Thread(target=V.pump_queue, daemon=True).start()
    sizes = re.findall(r'(\d+)x(\d+)', adb_run(ADB, SERIAL, 'shell', 'wm size'))
    width, height = map(int, sizes[-1])
    layout = ScreenLayout.from_size(width, height)
    print('waiting for a battle ...')
    while True:
        frame, health, _queue, accounts = snapshot()
        if frame and health and frame.get('battle_active') and health.get('can_control'):
            break
        time.sleep(0.2)
    side = health['local_side']
    allowed, reason = scope_gate.check(accounts, side)
    if not allowed:
        print(f'refusing: {reason}')
        return 1
    results = []
    for label, mode, gap, hold in GESTURES:
        rows = []
        tap = None
        if mode != 'adb':
            os.environ.update(CR_TAP_MODE=mode, CR_TAP_GAP_MS=str(gap),
                              CR_TAP_HOLD_MS=str(hold))
            tap = TAP.Tapper(ADB, SERIAL, width, height)
        for _ in range(trials):
            frame, health, queue, accounts = snapshot()
            if not frame or not frame.get('battle_active'):
                print('battle ended')
                break
            me = next(p for p in frame['players'] if p['side'] == side)
            deck = me['deck_card_ids']
            hand = [(V.CARDS.get(deck[i], {}).get('elixir') or 9, pos, deck[i])
                    for pos, i in enumerate(me['hand_deck_indices']) if 0 <= i < len(deck)]
            cost, position, card = min(hand)
            while me['elixir_raw'] / 10000.0 < cost + 0.2:       # wait for the elixir
                time.sleep(0.1)
                frame, health, queue, accounts = snapshot()
                me = next(p for p in frame['players'] if p['side'] == side)
            row = 2 if side == 0 else 29
            cell = row * X_TILES + 8 + len(rows) % 2          # own back row, centre
            before = {(e.get('issue_tick'), e.get('seq')) for e in queue}
            tap_tick = frame['game_tick']
            gesture_ms = None
            if mode == 'adb':
                started = time.time()
                send_card_taps(ADB, SERIAL, layout, position, cell, side=side)
                gesture_ms = (time.time() - started) * 1000.0
            else:
                tap.play(layout.hand_point(position), layout.deployment_point(cell, side))
            issued = None
            deadline = time.time() + 2.0
            while time.time() < deadline and issued is None:
                time.sleep(0.02)
                _f, _h, queue, _a = snapshot()
                for e in queue:
                    if ((e.get('issue_tick'), e.get('seq')) not in before
                            and e.get('card_id') == card and V.mine(e)):
                        issued = e['issue_tick']
            if tap is not None and tap.timings:
                gesture_ms = tap.timings[-1]['gesture_ms']
            rows.append({'accepted': issued is not None, 'gesture_ms': gesture_ms,
                         'tap_to_issue_ticks': None if issued is None else issued - tap_tick})
            time.sleep(1.0)       # let the unit land before the next trial
        if tap is not None:
            tap.close()
        accepted = [r for r in rows if r['accepted']]
        ticks = sorted(r['tap_to_issue_ticks'] for r in accepted)
        gestures = [r['gesture_ms'] for r in rows if r['gesture_ms'] is not None]
        summary = {'gesture': label, 'mode': mode, 'gap': gap, 'hold': hold,
                   'accepted': len(accepted), 'trials': len(rows),
                   'gesture_ms': round(sum(gestures) / len(gestures), 1) if gestures else None,
                   'tap_to_issue_ticks_median': ticks[len(ticks) // 2] if ticks else None}
        results.append(summary)
        print(f'{label:24} accepted {summary["accepted"]}/{summary["trials"]}  '
              f'gesture {summary["gesture_ms"]} ms  '
              f'tap->issued {summary["tap_to_issue_ticks_median"]} ticks')
    out = Path(__file__).resolve().parents[1] / 'build' / 'tap_bench.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f'written {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
