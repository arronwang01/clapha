"""One bounded placement to settle x-mirroring for the local side.

Upstream ScreenLayout mirrors x only when side == 1. The user's prior project found that
owner-0 games came out mirrored unless x' = 18000 - x was used. Both cannot be right.

This places a single Cannon at a deliberately off-centre cell and reports where it actually
landed, so the answer is measured rather than argued. One card, one battle, then it stops.

Usage: python3 mac012/mirror_test.py [--cell 165] [--execute]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_profile import ADB, CLAPHA, SERIAL, apply  # type: ignore  # noqa: E402

apply()
import layout_fix  # noqa: F401,E402  (side-0 x mirror fix)
from native_core.mumu_live_actions import (ScreenLayout, card_receipt,  # noqa: E402
                                           own_player, send_card_taps)
from native_core.mumu_live_protocol import (BattleClockGuard, adb_run,  # noqa: E402
                                            start_reader, verify_runtime)

CANNON = 27000000
NATIVE_W, NATIVE_H = 18000, 32000
CATALOG = json.loads((CLAPHA / 'live_card_catalog.json').read_text())
CARDS = {c['card_id']: c for c in CATALOG['cards']}


def pick(player: dict):
    """Cannon if it is in hand (stationary, exact read-back); else any affordable
    troop/building. A mirrored x is off by ~11000, far larger than any spawn spread."""
    deck = player.get('deck_card_ids') or []
    options = []
    for slot, index in enumerate(player['hand_deck_indices']):
        if not (0 <= index < len(deck)):
            continue
        card = CARDS.get(deck[index])
        if not card or card['type'] == 'spell':
            continue
        if player['elixir_raw'] < card['elixir'] * 10000:
            continue
        options.append((0 if deck[index] == CANNON else 1, card['elixir'], slot, deck[index]))
    return min(options) if options else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', type=int, default=165, help='canonical 18x32 cell (row*18+col)')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--seconds', type=float, default=60)
    args = parser.parse_args(argv)
    row, col = divmod(args.cell, 18)
    want = ((col + .5) * 1000, (row + .5) * 1000)

    runtime = verify_runtime(ADB, SERIAL)
    sizes = re.findall(r'(\d+)x(\d+)', adb_run(ADB, SERIAL, 'shell', 'wm size'))
    layout = ScreenLayout.from_size(*map(int, sizes[-1]))
    process = start_reader(ADB, SERIAL, runtime['pid'], interval_ms=100,
                           max_frames=int(args.seconds * 10))
    guard = BattleClockGuard()
    before = None
    deadline = time.monotonic() + args.seconds
    try:
        for line in process.stdout:
            if time.monotonic() > deadline:
                print('timed out waiting for a Cannon in hand')
                return 2
            if '"mumu_live_frame"' not in line:
                continue
            frame = json.loads(line)
            health = guard.observe(frame, now=frame['sample_monotonic_us'] / 1_000_000)
            side = health.get('local_side')
            if before is None:
                if not health.get('can_control'):
                    continue
                player = own_player(frame, side)
                choice = pick(player)
                if choice is None:
                    continue
                _, _, slot, card_id = choice
                chosen = {'slot': slot, 'card_id': card_id,
                          'name': CARDS[card_id]['display_name']}
                target = layout.deployment_point(args.cell, side)
                print(json.dumps({'local_side': side, 'cell': args.cell, 'card': chosen,
                                  'requested_canonical': want, 'hand_slot': slot,
                                  'tap_hand': layout.hand_point(slot), 'tap_target': target,
                                  'execute': args.execute}, indent=2))
                if not args.execute:
                    return 0
                send_card_taps(ADB, SERIAL, layout, slot, args.cell, side=side)
                before = frame
                continue
            receipt = card_receipt(before, frame, side=side,
                                   slot=chosen['slot'], card_id=chosen['card_id'])
            if not receipt.get('accepted'):
                continue
            landed = receipt['matching_new_entities']
            if not landed:
                continue
            native = (landed[0]['x'], landed[0]['y'])
            direct = native
            rotated = (NATIVE_W - native[0], NATIVE_H - native[1])
            print(json.dumps({
                'requested_canonical': want, 'native_read_back': native,
                'if_no_rotation': direct, 'if_rotated': rotated,
                'matches_no_rotation': [abs(direct[0] - want[0]), abs(direct[1] - want[1])],
                'matches_rotation': [abs(rotated[0] - want[0]), abs(rotated[1] - want[1])],
                'x_mirrored': abs(direct[0] - (NATIVE_W - want[0])) < 600,
            }, indent=2))
            return 0
    finally:
        process.terminate()
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
