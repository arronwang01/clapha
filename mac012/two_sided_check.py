"""Is this frame two-sided? Reports what is readable for each player.

Used to test spectate and replay, where the client holds both players' state, against a live
match, where the opponent's hand is not sent. Read-only.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_profile import ADB, CLAPHA, MANAGER_RVA, ROOT_CONTEXT_OFFSET, SERIAL, apply  # noqa: E402

apply()
from native_core.mumu_live_protocol import visible_sides, verify_runtime  # noqa: E402

CARDS = {c['card_id']: c['display_name']
         for c in json.loads((CLAPHA / 'live_card_catalog.json').read_text())['cards']}


def probe(tool: str, *args: str) -> dict | None:
    runtime = verify_runtime(ADB, SERIAL)
    command = f'/data/local/tmp/{tool} {runtime["pid"]} {hex(MANAGER_RVA)} {hex(ROOT_CONTEXT_OFFSET)}'
    if args:
        command += ' ' + ' '.join(args)
    out = subprocess.run([str(ADB), '-s', SERIAL, 'shell', command],
                         capture_output=True, text=True, timeout=30).stdout
    start = out.find('{')
    if start < 0:
        return None
    try:                      # these probes pretty-print across several lines
        return json.loads(out[start:])
    except json.JSONDecodeError:
        return None


def main() -> int:
    players = probe('player_dump')
    if not players or players.get('world') == '0x0':
        print('no live world — start the spectate/replay first')
        return 2
    print(f"world {players['world']}  local_account(world+0x2e4) {players['world_0x2e4']}")
    frame_players = []
    for p in players['players']:
        vectors = {v['off']: v['values'] for v in p['vector_candidates']}
        hand = vectors.get('0x210', [])
        cycle = vectors.get('0x220', [])
        revealed = [CARDS.get(c, c) for c in p['revealed_0x288_0x2a8'] if c > 0]
        print(f"\nplayer index {p['index']}  side {p['side_0x78']}  "
              f"elixir {p['elixir_0x2f8'] / 10000:.2f}")
        print(f"   hand  (+0x210): {hand or 'NOT READABLE'}")
        print(f"   cycle (+0x220): {cycle or 'NOT READABLE'}")
        print(f"   revealed: {revealed or 'none yet'}")
        frame_players.append({'side': p['side_0x78'],
                              'hand_deck_indices': hand or [-1, -1, -1, -1],
                              'next_deck_index': (cycle or [-1])[0]})

    sides = visible_sides({'players': frame_players})
    print(f"\nvisible_sides -> {sides}")
    print({0: 'neither hand readable', 1: 'ONE side readable (a live match: opponent hidden)',
           2: 'BOTH hands readable (spectate/replay/after-match)'}.get(len(sides), '?'))

    decks = probe('deck_probe')
    if decks:
        print('\ndecks by owner slot:')
        for p in decks['players'][:1]:
            for owner in p['owner_slots']:
                cards = owner.get('cards')
                if owner['owner'] == '0x0' or not cards:
                    continue
                named = [CARDS.get(c, c) for c in cards if c > 0]
                print(f"   owner {owner['i']}: "
                      f"{named if named else 'all -1 (not revealed)'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
