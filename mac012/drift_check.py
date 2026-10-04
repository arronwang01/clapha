"""What the FirstLight models cannot represent about the game build we play on.

FirstLight's catalog is frozen at Null's Royale 15.535.13; the game moves on. Run this after
every game update (and after regenerating live_card_catalog.json from the new build) to see,
without playing a match:

  * cards the game has that FirstLight's catalog does not -- the model cannot see them in
    hand, on the board, or in the opponent's revealed cards;
  * evolution / hero forms whose units have no FirstLight archetype -- shown as the base unit;
  * evolution cycle counts where our catalog and FirstLight's disagree -- the tracker counts
    with FirstLight's. The disagreement is not a constant offset, so it may be a different
    field in our catalog rather than a balance change; the progress vector read on the device
    (run_runtime_probe.py) is the ground truth;
  * elixir costs that changed.

    FIRSTLIGHT_ROOT=... python3 mac012/drift_check.py [--deck ID ID ...]

With --deck, only those cards are checked (e.g. your current deck).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import firstlight_obs as FLO  # noqa: E402


def main(argv: list[str]) -> int:
    from native_runner.training.v4.factory import production_semantic_bundle
    specs = production_semantic_bundle().card_specs
    known = FLO.known_cards()
    archetypes = FLO.archetype_by_card()
    cards = FLO.CARDS
    if '--deck' in argv:
        wanted = {int(value) for value in argv[argv.index('--deck') + 1:]}
        cards = {cid: info for cid, info in cards.items() if cid in wanted}
    else:
        cards = {cid: info for cid, info in cards.items() if info.get('standard_1v1')}
    missing, forms, cycles, costs = [], [], [], []
    for cid, info in sorted(cards.items()):
        name = info.get('display_name') or info.get('internal_name')
        if cid not in known:
            missing.append(f'{name} ({cid})')
            continue
        spec = specs[cid]
        for key, label in (('evolution_form_id', 'evolution'), ('hero_form_id', 'hero')):
            form = info.get(key)
            if form and int(form) not in archetypes:
                forms.append(f'{name} {label} ({form})')
        live_cycles = info.get('evolution_cycles')
        fl_cycles = spec.evolution.cycle_required if spec.evolution is not None else None
        if live_cycles and fl_cycles and int(live_cycles) != int(fl_cycles):
            cycles.append(f'{name}: our catalog {live_cycles}, FirstLight {fl_cycles}')
        if info.get('elixir') is not None and spec.elixir_cost is not None \
                and float(info['elixir']) != float(spec.elixir_cost):
            costs.append(f'{name}: game {info["elixir"]}, FirstLight {spec.elixir_cost:g}')
    sections = (('cards FirstLight does not know (invisible to the model)', missing),
                ('forms shown to the model as their base unit', forms),
                ('evolution cycles: our catalog and FirstLight disagree (tracker uses '
                 'FirstLight\'s; verify on the device)', cycles),
                ('elixir costs changed since 15.535', costs))
    print(f'checked {len(cards)} cards')
    for title, rows in sections:
        print(f'\n{title}: {len(rows)}')
        for row in rows:
            print(f'  {row}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
