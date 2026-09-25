"""Bring live_card_catalog.json up to the installed build, from that build's own tables.

Every card the tables list is kept or added: id, internal name, type (by table), elixir,
rarity and availability come from the build; metadata the old catalog carried for a card
(display name, hero/ability fields) is preserved. New cards get their internal name as display
name. The file records the build it was last checked against.

Evolutions come from the build's own spells_evolved table, not from the old catalog: a row
without NotInUse is a released evolution, and its id is 13,000,000 + its row. (Carrying the old
list over lost evo Elite Barbarians, AngryBarbarians_EV1, released after 15.535 and in 40% of
the IL_Replay battles.) Heroes are still carried over: spells_hero_form lists placeholder heroes
for nearly every card with no release flag, so it cannot say which heroes exist.

    python3 mac012/update_catalog.py [VERSION] [--write]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game_data as G  # noqa: E402

CLAPHA = Path(__file__).resolve().parents[1]
CATALOG = CLAPHA / 'live_card_catalog.json'
TYPES = {26: 'troop', 27: 'building', 28: 'spell'}


def truthy(value) -> bool:
    return str(value or '').strip().lower() in ('1', 'true', 'yes')


def main(argv: list[str]) -> int:
    version = next((a for a in argv if not a.startswith('--')), '160402012')
    data = json.loads(CATALOG.read_text())
    old = {c['card_id']: c for c in data['cards']}
    cards = []
    added, changed = [], []
    for cid, row in sorted(G.read_cards(version).items()):
        prior = dict(old.get(cid) or {})
        entry = {**prior, 'card_id': cid, 'internal_name': row['Name'],
                 'type': TYPES[cid // 1_000_000]}
        cost = row.get('ManaCost')
        entry['elixir'] = int(cost) if cost is not None else prior.get('elixir')
        entry['rarity'] = row.get('Rarity', prior.get('rarity'))
        entry['not_in_use'] = truthy(row.get('NotInUse'))
        entry['not_visible'] = truthy(row.get('NotVisible'))
        if not prior:
            entry['display_name'] = row['Name']
            entry['standard_1v1'] = not entry['not_in_use'] and not entry['not_visible']
            added.append(f"{cid} {row['Name']} ({entry['elixir']} elixir, {entry['type']})")
        else:
            for key in ('elixir', 'rarity', 'internal_name', 'type'):
                if prior.get(key) != entry.get(key):
                    changed.append(f"{cid} {row['Name']}: {key} {prior.get(key)} -> {entry.get(key)}")
        cards.append(entry)
    by_name = {entry['internal_name']: entry for entry in cards}
    for index, row in enumerate(G.read_table(version, 'spells_evolved.csv')):
        name = row.get('Name', '')
        base = by_name.get(name[:-4]) if name.endswith('_EV1') else None
        if base is None:
            continue
        released = not truthy(row.get('NotInUse'))
        form_id = 13_000_000 + index
        if released and (base.get('evolution_form'), base.get('evolution_form_id')) != (name, form_id):
            changed.append(f"{base['card_id']} {base['internal_name']}: evolution "
                           f"{base.get('evolution_form')} -> {name} ({form_id})")
            base['evolution_form'], base['evolution_form_id'] = name, form_id
            base.setdefault('evolution_cycles', None)
        elif not released and base.get('evolution_form') == name:
            changed.append(f"{base['card_id']} {base['internal_name']}: evolution {name} is not "
                           f"in use in this build -> removed")
            base['evolution_form'] = base['evolution_form_id'] = base['evolution_cycles'] = None
    print(f'{version}: {len(cards)} cards; added {len(added)}, changed {len(changed)}')
    for line in added + changed:
        print('   ', line)
    if '--write' in argv:
        data['cards'] = cards
        data['game_version_checked'] = version
        data['checked_utc'] = datetime.now(timezone.utc).isoformat()
        CATALOG.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
        print(f'wrote {CATALOG.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
