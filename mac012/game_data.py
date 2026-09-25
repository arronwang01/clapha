"""Read this build's own game data tables from the installed APK (pulled into runtime/).

Supercell ships the logic tables LZMA-compressed with a short header (5 property bytes + a
4-byte size where standard .lzma has 8). CSVs keep their classic shape: a header row, a types
row, then one row per entry, continuation rows having an empty Name. A card's global id is
table * 1_000_000 + its row index among named rows (26 troops, 27 buildings, 28 spells).

    python3 mac012/game_data.py [VERSION]      -> summary against live_card_catalog.json
"""
from __future__ import annotations

import csv
import io
import json
import lzma
import sys
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
CARD_TABLES = {26: 'spells_characters.csv', 27: 'spells_buildings.csv', 28: 'spells_other.csv'}


def decompress(raw: bytes) -> bytes:
    if raw[:1] == b']' and len(raw) > 9:
        return lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(
            raw[:9] + b'\x00\x00\x00\x00' + raw[9:])
    return raw


def logic_dir(version: str) -> Path:
    return CLAPHA / 'runtime' / version / 'assets' / 'assets' / 'csv_logic'


def read_table(version: str, name: str) -> list[dict]:
    """Named rows of one CSV table, in order, each a dict of its non-empty columns."""
    text = decompress((logic_dir(version) / name).read_bytes()).decode('utf-8-sig')
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    named = []
    for row in rows[2:]:
        if not row or not row[0].strip():
            continue
        named.append({key: value for key, value in zip(header, row) if key and value != ''})
    return named


def read_cards(version: str) -> dict[int, dict]:
    cards = {}
    for table, name in CARD_TABLES.items():
        for index, row in enumerate(read_table(version, name)):
            cards[table * 1_000_000 + index] = row
    return cards


def main(argv: list[str]) -> int:
    version = argv[0] if argv else '160402012'
    cards = read_cards(version)
    old = {c['card_id']: c for c in json.loads((CLAPHA / 'live_card_catalog.json').read_text())['cards']}
    same = renamed = 0
    new, changed_cost, gone = [], [], []
    for cid, row in cards.items():
        name = row.get('Name')
        prior = old.get(cid)
        if prior is None:
            new.append((cid, name, row.get('ManaCost'), row.get('NotInUse')))
            continue
        if prior['internal_name'] == name:
            same += 1
        else:
            renamed += 1
        cost = row.get('ManaCost')
        if cost is not None and prior.get('elixir') is not None and int(cost) != int(prior['elixir']):
            changed_cost.append((cid, name, prior['elixir'], int(cost)))
    for cid, prior in old.items():
        if cid not in cards:
            gone.append((cid, prior['internal_name']))
    print(f'{version}: {len(cards)} cards in the tables; old catalog (15.535.29) {len(old)}')
    print(f'  same id + name: {same}, id reused with another name: {renamed}')
    print(f'  new since the old catalog: {len(new)}')
    for row in new[:40]:
        print('    ', row)
    print(f'  elixir cost changed: {len(changed_cost)}', *changed_cost[:20], sep='\n    ')
    print(f'  in the old catalog but not in this build: {len(gone)}', *gone[:10], sep='\n    ')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
