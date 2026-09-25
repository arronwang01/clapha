"""Opponent deck lookup: account id from memory -> player tag -> official Clash Royale API.

The reader already knows each player's account id (queue_probe `accounts`: hi, lo). A player
tag is that id written in base 14 over the alphabet 0289PYLQGRJCUV, with id = lo * 256 + hi.
The official API (through RoyaleAPI's fixed-IP proxy, so the key works from any network) then
gives the player's *currently equipped* deck and recent battles. That is a strong prior for
the deck they are playing now -- not ground truth: they may have switched decks.

Token: build/cr_api_token or build/cr_api_token.txt (never committed). The key must allow
45.79.218.79 (proxy.royaleapi.dev).

    ./py mac012/opponent_intel.py --self-check     your own accounts' tags + decks (verify
                                                   the tags against your in-game profiles)
    ./py mac012/opponent_intel.py '#TAG'           one player's current deck
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
ALPHABET = '0289PYLQGRJCUV'
API = 'https://proxy.royaleapi.dev/v1'
CACHE = CLAPHA / 'build' / 'opponent_decks'
CACHE_SECONDS = 600


def tag_from_account(hi: int, lo: int) -> str:
    value = (int(lo) & 0xFFFFFFFF) * 256 + (int(hi) & 0xFF)
    digits = ''
    while True:
        value, digit = divmod(value, 14)
        digits = ALPHABET[digit] + digits
        if value == 0:
            break
    return '#' + digits


def account_from_tag(tag: str) -> tuple[int, int]:
    value = 0
    for char in tag.strip().lstrip('#').upper().replace('O', '0'):
        value = value * 14 + ALPHABET.index(char)
    return value % 256, value >> 8


def token() -> str | None:
    for name in ('cr_api_token', 'cr_api_token.txt'):
        path = CLAPHA / 'build' / name
        if path.is_file():
            text = path.read_text().strip()
            if text:
                return text
    return None


def _get(path: str) -> dict:
    key = token()
    if not key:
        raise RuntimeError('no API token: save it as build/cr_api_token.txt')
    request = urllib.request.Request(API + path, headers={'Authorization': f'Bearer {key}',
                                                          'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors='replace')[:200]
        hint = {403: ' (key not valid for this IP: it must allow 45.79.218.79)',
                404: ' (no such player)'}.get(error.code, '')
        raise RuntimeError(f'API {error.code}{hint}: {body}') from None


def player(tag: str) -> dict:
    """The API's player record (cached 10 min per tag in build/opponent_decks/)."""
    tag = '#' + tag.strip().lstrip('#').upper()
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f'{tag[1:]}.json'
    if cached.is_file() and time.time() - cached.stat().st_mtime < CACHE_SECONDS:
        return json.loads(cached.read_text())
    record = _get('/players/' + urllib.parse.quote(tag))
    cached.write_text(json.dumps(record))
    return record


def current_deck(tag: str) -> list[dict]:
    """[{'card_id', 'name', 'evolution_level', 'hero'}] for the player's equipped deck."""
    deck = []
    for card in player(tag).get('currentDeck') or []:
        deck.append({'card_id': int(card['id']), 'name': card.get('name'),
                     'evolution_level': int(card.get('evolutionLevel') or 0),
                     'raw': {k: v for k, v in card.items() if k not in ('iconUrls',)}})
    return deck


def lookup(hi: int, lo: int) -> dict:
    """Everything the console needs at battle start; never raises."""
    tag = tag_from_account(hi, lo)
    try:
        record = player(tag)
        return {'tag': tag, 'name': record.get('name'), 'trophies': record.get('trophies'),
                'deck': current_deck(tag), 'error': None}
    except Exception as error:  # noqa: BLE001
        return {'tag': tag, 'name': None, 'deck': [], 'error': str(error)}


def _self_check() -> int:
    """Tags of the accounts seen in the newest recorded session, with their decks."""
    sessions = sorted((CLAPHA / 'artifacts' / 'viewer-sessions').glob('*/queue.jsonl'))
    accounts = {}
    for path in reversed(sessions[-5:]):
        for line in path.read_text().splitlines()[-2000:]:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            for account in row.get('accounts') or []:
                if account and account.get('lo', 0) > 0:
                    accounts[(account.get('hi', 0), account['lo'])] = True
        if accounts:
            break
    if not accounts:
        print('no recorded accounts found; play a battle with the console running first')
        return 1
    for hi, lo in accounts:
        info = lookup(hi, lo)
        print(f'account hi={hi} lo={lo} -> {info["tag"]}  (check this against the profile)')
        if info['error']:
            print(f'   API: {info["error"]}')
        else:
            print(f'   {info["name"]}, {info["trophies"]} trophies; deck: '
                  + ', '.join(f'{c["name"]}{" (evo)" if c["evolution_level"] else ""}'
                              for c in info['deck']))
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--self-check':
        raise SystemExit(_self_check())
    if len(sys.argv) > 1:
        for c in current_deck(sys.argv[1]):
            print(c['card_id'], c['name'], 'evo' if c['evolution_level'] else '', c['raw'])
        raise SystemExit(0)
    print(__doc__)
