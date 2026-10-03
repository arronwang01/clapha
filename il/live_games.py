"""Live matches the user marked, whole, in the app's game viewer (Training games -> Live matches): the board as
the bot's client had it (the bot at the bottom), both players' plays, and every click the bot made -- the ones
the game never registered marked on the tile they were aimed at.

From the console's recordings of those matches (artifacts/viewer-sessions/<session>: frames.jsonl, queue.jsonl)
and its log (build/bot_<port>.log: one line per click, with its tile). A click counts as registered when a
command of that card from the bot's account shows in the game's queue within 8 ticks of it. Only the session and
battles named are read.

    ./py -m il.live_games artifacts/viewer-sessions/<session> --port 8778 --name "live 10-03" [--first N]
        [--last N] [--mirror-only]

The games go to runs/viewer/live.json; il/game_viewer.py --json (the app's Reload) adds them to its file, and
this does so at once.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))
STORE = CLAPHA / 'runs' / 'viewer' / 'live.json'
GAMES = CLAPHA / 'runs' / 'viewer' / 'games.json'
EVERY = 5                     # ticks between snapshots (the app smooths between)
HOG26 = {26000021, 26000014, 27000000, 26000038, 26000030, 26000010, 28000000, 28000011}
COMMAND_AGE = 21
LOG_NAMES = {'HogRider': 'Hog Rider', 'IceGolemite': 'Ice Golem', 'IceSpirits': 'Ice Spirit'}


def card_name(V, card: int) -> str:
    name = str(V.CARDS.get(card, {}).get('name', card))
    return LOG_NAMES.get(name, name)


def read(session: Path):
    """The session's battles in order: every EVERY-th frame whole, and the hand and towers of all."""
    battles, current = [], {}
    with open(session / 'frames.jsonl', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            frame, health = row['frame'], row['health']
            side = health.get('local_side')
            if side not in (0, 1) or not frame.get('battle_active'):
                continue
            pid, address, tick = frame['pid'], frame['chain']['battle'], int(frame['game_tick'])
            battle = current.get(pid)
            if battle is None or battle['battle'] != address or tick < battle['last'] - 100:
                battle = current[pid] = {'pid': pid, 'battle': address, 'side': side, 'frames': [], 'last': tick}
                battles.append(battle)
            battle['last'] = tick
            if tick % EVERY == 0 or not battle['frames']:
                battle['frames'].append(frame)
            battle['final'] = frame
    return [b for b in battles if len(b['frames']) > 40]


def commands_of(session: Path) -> dict[str, list[dict]]:
    import viewer as V
    battles: dict[str, dict] = {}
    with open(session / 'queue.jsonl', encoding='utf-8') as handle:
        for line in handle:
            if not line.startswith('{'):
                continue
            row = json.loads(line)
            store = battles.setdefault(str(row.get('battle')), {})
            for entry in (row.get('queue') or {}).get('entries', []):
                if int(entry.get('card_id') or 0) <= 0 or not isinstance(entry.get('issue_tick'), int):
                    continue
                key = (entry.get('account_lo'), entry.get('seq'), entry['issue_tick'], entry['card_id'])
                if key not in store:
                    card, _form, kind = V.card_identity(entry['card_id'])
                    store[key] = {'card': card, 'kind': kind, 'side': V.entry_side(entry, row.get('accounts')),
                                  'issue': entry['issue_tick'], 'x': entry.get('x'), 'y': entry.get('y')}
    return {battle: sorted(store.values(), key=lambda c: c['issue']) for battle, store in battles.items()}


def clicks_from_log(port: int, started: float, ended: float) -> list[dict]:
    """Per battle of the console's log between two times: the side it played, the model, and its clicks."""
    battles: list[dict] = []
    model = ''
    for line in open(CLAPHA / 'build' / f'bot_{port}.log', encoding='utf-8', errors='replace'):
        try:
            stamp = time.mktime(time.strptime(line[:19], '%Y-%m-%d %H:%M:%S'))
        except ValueError:
            continue
        loaded = re.search(r'  (\S+) loaded \(FirstLight', line)
        if loaded:
            model = loaded.group(1)
        if not started <= stamp <= ended:
            continue
        new = re.search(r'new battle, you are side ([01])', line)
        if new:
            battles.append({'side': int(new.group(1)), 'model': model, 'time': stamp, 'clicks': []})
        over = re.search(r'battle over at t=[\d.]+s: you (won|lost|drew)\S* - ([^.]*)', line)
        if over and battles:
            battles[-1]['result'] = f'{"draw" if over.group(1) == "drew" else over.group(1)} · {over.group(2).strip()}'
        tap = re.search(r't=\s*([\d.]+)s\s+(\S+)\s+row\s+(\d+) col\s+(\d+)', line)
        if tap and battles and 'would play' not in line:
            battles[-1]['clicks'].append({'tick': round(float(tap.group(1)) * 20), 'card': LOG_NAMES.get(tap.group(2), tap.group(2)),
                                          'row': int(tap.group(3)), 'column': int(tap.group(4))})
    return battles


def convert(battle: dict, commands: list[dict], logged: dict | None, V, label, tag: str, run: str,
            opponent: str = 'you') -> dict:
    side = battle['side']
    flip = side == 1

    def tile(x, y) -> tuple[int, int]:
        gx, gy = int(x // 1000), int(y // 1000)
        return (17 - gx, 31 - gy) if flip else (gx, gy)
    ids: dict[str, int] = {}
    out = []
    for frame in battle['frames']:
        players = {p['side']: p for p in frame['players']}
        me, them = players.get(side) or {}, players.get(1 - side) or {}
        deck = [V.card_identity(c)[0] for c in me.get('deck_card_ids') or ()]

        def name_of(slot) -> int:
            return label(card_name(V, deck[slot])) if isinstance(slot, int) and 0 <= slot < len(deck) else label('?')
        hand = [name_of(s) for s in (me.get('hand_deck_indices') or [-1] * 4)][:4] + [name_of(me.get('next_deck_index'))]
        objects = []
        for e in frame.get('entities') or ():
            base = V.card_identity(e['card_id'])[0] if e.get('card_id', 0) > 0 else 0
            family = base // 1000000
            if e.get('kind') in (12, 13) and e.get('card_id', 0) <= 0:      # a King reads 12 until it wakes
                kind, text = 2, 'King' if abs(int(e['x']) - 9000) < 600 else 'Tower'
            elif family == 27 and e.get('max_hp'):
                kind, text = 1, card_name(V, base)
            elif family == 28:
                kind, text = 3, card_name(V, base)
            elif family == 26 and e.get('max_hp'):
                kind, text = 0, card_name(V, base)
            else:
                continue
            x, y = (18000 - e['x'], 32000 - e['y']) if flip else (e['x'], e['y'])
            pct = round(100 * e['hp'] / e['max_hp']) if e.get('max_hp') else -1
            objects += [ids.setdefault(e['address'], len(ids)), (label(text) * 4 + kind) * 2 + (0 if e['side'] == side else 1),
                        round(x / 100), round(y / 100), pct]
        out.append([int(frame['game_tick']), round(int(me.get('elixir_raw') or 0) / 1000),
                    round(int(them.get('elixir_raw') or 0) / 1000), hand, [label('?')] * 5, objects])
    plays, ours = [], []
    for c in commands:
        if c['kind'] != 'card' or c['x'] is None or c['side'] not in (0, 1):
            continue
        gx, gy = tile(c['x'], c['y'])
        plays.append([c['issue'] + COMMAND_AGE + 1, 0 if c['side'] == side else 1, label(card_name(V, c['card'])), gx, gy])
        if c['side'] == side:
            ours.append(c)
    clicks, failed = [], 0
    if logged is not None and logged['side'] == side:
        claimed: set[int] = set()
        for click in logged['clicks']:
            match = next((i for i, c in enumerate(ours) if i not in claimed and card_name(V, c['card']) == click['card']
                          and -3 <= c['issue'] - click['tick'] <= 8), None)
            if match is not None:
                claimed.add(match)
            else:
                failed += 1
            gx, gy = (17 - click['column'], 31 - click['row']) if flip else (click['column'], click['row'])
            clicks.append([click['tick'], label(click['card']), gx, gy, 0 if match is not None else 1])
    # crowns from the towers standing at the end
    standing = {0: [], 1: []}
    for e in battle['final'].get('entities') or ():
        if e.get('kind') in (12, 13) and e.get('card_id', 0) <= 0:
            standing[e['side']].append(abs(int(e['x']) - 9000) < 600)
    def crowns_against(owner: int) -> int:
        # a King gone with a Princess tower still standing is the end of the match clearing the board
        princesses = standing[owner].count(False)
        return 3 if True not in standing[owner] and princesses == 0 else max(0, 2 - princesses)
    mine, theirs = crowns_against(1 - side), crowns_against(side)
    result = (logged or {}).get('result') or (('won' if mine > theirs else 'lost' if theirs > mine else 'draw') + f' {mine}-{theirs}')
    their_cards = sorted({card_name(V, c['card']) for c in commands if c['side'] == 1 - side and c['kind'] == 'card'})
    deck_names = [card_name(V, V.card_identity(c)[0]) for c in
                  next((p for p in battle['frames'][-1]['players'] if p['side'] == side), {}).get('deck_card_ids') or ()]
    end = out[-1][0]
    when = time.strftime('%H:%M', time.localtime(logged['time'])) if logged else ''
    model = (logged or {}).get('model') or 'the bot'
    return {'tag': tag, 'league': 'live', 'run': run, 'path': None,
            'title': f'{model} against {opponent}' + (f' · {when}' if when else ''), 'result': result,
            'sub': f'live · {end // 20 // 60}:{end // 20 % 60:02d}' + (f' · {failed} clicks failed' if clicks else ' · clicks not recorded'),
            'learner': f'{model} (the bot)', 'opponent': opponent, 'decks': [deck_names, their_cards],
            'frames': out, 'plays': sorted(plays), 'clicks': clicks}


def merge(data: dict) -> int:
    """Add the stored live games to the viewer's data (its labels extended); returns how many."""
    if not STORE.exists():
        return 0
    store = json.loads(STORE.read_text())
    index = {text: i for i, text in enumerate(data['labels'])}

    def remap(old: int) -> int:
        text = store['labels'][old]
        if text not in index:
            index[text] = len(data['labels'])
            data['labels'].append(text)
        return index[text]
    data['games'] = [g for g in data['games'] if g.get('league') != 'live']
    for game in store['games']:
        game = json.loads(json.dumps(game))
        for frame in game['frames']:
            frame[3] = [remap(v) for v in frame[3]]
            frame[4] = [remap(v) for v in frame[4]]
            objects = frame[5]
            for i in range(1, len(objects), 5):
                objects[i] = (remap(objects[i] >> 3) << 3) | (objects[i] & 7)
        for play in game['plays']:
            play[2] = remap(play[2])
        for click in game.get('clicks') or ():
            click[1] = remap(click[1])
        data['games'].insert(0, game)
    return len(store['games'])


def main(argv: list[str]) -> int:
    import firstlight_bot  # noqa: F401
    import viewer as V
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=Path)
    parser.add_argument('--port', type=int, required=True, help="the bot console's port (its log has the clicks)")
    parser.add_argument('--name', required=True, help='the run these games are listed under')
    parser.add_argument('--first', type=int)
    parser.add_argument('--last', type=int)
    parser.add_argument('--mirror-only', action='store_true')
    parser.add_argument('--opponent', default='you', help='who played the other side')
    args = parser.parse_args(argv)
    battles = read(args.session)
    commands = commands_of(args.session)
    import calendar
    started = calendar.timegm(time.strptime(args.session.name, '%Y%m%dT%H%M%SZ'))
    later = sorted(p.name for p in args.session.parent.iterdir() if p.name > args.session.name and len(p.name) == 16)
    ended = calendar.timegm(time.strptime(later[0], '%Y%m%dT%H%M%SZ')) if later else time.time()
    logged = clicks_from_log(args.port, started - 5, ended)
    aligned = len(logged) == len(battles) and all(l['side'] == b['side'] for l, b in zip(logged, battles))
    print(f'{len(battles)} battles in the session, {len(logged)} in the log' + ('' if aligned else ': not matched, clicks left out'))
    numbered = list(enumerate(battles))
    if args.first:
        numbered = numbered[:args.first]
    if args.last:
        numbered = numbered[-args.last:]
    store = json.loads(STORE.read_text()) if STORE.exists() else {'labels': [], 'games': []}
    store['games'] = [g for g in store['games'] if g.get('run') != args.name]
    index = {text: i for i, text in enumerate(store['labels'])}

    def label(text: str) -> int:
        if text not in index:
            index[text] = len(store['labels'])
            store['labels'].append(text)
        return index[text]
    added = []
    for number, battle in numbered:
        plays = commands.get(str(battle['battle']), [])
        theirs = {c['card'] for c in plays if c['side'] == 1 - battle['side'] and c['kind'] == 'card'}
        ours = {V.card_identity(c)[0] for c in
                next((p for p in battle['frames'][-1]['players'] if p['side'] == battle['side']), {}).get('deck_card_ids') or ()}
        if args.mirror_only and not (ours == HOG26 and theirs and theirs <= HOG26):
            continue
        game = convert(battle, plays, logged[number] if aligned else None, V, label,
                       f'live-{args.session.name}-{number + 1}', args.name, args.opponent)
        added.append(game)
        print(f"  battle {number + 1}: {game['title']} -- {game['result']}, {game['sub']}")
    store['games'] = added[::-1] + store['games']
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(store, separators=(',', ':')))
    data = json.loads(GAMES.read_text()) if GAMES.exists() else {'labels': [], 'games': [], 'about': ''}
    total = merge(data)
    GAMES.write_text(json.dumps(data, separators=(',', ':')))
    print(f'{len(added)} games stored ({total} live games in all) -> {GAMES} ({GAMES.stat().st_size // 1024} KB)')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
