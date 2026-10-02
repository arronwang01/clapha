"""Watch training games: the whole games an RL run keeps (il/rl.py --save-every, 1 game in 50, in
runs/rl/<run>/recordings) as one standalone page -- the board every half second with smooth motion
between (troops, buildings, towers with health, spells in flight), both players' elixir and hands
(and the next card), where each card landed, and a log of plays. The learner is always at the
bottom.

    ./py -m il.game_viewer runs/rl/pilot3/recordings [--league general] [--result lost] [--max 40]
        [--out runs/viewer/pilot3.html]

The Clapha app's "Training games" window shows the same games natively: its Reload runs this with
--json runs/viewer/games.json (the data alone) and reads that file.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))
TEMPLATE = CLAPHA / 'tools' / 'game_viewer.tpl.html'
TOWER_IDS = range(5000000, 5000006)
KINGS = (5000000, 5000003)
# the card that names a real deck (the first one found, in this order)
WIN_CONDITIONS = ('Golem', 'Lava Hound', 'Elixir Golem', 'Electro Giant', 'Giant', 'Goblin Giant', 'Royal Giant',
                  'Mega Knight', 'P.E.K.K.A', 'X-Bow', 'Mortar', 'Graveyard', 'Balloon', 'Hog Rider', 'Royal Hogs',
                  'Ram Rider', 'Battle Ram', 'Goblin Drill', 'Miner', 'Goblin Barrel', 'Skeleton Barrel', 'Wall Breakers',
                  'Sparky', 'Three Musketeers', 'Elite Barbarians', 'Royal Recruits', 'Bowler', 'Giant Skeleton')


def _header(path: Path) -> dict:
    import zstandard
    with path.open('rb') as handle:
        return json.loads(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(handle), encoding='utf-8').readline())


def _version(played: dict) -> int:
    match = re.search(r'v(\d+)', str(played.get('a', '')))
    return int(match.group(1)) if match else -1


def _opponent(played: dict) -> str:
    who = str(played.get('b', ''))
    if who == 'fl:general':
        return 'General'
    if who == 'fl:hog2':
        return 'hog2 (no delay)'
    if who == 'latest':
        return 'itself'
    if 'distill-v2' in who:
        return 'v2 (the start)'
    match = re.search(r'policy-(\d+)', who)
    if match:
        return f'snapshot {int(match.group(1))}'
    if 'ex1-target' in who:
        return 'frozen pilot3 (ex1\'s target)'
    return Path(who.replace('\\', '/')).stem or who


def game(path: Path, label, name, every: int) -> dict:
    from il.frames import load_replay
    header, frames = load_replay(path, with_replay=False)
    played = header.get('played') or {}
    side = int(played.get('a_side', 0))
    flip = side == 1                          # the learner at the bottom

    def pos(x: int, y: int) -> tuple[int, int]:
        return (18000 - x, 32000 - y) if flip else (x, y)
    frames = sorted(frames, key=lambda f: int(f['tick']))
    ids: dict[int, int] = {}
    crowns = [0, 0]
    out = []
    for frame in frames:
        tick = int(frame['tick'])
        players = {int(p['owner']): p for p in (frame.get('state') or {}).get('players') or ()}
        raw = (frame.get('state') or {}).get('crownsRaw')
        if isinstance(raw, list) and len(raw) == 2:
            crowns = [max(crowns[0], int(raw[side])), max(crowns[1], int(raw[1 - side]))]
        if tick % every:
            continue

        def hand(owner: int) -> list[int]:
            player = players.get(owner) or {}
            cards = [label(name(int(c['cardId']))) for c in player.get('hand') or ()]
            upcoming = player.get('nextCard')
            card = upcoming.get('cardId') if isinstance(upcoming, dict) else upcoming
            return (cards + [label('?')] * 4)[:4] + [label(name(int(card))) if card else label('?')]

        def elixir(owner: int) -> int:
            return round(int((players.get(owner) or {}).get('elixirRaw') or 0) / 1000)
        objects = []
        for o in frame.get('objects') or ():
            oid, card = int(o['nativeObjectId']), int(o.get('cardId') or 0)
            family = card // 1000000
            if oid in TOWER_IDS:
                kind = 2
            elif family == 28:
                kind = 3                       # a spell: a Fireball in flight, a rolling Log
            elif o.get('projectile') is None and o.get('hp') is not None and family in (26, 27, 203):
                kind = 1 if family == 27 else 0
            else:
                continue
            x, y = pos(int(o['x']), int(o['y']))
            hp, top = o.get('hp'), o.get('maxHp')
            pct = round(100 * int(hp) / int(top)) if hp is not None and top else -1
            text = ('King' if oid in KINGS else 'Tower') if kind == 2 else name(card)
            objects += [ids.setdefault(oid, len(ids)), (label(text) * 4 + kind) * 2 + (0 if int(o['owner']) == side else 1),
                        round(x / 100), round(y / 100), pct]
        out.append([tick, elixir(side), elixir(1 - side), hand(side), hand(1 - side), objects])
    plays = []
    for play in header['timeline']['plays']:
        if not play.get('grid'):
            continue
        gx, gy = play['grid']
        if flip:
            gx, gy = 17 - gx, 31 - gy
        plays.append([int(play['lands']) + 1, 0 if play['owner'] == side else 1, label(name(int(play['card_id']))), gx, gy])
    decks = [[name(int(c)) for c in header['timeline']['decks'][s]] for s in (side, 1 - side)]
    opponent = _opponent(played)
    if played.get('league') == 'general':
        opponent += f" ({next((w for w in WIN_CONDITIONS if w in decks[1]), 'real deck')})"
    result = {'a': 'won', 'b': 'lost'}.get(played.get('result'), 'draw')
    if frames:
        # crowns from the towers gone at the end (a King is all three); the last snapshot can come
        # just before the deciding tower falls, so an equal score before 5:00 means a crown in
        # sudden death
        standing = {int(o['nativeObjectId']) for o in frames[-1].get('objects') or ()}
        lost = [0, 0]
        for oid in TOWER_IDS:
            owner = 0 if oid < 5000003 else 1
            if oid not in standing:
                lost[0 if owner == side else 1] += 3 if oid in KINGS else 1
        crowns = [max(crowns[0], min(3, lost[1])), max(crowns[1], min(3, lost[0]))]
        if crowns[0] == crowns[1] and result != 'draw' and int(frames[-1]['tick']) < 5990:
            crowns[0 if result == 'won' else 1] += 1
    score = f'{crowns[0]}-{crowns[1]}' + (' on tower health' if crowns[0] == crowns[1] and result != 'draw' else '')
    return {'tag': header.get('replay_tag') or path.stem, 'league': played.get('league', ''),
            'title': f'update {_version(played)} vs {opponent}', 'result': f'{result} {score}',
            'sub': f"{played.get('league', '')} game · {clock(out[-1][0]) if out else ''}",
            'learner': f'learner (update {_version(played)})', 'opponent': opponent, 'decks': decks,
            'frames': out, 'plays': sorted(plays)}


def clock(tick: int) -> str:
    seconds = tick / 20
    return f'{int(seconds // 60)}:{int(seconds % 60):02d}'


def main(argv: list[str]) -> int:
    import firstlight_bot  # noqa: F401  (puts FirstLight's native_runner on the path)
    import viewer as V
    from il.attacks import _base, card_name
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', type=Path, help='recordings folders or files')
    parser.add_argument('--league', help='only these games: self, snap, anchor, hog2, general')
    parser.add_argument('--result', choices=('won', 'lost'), help="only the learner's wins or losses")
    parser.add_argument('--max', type=int, default=40, help='the newest N games')
    parser.add_argument('--every', type=int, default=10, help='ticks between snapshots (motion is smoothed between)')
    parser.add_argument('--out', type=Path, default=CLAPHA / 'runs' / 'viewer' / 'games.html')
    parser.add_argument('--json', type=Path, help='write the data alone here (for the app) instead of a page')
    args = parser.parse_args(argv)
    files = [p for path in args.paths if path.exists()
             for p in ([path] if path.is_file() else sorted(path.rglob('*.jsonl.zst')))]
    chosen, unreadable = [], 0
    for path in files:
        try:
            played = _header(path).get('played') or {}
        except Exception:  # noqa: BLE001  (a file still being copied, or cut short)
            unreadable += 1
            continue
        if args.league and played.get('league') != args.league:
            continue
        if args.result and {'a': 'won', 'b': 'lost'}.get(played.get('result')) != args.result:
            continue
        chosen.append((_version(played), path))
    chosen = [path for _version_, path in sorted(chosen, reverse=True)[:args.max]]
    labels: list[str] = []
    index: dict[str, int] = {}

    def label(text: str) -> int:
        if text not in index:
            index[text] = len(labels)
            labels.append(text)
        return index[text]

    def name(card: int) -> str:
        return card_name(V, _base(card)) if card > 0 else '?'
    games = []
    for path in chosen:
        try:
            games.append(game(path, label, name, args.every))
        except Exception as error:  # noqa: BLE001
            unreadable += 1
            print(f'{path.name}: {type(error).__name__}: {error}')
    if unreadable:
        print(f'{unreadable} recordings could not be read (still copying?) and are left out')
    data = {'labels': labels, 'games': games,
            'about': f'{len(games)} games from {", ".join(str(p) for p in args.paths)}, newest first'
                     + (f'; {args.league} only' if args.league else '') + (f'; {args.result} only' if args.result else '')
                     + '. The learner plays from the bottom; rings mark where cards landed.'}
    target = args.json or args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, separators=(',', ':'))
    target.write_text(body if args.json else TEMPLATE.read_text().replace('__DATA__', body))
    print(f'{len(games)} games -> {target} ({target.stat().st_size // 1024} KB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
