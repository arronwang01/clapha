"""Every Hog Rider played against the bot in live matches the user marked for it: what it did to the bot's
tower and what the bot did about it. Read from the console's own recordings of those matches
(artifacts/viewer-sessions/<session>: frames.jsonl, the reader's frames; queue.jsonl, both players' commands)
and its timing record (build/timing_<port>.jsonl). Only sessions and battles the user named are read.

Per Hog: when it was issued and when the bot's client first showed it, the bot's hand, elixir and Cannon
(standing, in hand, how many cards away) at that moment, what the bot played until the Hog was dealt with,
whether a Cannon pulled it, how many hits it got on a tower -- and why: no Cannon although it could
(decision), Cannon not in hand (cycle), Cannon down too late (after the Hog reached the tower), down in time
without pulling (placement), or the opponent removed it.

    ./py -m il.live_hogs artifacts/viewer-sessions/<session> [--port 8778] [--mirror-only] [--last N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))

HOG, CANNON, FIREBALL, LOG = 26000021, 27000000, 28000000, 28000011
HOG26 = {26000021, 26000014, 27000000, 26000038, 26000030, 26000010, 28000000, 28000011}
NAMES = {26000021: 'Hog', 26000014: 'Musketeer', 27000000: 'Cannon', 28000000: 'Fireball', 28000011: 'Log',
         26000010: 'Skeletons', 26000038: 'Ice Golem', 26000030: 'Ice Spirit'}
COST = {26000021: 4, 26000014: 4, 27000000: 3, 28000000: 4, 28000011: 2, 26000010: 1, 26000038: 2, 26000030: 1}
COMMAND_AGE = 21


def name(card) -> str:
    return NAMES.get(card, str(card))


def clock(tick: int) -> str:
    return f'{tick // 20 // 60}:{tick // 20 % 60:02d}'


def plain(address) -> int:
    """An object's address without its pointer tag: a unit's `target` carries none, its `address` does."""
    return int(address or '0x0', 16) & 0x00FFFFFFFFFFFFFF


def tiles(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) / 1000.0


def read_battles(session: Path, cards=(HOG, CANNON)):
    """The session's battles in order, each with what this analysis needs from every frame: the towers
    and the units of `cards`, both sides."""
    import viewer as V
    current: dict[int, dict] = {}
    done = []
    with open(session / 'frames.jsonl', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            frame, health = row['frame'], row['health']
            side = health.get('local_side')
            if side not in (0, 1) or not frame.get('battle_active'):
                continue
            pid, address, tick = frame['pid'], frame['chain']['battle'], int(frame['game_tick'])
            battle = current.get(pid)
            if battle is None or battle['battle'] != address or tick < battle['frames'][-1]['tick'] - 100:
                if battle is not None:
                    done.append(battle)
                battle = current[pid] = {'pid': pid, 'battle': address, 'side': side, 'frames': [], 'deck': None}
            me = next((p for p in frame['players'] if p['side'] == side), None)
            if not me:
                continue
            deck = me.get('deck_card_ids') or []
            if len(deck) == 8 and battle['deck'] is None:
                battle['deck'] = [V.card_identity(c)[0] for c in deck]
            units = []
            for e in frame.get('entities') or ():
                base = V.card_identity(e['card_id'])[0] if e.get('card_id', 0) > 0 else 0
                tower = e.get('kind') == 13 and base != CANNON      # kind 13 is every building, a Cannon too
                if tower or base in cards:
                    units.append((plain(e['address']), 'tower' if tower else base, e['side'], e['x'], e['y'], e['hp'],
                                  plain(e.get('target'))))
            hand = [deck[i] if 0 <= i < len(deck) else -1 for i in me.get('hand_deck_indices') or ()]
            cycle = [deck[i] for i in me.get('cycle_deck_indices') or () if 0 <= i < len(deck)]
            battle['frames'].append({'tick': tick, 'elixir': me['elixir_raw'] / 10000.0,
                                     'hand': [V.card_identity(c)[0] if c > 0 else -1 for c in hand],
                                     'cycle': [V.card_identity(c)[0] for c in cycle], 'units': units})
    done.extend(current.values())
    return [b for b in done if b['deck'] and len(b['frames']) > 200]


def read_commands(session: Path) -> dict[str, list[dict]]:
    """Every command either player issued, per battle address: card, side, issue tick, tile point, and the
    tick this client first showed it."""
    import viewer as V
    battles: dict[str, dict] = {}
    with open(session / 'queue.jsonl', encoding='utf-8') as handle:
        for line in handle:
            if not line.startswith('{'):
                continue
            row = json.loads(line)
            tick = row.get('tick_0x60')
            store = battles.setdefault(str(row.get('battle')), {})
            for entry in (row.get('queue') or {}).get('entries', []):
                if int(entry.get('card_id') or 0) <= 0 or not isinstance(entry.get('issue_tick'), int):
                    continue
                key = (entry.get('account_lo'), entry.get('seq'), entry['issue_tick'], entry['card_id'])
                if key in store:
                    continue
                card, form, kind = V.card_identity(entry['card_id'])
                store[key] = {'card': card, 'kind': kind, 'side': V.entry_side(entry, row.get('accounts')),
                              'issue': entry['issue_tick'], 'lands': entry['issue_tick'] + COMMAND_AGE,
                              'x': entry.get('x'), 'y': entry.get('y'), 'seen': tick}
    return {battle: sorted(store.values(), key=lambda c: c['issue']) for battle, store in battles.items()}


def read_timing(port: int | None) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for path in sorted((CLAPHA / 'build').glob(f'timing_{port or "*"}.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['kind'] != 'turn':
                rows.setdefault(str(row.get('battle')), []).append(row)
    return rows


def analyse(battle: dict, commands: list[dict], timing: list[dict]) -> list[dict]:
    side = battle['side']
    frames = battle['frames']
    by_tick = {f['tick']: f for f in frames}
    ticks = sorted(by_tick)

    def at(tick: int) -> dict:
        """The frame at `tick`, or the last one before it."""
        index = max(0, min(len(ticks) - 1, next((i for i, t in enumerate(ticks) if t > tick), len(ticks)) - 1))
        return by_tick[ticks[index]]

    theirs = [c for c in commands if c['side'] == 1 - side and c['kind'] == 'card']
    ours = [c for c in commands if c['side'] == side and c['kind'] == 'card']
    # units over time
    tracks: dict[str, list] = {}
    kinds: dict[str, tuple] = {}
    for f in frames:
        for address, kind, owner, x, y, hp, target in f['units']:
            tracks.setdefault((address, kind, owner), []).append((f['tick'], x, y, hp, target))
            kinds[(address, kind, owner)] = (address, kind, owner)
    towers = {kinds[k][0]: k for k in tracks if kinds[k][1] == 'tower' and kinds[k][2] == side}
    cannons = {k: tracks[k] for k in tracks if kinds[k][1] == CANNON and kinds[k][2] == side}
    hogs = sorted(((k, tracks[k]) for k in tracks if kinds[k][1] == HOG and kinds[k][2] == 1 - side),
                  key=lambda item: item[1][0][0])
    rows = []
    hog_commands = [c for c in theirs if c['card'] == HOG]
    for key, track in hogs:
        spawn, died = track[0][0], track[-1][0]
        command = min(hog_commands, key=lambda c: abs(c['lands'] + 1 - spawn), default=None)
        if command is not None and abs(command['lands'] + 1 - spawn) > 6:
            command = None
        seen = command['seen'] if command and command.get('seen') is not None else spawn
        seen = min(seen, spawn)
        row = {'clock': clock(spawn), 'spawn': spawn, 'seen_ticks_before': spawn - seen, 'lived': died - spawn}
        # the Hog's own course: first frame it stands still by a tower with that tower as its target
        arrived = None
        for i in range(1, len(track)):
            tick, x, y, hp, target = track[i]
            if target in towers and (x, y) == track[i - 1][1:3] and tiles((x, y), tracks[towers[target]][0][1:3]) < 4.0:
                arrived = (tick, target)
                break
        pulled = next(((t, tg) for t, x, y, hp, tg in track
                       if any(kinds[k][0] == tg and tracks[k][0][0] <= t <= tracks[k][-1][0] for k in cannons)), None)
        # hits: drops of the tower's hitpoints while the Hog stands by it with it as its target
        hits, damage = 0, 0
        for i in range(1, len(track)):
            tick, x, y, hp, target = track[i]
            if target in towers and tiles((x, y), tracks[towers[target]][0][1:3]) < 4.0:
                tower = {t: h for t, _x, _y, h, _g in tracks[towers[target]]}
                before, after = tower.get(track[i - 1][0]), tower.get(tick)
                if before is not None and after is not None and after < before:
                    hits, damage = hits + 1, damage + before - after
        row.update(hits=hits, damage=damage, arrived=arrived[0] if arrived else None,
                   pulled=pulled[0] if pulled else None)
        # the bot's situation when its client first showed the Hog
        f = at(seen)
        standing = [k for k, t in cannons.items() if t[0][0] <= seen <= t[-1][0]]
        cycle_away = f['cycle'].index(CANNON) + 1 if CANNON in f['cycle'] else None
        row.update(elixir=round(f['elixir'], 1), hand=[name(c) for c in f['hand']],
                   cannon_in_hand=CANNON in f['hand'], cannon_standing=bool(standing), cannon_cards_away=cycle_away)
        # what the bot played from then until the Hog was gone (or 3 s after it arrived)
        until = min(died, (arrived[0] + 60) if arrived else died)
        answer = [c for c in ours if seen <= c['issue'] <= until]
        row['answer'] = [(name(c['card']), c['lands'] + 1 - spawn) for c in answer]
        cannon_plays = [c for c in answer if c['card'] == CANNON]
        row['their_spells'] = [(name(c['card']), c['lands'] + 1 - spawn) for c in theirs
                               if c['card'] in (FIREBALL, LOG) and seen - 40 <= c['issue'] <= until]
        # the Cannon it played, if any: when it was down, how far from the Hog's line, and its timing record
        if cannon_plays:
            c = cannon_plays[0]
            down = c['lands'] + 1
            row.update(cannon_down_after_spawn=down - spawn,
                       cannon_down_vs_arrival=(down - arrived[0]) if arrived else None,
                       cannon_tile=(round((c['x'] or 0) / 1000 - 0.5), round((c['y'] or 0) / 1000 - 0.5)))
            hog_then = min(track, key=lambda p: abs(p[0] - down))
            row['cannon_to_hog_tiles'] = round(tiles((c['x'], c['y']), hog_then[1:3]), 1)
            record = next((t for t in timing if t['kind'] == 'play' and t.get('card') == 'Cannon'
                           and t.get('issue_tick') == c['issue']), None)
            if record:
                row.update(late=record.get('late'), elixir_wait=record.get('elixir_wait'),
                           decided_turn=record.get('turn'), offset=record.get('offset'))
                row['decided_after_seen'] = record['turn'] + (record.get('offset') or 0) - seen
            life = next((t for k, t in cannons.items() if abs(t[0][0] - down) <= 3), None)
            if life:
                row['cannon_lived'] = life[-1][0] - life[0][0]
        dropped = [t for t in timing if t['kind'] in ('dropped', 'lost') and t.get('card') == 'Cannon'
                   and seen <= (t.get('turn') or -1) <= until]
        if dropped:
            row['cannon_dropped'] = [(t['kind'], t.get('wait')) for t in dropped]
        # the verdict
        if hits == 0:
            row['verdict'] = 'no hit' + (' (pulled)' if pulled else '')
        elif standing and not cannon_plays:
            row['verdict'] = 'Cannon was standing; the Hog still hit' + (' after being pulled' if pulled else '')
        elif cannon_plays:
            when = row.get('cannon_down_vs_arrival')
            if pulled:
                row['verdict'] = 'pulled, hit later (Cannon gone)'
            elif when is not None and when > 0:
                row['verdict'] = f'Cannon down {when} ticks AFTER the Hog reached the tower'
            else:
                row['verdict'] = f'Cannon down in time, did not pull ({row.get("cannon_to_hog_tiles")} tiles from the Hog)'
        elif row['cannon_in_hand'] and f['elixir'] >= COST[CANNON]:
            row['verdict'] = 'Cannon in hand and affordable, not played'
        elif row['cannon_in_hand']:
            row['verdict'] = f'Cannon in hand, {f["elixir"]:.1f} elixir, not played'
        else:
            row['verdict'] = f'Cannon not in hand ({cycle_away} cards away)' if cycle_away else 'Cannon not in hand'
        rows.append(row)
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=Path)
    parser.add_argument('--port', type=int, help="the bot console's timing record (default: any)")
    parser.add_argument('--mirror-only', action='store_true', help='battles where both decks are Hog 2.6')
    parser.add_argument('--last', type=int, help='only the last N battles')
    parser.add_argument('--json', type=Path, help='also write every row here')
    args = parser.parse_args(argv)
    battles = read_battles(args.session)
    commands = read_commands(args.session)
    timing = read_timing(args.port)
    if args.last:
        battles = battles[-args.last:]
    everything = []
    for number, battle in enumerate(battles, 1):
        plays = commands.get(str(battle['battle']), [])
        ours = set(battle['deck'])
        theirs = {c['card'] for c in plays if c['side'] == 1 - battle['side'] and c['kind'] == 'card'}
        mirror = ours == HOG26 and theirs and theirs <= HOG26
        if args.mirror_only and not mirror:
            continue
        rows = analyse(battle, plays, timing.get(str(battle['battle']), []))
        if not rows:
            continue
        end = battle['frames'][-1]['tick']
        print(f'\n=== battle {number}: {clock(end)} long, bot on side {battle["side"]}, '
              f'{"2.6 mirror" if mirror else "opponent deck " + ", ".join(sorted(name(c) for c in theirs))} -- '
              f'{len(rows)} Hogs, {sum(r["hits"] for r in rows)} hits, {sum(r["damage"] for r in rows)} damage')
        for r in rows:
            cannon = ('standing' if r['cannon_standing'] else 'in hand' if r['cannon_in_hand']
                      else f'{r["cannon_cards_away"]} away' if r['cannon_cards_away'] else 'not in hand')
            extra = ''
            if 'cannon_down_after_spawn' in r:
                extra = f' | Cannon down +{r["cannon_down_after_spawn"]} at tile {r["cannon_tile"]}'
                if r.get('late') is not None:
                    extra += f', pipeline late {r["late"]}' + (' (waited for elixir)' if r.get('elixir_wait') else '')
                if r.get('decided_after_seen') is not None:
                    extra += f', decided {r["decided_after_seen"]} ticks after it saw the Hog'
            if r.get('cannon_dropped'):
                extra += f' | Cannon decision lost: {r["cannon_dropped"]}'
            print(f'  {r["clock"]}  hits {r["hits"]} ({r["damage"]:4})  {r["verdict"]}')
            print(f'         saw it {r["seen_ticks_before"]} ticks early; {r["elixir"]} elixir, Cannon {cannon}, hand {r["hand"]}')
            print(f'         bot played {r["answer"]}; their spells {r["their_spells"]}{extra}')
        everything += [dict(r, battle=number) for r in rows]
    if everything:
        print(f'\n{len(everything)} Hogs: ' + ', '.join(f'{k} {v}' for k, v in Counter(
            r['verdict'].split(' (')[0].split(' AFTER')[0] if not r['verdict'].startswith('Cannon down') else
            r['verdict'].split(' ticks')[0].rsplit(' ', 1)[0] if 'AFTER' in r['verdict'] else 'Cannon down in time, did not pull'
            for r in everything).most_common()))
    if args.json:
        args.json.write_text(json.dumps(everything, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
