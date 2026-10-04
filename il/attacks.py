"""Opponent attacks on the Hog 2.6 side in real games (the converted replays: other players' public
games, never the user's) and how the real player answered them: the survey behind the defense
drills.

An attack (the user's rule, 2026-09-27): from the first snapshot with an opponent troop on the Hog
side's half (past the river) until that half has had none for QUIET_TICKS. Troops are units
(card ids 26..., 27..., 203...); towers, projectiles, spell areas and effects are not, and a Goblin
Barrel or Graveyard counts through the units it spawns. After the user looked at examples: troops
that could not do damage are not an attack (a lone Skeleton walking over) -- a crossing counts when
the troops on the Hog side's half held at least THREAT_HP health together at some moment, or the
towers lost at least 50 -- and only answers near the attackers count as defense (an Ice Golem in the
other lane was being counted). Per attack:
  cards      the opponent cards whose units crossed; the main one (dearest) names the attack,
             small = one card of <= 4 elixir, big = anything more
  threat_hp  the most health the crossing troops had on the Hog side's half at once
  attack     the opponent's elixir behind it: their plays of those cards from 20 s before the
             crossing to the end, plus anything they placed on the Hog side's half meanwhile
  defense    the Hog side's plays from 3 s before the crossing to the end that landed within
             NEAR_TILES of an attacking troop (Hog Rider plays are counter-pushes and not counted);
             far      the ones further away (not counted)
  lost       tower health the Hog side lost, from the crossing to 1 s after the end

    ./py -m il.attacks [--sample 3000] [--out runs/attacks]
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))

QUIET_TICKS = 30            # 1.5 s without an opponent troop on our half ends an attack
LEAD_ATTACK = 400           # 20 s: troops that defended first and then walk over (counter-pushes)
#                             were played this long before they cross; each play counts once
LEAD_DEFENSE = 60           # 3 s: the defender's answer can start this early
THREAT_HP = 400            # a lone Skeleton (~80), Goblin (~200), Ice Spirit (~230) is below it
NEAR_TILES = 8.0           # a defense play lands this close to an attacking troop (the other lane is ~11)
HOG_RIDER = 26000021
TOWER_IDS = range(5000000, 5000006)
NAMES = {'Assassin': 'Bandit', 'IceGolemite': 'Ice Golem', 'MovingCannon': 'Cannon Cart', 'WitchMother': 'Mother Witch',
         'Pekka': 'P.E.K.K.A', 'AxeMan': 'Executioner', 'MiniSparkys': 'Zappies', 'SkeletonWarriors': 'Guards',
         'DarkWitch': 'Night Witch', 'CHAR_DISABLED_1': 'Berserker', 'CHAR_DISABLED_3': 'Berserker',
         'AngryBarbarian': 'Elite Barbarians', 'AngryBarbarians': 'Elite Barbarians', 'Ghost': 'Royal Ghost', 'BlowdartGoblin': 'Dart Goblin',
         'SuperMiniPekka': 'Mini P.E.K.K.A', 'MiniPekka': 'Mini P.E.K.K.A', 'ZapMachine': 'Sparky', 'Bowler': 'Bowler',
         'SkeletonBalloon': 'Skeleton Barrel', 'RageBarbarian': 'Lumberjack', 'BattleHealer': 'Battle Healer',
         'MergeMaiden': 'Merge Maiden', 'goblinstein': 'Goblinstein'}


def card_name(V, card: int) -> str:
    """A readable name: the table above, else the internal name split at capitals."""
    import re
    raw = str(V.CARDS.get(card, {}).get('name', card))
    return NAMES.get(raw, re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', raw))


# units the engine files under a placeholder card id: the card that was played (checked 2026-09-27:
# 33 of 40 crossings by these units followed a Berserker play)
UNIT_CARD = {26000076: 26000102, 26000088: 26000102}


def _base(card_id: int) -> int:
    card_id = 26000000 + card_id % 1000000 if 203000000 <= card_id < 204000000 else card_id
    return UNIT_CARD.get(card_id, card_id)


def _is_troop(o: dict) -> bool:
    card = int(o.get('cardId') or -1)
    return (o.get('projectile') is None and o.get('hp') is not None and int(o['nativeObjectId']) not in TOWER_IDS
            and (26000000 <= card < 28000000 or 203000000 <= card < 204000000))


def _our_half(side: int, y: int) -> bool:
    return y < 15000 if side == 0 else y > 17000       # past the river (rows 15-16)


def _row_ours(side: int, row: int) -> bool:
    return row < 15 if side == 0 else row > 16


def _tower_hp(frame: dict, owner: int) -> int:
    return sum(int(o['hp']) for o in frame.get('objects') or ()
               if int(o['nativeObjectId']) in TOWER_IDS and int(o['owner']) == owner and o.get('hp') is not None)


def attacks_in(path: Path) -> list[dict]:
    import il.samples as S
    from il.frames import load_replay
    header, frames = load_replay(path, with_replay=False)
    timeline = header['timeline']
    plays = [p for p in timeline['plays'] if p.get('kind') == 'card' and p.get('card_id')]
    frames = sorted(frames, key=lambda f: int(f['tick']))
    out = []
    for side in (0, 1):
        if not S.HOG26_CARDS <= set(timeline['decks'][side]):
            continue
        enemy = 1 - side
        current, windows = None, []
        for index, frame in enumerate(frames):
            tick = int(frame['tick'])
            crossed = [o for o in frame.get('objects') or ()
                       if int(o['owner']) == enemy and _is_troop(o) and _our_half(side, int(o['y']))]
            if crossed:
                if current is None:
                    current = {'start': tick, 'start_index': index, 'last': tick, 'cards': set(), 'threat_hp': 0}
                current['last'] = tick
                current['cards'].update(_base(int(o['cardId'])) for o in crossed)
                current['threat_hp'] = max(current['threat_hp'], sum(int(o['hp']) for o in crossed))
            elif current is not None and tick - current['last'] >= QUIET_TICKS:
                windows.append(current)
                current = None
        if current is not None:
            windows.append(current)
        used: set[int] = set()
        ticks = [int(f['tick']) for f in frames]
        for window in windows:
            out.append(_score(window, frames, ticks, plays, side, header['replay_tag'], S, used))
    return out


def _distance_to_attackers(frames: list[dict], ticks: list[int], tick: int, cards: set, enemy: int,
                           cell: tuple[int, int]) -> float:
    """Tiles from a play's cell to the nearest troop of the attack's cards when the play landed
    (or, before any is on the board, when the attack started)."""
    x0, y0 = cell[0] * 1000 + 500, cell[1] * 1000 + 500
    index = min(bisect.bisect_left(ticks, tick), len(frames) - 1)
    troops = [o for o in frames[index].get('objects') or ()
              if int(o['owner']) == enemy and _is_troop(o) and _base(int(o['cardId'])) in cards]
    return min((math.hypot(int(o['x']) - x0, int(o['y']) - y0) / 1000 for o in troops), default=math.inf)


def _score(attack: dict, frames: list[dict], ticks: list[int], plays: list[dict], side: int, tag: str, S,
           used: set) -> dict:
    enemy = 1 - side
    start, last = attack['start'], attack['last']
    cards = attack['cards']
    attack_plays = [p for p in plays if p['owner'] == enemy and p['index'] not in used
                    and start - LEAD_ATTACK <= p['lands'] + 1 <= last
                    and (p['card_id'] in cards or (p.get('grid') and _row_ours(side, p['grid'][1])
                                                   and p['lands'] + 1 >= start - LEAD_DEFENSE))]
    used.update(p['index'] for p in attack_plays)
    candidates = [p for p in plays if p['owner'] == side and start - LEAD_DEFENSE <= p['lands'] + 1 <= last
                  and p['card_id'] != HOG_RIDER and p.get('grid')]
    defense_plays, far = [], []
    for play in candidates:
        tick = max(play['lands'] + 1, start)
        near = _distance_to_attackers(frames, ticks, tick, cards, enemy, play['grid']) <= NEAR_TILES
        (defense_plays if near else far).append(play)
    before = frames[max(0, attack['start_index'] - 1)]
    after = next((f for f in frames if int(f['tick']) >= last + 20), frames[-1])
    costs = {}
    for card in cards:
        costs[card] = S._card_cost(card) or 0.0
    main = max(cards, key=lambda c: (costs.get(c, 0), c))
    attack_elixir = sum(S._card_cost(p['card_id']) or 0.0 for p in attack_plays)
    lost = max(0, _tower_hp(before, side) - _tower_hp(after, side))
    return {'tag': tag, 'side': side, 'start': start, 'end': last, 'cards': sorted(cards), 'main': main,
            'size': 'small' if len(cards) == 1 and costs.get(main, 0) <= 4 else 'big',
            'threat_hp': attack['threat_hp'], 'attack': attack['threat_hp'] >= THREAT_HP or lost >= 50,
            'attack_elixir': attack_elixir,
            'defense_elixir': sum(S._card_cost(p['card_id']) or 0.0 for p in defense_plays),
            'defense_cards': [p['card_id'] for p in defense_plays],
            'far_cards': [p['card_id'] for p in far],
            'lost': lost}


def main(argv: list[str]) -> int:
    import firstlight_bot  # noqa: F401  (puts FirstLight's native_runner on the path)
    import viewer as V
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=Path, default=CLAPHA / 'runs/conv-hog26')
    parser.add_argument('--sample', type=int, default=3000, help='games to survey (0: all)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', type=Path, default=CLAPHA / 'runs/attacks')
    args = parser.parse_args(argv)
    paths = sorted((args.frames / 'frames').glob('*/*.jsonl.zst'))
    random.Random(args.seed).shuffle(paths)
    if args.sample:
        paths = paths[:args.sample]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.out / 'attacks.jsonl').open('w') as handle:
        for number, path in enumerate(paths, 1):
            try:
                found = attacks_in(path)
            except Exception as error:  # noqa: BLE001  (one unreadable game must not stop the survey)
                print(f'{path.name}: {type(error).__name__}: {error}', flush=True)
                continue
            for row in found:
                handle.write(json.dumps(row) + '\n')
            rows += found
            if number % 500 == 0:
                print(f'{number} games, {len(rows)} attacks', flush=True)

    def name(card: int) -> str:
        return card_name(V, card)
    crossings = rows
    rows = [r for r in crossings if r['attack']]
    far = sum(len(r['far_cards']) for r in rows)
    print(f'\n{len(crossings)} crossings; {len(crossings) - len(rows)} not attacks (troops under {THREAT_HP} health '
          f'together and under 50 tower damage); {far} of {far + sum(len(r["defense_cards"]) for r in rows)} '
          f'Hog-side plays during attacks landed over {NEAR_TILES:.0f} tiles from the attackers (not counted)')
    by_main: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_main[row['main']].append(row)
    print(f'\n{len(paths)} games, {len(rows)} attacks on the Hog 2.6 side '
          f'({sum(r["size"] == "small" for r in rows)} small, {sum(r["size"] == "big" for r in rows)} big)')
    print('main card | attacks | small | median attack elixir | median defense elixir | mean elixir balance '
          '(attack - defense) | mean tower health lost | no damage')
    for card, group in sorted(by_main.items(), key=lambda kv: -len(kv[1]))[:30]:
        balance = [r['attack_elixir'] - r['defense_elixir'] for r in group]
        print(f"{name(card)} | {len(group)} | {100 * sum(r['size'] == 'small' for r in group) / len(group):.0f}% | "
              f"{statistics.median(r['attack_elixir'] for r in group):.0f} | "
              f"{statistics.median(r['defense_elixir'] for r in group):.0f} | {statistics.mean(balance):+.1f} | "
              f"{statistics.mean(r['lost'] for r in group):.0f} | "
              f"{100 * sum(r['lost'] == 0 for r in group) / len(group):.0f}%")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))


def examples(attacks_file: Path, frames_dir: Path, out: Path, kinds: int = 6, every: int = 10) -> int:
    """A handful of detected attacks, snapshot by snapshot, for checking the detection by eye: for
    the most common main cards, one attack where the tower lost health and one defended clean.
    Drawn from the defender's side (its half at the bottom). Writes a compact JSON for the viewer."""
    import firstlight_bot  # noqa: F401
    import viewer as V
    from il.frames import load_replay
    everything = [json.loads(line) for line in attacks_file.read_text().splitlines() if line.strip()]
    rows = [r for r in everything if r['attack']]
    by_main: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_main[row['main']].append(row)

    def name(card: int) -> str:
        return card_name(V, card)
    picked = []
    for card, group in sorted(by_main.items(), key=lambda kv: -len(kv[1]))[:kinds]:
        hit = next((r for r in group if r['lost'] > 0 and r['end'] - r['start'] <= 400), None)
        clean = next((r for r in group if r['lost'] == 0 and r['end'] - r['start'] <= 400), None)
        picked += [r for r in (hit, clean) if r is not None]
    # the new rules at work: a defense with a play too far away to count, crossings that are no attack
    picked += [r for r in rows if r['far_cards'] and r['end'] - r['start'] <= 400][:2]
    picked += [r for r in everything if not r['attack'] and r['end'] - r['start'] <= 200][:3]
    labels: list[str] = []
    index_of: dict[str, int] = {}

    def label(text: str) -> int:
        if text not in index_of:
            index_of[text] = len(labels)
            labels.append(text)
        return index_of[text]
    paths = {p.stem.split('.')[0]: p for p in (frames_dir / 'frames').glob('*/*.jsonl.zst')}
    shown = []
    for row in picked:
        path = paths.get(row['tag'])
        if path is None:
            continue
        _header, frames = load_replay(path, with_replay=False)
        side = row['side']
        low, high = row['start'] - 60, row['end'] + 20
        full: dict[int, int] = {}
        snaps = []
        for frame in sorted(frames, key=lambda f: int(f['tick'])):
            tick = int(frame['tick'])
            if tick < low or tick > high or (tick - low) % every:
                continue
            items = []
            for o in frame.get('objects') or ():
                oid, owner = int(o['nativeObjectId']), int(o['owner'])
                tower = oid in TOWER_IDS
                if not tower and not _is_troop(o):
                    continue
                x, y = int(o['x']), int(o['y'])
                if side == 1:                        # the defender at the bottom
                    x, y = 18000 - x, 32000 - y
                hp = int(o['hp'])
                full.setdefault(oid, hp)
                text = ('King' if oid in (5000000, 5000003) else 'Tower') if tower else name(_base(int(o['cardId'])))
                items.append([0 if owner == side else 1, label(text), round(x / 100), round(y / 100),
                              round(100 * hp / max(1, full[oid])) if tower else -1])
            snaps.append([tick, items])
        title = (f"{name(row['main'])} ({row['size']})" if row['attack'] else
                 f"NOT an attack: {name(row['main'])} ({row['threat_hp']} health, no damage)")
        shown.append({'title': title, 'cards': [name(c) for c in row['cards']],
                      'start': row['start'], 'end': row['end'], 'attack_elixir': row['attack_elixir'],
                      'defense_elixir': row['defense_elixir'],
                      'defense': [name(c) for c in row['defense_cards']]
                                 + [f'{name(c)} (too far from the attack: not counted)' for c in row['far_cards']],
                      'lost': row['lost'], 'frames': snaps})
    out.write_text(json.dumps({'labels': labels, 'examples': shown}, separators=(',', ':')))
    print(f'{len(shown)} examples -> {out} ({out.stat().st_size // 1024} KB)')
    return 0
