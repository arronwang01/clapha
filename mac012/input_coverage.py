"""Which of the model's inputs do we actually fill? Measured, not recalled.

Replays a recorded match (a viewer session's frames.jsonl) through our live pipeline -- the same
FLO.build + FirstLight tensorizer the console uses, on the policy's own five-tick grid -- and for
every feature slot of every input tensor reports how often it is non-zero on real rows.

Slot names come from FirstLight's own tensorizer source: each slot is assigned in one builder
function as `name[k] = <expression>`, and that expression is printed as the slot's meaning.

A slot that is always zero is either (a) data we do not supply, or (b) a situation that did not
occur in this match (no air units, no overtime...). The report marks which runtime domain each
slot reads, and whether our observation ever fills that domain, so (a) and (b) can be told apart.

    python3 mac012/input_coverage.py [SESSION_DIR] [--all]
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac012 import firstlight_obs as FLO  # noqa: E402
from mac012 import firstlight_bot as FLB  # noqa: E402

CLAPHA = Path(__file__).resolve().parents[1]
TENSORIZER = FLO.FIRSTLIGHT / 'native_runner/training/v4/tensorizer.py'

# tensor path -> (builder function, array variable in it, row mask path)
SPEC = {
    'match_scalars': ('_scalars', 'result', None),
    'own_cards.runtime_features': ('_card_runtime_features', 'features', 'own_cards.mask'),
    'opponent_cards.runtime_features': ('_card_runtime_features', 'features',
                                        'opponent_cards.mask'),
    'towers.features': ('_towers', 'feature', 'towers.mask'),
    'groups.features': ('_groups', 'features', 'groups.mask'),
    'groups.child_features': ('_groups', 'dynamic', 'groups.child_mask'),
    'events.features': ('_events', 'features', 'events.mask'),
}

# Which observation domain an expression reads, and whether we ever fill it.
DOMAINS = [
    ('attack', r'attack', True), ('movement', r'movement', True),
    ('deployment', r'deployment', True), ('target', r'visible_target', True),
    ('velocity', r'velocity|vx|vy', True), ('hitpoints', r'hitpoints|hp', True),
    ('shield', r'shield', False), ('effects/buffs', r'effect|buff|stun|slow|freeze', False),
    ('projectile', r'projectile', False), ('visibility', r'visibility|hidden|burrow', False),
    ('ability', r'ability', False), ('evolution', r'evolution', False),
    ('capture', r'capture', False), ('relocation', r'relocation', False),
    ('periodic modifier', r'modifier', False), ('resource', r'resource', False),
    ('events', r'event', False), ('tower troop runtime', r'tower_runtime', False),
    ('opponent tracker', r'tracker', True),
]


def slot_labels() -> dict[tuple[str, str], dict[int, str]]:
    source = TENSORIZER.read_text().splitlines()
    labels: dict[tuple[str, str], dict[int, str]] = collections.defaultdict(dict)
    function = None
    pattern = re.compile(r'^\s+(\w+)\[(?:0,\s*(?:\w+),\s*)?(\d+)\]\s*=\s*(.+)$')
    for line in source:
        match_def = re.match(r'^    def (\w+)\(', line)
        if match_def:
            function = match_def.group(1)
            continue
        match = pattern.match(line)
        if function and match:
            variable, index, expression = match.group(1), int(match.group(2)), match.group(3)
            existing = labels[(function, variable)].get(index)
            text = expression.strip()
            labels[(function, variable)][index] = text if not existing else (
                existing if text in existing else f'{existing}  |  {text}')
    return labels


def domain_of(expression: str) -> tuple[str, bool] | None:
    lowered = expression.lower()
    for name, regex, filled in DOMAINS:
        if re.search(regex, lowered):
            return name, filled
    return None


def get(batch, path: str):
    value = batch
    for part in path.split('.'):
        value = getattr(value, part)
    return value


def load_session(directory: Path):
    frames = []
    for line in (directory / 'frames.jsonl').open():
        row = json.loads(line)
        frames.append((row['frame'], row['health']))
    revealed_by_tick = []
    queue = directory / 'queue.jsonl'
    if queue.is_file():
        for line in queue.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tick = row.get('tick_0x60', row.get('tick'))
            if row.get('revealed') is not None and tick is not None:
                revealed_by_tick.append((int(tick), row['revealed']))
    revealed_by_tick.sort(key=lambda item: item[0])
    return frames, revealed_by_tick


def main() -> int:
    sessions = sorted((CLAPHA / 'artifacts/viewer-sessions').iterdir())
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    directory = Path(args[0]) if args else max(
        (d for d in sessions if (d / 'frames.jsonl').is_file()
         and (d / 'frames.jsonl').stat().st_size > 0), key=lambda d: d.name)
    frames, revealed_by_tick = load_session(directory)
    battle_rows = [json.loads(l) for l in (directory / 'queue.jsonl').open() if l.strip()]
    side = next((h.get('local_side') for _, h in frames if h.get('local_side') in (0, 1)), 0)
    # A session can hold several battles; replay only the first (the chosen one), since each
    # battle needs its own episode and deck.
    first = next(f for f, h in frames if f.get('battle_active'))
    battle_id = (first.get('chain') or {}).get('battle')
    frames = [(f, h) for f, h in frames if (f.get('chain') or {}).get('battle') == battle_id]
    cut = next((i for i in range(1, len(frames))
                if frames[i][0]['game_tick'] < frames[i - 1][0]['game_tick']), len(frames))
    frames = frames[:cut]
    me = next(p for p in first['players'] if p['side'] == side)
    deck = me['deck_card_ids']
    # the queue rows of this battle only: a session can hold several
    first_tick, last_tick = frames[0][0]['game_tick'], frames[-1][0]['game_tick']
    ours = [r for r in battle_rows if r.get('battle') == battle_id]
    if not ours:  # queue rows name the battle object differently; bound by time order instead
        cut_at = next((i for i in range(1, len(battle_rows))
                       if (battle_rows[i].get('tick_0x60') or 0) < (battle_rows[i - 1].get('tick_0x60') or 0)),
                      len(battle_rows))
        ours = battle_rows[:cut_at]
    revealed_by_tick = sorted((int(r['tick_0x60']), r['revealed']) for r in ours
                              if r.get('revealed') is not None and r.get('tick_0x60') is not None)
    print(f'session {directory.name}: {len(frames)} frames of battle {battle_id}, local side '
          f'{side}, reader has attack fields: {"has_attack" in (first["entities"] or [{}])[0]}')

    # Plays, exactly as the live queue pump derives them, and the opponent's deck as the set of
    # cards they played (complete once all eight have been seen).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import viewer as V  # noqa: E402
    plays, pending = [], {}
    for row in ours:
        plays += V.executed_plays(row, row.get('queue', {}).get('entries', []), pending)
    opponent_cards = []
    for play in plays:
        if play['side'] != side and play.get('kind') == 'card' and play['card_id'] not in opponent_cards:
            opponent_cards.append(play['card_id'])
    opponent_deck = opponent_cards if len(opponent_cards) == 8 else None
    if opponent_deck is None and opponent_cards and '--pad-deck' in sys.argv:
        # cards never played can be anything: padding is consistent with every real play
        fillers = [c for c in sorted(FLO.known_cards()) if c not in opponent_cards
                   and 26000000 <= c < 29000000 and c != FLO.MIRROR_CARD_ID]
        opponent_deck = opponent_cards + fillers[:8 - len(opponent_cards)]
    print(f'plays recovered: {len(plays)}; opponent deck '
          f'{"complete" if opponent_deck else f"incomplete ({len(opponent_cards)} seen)"}')
    decks = {side: tuple(deck), 1 - side: tuple(opponent_deck or ())}
    runner = FLB.FirstLightRunner('fl:hog1')
    counts: dict[str, torch.Tensor] = {}
    rows: dict[str, int] = collections.Counter()
    battle = None
    last_turn = -10 ** 9
    started = False
    turns = 0
    for frame, health in frames:
        if not frame.get('battle_active'):
            continue
        hand = next((p for p in frame['players'] if p['side'] == side), {})
        if len(hand.get('hand_deck_indices') or []) != 4 or min(hand['hand_deck_indices']) < 0:
            continue
        turn = frame['game_tick'] - frame['game_tick'] % 5
        if turn <= last_turn:
            continue
        last_turn = turn
        revealed = None
        for tick, value in revealed_by_tick:
            if tick > frame['game_tick']:
                break
            revealed = value
        seen = {index: list(cards) for index, cards in enumerate(revealed or [[], []])}
        health = {**health, 'local_side': side}
        observation, battle = FLO.build(frame, health, '1', battle=battle, revealed=seen,
                                        plays=plays, decks=decks)
        if not started:
            runner.start_battle(deck, opponent_deck or deck, side, observation,
                                {p['side']: p['elixir_raw'] / 10000.0 for p in frame['players']},
                                our_forms=me.get('deck_form_flags'))
            started = True
        batch = runner.session.tensorizer.tensorize(observation, validate=False)
        turns += 1
        for path, (_function, _variable, mask_path) in SPEC.items():
            tensor = get(batch, path)[0]
            if tensor.numel() == 0:
                continue
            if mask_path is None:
                valid = tensor.reshape(1, -1)
            else:
                mask = get(batch, mask_path)[0].bool()
                valid = tensor[mask]
            if valid.shape[0] == 0:
                continue
            nonzero = (valid != 0).sum(dim=0).to(torch.float64)
            counts[path] = counts.get(path, torch.zeros_like(nonzero)) + nonzero
            rows[path] += valid.shape[0]

    labels = slot_labels()
    show_all = '--all' in sys.argv
    print(f'{turns} decision turns replayed\n')
    summary = []
    for path, (function, variable, _mask) in SPEC.items():
        names = labels.get((function, variable), {})
        total = rows.get(path, 0)
        width = counts[path].shape[0] if path in counts else (max(names) + 1 if names else 0)
        filled = [i for i in range(width) if total and counts[path][i] > 0]
        print(f'=== {path}: {len(filled)}/{width} slots ever non-zero '
              f'({total} rows seen) ===')
        for index in range(width):
            share = (float(counts[path][index]) / total * 100) if total else 0.0
            label = names.get(index, '(unassigned in source)')
            domain = domain_of(label)
            tag = ''
            if share == 0:
                if domain and not domain[1]:
                    tag = f'MISSING: we never supply {domain[0]}'
                elif domain:
                    tag = f'zero this match ({domain[0]})'
                else:
                    tag = 'zero this match'
                summary.append((path, index, label, tag))
            if share > 0 and not show_all:
                continue
            print(f'  [{index:2}] {share:5.1f}%  {label[:95]:95}  {tag}')
        print()
    reader_new = 'has_attack' in (first['entities'] or [{}])[0]
    import datetime
    when = datetime.datetime.strptime(directory.name, '%Y%m%dT%H%M%SZ').replace(
        tzinfo=datetime.timezone.utc).astimezone()
    print(f'=== MATCH: {directory.name} (recorded {when:%b %d %H:%M}), {turns} decision turns ===')
    if not reader_new:
        print('!!! This recording predates the attack/charge/deploy reader: those slots show 0%')
        print('!!! because they were not recorded, not because they are missing now.')
        print('!!! Play a match with the current consoles, then run this again.')
    missing = [row for row in summary if row[3].startswith('MISSING')]
    print(f'=== {len(missing)} slots empty because we do not supply the data ===')
    by_domain = collections.Counter(row[3].split('supply ')[1] for row in missing)
    for domain, count in by_domain.most_common():
        print(f'  {count:3}  {domain}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
