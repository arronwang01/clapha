"""Every card FirstLight's catalog knows, through the whole live pipeline, both sides.

Instead of finding problems one card at a time in real matches, this forces each card to be
the only legal play and runs real decisions until the policy plays it (or gives up), checking:

  1. the placement mask builds (their card_placement_mask) and is non-empty
  2. the observation builds with that card's units on the board (archetype resolution)
  3. decide() runs without raising while that card is the only legal play
  4. when the policy does play it, the tile it chose is legal under the mask

Run:  python3 mac012/test_all_cards.py            (all cards, fl:il)
      python3 mac012/test_all_cards.py fl:hog2    (another checkpoint)
"""
import collections
import dataclasses
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac012 import firstlight_obs as FLO  # noqa: E402
from mac012 import firstlight_bot as FLB  # noqa: E402
from native_runner.training.v4.factory import production_semantic_bundle  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'fl:il'
TURNS = 40
bundle = production_semantic_bundle()
SPECS = bundle.card_specs
CARDS = sorted(FLO.known_cards())
FILLER = [c for c in CARDS if SPECS[c].kind.value == 'troop' and SPECS[c].elixir_cost][:12]
TOW = [{'address': f'0xt{i}', 'category': 1, 'kind': 0, 'side': o, 'x': x, 'y': y,
        'card_id': -1, 'level': 14, 'hp': 2600 if k != 'king' else 4800,
        'max_hp': 2600 if k != 'king' else 4800, 'behavior_state_raw': 0}
       for i, (x, y, o, k) in enumerate(FLO.TOWERS)]


def deck_for(card):
    rest = [c for c in FILLER if c != card][:7]
    return [card] + rest


def frame(tick, side, deck, with_units):
    ents = [dict(t) for t in TOW]
    if with_units:
        # the card's own units on both halves, so archetype resolution is exercised too
        for n, owner in enumerate((0, 1)):
            ents.append({'address': f'0xu{n}', 'category': 20, 'kind': 1, 'side': owner,
                         'x': 6000 + n * 6000, 'y': 11000 if owner == 0 else 21000,
                         'card_id': deck[0], 'level': 11, 'hp': 900, 'max_hp': 1000,
                         'behavior_state_raw': 3})
    ps = [{'side': s, 'elixir_raw': 100000, 'deck_card_ids': deck, 'deck_form_flags': [0] * 8,
           'hand_deck_indices': [0, 1, 2, 3] if s == side else [-1] * 4,
           'cycle_deck_indices': [4, 5, 6, 7] if s == side else [],
           'next_deck_index': 4 if s == side else -1} for s in (0, 1)]
    return {'game_tick': tick, 'battle_active': True, 'players': ps,
            'entities': ents, 'chain': {'battle': 1}}


def only_slot_zero(observation):
    """Leave slot 0 (the card under test) as the only legal play."""
    mask = observation.action_mask
    return dataclasses.replace(observation, action_mask=dataclasses.replace(
        mask, hand_slots=(mask.hand_slots[0], False, False, False)))


def check(card, side, runner):
    deck = deck_for(card)
    obs, battle = FLO.build(frame(0, side, deck, False), {'local_side': side}, '1')
    entry = obs.action_mask.placement_masks.get('0')
    if not entry:
        return 'no placement entry', 0
    if not any(any(r) for r in entry['row_major']):
        return f'empty mask ({entry.get("accuracy")}: {list(entry.get("reasons") or [])[:2]})', 0
    runner.start_battle(deck, deck, side, obs, {0: 10.0, 1: 10.0}, our_forms=[0] * 8)
    for tick in range(0, 90, 5):
        o, battle = FLO.build(frame(tick, side, deck, False), {'local_side': side}, '1',
                              battle=battle)
        runner.observe(o)
    for step in range(TURNS):
        tick = 90 + step * 5
        o, battle = FLO.build(frame(tick, side, deck, True), {'local_side': side}, '1',
                              battle=battle)
        o = only_slot_zero(o)
        moves = runner.decide(o)
        for kind, slot, played, grid, _ in moves:
            if played == card and grid is not None:
                col, row = int(grid[0]), int(grid[1])
                if not entry['row_major'][row][col]:
                    return f'ILLEGAL tile col {col} row {row}', step + 1
                return 'ok', step + 1
    return 'never chosen', TURNS


results = collections.defaultdict(list)
problems = []
runner = FLB.FirstLightRunner(MODEL)
for index, card in enumerate(CARDS):
    for side in (0, 1):
        try:
            verdict, turns = check(card, side, runner)
        except Exception as error:  # noqa: BLE001
            verdict = f'ERROR {type(error).__name__}: {str(error)[:100]}'
            turns = 0
        runner.end_battle()
        results[verdict.split(' (')[0].split(':')[0]].append((card, side))
        if verdict not in ('ok', 'never chosen'):
            problems.append((SPECS[card].name, card, SPECS[card].kind.value, side, verdict))
    print(f'  {index + 1}/{len(CARDS)} {SPECS[card].name}', flush=True)

print(f'\n=== {MODEL}: {len(CARDS)} cards x 2 sides ===')
for verdict, rows in sorted(results.items(), key=lambda kv: -len(kv[1])):
    print(f'  {len(rows):4}  {verdict}')
print('\nproblems:')
for row in problems:
    print('  ', row)
if not problems:
    print('   none')
