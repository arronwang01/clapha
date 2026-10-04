"""All five checkpoints, both sides, with entities that spawn, move, die and reuse addresses."""
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac012 import firstlight_obs as FLO
from mac012 import firstlight_bot as FLB

C = FLO.CARDS
troops = [c for c, i in C.items() if i.get('type') == 'troop' and i.get('elixir')
          and i.get('standard_1v1') and (i.get('elixir') or 9) <= 4][:5]
bld = [c for c, i in C.items() if i.get('type') == 'building' and i.get('elixir') and i.get('standard_1v1')][:1]
spells = [c for c, i in C.items() if i.get('type') == 'spell' and i.get('elixir') and i.get('standard_1v1')][:2]
deck = troops + bld + spells
TOW = [{'address': f'0xf00{i}', 'category': 1, 'kind': 0, 'side': o, 'x': x, 'y': y,
        'card_id': -1, 'level': 14, 'hp': 2600, 'max_hp': 2600, 'behavior_state_raw': 0}
       for i, (x, y, o, k) in enumerate(FLO.TOWERS)]


def run(model, side, seed):
    rng = random.Random(seed)
    runner = FLB.FirstLightRunner(model)
    # A small pool of heap addresses that units are born into and die out of, so addresses
    # get reused exactly as they do on the heap.
    pool = [f'0x{i:04x}' for i in range(6)]
    alive = {}
    battle = None
    plays, errors, wrong, spell_plays = 0, [], 0, 0
    wrong_tiles = []

    def frame(tick, elixir, dead_lane):
        ents = [dict(t) for t in TOW]
        if dead_lane:
            ents = [t for t in ents if not (t['side'] == 1 - side and t['x'] == 3500)]
        for address, unit in alive.items():
            ents.append({'address': address, 'category': 20, 'kind': 1, 'side': unit['side'],
                         'x': unit['x'], 'y': unit['y'], 'card_id': unit['card'],
                         'level': 11, 'hp': unit['hp'], 'max_hp': 1600,
                         'behavior_state_raw': 3})
        if rng.random() < 0.4:      # a spell effect, which has no archetype
            ents.append({'address': '0xspell', 'category': 30, 'kind': 5, 'side': 1 - side,
                         'x': 6000, 'y': 12000, 'card_id': rng.choice(spells),
                         'level': 11, 'hp': 0, 'max_hp': 0, 'behavior_state_raw': 1})
        ps = [{'side': s, 'elixir_raw': int(elixir * 10000), 'deck_card_ids': deck,
               'hand_deck_indices': [0, 1, 5, 6] if s == side else [-1] * 4,
               'cycle_deck_indices': [2, 3, 4, 7] if s == side else [],
               'next_deck_index': 2 if s == side else -1} for s in (0, 1)]
        return {'game_tick': tick, 'battle_active': True, 'players': ps,
                'entities': ents, 'chain': {'battle': 1}}

    obs, battle = FLO.build(frame(0, 5.0, False), {'local_side': side}, '1')
    runner.start_battle(deck, deck, side, obs, {0: 5.0, 1: 5.0})
    for step in range(1, 121):
        tick = step * 5
        # churn: kill some, spawn some into freed addresses
        for address in list(alive):
            unit = alive[address]
            unit['hp'] -= rng.randint(0, 400)
            unit['y'] += -300 if unit['side'] == 1 else 300
            unit['x'] = max(500, min(17500, unit['x'] + rng.randint(-200, 200)))
            if unit['hp'] <= 0 or not (0 < unit['y'] < 32000):
                del alive[address]
        for address in pool:
            if address not in alive and rng.random() < 0.25:
                owner = rng.choice((0, 1))
                alive[address] = {'side': owner, 'card': rng.choice(troops),
                                  'x': rng.randint(2000, 16000),
                                  'y': 8000 if owner == 0 else 24000,
                                  'hp': rng.randint(600, 1600)}
        obs, battle = FLO.build(frame(tick, min(10.0, 2.0 + step * 0.1), step > 80),
                                {'local_side': side}, '1', battle=battle)
        try:
            move = runner.decide(obs)
        except Exception as error:  # noqa: BLE001
            errors.append(f'{type(error).__name__}: {error}')
            continue
        for kind, slot, card, grid, _offset in move:
            if str(getattr(kind, 'value', kind)) != 'play_card' or grid is None:
                continue
            plays += 1
            col, row = int(grid[0]), int(grid[1])
            if C.get(card, {}).get('type') == 'spell':
                spell_plays += 1
            # Legal means legal in the placement mask the policy was given -- FirstLight's own
            # card_placement_mask, which includes the bridge row and destroyed-lane pockets. A
            # fixed half-board rule (row < 16 / row >= 16) called bridge plays wrong.
            entry = obs.action_mask.placement_masks.get(str(slot))
            if not entry or not entry['row_major'][row][col]:
                wrong += 1
                wrong_tiles.append((C.get(card, {}).get('display_name'), col, row))
    if wrong_tiles:
        print('   illegal:', wrong_tiles[:5])
    return plays, spell_plays, wrong, errors, dict(battle.unresolved)


for model in ('fl:general', 'fl:hog1', 'fl:hog2', 'fl:il', 'fl:active-il'):
    for side in (0, 1):
        plays, spell_plays, wrong, errors, unresolved = run(model, side, 7 + side)
        flag = f'  ERRORS {len(errors)}: {errors[0][:100]}' if errors else ''
        print(f'{model:14} side {side}: {plays:3} plays ({spell_plays} spells) / 120'
              f', illegal tiles {wrong}, skipped {sum(unresolved.values())}{flag}')
