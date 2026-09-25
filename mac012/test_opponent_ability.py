"""Offline check that an opponent's hero ability is charged to their elixir.

The opponent (deck unknown, learned) plays Musketeer; its controller shows Hero Musketeer; the
queue reports an ability activation (card id 65535). The runner must join it to Hero Musketeer,
the tracker must take FirstLight's exact ability cost off the opponent's elixir ceiling, and
the policy must keep deciding. A second run gives the opponent two heroes (Musketeer, Ice
Golem) with only Musketeer's charges spent: the activation must go to Musketeer.

    FIRSTLIGHT_ROOT=... PYTHONPATH=<cr-native-sandbox> python3 mac012/test_opponent_ability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import firstlight_bot as FLB  # noqa: E402
import firstlight_obs as FLO  # noqa: E402

HOG = [26000021, 26000014, 27000000, 28000000, 28000011, 26000010, 26000038, 26000030]
FORMS = [0, 2, 1, 0, 0, 1, 0, 0]
MUSKETEER, ICE_GOLEM = 26000014, 26000038
HERO_MUSK_CHAR, HERO_GOLEM_CHAR = 130283371, 2979504115
TOWERS = [{'address': f'0xf00{i}', 'category': 1, 'kind': 0, 'side': o, 'x': x, 'y': y,
           'card_id': -1, 'level': 14, 'hp': 3000, 'max_hp': 3000, 'behavior_state_raw': 0}
          for i, (x, y, o, _k) in enumerate(FLO.TOWERS)]


def run(side: int, two_heroes: bool) -> dict:
    runner = FLB.FirstLightRunner('fl:hog2')
    opponent = 1 - side
    plays, notes, errors, ceilings = [], [], [], []

    def frame(tick, spent):
        abilities = [{'controller_slot': 1, 'charges': 0 if spent else 1, 'button': 6 if spent else 2,
                      'cooldown_ms': 0, 'configured_ms': 0, 'character_id': HERO_MUSK_CHAR}]
        if two_heroes:
            abilities.append({'controller_slot': 2, 'charges': 1, 'button': 2, 'cooldown_ms': 0,
                              'configured_ms': 0, 'character_id': HERO_GOLEM_CHAR})
        players = [{'side': s, 'elixir_raw': 70000, 'deck_card_ids': HOG, 'deck_form_flags': FORMS,
                    'hand_deck_indices': [0, 1, 2, 3] if s == side else [-1] * 4,
                    'cycle_deck_indices': [4, 5, 6, 7] if s == side else [],
                    'next_deck_index': 4 if s == side else -1,
                    'abilities': abilities if s == opponent else []} for s in (0, 1)]
        return {'game_tick': tick, 'battle_active': True, 'players': players,
                'entities': [dict(t) for t in TOWERS], 'chain': {'battle': 1}}

    health = {'local_side': side}
    observation, battle = FLO.build(frame(0, False), health, '1')
    runner.start_battle(HOG, None, side, observation, {0: 5.0, 1: 5.0}, our_forms=FORMS)
    tracker = runner.session.tensorizer.tracker
    plays.append({'tick': 200, 'side': opponent, 'card_id': MUSKETEER, 'raw_card_id': 203000014,
                  'form_code': 2, 'kind': 'card', 'x': 9000, 'y': 20000, 'seq': 1, 'issue_tick': 179})
    if two_heroes:
        plays.append({'tick': 260, 'side': opponent, 'card_id': ICE_GOLEM,
                      'raw_card_id': 203000038, 'form_code': 2, 'kind': 'card', 'x': 9000,
                      'y': 20000, 'seq': 2, 'issue_tick': 239})
    plays.append({'tick': 400, 'side': opponent, 'card_id': 65535, 'raw_card_id': 65535,
                  'form_code': 0, 'kind': 'ability', 'x': 0, 'y': 0, 'seq': 3, 'issue_tick': 379})
    for tick in range(0, 600, 5):
        visible = [p for p in plays if p['tick'] <= tick]
        runner.register_plays(visible, {})
        current = frame(tick, spent=tick >= 400)
        notes += runner.attribute_opponent_abilities(visible, current['players'][opponent])
        observation, battle = FLO.build(current, health, '1', battle=battle, plays=visible,
                                        decks=runner.tracked_decks(),
                                        hand_forms=runner.hand_forms(HOG, FORMS))
        if tick < runner.first_decision_tick:
            runner.observe(observation)
        else:
            try:
                runner.decide(observation)
            except Exception as error:  # noqa: BLE001
                errors.append(f'{type(error).__name__}: {error}')
        if tick in (395, 405):
            ceilings.append(tracker.elixir_interval(opponent)[1])
    return {'notes': notes, 'errors': errors, 'drop': ceilings[0] - ceilings[1],
            'cost': FLO.ability_by_card()[MUSKETEER][1].elixir_cost}


def main() -> int:
    failed = False
    for two in (False, True):
        for side in (0, 1):
            r = run(side, two)
            # the ceiling also regenerates over the 10 ticks between readings (~0.2 elixir)
            ok = not r['errors'] and r['drop'] >= r['cost'] - 0.4 and any(
                'Musketeer' in n and 'charged to them' in n for n in r['notes'])
            failed |= not ok
            print(f'two heroes {two!s:5} side {side}: ceiling fell {r["drop"]:.2f} '
                  f'(ability cost {r["cost"]:g}), errors {len(r["errors"])}, notes {r["notes"]}'
                  + (f' first error: {r["errors"][0][:120]}' if r['errors'] else ''))
    print('FAIL' if failed else 'OK')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
