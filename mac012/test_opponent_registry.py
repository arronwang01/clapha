"""Offline check that every opponent play reaches FirstLight's tracker.

Scenarios, each a synthetic match driven the way the console drives it (plays come from
viewer.executed_plays-shaped dicts, registered with FirstLightRunner.register_plays before the
observation is built, and filtered through runner.tracked_decks()):

  unknown   opponent deck not published (Training Camp / other console not running)
  stale     a published deck that is not the one being played (an old publication)
  exact     the published deck is right

The opponent plays eight distinct cards, one of them as an evolution form id (normalised by
viewer.card_identity), then a ninth distinct card, which must be refused *loudly* while the
policy keeps deciding. An evolved Skeleton (form 13000010) stands on our side of the board and
must reach the model as a unit. Checks: every one of the eight cards registered and recorded by
the tracker, the tracker's opponent elixir estimate falls when they spend, 0 decide errors.

    FIRSTLIGHT_ROOT=... PYTHONPATH=<cr-native-sandbox> python3 mac012/test_opponent_registry.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import firstlight_bot as FLB  # noqa: E402
import firstlight_obs as FLO  # noqa: E402
import viewer as V  # noqa: E402

HOG = [26000021, 26000014, 27000000, 28000000, 28000011, 26000010, 26000038, 26000030]
FORMS = [0, 2, 1, 0, 0, 1, 0, 0]
# The opponent's real deck: Knight, Archers, Goblins, Giant, Minions, Barbarians (played
# evolved, arriving as form id 13000008), Arrows, Zap. Valkyrie comes ninth.
THEIRS = [26000000, 26000001, 26000002, 26000003, 26000005, 26000008, 28000001, 28000008]
NINTH = 26000011
STALE = [26000004, 26000006, 26000007, 26000009, 26000012, 26000013, 28000003, 28000004]
TOWERS = [{'address': f'0xf00{i}', 'category': 1, 'kind': 0, 'side': o, 'x': x, 'y': y,
           'card_id': -1, 'level': 14, 'hp': 3000, 'max_hp': 3000, 'behavior_state_raw': 0}
          for i, (x, y, o, _k) in enumerate(FLO.TOWERS)]


def run(model: str, side: int, scenario: str) -> dict:
    rng = random.Random(5 + side)
    runner = FLB.FirstLightRunner(model)
    opponent = 1 - side
    plays: list[dict] = []
    elixir = {0: 5.0, 1: 5.0}
    stats = {'errors': [], 'refused': [], 'low_after_spend': []}

    def frame(tick):
        ents = [dict(t) for t in TOWERS]
        # an evolved Skeleton of ours, standing in our half
        ents.append({'address': '0x5e1', 'category': 20, 'kind': 1, 'side': side,
                     'x': 9000, 'y': 9000 if side == 0 else 23000, 'card_id': 13000010,
                     'level': 11, 'hp': 30, 'max_hp': 32, 'behavior_state_raw': 3})
        players = [{'side': s, 'elixir_raw': int(elixir[s] * 10000), 'deck_card_ids': HOG,
                    'deck_form_flags': FORMS,
                    'hand_deck_indices': [0, 1, 2, 3] if s == side else [-1] * 4,
                    'cycle_deck_indices': [4, 5, 6, 7] if s == side else [],
                    'next_deck_index': 4 if s == side else -1} for s in (0, 1)]
        return {'game_tick': tick, 'battle_active': True, 'players': players,
                'entities': ents, 'chain': {'battle': 1}}

    health = {'local_side': side}
    published = {'unknown': None, 'stale': STALE, 'exact': THEIRS}[scenario]
    observation, battle = FLO.build(frame(0), health, '1')
    runner.start_battle(HOG, published, side, observation, {0: 5.0, 1: 5.0},
                        our_forms=FORMS, opponent_forms=None)
    tracker = runner.session.tensorizer.tracker
    schedule = {150 + 120 * i: card for i, card in enumerate(THEIRS)}
    schedule[150 + 120 * len(THEIRS)] = NINTH
    seq = 0
    for tick in range(0, 1500, 5):
        for s in (0, 1):
            elixir[s] = min(10.0, elixir[s] + 5 / 56)
        if tick in schedule:
            card = schedule[tick]
            raw = 13000008 if card == 26000008 else card        # evolved Barbarians
            base, form, kind = V.card_identity(raw)
            seq += 1
            plays.append({'tick': tick, 'side': opponent, 'card_id': base, 'raw_card_id': raw,
                          'form_code': form, 'kind': kind, 'x': 9000, 'y': 20000,
                          'seq': seq, 'issue_tick': tick - 21})
            elixir[opponent] = max(0.0, elixir[opponent] - FLO.CARDS[base]['elixir'])
        refused = runner.register_plays(plays, {})
        stats['refused'] += refused
        observation, battle = FLO.build(frame(tick), health, '1', battle=battle, plays=plays,
                                        decks=runner.tracked_decks(),
                                        hand_forms=runner.hand_forms(HOG, FORMS))
        if tick < runner.first_decision_tick:
            runner.observe(observation)
            continue
        try:
            runner.decide(observation)
        except Exception as error:  # noqa: BLE001
            stats['errors'].append(f'{type(error).__name__}: {error}')
        if tick - 5 in schedule or tick == 150:
            stats['low_after_spend'].append(round(tracker.elixir_interval(opponent)[1], 2))
    stats['recorded'] = list(tracker.revealed_cards(opponent))
    stats['seen'] = list(runner.opponent_seen)
    stats['skeleton_shown'] = any(e.card_id == 13000010 for e in observation.entities)
    stats['unresolved'] = dict(battle.unresolved)
    return stats


def main(models) -> int:
    failed = False
    for model in models:
        for scenario in ('unknown', 'stale', 'exact'):
            for side in (0, 1):
                s = run(model, side, scenario)
                missing = [c for c in THEIRS if c not in s['recorded']]
                ninth_refused = any(card == NINTH for _o, card, _r in s['refused'])
                bad = (s['errors'] or missing or not ninth_refused or NINTH in s['recorded']
                       or not s['skeleton_shown'] or len(s['refused']) != 1)
                failed |= bool(bad)
                print(f'{model:10} {scenario:7} side {side}: recorded {len(s["recorded"])}/8, '
                      f'missing {missing}, refused {[(c, r[:28]) for _o, c, r in s["refused"]]}, '
                      f'evo skeleton shown {s["skeleton_shown"]}, '
                      f'opponent elixir ceiling {s["low_after_spend"][:5]}..., '
                      f'errors {len(s["errors"])}'
                      + (f' first: {s["errors"][0][:100]}' if s['errors'] else ''))
    print('FAIL' if failed else 'OK')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:] or ['fl:hog2']))
