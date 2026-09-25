"""Offline check of the API-deck rule: trusted from the start, discarded on the first card
that contradicts it, after which the deck is learned from play only.

    FIRSTLIGHT_ROOT=... PYTHONPATH=<cr-native-sandbox> python3 mac012/test_api_deck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_opponent_registry as R  # noqa: E402  (reuses its decks and frames)
import firstlight_bot as FLB  # noqa: E402
import firstlight_obs as FLO  # noqa: E402


def run(api_deck, side=0):
    runner = FLB.FirstLightRunner('fl:hog2')
    opponent = 1 - side
    frame = {'game_tick': 0, 'battle_active': True, 'chain': {'battle': 1},
             'entities': [dict(t) for t in R.TOWERS],
             'players': [{'side': s, 'elixir_raw': 50000, 'deck_card_ids': R.HOG,
                          'deck_form_flags': R.FORMS,
                          'hand_deck_indices': [0, 1, 2, 3] if s == side else [-1] * 4,
                          'cycle_deck_indices': [4, 5, 6, 7] if s == side else [],
                          'next_deck_index': 4 if s == side else -1} for s in (0, 1)]}
    obs, battle = FLO.build(frame, {'local_side': side}, '1')
    runner.start_battle(R.HOG, None, side, obs, {0: 5.0, 1: 5.0}, our_forms=R.FORMS)
    note = runner.adopt_api_deck(api_deck)
    tracker = runner.session.tensorizer.tracker
    after_adopt = set(tracker._card_states[opponent])
    plays = []
    for i, card in enumerate(R.THEIRS):
        plays.append({'tick': 150 + 60 * i, 'side': opponent, 'card_id': card, 'kind': 'card',
                      'seq': i, 'issue_tick': 129 + 60 * i})
        runner.register_plays(plays, {})
    return note, after_adopt, runner, set(tracker._card_states[opponent])


note, adopted, runner, final = run(R.THEIRS)
ok1 = adopted == set(R.THEIRS) and runner.api_deck_state == 'trusted' and final == set(R.THEIRS)
print(f'right API deck: {note!r}; tracker had all 8 from the start: {adopted == set(R.THEIRS)}; '
      f'still trusted: {runner.api_deck_state}')
note, adopted, runner, final = run(R.STALE)
ok2 = (adopted == set(R.STALE) and runner.api_deck_state == 'discarded'
       and final == set(R.THEIRS) and any('discarded' in m for m in runner.messages))
print(f'stale API deck: adopted {len(adopted)}, then {runner.api_deck_state}; tracker now '
      f'holds exactly the played cards: {final == set(R.THEIRS)}; message: {runner.messages[:1]}')
print('OK' if ok1 and ok2 else 'FAIL')
raise SystemExit(0 if ok1 and ok2 else 1)
