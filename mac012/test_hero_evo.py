"""Hero abilities and evolution progress, read from memory, through to the model and back.

Frames are shaped like live_sampler_tbi's output (players[].abilities / evo_progress) with
values taken from the device run in build/runtime_probe.jsonl: Hero Musketeer character
130283371, Hero Ice Golem 2979504115; button 1 absent / 2 ready / 6 used / 9 no elixir.

Checks:
  1. each bound controller becomes one ability state with the right ability, cost and phase,
     bound to the live hero unit;
  2. the mask offers an ability exactly when FirstLight's rule allows it (ready, charges,
     affordable, hero on board, not already in flight);
  3. with only the ability legal, the policy can choose it and decoding returns the hero unit;
  4. evolution: progress vector -> ready / cycle_remaining, and the hand form follows memory.

    python3 mac012/test_hero_evo.py [MODEL ...]
"""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac012 import firstlight_obs as FLO  # noqa: E402
from mac012 import firstlight_bot as FLB  # noqa: E402

MODELS = sys.argv[1:] or ['fl:hog2', 'fl:il']
DECK = [26000010, 26000014, 27000000, 28000011, 26000038, 26000030, 28000000, 26000021]
# The user's deck in the device run: Hero Musketeer, Hero Ice Golem, Evo Skeletons. FirstLight's
# V4 contract allows <= 2 heroes, <= 2 evolutions, <= 3 special cards -- as does the game.
FLAGS = [1, 2, 0, 0, 2, 0, 0, 0]
MUSKETEER, ICE_GOLEM = 130283371, 2979504115
TOW = [{'address': f'0xb4000000000a{i:04x}', 'category': 1, 'kind': 0, 'side': o, 'x': x, 'y': y,
        'card_id': -1, 'level': 14, 'hp': 2600, 'max_hp': 2600, 'behavior_state_raw': 0,
        'deploy_remaining': 0, 'has_attack': 1, 'target': '0x0', 'atk_stage': 0,
        'atk_timeline': 0, 'atk_load': 0, 'has_move': 0, 'charge': -1}
       for i, (x, y, o, k) in enumerate(FLO.TOWERS)]
failures = []


def check(condition, label):
    print(f"   {'ok ' if condition else 'FAIL'} {label}")
    if not condition:
        failures.append(label)


def frame(tick, side, *, elixir, musketeer_button, golem_button, hero_on_board, evo,
          threat=False):
    ents = [dict(t) for t in TOW]
    if hero_on_board:
        ents.append({'address': '0xb400000000009999', 'category': 20, 'kind': 1, 'side': side,
                     'x': 9000, 'y': 12000 if side == 0 else 20000, 'card_id': 203000014,
                     'level': 11, 'hp': 700, 'max_hp': 720, 'behavior_state_raw': 3,
                     'deploy_remaining': 0, 'has_attack': 1, 'target': '0x0', 'atk_stage': 0,
                     'atk_timeline': 0, 'atk_load': 0, 'has_move': 1, 'charge': -1})
    if threat:
        # enemy troops closing on our hero: the situation the ability exists for
        for n in range(4):
            ents.append({'address': f'0xb40000000000e{n:03x}', 'category': 20, 'kind': 1,
                         'side': 1 - side, 'x': 7800 + n * 700,
                         'y': (14500 - n * 300) if side == 0 else (17500 + n * 300),
                         'card_id': (26000021, 26000055, 26000010, 26000010)[n], 'level': 11,
                         'hp': 1200, 'max_hp': 1400, 'behavior_state_raw': 3,
                         'deploy_remaining': 0, 'has_attack': 1, 'target': '0xb400000000009999',
                         'atk_stage': 0, 'atk_timeline': 300, 'atk_load': 0, 'has_move': 1,
                         'charge': -1})
    players = []
    for s in (0, 1):
        mine = s == side
        players.append({
            'side': s, 'elixir_raw': int((elixir if mine else 5.0) * 10000),
            'deck_card_ids': DECK if mine else [], 'deck_form_flags': FLAGS if mine else [],
            'hand_deck_indices': [0, 1, 2, 3] if mine else [-1] * 4,
            'cycle_deck_indices': [4, 5, 6, 7] if mine else [],
            'next_deck_index': 4 if mine else -1,
            'evo_progress': evo if mine else [],
            'abilities': ([
                {'controller_slot': 1, 'charges': 1 if golem_button == 2 else -1,
                 'button': golem_button, 'cooldown_ms': 0, 'configured_ms': 0,
                 'character_id': ICE_GOLEM},
                {'controller_slot': 2, 'charges': 1 if musketeer_button in (2, 9) else
                 (0 if musketeer_button == 6 else -1), 'button': musketeer_button,
                 'cooldown_ms': 0, 'configured_ms': 0, 'character_id': MUSKETEER}] if mine else [])})
    return {'game_tick': tick, 'battle_active': True, 'players': players, 'entities': ents,
            'chain': {'battle': 1}}


def mask_for(side, **kw):
    obs, _ = FLO.build(frame(200, side, **kw), {'local_side': side}, '1')
    return obs



def ability_candidates(runner, observation):
    """How many ability candidates the tensorizer hands the policy for this observation."""
    from native_runner.training.v4.tensorizer import CANDIDATE_ABILITY
    batch = runner.session.tensorizer.tensorize(observation, validate=False)
    mask = batch.candidates.mask[0]
    return int((mask & (batch.candidates.variant[0] == CANDIDATE_ABILITY)).sum())


def main() -> int:
    decoded_ok = []
    for side in (0, 1):
        print(f'\n== side {side}: ability states and mask ==')
        obs = mask_for(side, elixir=6.0, musketeer_button=2, golem_button=1, hero_on_board=True,
                       evo=[0] * 8)
        me = next(p for p in obs.players if p.owner == side)
        by_id = {a.ability_id: a for a in me.ability_runtime_states}
        musk = by_id.get('Musketeer_hero_Ability')
        golem = by_id.get('IceGolemiteHero_Ability')
        check(musk is not None and golem is not None, 'both controllers joined to their ability')
        hero_unit = next((e.entity_id for e in obs.entities if e.card_id == 203000014), None)
        check(musk is not None and musk.source_entity == hero_unit, 'Musketeer ability bound to the hero unit')
        check(musk is not None and musk.phase.value == 'ready' and musk.available is True
              and musk.elixir_cost == 3.0, 'Musketeer: ready, available, cost 3')
        check(golem is not None and golem.phase.value == 'unavailable' and golem.source_entity is None,
              'Ice Golem: absent hero -> unavailable, no source')
        check(obs.action_mask.ability_sources == (hero_unit,), 'mask offers exactly the ready hero')
        cases = [
            ('not enough elixir', dict(elixir=2.0, musketeer_button=9, golem_button=1,
                                       hero_on_board=True, evo=[0] * 8), ()),
            ('already used', dict(elixir=8.0, musketeer_button=6, golem_button=1,
                                  hero_on_board=True, evo=[0] * 8), ()),
            ('hero not on board', dict(elixir=8.0, musketeer_button=1, golem_button=1,
                                       hero_on_board=False, evo=[0] * 8), ()),
        ]
        for label, kw, want in cases:
            check(mask_for(side, **kw).action_mask.ability_sources == want, f'not offered when {label}')
        in_flight, _ = FLO.build(frame(200, side, elixir=6.0, musketeer_button=2, golem_button=1,
                                       hero_on_board=True, evo=[0] * 8), {'local_side': side}, '1',
                                 pending_ability_sources=(hero_unit,))
        check(in_flight.action_mask.ability_sources == (), 'not offered while its tap is in flight')

        print(f'== side {side}: evolution from memory ==')
        obs = mask_for(side, elixir=6.0, musketeer_button=1, golem_button=1, hero_on_board=False,
                       evo=[2, 0, 1, 0, 0, 0, 0, 0])
        me = next(p for p in obs.players if p.owner == side)
        evo = {e.card_id: e for e in me.evolution_runtime_states}
        check(set(evo) == {26000010}, 'states for exactly the evolution-enabled slot (Skeletons)')
        check(evo[26000010].ready is True and evo[26000010].cycle_remaining == 0,
              'Skeletons progress 2/2 -> ready')
        obs1 = mask_for(side, elixir=6.0, musketeer_button=1, golem_button=1, hero_on_board=False,
                        evo=[1, 0, 0, 0, 0, 0, 0, 0])
        one = next(p for p in obs1.players if p.owner == side).evolution_runtime_states[0]
        check(one.ready is False and one.cycle_remaining == 1 and one.phase.value == 'cycling',
              'Skeletons progress 1/2 -> cycling, one cycle left')
        runner = FLB.FirstLightRunner(MODELS[0])
        forms = runner.hand_forms(DECK, FLAGS, [2, 0, 1, 0, 0, 0, 0, 0])
        check(forms.get(26000010) == 1 and 27000000 not in forms and forms.get(26000014) == 2
              and forms.get(26000038) == 2, 'hand forms: evo Skeletons 1, heroes 2, Cannon plain')
        early = runner.hand_forms(DECK, FLAGS, [1, 0, 0, 0, 0, 0, 0, 0])
        check(early.get(26000010) == 0, 'hand form: Skeletons not yet evolved at progress 1')

    for model in MODELS:
        for side in (0, 1):
            print(f'\n== {model} side {side}: the policy chooses the ability ==')
            runner = FLB.FirstLightRunner(model)
            kw = dict(elixir=8.0, musketeer_button=2, golem_button=1, hero_on_board=True,
                      evo=[0] * 8, threat=True)
            first = frame(0, side, **kw)
            obs, battle = FLO.build(first, {'local_side': side}, '1')
            runner.start_battle(DECK, DECK, side, obs, {0: 6.0, 1: 5.0}, our_forms=FLAGS)
            for tick in range(0, 90, 5):
                o, battle = FLO.build(frame(tick, side, **kw), {'local_side': side}, '1',
                                      battle=battle, hand_forms=runner.hand_forms(DECK, FLAGS, [0] * 8))
                runner.observe(o)
            chosen = errors = offered = 0
            source = None
            for step in range(60):
                o, battle = FLO.build(frame(90 + step * 5, side, **kw), {'local_side': side}, '1',
                                      battle=battle, hand_forms=runner.hand_forms(DECK, FLAGS, [0] * 8))
                # leave the ability as the only legal action, so the policy's choice is observable
                mask = o.action_mask
                o = dataclasses.replace(o, action_mask=dataclasses.replace(
                    mask, hand_slots=(False,) * 4,
                    kinds={**dict(mask.kinds), 'play_card': False}))
                if step == 0:
                    offered = ability_candidates(runner, o)
                try:
                    for move in runner.decide(o):
                        if str(getattr(move[0], 'value', move[0])) == 'activate_ability':
                            chosen += 1
                            source = move.source_entity
                except Exception as error:  # noqa: BLE001
                    errors += 1
                    print('   error:', type(error).__name__, str(error)[:120])
                if chosen:
                    break
            hero_unit = next((e.entity_id for e in o.entities if e.card_id == 203000014), None)
            check(errors == 0, f'no decide errors ({errors})')
            check(offered > 0, 'the ability is a candidate the policy sees')
            if chosen:
                check(source == hero_unit, f'chosen on turn {step + 1}; decoded source = the hero unit')
                decoded_ok.append(f'{model} side {side}')
            else:
                print('   ---  not chosen in 60 turns (the policy\'s call, not an error)')

    check(bool(decoded_ok), f'some policy chose the ability and it decoded: {decoded_ok}')
    print('\nFAILED:' if failures else '\nOK', *failures, sep='\n  ' if failures else '')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
