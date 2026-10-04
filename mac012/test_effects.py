"""Projectiles, tower shots and spawned units, resolved by each object's own data id.

Checks, both sides: all six towers survive a tower shot sitting on the tower's tile; a spell in
flight becomes a `projectile` entity with its FirstLight archetype and velocity; a Battle Ram's
Barbarian and a Tombstone's Skeleton get their own unit archetype (not the spawner's); the model
decides on such boards without errors and the tensorizer marks projectiles in flight.

    python3 mac012/test_effects.py [MODEL]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac012 import firstlight_obs as FLO  # noqa: E402
from mac012 import firstlight_bot as FLB  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'fl:hog2'
DECK = [26000021, 27000000, 26000014, 26000038, 26000030, 26000010, 28000011, 28000000]
failures = []


def check(ok, label):
    print(f"   {'ok ' if ok else 'FAIL'} {label}")
    if not ok:
        failures.append(label)


def obj(addr, card, data, side, x, y, kind=15, **extra):
    unit = kind != 0
    row = {'address': addr, 'category': 5000000 if unit else 4000000, 'kind': kind, 'side': side,
           'x': x, 'y': y, 'card_id': card, 'data_id': data, 'level': 11 if unit else 0,
           'hp': 600 if unit else -1, 'max_hp': 700 if unit else -1, 'behavior_state_raw': 3,
           'deploy_remaining': 0, 'has_attack': int(unit), 'target': '0x0', 'atk_stage': 0,
           'atk_timeline': 0, 'atk_load': 0, 'has_move': int(unit), 'charge': -1}
    row.update(extra)
    return row


TOWERS = [obj(f'0xb4000000000a{i:04x}', -1, 35000000 if k == 'king' else 35000001, o, x, y,
              kind=12 if k == 'king' else 13, hp=2600, max_hp=2600)
          for i, (x, y, o, k) in enumerate(FLO.TOWERS)]


def frame(tick, side, fireball_y):
    enemy = 1 - side
    shot_tower = next(t for t in TOWERS if t['side'] == enemy and t['kind'] == 13)
    entities = [dict(t) for t in TOWERS] + [
        obj('0xb400000000001001', 28000000, 10000000, enemy, 9000, fireball_y, kind=0),
        obj('0xb400000000001002', -1, 10000003, enemy, shot_tower['x'], shot_tower['y'], kind=0),
        obj('0xb400000000001003', 26000036, 34000009, enemy, 5000, 18000),
        obj('0xb400000000001004', 27000009, 34000008, enemy, 12000, 20000),
        obj('0xb400000000001005', 26000021, 34000017, side, 14500, 12000)]
    players = [{'side': s, 'elixir_raw': 70000, 'deck_card_ids': DECK, 'deck_form_flags': [0] * 8,
                'hand_deck_indices': [0, 1, 2, 3] if s == side else [-1] * 4,
                'cycle_deck_indices': [4, 5, 6, 7] if s == side else [],
                'next_deck_index': 4 if s == side else -1, 'evo_progress': [0] * 8,
                'abilities': []} for s in (0, 1)]
    return {'game_tick': tick, 'battle_active': True, 'players': players, 'entities': entities,
            'chain': {'battle': 1}}


def main() -> int:
    for side in (0, 1):
        print(f'== side {side}')
        obs, battle = FLO.build(frame(100, side, 20000), {'local_side': side}, '1')
        obs, battle = FLO.build(frame(105, side, 19000), {'local_side': side}, '1', battle=battle)
        check(len(obs.towers) == 6 and all(t.active and t.hitpoints == 2600 for t in obs.towers),
              'six towers intact with a tower shot on a tower tile')
        by_archetype = {e.native_data_global_id: e for e in obs.entities}
        fireball = by_archetype.get(10000000)
        check(fireball is not None and fireball.entity_kind == 'projectile'
              and fireball.velocity is not None and fireball.velocity[1] != 0,
              'Fireball in flight: projectile, moving')
        shot = by_archetype.get(10000003)
        check(shot is not None and shot.entity_kind == 'projectile' and shot.card_id is None,
              'tower shot: projectile, no card')
        check(by_archetype.get(34000009) is not None and by_archetype[34000009].entity_kind == 'troop',
              "Battle Ram's Barbarian resolved as Barbarian")
        check(by_archetype.get(34000008) is not None and by_archetype[34000008].entity_kind == 'troop',
              "Tombstone's Skeleton resolved as Skeleton")
        runner = FLB.FirstLightRunner(MODEL)
        obs, battle = FLO.build(frame(0, side, 20000), {'local_side': side}, '1')
        runner.start_battle(DECK, DECK, side, obs, {0: 7.0, 1: 7.0})
        errors = 0
        for tick in range(5, 160, 5):
            obs, battle = FLO.build(frame(tick, side, 20000 - max(0, tick - 90) * 150),
                                    {'local_side': side}, '1', battle=battle)
            try:
                runner.observe(obs) if tick < runner.first_decision_tick else runner.decide(obs)
            except Exception as error:  # noqa: BLE001
                errors += 1
                print('     ', type(error).__name__, str(error)[:120])
        check(errors == 0, f'{MODEL}: decisions on boards with projectiles, {errors} errors')
        batch = runner.session.tensorizer.tensorize(obs, validate=True)
        rows = batch.groups.child_features[0][batch.groups.child_mask[0]]
        check(int((rows[:, 32] != 0).sum()) >= 2, 'tensorizer marks projectiles in flight (slot 32)')
    print('\nFAILED:' if failures else '\nOK', *failures, sep='\n  ' if failures else '')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
