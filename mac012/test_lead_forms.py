"""Offline check of the latency lead, in-flight hand cycling and evolution/hero forms.

A synthetic three-minute match with the Hog 2.6 deck the specialists trained on (Evo Cannon,
Evo Skeletons, Hero Musketeer), driven the way the console drives it: a tap is issued at the
decision, reaches the queue RTT ticks later, and executes COMMAND_AGE ticks after that; the
reader's hand and elixir only change on execution. Checks, per checkpoint and side:

  * no decide() errors with the lead on and off;
  * the hand the policy sees never contains a card that is already in flight;
  * an evolution card is offered evolved exactly when FirstLight's own tracker says so,
    and the hero card is always offered as its hero form;
  * elixir is spent in the right amount (utilisation, like the NOTES table).

    FIRSTLIGHT_ROOT=... python3 mac012/test_lead_forms.py [model ...]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import firstlight_bot as FLB  # noqa: E402
import firstlight_obs as FLO  # noqa: E402

HOG = [26000021, 26000014, 27000000, 28000000, 28000011, 26000010, 26000038, 26000030]
FORMS = [0, 2, 1, 0, 0, 1, 0, 0]          # Hero Musketeer, Evo Cannon, Evo Skeletons
COMMAND_AGE, RTT = 21, 3
END_TICK = 3600
TOWERS = [{'address': f'0xf00{i}', 'category': 1, 'kind': 0, 'side': o, 'x': x, 'y': y,
           'card_id': -1, 'level': 14, 'hp': 3000, 'max_hp': 3000, 'behavior_state_raw': 0}
          for i, (x, y, o, _k) in enumerate(FLO.TOWERS)]


def regen(tick: int) -> float:
    return 1.0 / (56 if tick < 2400 else 28)     # 2.8 s per elixir at 1x, 1.4 s at 2x


def run(model: str, side: int, lead: int, seed: int) -> dict:
    rng = random.Random(seed)
    runner = FLB.FirstLightRunner(model)
    costs = {c: float(FLO.CARDS[c]['elixir']) for c in HOG}
    hand, cycle = [0, 1, 2, 3], [4, 5, 6, 7]
    elixir = {0: 5.0, 1: 5.0}
    queue: list[dict] = []        # issued commands, not yet executed
    executed: list[dict] = []     # the viewer's STATE['plays']
    in_flight: list[dict] = []
    units: dict[str, dict] = {}
    seq = 0
    stats = {'errors': [], 'plays': 0, 'spent': 0.0, 'in_flight_in_hand': 0,
             'evolved_offers': 0, 'evolved_plays': 0, 'hero_offers': 0, 'form_mismatch': 0}

    def frame(tick):
        ents = [dict(t) for t in TOWERS]
        for address, unit in units.items():
            ents.append({'address': address, 'category': 20, 'kind': 1, 'side': unit['side'],
                         'x': unit['x'], 'y': unit['y'], 'card_id': unit['card'], 'level': 11,
                         'hp': unit['hp'], 'max_hp': 1600, 'behavior_state_raw': 3})
        players = [{'side': s, 'elixir_raw': int(elixir[s] * 10000), 'deck_card_ids': HOG,
                    'deck_form_flags': FORMS,
                    'hand_deck_indices': list(hand) if s == side else [-1] * 4,
                    'cycle_deck_indices': list(cycle) if s == side else [],
                    'next_deck_index': cycle[0] if s == side else -1} for s in (0, 1)]
        return {'game_tick': tick, 'battle_active': True, 'players': players,
                'entities': ents, 'chain': {'battle': 1}}

    health = {'local_side': side}
    observation, battle = FLO.build(frame(0), health, '1', lead_ticks=lead)
    runner.start_battle(HOG, HOG, side, observation, {0: 5.0, 1: 5.0},
                        our_forms=FORMS, opponent_forms=FORMS)
    tracker = runner.session.tensorizer.tracker
    for tick in range(0, END_TICK, 5):
        # the world advances five ticks
        for t in range(tick - 4 if tick else 0, tick + 1):
            for s in (0, 1):
                elixir[s] = min(10.0, elixir[s] + regen(t))
            for command in [c for c in queue if c['issue_tick'] + COMMAND_AGE == t]:
                queue.remove(command)
                executed.append({'tick': t, 'side': command['side'], 'card_id': command['card_id'],
                                 'kind': 'card', 'x': command['x'], 'y': command['y'],
                                 'seq': command['seq'], 'issue_tick': command['issue_tick']})
                if command['side'] == side:
                    slot = HOG.index(command['card_id'])
                    hand[hand.index(slot)] = cycle.pop(0)
                    cycle.append(slot)
                    elixir[side] -= costs[command['card_id']]
                    in_flight[:] = [f for f in in_flight if f['seq'] != command['seq']]
                if FLO.CARDS[command['card_id']]['type'] == 'troop':
                    units[f'0x{0x1000 + command["seq"]:x}'] = {
                        'side': command['side'], 'card': command['card_id'],
                        'x': command['x'], 'y': command['y'], 'hp': 1200}
            for tap in [f for f in in_flight if f.get('seq') is None and f['tap'] + RTT == t]:
                seq += 1
                tap['seq'] = seq
                queue.append({'issue_tick': t, 'seq': seq, 'side': side, 'card_id': tap['card'],
                              'x': tap['x'], 'y': tap['y']})
        for address in list(units):
            unit = units[address]
            unit['y'] += 150 if unit['side'] == 0 else -150
            unit['hp'] -= rng.randint(0, 60)
            if unit['hp'] <= 0 or not 0 < unit['y'] < 32000:
                del units[address]
        # a scripted opponent: plays a random affordable card now and then
        if rng.random() < 0.06:
            card = rng.choice(HOG)
            if elixir[1 - side] >= costs[card]:
                elixir[1 - side] -= costs[card]
                seq += 1
                queue.append({'issue_tick': tick, 'seq': seq, 'side': 1 - side, 'card_id': card,
                              'x': rng.randint(2000, 16000), 'y': 20000 if side == 0 else 12000})

        plays = list(executed)
        if lead:
            plays += [{'tick': c['issue_tick'] + COMMAND_AGE, 'side': c['side'],
                       'card_id': c['card_id'], 'kind': 'card', 'x': c['x'], 'y': c['y'],
                       'seq': c['seq'], 'issue_tick': c['issue_tick']}
                      for c in queue if c['issue_tick'] + COMMAND_AGE <= tick + lead]
        reserved = sum(costs[f['card']] for f in in_flight)
        forms = runner.hand_forms(HOG, FORMS)
        observation, battle = FLO.build(
            frame(tick), health, '1', battle=battle, reserved=reserved, plays=plays,
            decks={0: tuple(HOG), 1: tuple(HOG)}, lead_ticks=lead,
            in_flight=[HOG.index(f['card']) for f in in_flight], hand_forms=forms)
        me = next(p for p in observation.players if p.owner == side)
        if any(f['card'] in me.hand for f in in_flight) and lead:
            stats['in_flight_in_hand'] += 1
        for position, entry in observation.action_mask.placement_masks.items():
            card = int(entry['visible_card_id'])
            if card in (27000000, 26000010) and entry['form_code'] == 1:
                stats['evolved_offers'] += 1
            if card == 26000014 and entry['form_code'] == 2:
                stats['hero_offers'] += 1
            if card in (27000000, 26000010):
                ready = tracker.card_state(side, card).evolution_ready
                if entry['form_code'] != int(ready):
                    stats['form_mismatch'] += 1
        if observation.tick < runner.first_decision_tick:
            runner.observe(observation)
            continue
        try:
            moves = runner.decide(observation)
        except Exception as error:  # noqa: BLE001
            stats['errors'].append(f'{type(error).__name__}: {error}')
            continue
        available = elixir[side] - reserved
        for kind, _slot, card, grid, _offset in moves:
            if str(getattr(kind, 'value', kind)) != 'play_card' or grid is None:
                continue
            if any(f['card'] == card for f in in_flight) or HOG.index(card) not in hand:
                continue
            if available < costs[card]:
                continue
            available -= costs[card]
            if card in (27000000, 26000010) and forms.get(card) == 1:
                stats['evolved_plays'] += 1
            in_flight.append({'card': card, 'tap': tick, 'x': 500 + 1000 * int(grid[0]),
                              'y': 500 + 1000 * int(grid[1])})
            stats['plays'] += 1
            stats['spent'] += costs[card]
    return stats


def main(models) -> int:
    failed = False
    for model in models:
        for side in (0, 1):
            for lead in (0, 24):
                s = run(model, side, lead, 11 + side)
                bad = s['errors'] or s['in_flight_in_hand'] or s['form_mismatch'] \
                    or not s['hero_offers']
                failed |= bool(bad)
                print(f'{model:12} side {side} lead {lead:2}: {s["plays"]:3} plays, '
                      f'{s["spent"]:5.1f} elixir ({s["spent"] / 91 * 100:3.0f}%), '
                      f'evolved offers {s["evolved_offers"]:3} plays {s["evolved_plays"]:2}, '
                      f'hero offers {s["hero_offers"]:3}, in-flight-in-hand '
                      f'{s["in_flight_in_hand"]}, form mismatch {s["form_mismatch"]}, '
                      f'errors {len(s["errors"])}'
                      + (f'  first: {s["errors"][0][:120]}' if s['errors'] else ''))
    print('FAIL' if failed else 'OK')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:] or ['fl:hog2', 'fl:il']))
