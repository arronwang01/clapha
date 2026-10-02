"""Our model against a FirstLight model at its full strength, played in Null's engine and kept for watching.

FirstLight's side runs as in its own evaluator and console (native_runner/training/v4/evaluate.py,
offline_agent.py): its BattleEnvV1 owns the match and builds its native observation every five ticks,
its policy is sampled at the checkpoint's temperatures, and its cards execute 1-4 ticks after it decides
(FirstLight's sandbox: no delay), at its exact sub-tile points.

Our side is il/duel.py's live path: the reader's frame (il/samples.reader_frame) built from the same
atomic snapshot the environment takes each turn (the rich envelope's units and players -- the lean
snapshot our training reads is that minus histories -- plus the plain one), the extended inputs, the tap
after the measured pipeline overhead with elixir checked at the tap, and the play landing TARGET_DELAY
ticks after its moment: how it trained, and how the console plays live. Turns are FirstLight's five-tick
grid, so a tap due between two turns is checked for elixir at the turn before it; its execute tick is exact.

Needs CR_4k with FirstLight's attested probe and the user's FirstLight_CR -- its console's own setup
(il/watch_nulls.prepare_engine does it; FirstLight's console must be closed). Each game is saved as a
recording in runs/rl/vs-firstlight (played.config and the exact points included): Training games lists it,
shows the board, and plays it in Null's.

    ./py -m il.vs_firstlight [--ours clapha:p3] [--theirs fl:hog2] [--games 2] [--level 16]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
# FirstLight's console setup: the user's FirstLight_CR, whose attested probe its environment requires
os.environ.setdefault('FIRSTLIGHT_ROOT', str(Path.home() / 'Documents/GitHub/FirstLight_CR'))
for _path in (CLAPHA, CLAPHA / 'mac012'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import firstlight_bot as FLB  # noqa: E402  (FirstLight_CR on the path before anything imports native_runner)
import il.duel as D  # noqa: E402
import il.samples as S  # noqa: E402
from il.params import COMMAND_AGE_TICKS, REPLAY_TICK_AFTER_ISSUE, elixir_lead  # noqa: E402

HOG26 = (27000000, 28000000, 26000021, 26000038, 26000030, 26000014, 26000010, 28000011)
HOG26_FORMS = (1, 0, 0, 0, 0, 2, 1, 0)      # Evo Cannon, Hero Musketeer, Evo Skeletons: the specialist's deck
TARGET_DELAY = 26                            # mac012/console.py, il/rl.py: plays land 26 ticks after their moment
DEFER_TICKS = 30                             # a tap not affordable for this long is dropped (the console's 1.5 s)
RUN = CLAPHA / 'runs' / 'rl' / 'vs-firstlight'


def _kept(frame: dict) -> dict:
    """What a recording keeps of one atomic capture: the units, the players, the plain state."""
    return {'tick': frame['tick'], 'objects': frame['objects'], 'players': frame['players'], 'state': frame['state']}


def play_game(env, native, theirs, ours, config, their_side: int, rng: random.Random, sample: bool,
              match_id: str, ours_delay: str = 'live') -> dict:
    """One whole game; the result, with the recording (il/duel's format)."""
    import firstlight_obs as FLO
    from native_runner.arena import cell_to_world
    from native_runner.contracts import ActionKind, ActionV1, TargetKind
    from native_runner.cr_native_env import NATIVE_OBJECT_ID_ENTITY_KEY_TAG, AbilityAction
    from native_runner.training.v4.expert import FIRST_POLICY_DECISION_TICK, POLICY_DECISION_TICKS
    from native_runner.training.v4.policy_session import build_policy_session_v4
    from il.extras import AbilityClock, build_extras, install_session_hook
    from il.flight import flight_ticks
    from il.params import OPPONENT_LEAD_TICKS
    side = 1 - their_side
    deck_forms = {0: list(config.deck0_form_availability), 1: list(config.deck1_form_availability)}
    observations, _ = env.reset(match_config=config, options={
        'render_mode': 'headless', 'decision_ticks': POLICY_DECISION_TICKS, 'event_driven_decisions': False,
        'allow_initial_remaining_runtime_rejections': True})
    initial_elixir = {owner: next(p.elixir_exact for p in obs.players if p.owner == owner)
                      for owner, obs in observations.items()}
    session = build_policy_session_v4(env, theirs.model, actor_owner=their_side, device='cpu', sample=sample)
    session.start_episode(observations[their_side], initial_elixir=initial_elixir)

    battle = None
    commands: list[D.Command] = []           # both sides' plays on their way
    done: list[D.Command] = []
    executed_rows: list[dict] = []
    frames: list[dict] = []
    exact_xy: dict[int, list[int]] = {}      # their plays' points (FirstLight places inside a tile)
    leads_drawn: dict[int, int] = {}
    clock = AbilityClock()
    evolution_ready: dict[tuple[int, int], bool] = {}
    last_players: list = []
    counters = {'ours': {'decided': 0, 'tapped': 0, 'dropped_elixir': 0, 'dropped_hand': 0, 'abilities': 0},
                'theirs': {'plays': 0, 'abilities': 0}}
    seq = 0

    def form_now(owner: int, card_id: int) -> int:
        state = next((q for q in last_players if q['owner'] == owner), None)
        if state is None:
            return 0
        deck = [d['cardId'] for d in sorted(state['deck'], key=lambda d: d['deckSlot'])]
        flag = int(deck_forms[owner][deck.index(card_id)]) if card_id in deck else 0
        if flag & S.HERO_FLAG:
            return 2
        return 1 if flag & S.EVOLUTION_FLAG and evolution_ready.get((owner, card_id)) else 0

    def extras(frame: dict, tick: int) -> None:
        """il/duel's session_extras for our side: pending cards, the opponent's exact elixir, the delay."""
        install_session_hook(ours.session)
        pending = []
        for c in commands:
            if c.kind != 'card':
                continue
            execute = c.execute if c.execute is not None else c.tap + COMMAND_AGE_TICKS
            arrival = execute - tick + flight_ticks(c.card_id, c.side, c.grid)
            if c.side == side:
                pending.append((c.card_id, form_now(c.side, c.card_id), 0, c.grid, execute - tick, arrival))
            elif c.execute is not None:
                lead = leads_drawn.setdefault(c.seq, rng.choices(*zip(*sorted(OPPONENT_LEAD_TICKS.items())))[0])
                if c.execute - lead <= tick < c.execute:
                    pending.append((c.card_id, form_now(c.side, c.card_id), 1, c.grid, c.execute - tick, arrival))
        opponent_raw = {p['owner']: p['elixirRaw'] for p in frame['state']['players']}[their_side]
        ours.session.next_extras = build_extras(ours.session.tensorizer, pending,
                                                opponent_elixir=opponent_raw / 10000.0, delay=TARGET_DELAY,
                                                abilities=S.ability_rows(frame, side, clock, tick))

    def our_observation(frame: dict, tick: int, revealed: dict):
        nonlocal battle
        sent = [D._Sent(c.card_id) for c in sorted(commands, key=lambda c: c.seq) if c.side == side and c.kind == 'card']
        raw = S.reader_frame(frame, side, deck_forms, sent, strict=False)
        health = {'local_side': side, 'visible_sides': [side]}
        if battle is None:
            observation, battle = FLO.build(raw, health, episode_id=f'{match_id}:{side}')
            me = raw['players'][side]
            ours.start_battle(me['deck_card_ids'], None, side, observation,
                              {p['side']: p['elixir_raw'] / 10000.0 for p in raw['players']},
                              our_forms=me['deck_form_flags'])
            opponent = {p['owner']: p for p in frame['state']['players']}[their_side]
            ours.adopt_api_deck([d['cardId'] for d in sorted(opponent['deck'], key=lambda d: d['deckSlot'])])
        ours.register_plays(executed_rows, revealed)
        ours.attribute_opponent_abilities(executed_rows, raw['players'][their_side])
        seen = {side: revealed[side], their_side: [c for c in revealed[their_side] if c in ours.opponent_seen]}
        me = raw['players'][side]
        observation, battle = FLO.build(
            raw, health, episode_id=f'{match_id}:{side}', battle=battle, revealed=seen, reserved=0.0,
            plays=executed_rows, decks=ours.tracked_decks(),
            hand_forms=ours.hand_forms(me['deck_card_ids'], me['deck_form_flags'], me['evo_progress']),
            pending_ability_sources=tuple(c.entity_id for c in commands if c.side == side and c.kind == 'ability'),
            elixir_lead_ticks=elixir_lead(TARGET_DELAY) if ours_delay == 'live' else 0)
        if tick < ours.first_decision_tick:
            ours.observe(observation)
            return None
        if ours.model is not None and '_extras_head' in ours.model.__dict__:
            extras(frame, tick)
        return observation

    def plan(moves, tick: int) -> None:
        """Our decided plays: tapped after the pipeline's overhead, held to land TARGET_DELAY after the moment."""
        nonlocal seq
        hold = TARGET_DELAY - REPLAY_TICK_AFTER_ISSUE
        for move in moves:
            kind, _slot, card_id, target_grid, offset = move
            kind = str(getattr(kind, 'value', kind))
            tap = tick + int(offset) + max(D._overhead(rng), hold) if ours_delay == 'live' else tick
            seq += 1
            if kind == 'activate_ability':
                entity = getattr(move, 'source_entity', None)
                address = {eid: addr for addr, eid in battle.ids.items()}.get(entity)
                if address is None:
                    continue
                joined = FLO.ability_by_card().get(FLO.base_card(int(card_id))) if card_id is not None else None
                commands.append(D.Command(side=side, card_id=int(card_id or 0), grid=(0, 0), decided=tick, tap=tap,
                                          seq=seq, kind='ability',
                                          cost=float(getattr(joined[1], 'elixir_cost', 0) or 0) if joined else 0.0,
                                          native_object_id=int(address, 16) & 0x00FFFFFFFFFFFFFF, entity_id=int(entity)))
            elif kind == 'play_card' and target_grid is not None:
                counters['ours']['decided'] += 1
                commands.append(D.Command(side=side, card_id=int(card_id), grid=(int(target_grid[0]), int(target_grid[1])),
                                          decided=tick, tap=tap, seq=seq, cost=S._card_cost(int(card_id)) or 0.0,
                                          # no delay (FirstLight's sandbox, the league's hog2): it executes here
                                          moment=tick + (int(offset) if ours_delay == 'live' else max(1, int(offset)))))

    def due_taps(frame: dict, tick: int) -> list:
        """Our taps due before the next turn, elixir checked as the console checks it; their ActionV1s."""
        actions = []
        state = {p['owner']: p for p in frame['state']['players']}[side]
        in_flight = sum(c.cost for c in commands if c.side == side and c.execute is not None and c.execute > tick)
        screen = state['elixirRaw'] / 10000.0 - in_flight
        for command in sorted((c for c in commands if c.side == side and c.execute is None
                               and c.tap < tick + POLICY_DECISION_TICKS), key=lambda c: c.seq):
            if screen < command.cost - 1e-6 and ours_delay == 'live':
                if tick - command.tap > DEFER_TICKS:
                    counters['ours']['dropped_elixir'] += 1
                    commands.remove(command)
                continue
            execute = max(command.tap, tick) + COMMAND_AGE_TICKS if ours_delay == 'live' else max(command.moment, tick + 1)
            if command.kind == 'ability':
                try:
                    native.queue_ability_action_at(AbilityAction(side, NATIVE_OBJECT_ID_ENTITY_KEY_TAG,
                                                                 command.native_object_id), execute_in_ticks=execute - tick)
                    counters['ours']['abilities'] += 1
                except Exception:  # noqa: BLE001  (the unit died or its ability is not ready)
                    commands.remove(command)
                    continue
            else:
                slot = next((int(h['handIndex']) for h in state['hand'] if int(h['cardId']) == command.card_id), None)
                if slot is None:
                    counters['ours']['dropped_hand'] += 1
                    commands.remove(command)
                    continue
                command.hand_slot = slot
                candidate = ActionV1(owner=side, kind=ActionKind.PLAY_CARD, hand_slot=slot, card_id=command.card_id,
                                     target_kind=TargetKind.GRID, target_grid=tuple(command.grid),
                                     subcell_offset=(0.0, 0.0), execute_offset_ticks=execute - tick,
                                     # the next decision on the five-tick grid, as FirstLight's own actions
                                     # say (the environment advances to the earliest side's next decision)
                                     next_decision_ticks=POLICY_DECISION_TICKS,
                                     action_id=f'{match_id}-ours-{command.seq}')
                try:
                    env._validate_action(candidate, env._raw)
                except Exception:  # noqa: BLE001  (an illegal target or not affordable now: the engine refuses it)
                    counters['ours']['rejected'] = counters['ours'].get('rejected', 0) + 1
                    commands.remove(command)
                    continue
                actions.append(candidate)
            command.execute = execute
            command.injected = True
            screen -= command.cost
            counters['ours']['tapped'] += 1
        return actions

    terminated = truncated = False
    while not (terminated or truncated):
        frame = dict(native.last_atomic['rich'])
        frame['state'] = native.last_atomic['ordinary']
        frame['tick'] = int(frame['state']['tick'])
        tick = frame['tick']
        frames.append(_kept(frame))
        last_players[:] = frame['state']['players']
        for player in frame.get('players') or ():
            for evolution in player.get('evolutionRuntime') or ():
                evolution_ready[(int(player['owner']), int(evolution['cardId']))] = bool(evolution.get('ready'))
        # executed by now: as il/duel dates them (the engine's hand still holds a card at execute - 1)
        for command in [c for c in commands if c.execute is not None and c.execute <= tick]:
            if command.kind == 'ability':
                executed_rows.append({'tick': command.execute, 'side': command.side, 'card_id': 65535, 'form_code': 0,
                                      'kind': 'ability', 'x': None, 'y': None, 'seq': command.seq,
                                      'issue_tick': command.execute - COMMAND_AGE_TICKS})
            else:
                x, y = exact_xy.get(command.seq) or cell_to_world(command.grid)
                executed_rows.append({'tick': command.execute, 'side': command.side, 'card_id': command.card_id,
                                      'form_code': 0, 'kind': 'card', 'x': x, 'y': y, 'seq': command.seq,
                                      'issue_tick': command.execute - COMMAND_AGE_TICKS})
            commands.remove(command)
            done.append(command)
        for row in executed_rows:
            if row['kind'] == 'card':
                row['form_code'] = form_now(row['side'], row['card_id'])
        revealed = {0: [], 1: []}
        for row in executed_rows:
            if row['card_id'] not in revealed[row['side']]:
                revealed[row['side']].append(row['card_id'])

        # both decide on this turn's state
        observation = observations[their_side]
        chosen = ()
        if observation.tick >= FIRST_POLICY_DECISION_TICK:
            chosen = tuple(session.decide(observation).decoded.actions)
        else:
            session.tensorizer.tensorize(observation, validate=False)
        mine = our_observation(frame, tick, revealed)
        if mine is not None:
            plan(ours.decide(mine), tick)
        our_actions = due_taps(frame, tick)

        # their plays, for our model's view and the recording: executed 1-4 ticks from now
        hands = {p['owner']: p for p in frame['state']['players']}[their_side]['hand']
        for action in chosen:
            if action.kind == ActionKind.PLAY_CARD and action.target_grid is not None:
                card = next((int(h['cardId']) for h in hands if int(h['handIndex']) == action.hand_slot), None)
                if card is None:
                    continue
                seq += 1
                exact_xy[seq] = list(env._world_target(action))
                commands.append(D.Command(side=their_side, card_id=card, grid=tuple(action.target_grid), decided=tick,
                                          tap=tick, execute=tick + max(1, action.execute_offset_ticks or 1), seq=seq,
                                          injected=True, hand_slot=action.hand_slot, cost=S._card_cost(card) or 0.0))
                counters['theirs']['plays'] += 1
            elif action.kind == ActionKind.ACTIVATE_ABILITY:
                seq += 1
                commands.append(D.Command(side=their_side, card_id=0, grid=(0, 0), decided=tick, tap=tick,
                                          execute=tick + max(1, action.execute_offset_ticks or 1), seq=seq,
                                          kind='ability', injected=True))
                counters['theirs']['abilities'] += 1
        step = {their_side: chosen or (ActionV1.wait(their_side, ticks=POLICY_DECISION_TICKS),),
                side: tuple(our_actions) or (ActionV1.wait(side, ticks=POLICY_DECISION_TICKS),)}
        observations, _, terms, truncs, _ = env.step(step, record_trace=False)
        terminated, truncated = bool(terms['__all__']), bool(truncs['__all__'])

    final = native.observe()
    frame = dict(native.last_atomic['rich'])
    frame['state'] = native.last_atomic['ordinary']
    frame['tick'] = int(frame['state']['tick'])
    frames.append(_kept(frame))
    for command in commands:
        if command.execute is not None and command.execute <= int(final.get('tick') or 0):
            done.append(command)
    session.end_episode()
    ours.end_battle()
    recording = D._recording(frames, done, deck_forms, final)
    recording['header']['play_xy'] = {str(k): v for k, v in exact_xy.items()}
    return {'winner': final.get('winner'), 'crowns': final.get('crownsRaw'), 'tick': final.get('tick'),
            'counters': counters, 'recording': recording}


def _update_of(name: str) -> int:
    import torch
    try:
        return int(torch.load(FLB.CHECKPOINTS[name], map_location='cpu', weights_only=False).get('update_step') or 0)
    except Exception:  # noqa: BLE001
        return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ours', default='clapha:p3', help='our model (a firstlight_bot name)')
    parser.add_argument('--theirs', default='fl:hog2', help="FirstLight's model, at its full strength")
    parser.add_argument('--games', type=int, default=2)
    parser.add_argument('--level', type=int, default=16, help='card, King and princess tower level (both sides)')
    parser.add_argument('--seed', type=int, default=0, help='first deal seed (0: random)')
    parser.add_argument('--deterministic', action='store_true', help="FirstLight's argmax instead of its sampling")
    parser.add_argument('--ours-delay', default='live', choices=('live', 'none'),
                        help="none: our side's plays execute 1-4 ticks after deciding (FirstLight's sandbox, how the "
                             "league's hog2 played); with --ours fl:hog2 it is the league's hog2 against the real one")
    parser.add_argument('--run', default=RUN.name, help='the runs/rl folder the recordings go to')
    args = parser.parse_args(argv)
    from native_runner.battle_env import BattleEnvV1
    from native_runner.cr_native_env import NativeClashEnv
    from native_runner.match_factory import MatchConfig
    from native_runner.rich_telemetry_adapter import RuntimeEffectCatalog
    from native_runner.training.v4.factory import production_semantic_bundle
    from native_runner.training.v4.policy_session import load_policy_v4
    from il.frames import save_recorded
    from il.rl import battle_setup
    from il.speed import install
    import il.watch_nulls as W
    install()
    W.prepare_engine()
    theirs = load_policy_v4(FLB.CHECKPOINTS[args.theirs], device='cpu')
    ours = FLB.FirstLightRunner(args.ours)
    update = _update_of(args.ours)
    bundle = production_semantic_bundle()
    native = NativeClashEnv('127.0.0.1', W.PORT, timeout=60.0)
    native.public_card_play_events_from_combat_ring = True
    original = native.observe_atomic

    def observe_atomic():
        # the environment's own capture each turn, kept for our side's frame
        native.last_atomic = original()
        return native.last_atomic
    native.observe_atomic = observe_atomic
    env = BattleEnvV1(native=native, card_catalog=bundle.native_card_catalog,
                      semantic_subset_contract=bundle.semantic_subset_contract,
                      effect_catalog=RuntimeEffectCatalog.from_catalog(bundle.native_effect_catalog))
    rng = random.Random(args.seed or None)
    score = {'ours': 0, 'theirs': 0, 'draw': 0}
    run = RUN.parent / args.run
    for index in range(args.games):
        seed = args.seed + index if args.seed else rng.randrange(1, 2 ** 31)
        their_side = 1 - seed % 2                # ours on side seed % 2, as il.duel --specialist puts its a
        config = MatchConfig(deck0=HOG26, deck1=HOG26, deck0_form_availability=HOG26_FORMS,
                             deck1_form_availability=HOG26_FORMS, seed=seed, level_cap=args.level,
                             minimum_card_level=args.level, king_tower_level=args.level,
                             owner0_name='FirstLight' if their_side == 0 else f'Ours u{update}',
                             owner1_name=f'Ours u{update}' if their_side == 0 else 'FirstLight')
        started = time.time()
        result = play_game(env, native, theirs, ours, config, their_side, rng, not args.deterministic,
                           match_id=f'vsfl-{seed}', ours_delay=args.ours_delay)
        winner = result['winner']
        label = 'draw' if winner not in (0, 1) else ('ours' if winner == 1 - their_side else 'theirs')
        score[label] += 1
        header = result['recording']['header']
        header['played'] = {'a': f'{args.ours} v{update}', 'b': args.theirs, 'a_side': 1 - their_side,
                            'league': 'firstlight', 'result': {'ours': 'a', 'theirs': 'b'}.get(label, 'draw'),
                            'run': run.name, 'config': battle_setup(config), 'counters': result['counters'],
                            'ours_delay': args.ours_delay}
        save_recorded(run / 'recordings', header, result['recording']['frames'])
        print(json.dumps({'game': index + 1, 'winner': label, 'crowns': result['crowns'], 'end_tick': result['tick'],
                          'ours_side': 1 - their_side, 'seconds': round(time.time() - started),
                          'counters': result['counters'], 'score': score}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
