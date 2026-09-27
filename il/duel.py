"""Model against model in Null's engine, both sides fed by the live code, with a real command delay.

Each side is a FirstLightRunner (a FirstLight checkpoint or one of ours) given exactly what the
console gives it: the engine snapshot as the reader's frame (il/samples.reader_frame: its own hand
and elixir as the screen shows them, its sent commands taken out), firstlight_obs.build, the
tracker with the opponent's deck adopted as from the API. What differs between the sides is only
when their plays execute:
  delay 'live'  the console's timing: the tap goes out `overhead` ticks after the decision (drawn
                per play from the measured live distribution, il/params.OWN_OVERHEAD_TICKS) plus
                the play's offset in its window, waits for elixir at the tap as the console does,
                and executes 21 ticks after the tap (the game's command age)
  delay 'none'  FirstLight's sandbox: the play executes on the next tick after the decision
Decks come from converted Hog 2.6 replays (il/frames.py): the Hog side's deck for both sides
(a mirror), sides swapped every match.

    ./py -m il.duel --a fl:hog2 --a-delay live --b fl:hog2 --b-delay none --matches 20
Needs the engine free (CR_4k with the run-lean probe); the conversion uses the same engine.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
if str(CLAPHA) not in sys.path:
    sys.path.insert(0, str(CLAPHA))

import il.samples as S                       # noqa: E402  (live code on sys.path first)
from il.params import (COMMAND_AGE_TICKS, DECISION_TICKS, OWN_OVERHEAD_TICKS, REPLAY_TICK_AFTER_ISSUE,  # noqa: E402
                       elixir_lead, live_delay)

DEFER_TICKS = 30        # a tap that cannot be afforded is dropped after this long (console: 1.5 s)


@dataclass
class Command:
    """One decided play on its way: tapped at `tap`, executes at `execute` (the engine step after)."""
    side: int
    card_id: int
    grid: tuple[int, int]
    decided: int
    tap: int
    execute: int | None = None
    kind: str = 'card'
    seq: int = 0
    injected: bool = False
    cost: float = 0.0
    native_object_id: int | None = None      # an ability's source unit in the engine
    entity_id: int | None = None             # the same unit in the live code's ids
    hand_slot: int | None = None             # the engine hand index it was queued from
    moment: int = 0                          # decision tick + the model's offset: the tap without delay
    native_xy: tuple[int, int] | None = None  # a replayed play's exact point (tile centre otherwise)


def _lead_arg(value: str):
    return 'auto' if value == 'auto' else int(value)


def _overhead(rng: random.Random) -> int:
    values, weights = zip(*sorted(OWN_OVERHEAD_TICKS.items()))
    return rng.choices(values, weights)[0]


def play_match(native, runners, delays, leads, config, deck_forms, rng, match_id: str, record: bool = False,
               extra_delay: dict | None = None, measured: dict | None = None,
               script: dict[int, list[dict]] | None = None, target_delay: int | None = None,
               decide_many=None) -> dict:
    """One battle to the end; returns the result and per-side counters.

    The engine is stepped to the next thing that happens: a decision turn (every 5 ticks), a tap
    coming due (checked for elixir at that tick; one that cannot be afforded is re-checked every
    tick for up to DEFER_TICKS, as the console defers it), or a command to inject the tick before
    it executes (as il/engine_convert queues recorded plays)."""
    import firstlight_obs as FLO
    from il.engine_convert import run_lean
    from native_runner.arena import cell_to_world
    from native_runner.cr_native_env import NATIVE_OBJECT_ID_ENTITY_KEY_TAG, AbilityAction, HandAction

    native.create_match(config)
    battles = {0: None, 1: None}
    commands: list[Command] = []
    executed_rows: list[dict] = []
    counters = {side: {'decided': 0, 'tapped': 0, 'executed': 0, 'dropped_elixir': 0, 'dropped_hand': 0,
                       'abilities': 0, 'abilities_dropped': 0} for side in (0, 1)}
    tick, ended, seq = 0, False, 0
    # a replay opponent (script: side -> il.engine_convert.replay_commands card plays): the real
    # player's plays go in at their recorded ticks, as the conversion queues them; they sit in the
    # command list like anyone's, so the other side sees them pending and executed
    for scripted_side, plays in (script or {}).items():
        for play in plays:
            seq += 1
            issue = int(play['tick']) - COMMAND_AGE_TICKS
            commands.append(Command(side=scripted_side, card_id=int(play['card_id']),
                                    grid=(int(play['x']) // 1000, int(play['y']) // 1000), decided=issue, tap=issue,
                                    execute=int(play['tick']), seq=seq, cost=S._card_cost(int(play['card_id'])) or 0.0,
                                    moment=issue, native_xy=(int(play['x']), int(play['y']))))
            counters[scripted_side]['tapped'] += 1
    extra_delay = extra_delay or {0: 0, 1: 0}   # ticks added to a live side's tap (a slower pipeline)
    # lead 'auto': as the console does, each side's delay is measured from its own plays (moment
    # -> issue, il.params.live_delay), told to an extended model, and sets the mask's elixir lead
    # per pipeline, kept across matches as the console keeps it across battles
    measured = measured if measured is not None else {0: [], 1: []}

    def told_delay(side: int) -> int:
        # target_delay: the console's held landing (TARGET_DELAY); otherwise the measured rule
        return int(target_delay) if target_delay else live_delay(measured[side])

    def lead_now(side: int) -> int:
        return elixir_lead(told_delay(side)) if leads[side] == 'auto' else int(leads[side])
    recorded_frames: list[dict] = []          # record=True: every decision-tick snapshot, as a conversion saves
    done: list[Command] = []                   # executed commands, for the recording

    def screen_elixir(side: int, logic_raw: int) -> float:
        in_flight = [c for c in commands if c.side == side and c.execute is not None]
        return logic_raw / 10000.0 - sum(c.cost for c in in_flight)

    def process_taps(state_players) -> None:
        for command in sorted((c for c in commands if c.execute is None and c.tap <= tick), key=lambda c: c.seq):
            logic = {p['owner']: p['elixirRaw'] for p in state_players}[command.side]
            if screen_elixir(command.side, logic) >= command.cost - 1e-6:
                command.execute = tick + COMMAND_AGE_TICKS
                counters[command.side]['tapped'] += 1
                if command.kind == 'card' and tick == command.tap:
                    measured[command.side].append(tick - command.moment)
            elif tick - command.tap > DEFER_TICKS:
                counters[command.side]['dropped_elixir'] += 1
                commands.remove(command)

    def session_extras(runner, side: int, tick: int, frame: dict) -> None:
        """Stage B inputs as the console would build them: this side's commands not executed yet
        (execute time estimated as tap + 21 until tapped), the opponent's once visible in the queue
        (issue + 21 - lead, lead drawn per command from the measured table), exact opponent elixir."""
        from il.extras import build_extras, install_session_hook
        from il.flight import flight_ticks
        from il.params import OPPONENT_LEAD_TICKS, OWN_OVERHEAD_TICKS
        install_session_hook(runner.session)
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
        opponent_raw = {p['owner']: p['elixirRaw'] for p in frame['state']['players']}[1 - side]
        overhead = sum(k * v for k, v in OWN_OVERHEAD_TICKS.items()) / sum(OWN_OVERHEAD_TICKS.values())
        delay = round(COMMAND_AGE_TICKS - 1 + overhead + extra_delay[side]) if delays[side] == 'live' else 1
        if delays[side] == 'live' and leads[side] == 'auto':
            delay = told_delay(side)
        runner.session.next_extras = build_extras(runner.session.tensorizer, pending,
                                                  opponent_elixir=opponent_raw / 10000.0, delay=delay,
                                                  abilities=S.ability_rows(frame, side, ability_clocks[side], tick))

    leads_drawn: dict[int, int] = {}
    from il.extras import AbilityClock
    ability_clocks = {0: AbilityClock(), 1: AbilityClock()}
    evolution_ready: dict[tuple[int, int], bool] = {}
    last_players: list = []

    def form_now(owner: int, card_id: int) -> int:
        """A play's form by il/samples' rule (form_at), so the duel feeds what training fed: hero
        -> 2; evolution flag with the evolution ready as of the latest decision frame -> 1."""
        state = next((q for q in last_players if q['owner'] == owner), None)
        if state is None:
            return 0
        deck = [d['cardId'] for d in sorted(state['deck'], key=lambda d: d['deckSlot'])]
        flag = int(deck_forms[owner][deck.index(card_id)]) if card_id in deck else 0
        if flag & S.HERO_FLAG:
            return 2
        return 1 if flag & S.EVOLUTION_FLAG and evolution_ready.get((owner, card_id)) else 0

    def inject_due() -> None:
        for command in [c for c in commands if c.execute is not None and not c.injected and c.execute - 1 == tick]:
            if command.kind == 'ability':
                try:
                    native.queue_ability_action_at(AbilityAction(command.side, NATIVE_OBJECT_ID_ENTITY_KEY_TAG,
                                                                 command.native_object_id), execute_in_ticks=1)
                    command.injected = True
                except Exception:  # noqa: BLE001  (the source unit died or the ability is not ready)
                    counters[command.side]['abilities_dropped'] += 1
                    commands.remove(command)
                continue
            state = next(p for p in native.observe()['players'] if p['owner'] == command.side)
            slot = next((h['handIndex'] for h in state['hand'] if h['cardId'] == command.card_id), None)
            if slot is None:
                counters[command.side]['dropped_hand'] += 1
                commands.remove(command)
                continue
            x, y = command.native_xy or cell_to_world(command.grid)
            native.queue_hand_action_at(HandAction(command.side, slot, x, y), execute_tick=command.execute)
            command.injected = True
            command.hand_slot = int(slot)

    while True:
        waiting = any(c.execute is None and c.tap <= tick for c in commands)
        candidates = [tick + DECISION_TICKS - tick % DECISION_TICKS]
        candidates += [c.tap for c in commands if c.execute is None and c.tap > tick]
        candidates += [c.execute - 1 for c in commands if c.execute is not None and not c.injected
                       and c.execute - 1 > tick]
        if waiting:
            candidates.append(tick + 1)
        frames, tick, ended = run_lean(native, min(candidates))
        if record:
            recorded_frames.extend(frames)
        if ended:
            break
        decision = tick % DECISION_TICKS == 0 and frames
        players = frames[-1]['state']['players'] if decision else native.observe()['players']
        process_taps(players if decision else [{'owner': p['owner'], 'elixirRaw': p['elixirRaw']} for p in players])
        inject_due()
        if not decision:
            continue
        frame = frames[-1]
        last_players[:] = frame['state']['players']
        for player in frame.get('players') or ():
            for evolution in player.get('evolutionRuntime') or ():
                evolution_ready[(player['owner'], int(evolution['cardId']))] = bool(evolution.get('ready'))
        # executed: out of the hand in the snapshot at their execute tick; dated like the viewer's
        for command in [c for c in commands if c.execute is not None and c.execute <= tick]:
            if command.kind == 'ability':
                # the viewer's row for an ability: no card (65535); the console joins it to a hero
                executed_rows.append({'tick': command.execute, 'side': command.side, 'card_id': 65535,
                                      'form_code': 0, 'kind': 'ability', 'x': None, 'y': None, 'seq': command.seq,
                                      'issue_tick': command.execute - COMMAND_AGE_TICKS})
            else:
                x, y = cell_to_world(command.grid)
                executed_rows.append({'tick': command.execute, 'side': command.side, 'card_id': command.card_id,
                                      'form_code': 0, 'kind': 'card', 'x': x, 'y': y, 'seq': command.seq,
                                      'issue_tick': command.execute - COMMAND_AGE_TICKS})
            counters[command.side]['executed'] += 1
            commands.remove(command)
            done.append(command)
        for row in executed_rows:
            # training recomputes every executed play's form each turn (il/samples.executed_plays)
            if row['kind'] == 'card':
                row['form_code'] = form_now(row['side'], row['card_id'])
        revealed = {0: [], 1: []}
        for row in executed_rows:
            if row['card_id'] not in revealed[row['side']]:
                revealed[row['side']].append(row['card_id'])
        def prepare(side):
            runner = runners[side]
            if runner is None:
                return None     # a replay opponent: its plays are scripted
            # every play this side has decided and that has not executed is gone from its screen
            # hand, tapped or not: the console taps within the turn, and training counts a play as
            # sent from the turn after its decision (il/samples in_flight)
            sent = [_Sent(c.card_id) for c in sorted(commands, key=lambda c: c.seq) if c.side == side and c.kind == 'card']
            try:
                raw = S.reader_frame(frame, side, deck_forms, sent, strict=False)
            except ValueError:
                state = {p['owner']: p for p in frame['state']['players']}[side]
                print('DEBUG tick', tick, 'side', side, 'hand', [(h['handIndex'], h['deckSlot'], h['cardId']) for h in state['hand']],
                      'cycle', [c['cardId'] for c in state['cycle']],
                      'commands', [(c.side, c.card_id, c.decided, c.tap, c.execute, c.injected) for c in commands],
                      'executed', [(r['side'], r['card_id'], r['tick']) for r in executed_rows[-6:]], flush=True)
                raise
            health = {'local_side': side, 'visible_sides': [side]}
            if battles[side] is None:
                observation, battles[side] = FLO.build(raw, health, episode_id=f'{match_id}:{side}')
                me = raw['players'][side]
                runner.start_battle(me['deck_card_ids'], None, side, observation,
                                    {p['side']: p['elixir_raw'] / 10000.0 for p in raw['players']},
                                    our_forms=me['deck_form_flags'])
                opponent = {p['owner']: p for p in frame['state']['players']}[1 - side]
                runner.adopt_api_deck([d['cardId'] for d in sorted(opponent['deck'], key=lambda d: d['deckSlot'])])
            runner.register_plays(executed_rows, revealed)
            runner.attribute_opponent_abilities(executed_rows, raw['players'][1 - side])
            seen = {side: revealed[side], 1 - side: [c for c in revealed[1 - side] if c in runner.opponent_seen]}
            me = raw['players'][side]
            observation, battles[side] = FLO.build(
                raw, health, episode_id=f'{match_id}:{side}', battle=battles[side], revealed=seen, reserved=0.0,
                plays=executed_rows, decks=runner.tracked_decks(),
                hand_forms=runner.hand_forms(me['deck_card_ids'], me['deck_form_flags'], me['evo_progress']),
                pending_ability_sources=tuple(c.entity_id for c in commands if c.side == side and c.kind == 'ability'),
                elixir_lead_ticks=lead_now(side))
            if tick < runner.first_decision_tick:
                runner.observe(observation)
                return None
            if runner.model is not None and '_extras_head' in runner.model.__dict__:
                session_extras(runner, side, tick, frame)
            return runner, observation

        def apply(side, runner, moves):
            nonlocal seq
            for move in moves:
                kind, _slot, card_id, target_grid, offset = move
                kind = str(getattr(kind, 'value', kind))
                if delays[side] == 'live':
                    # the pipeline's own time, or held to the target landing when it is quicker
                    pipeline = _overhead(rng) + extra_delay[side]
                    hold = int(target_delay) - REPLAY_TICK_AFTER_ISSUE if target_delay else 0
                    tap = tick + int(offset) + max(pipeline, hold)
                else:
                    tap = tick + int(offset)
                seq += 1
                if kind == 'activate_ability':
                    entity = getattr(move, 'source_entity', None)
                    address = {eid: addr for addr, eid in battles[side].ids.items()}.get(entity)
                    joined = FLO.ability_by_card().get(FLO.base_card(int(card_id))) if card_id is not None else None
                    if address is None:
                        counters[side]['abilities_dropped'] += 1
                        continue
                    counters[side]['abilities'] += 1
                    command = Command(side=side, card_id=int(card_id or 0), grid=(0, 0), decided=tick, tap=tap,
                                      seq=seq, kind='ability',
                                      cost=float(getattr(joined[1], 'elixir_cost', 0) or 0) if joined else 0.0,
                                      native_object_id=int(address, 16) & 0x00FFFFFFFFFFFFFF, entity_id=int(entity))
                elif kind == 'play_card' and target_grid is not None:
                    counters[side]['decided'] += 1
                    command = Command(side=side, card_id=int(card_id), grid=(int(target_grid[0]), int(target_grid[1])),
                                      decided=tick, tap=tap, seq=seq, cost=S._card_cost(int(card_id)) or 0.0,
                                      moment=tick + int(offset))
                else:
                    continue
                if delays[side] != 'live':
                    # FirstLight's sandbox: execute_in_ticks = max(1, offset), no tap and no wait
                    command.execute = tick + max(1, int(offset))
                    counters[side]['tapped'] += 1
                commands.append(command)
        if decide_many is None:
            # one side after the other: side 1 is prepared after side 0's plays of this turn exist
            for side in (0, 1):
                prepared = prepare(side)
                if prepared is not None:
                    apply(side, prepared[0], prepared[0].decide(prepared[1]))
        else:
            # both sides prepared from the same state, then decided together (il/rl_serve.py)
            ready = []
            for side in (0, 1):
                prepared = prepare(side)
                if prepared is not None:
                    ready.append((side, *prepared))
            if ready:
                decided = decide_many(ready)
                for side, runner, _observation in ready:
                    apply(side, runner, decided[side])
        # a tap due right now (no-delay side, offset 0) goes out this tick
        process_taps(frame['state']['players'])
        inject_due()
    final = native.observe()
    for runner in runners.values():
        if runner is not None:
            runner.end_battle()
    # each side's standing tower health at the end (a destroyed tower is gone): with a replay
    # opponent, wins alone saturate, and what a model loses to real pushes is the finer measure
    tower_hp = {0: 0, 1: 0}
    for o in final.get('objects') or ():
        if 5000000 <= int(o.get('nativeObjectId', 0)) <= 5000005 and o.get('hp') is not None:
            tower_hp[int(o['owner'])] += int(o['hp'])
    result = {'winner': final.get('winner'), 'crowns': final.get('crownsRaw'), 'tick': final.get('tick'),
              'counters': counters, 'tower_hp': tower_hp}
    if record and recorded_frames:
        result['recording'] = _recording(recorded_frames, done, deck_forms, final)
    return result


def _recording(frames: list[dict], done: list[Command], deck_forms: dict, final: dict) -> dict:
    """A played match in the conversion's format (il/frames.save_recorded): the timeline of what
    executed and each card play as a FirstLight expert action (the tick before it executed, the
    hand slot it was queued from), so il/samples and il/teacher treat it like a replay."""
    import base64
    import pickle
    import uuid
    from il.frames import FRAME_FORMAT
    from il.timeline import Play, Timeline, timeline_to_json
    from native_runner.contracts import ActionKind, ActionV1, TargetKind
    from native_runner.training.v4.expert import TimedExpertActionV4
    tag = f'd{uuid.uuid4().hex}'
    first = {p['owner']: p for p in frames[0]['state']['players']}
    decks, deals = [], {}
    for owner in (0, 1):
        state = first[owner]
        decks.append(tuple(d['cardId'] for d in sorted(state['deck'], key=lambda d: d['deckSlot'])))
        deals[owner] = (tuple(h['cardId'] for h in sorted(state['hand'], key=lambda h: h['handIndex'])),
                        tuple(c['cardId'] for c in sorted(state['cycle'], key=lambda c: c['cycleIndex'])))
    executed = sorted(done, key=lambda c: (c.execute, c.seq))
    plays = tuple(Play(owner=c.side, kind=c.kind, card_id=c.card_id if c.kind == 'card' else None,
                       grid=tuple(c.grid) if c.kind == 'card' else None, lands=c.execute - 1, index=c.seq)
                  for c in executed)
    end_tick = int(final.get('tick') or frames[-1]['tick'])
    timeline = Timeline(replay_tag=tag, decks=(decks[0], decks[1]),
                        form_availability=(tuple(deck_forms[0]), tuple(deck_forms[1])), tower_troops=(None, None),
                        deals=deals, first_certain_play={0: 0, 1: 0}, plays=plays, end_tick=end_tick,
                        winner=final.get('winner'))
    expert = [TimedExpertActionV4(
        source_tick=c.execute - 1, source_index=c.seq,
        action=ActionV1(owner=c.side, kind=ActionKind.PLAY_CARD, hand_slot=c.hand_slot, card_id=c.card_id,
                        target_kind=TargetKind.GRID, target_grid=tuple(c.grid), subcell_offset=(0.0, 0.0),
                        execute_offset_ticks=1, next_decision_ticks=1, action_id=f'{tag}-{c.seq}',
                        metadata={'source': 'clapha-duel', 'replay_tag': tag, 'source_index': c.seq,
                                  'source_event_index': c.seq, 'source_command_tick': c.execute - 1,
                                  'native_observable_tick': c.execute}))
              for c in executed if c.kind == 'card' and c.hand_slot is not None]
    header = {'format': FRAME_FORMAT, 'replay_tag': tag, 'end_tick': end_tick, 'ended_at': end_tick, 'every': DECISION_TICKS,
              'frames': len(frames), 'first_tick': frames[0]['tick'], 'commands': len(executed), 'queued': len(executed),
              'failures': [], 'fidelity': {'source': 'played', 'tower_hp_error': 0, 'winner_match': True,
                                           'crowns_match': True},
              'timeline': timeline_to_json(timeline),
              'expert_actions_pickle': base64.b64encode(pickle.dumps(expert, protocol=5)).decode()}
    return {'header': header, 'frames': frames}


@dataclass
class _Sent:
    card_id: int
    kind: str = 'card'


def hog26_matchups(frames_dir: Path, count: int, seed: int):
    """(match config, deck forms) for Hog 2.6 mirrors, from converted replays' Hog decks."""
    from il.frames import load_replay
    from native_runner.royaleapi_replay import _episode_match_config
    from dataclasses import replace
    rng = random.Random(seed)
    paths = sorted((frames_dir / 'frames').glob('*/*.jsonl.zst'))
    rng.shuffle(paths)
    out = []
    for path in paths:
        header, _frames = load_replay(path)
        timeline = header['timeline']
        sides = [i for i, deck in enumerate(timeline['decks']) if S.HOG26_CARDS <= set(deck)]
        if not sides:
            continue
        hog = sides[0]
        episode = header['calibrated'].replay.episode_config
        tags = dict(episode.tags)
        deck = tuple(timeline['decks'][hog])
        forms = tuple(timeline['form_availability'][hog])
        tags['deck0_form_availability'] = tags['deck1_form_availability'] = forms
        mirrored = replace(episode, deck0=deck, deck1=deck, tags=tags)
        out.append((_episode_match_config(mirrored), {0: list(forms), 1: list(forms)}, header['replay_tag']))
        if len(out) >= count:
            break
    return out


def replay_matchups(frames_dir: Path, count: int, seed: int, opponent_has: set[int] | None = None):
    """Real games with one Hog 2.6 side: (match config with both real decks and the deal, deck
    forms, tag, the Hog side, the other side's recorded card plays). The Hog side is the model's;
    the other is played back as the real player played it (a replay opponent: real decks and real
    habits, not reactive -- it does what it did against the original Hog player)."""
    from il.engine_convert import replay_commands
    from il.frames import load_replay
    from native_runner.royaleapi_replay import _episode_match_config
    rng = random.Random(seed)
    paths = sorted((frames_dir / 'frames').glob('*/*.jsonl.zst'))
    rng.shuffle(paths)
    out = []
    for path in paths:
        header, _frames = load_replay(path)
        timeline = header['timeline']
        sides = [i for i, deck in enumerate(timeline['decks']) if S.HOG26_CARDS <= set(deck)]
        if len(sides) != 1:
            continue
        hog = sides[0]
        if opponent_has and not opponent_has & set(timeline['decks'][1 - hog]):
            continue
        replay = header['calibrated'].replay
        plays = [c for c in replay_commands(replay) if c['owner'] == 1 - hog and c['kind'] == 'card']
        forms = {side: list(timeline['form_availability'][side]) for side in (0, 1)}
        out.append((_episode_match_config(replay.episode_config), forms, header['replay_tag'], hog, plays))
        if len(out) >= count:
            break
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', default='fl:hog2')
    parser.add_argument('--a-delay', default='live', choices=('live', 'none'))
    parser.add_argument('--b', default='fl:hog2',
                        help="a checkpoint, or replay: real games' opponents played back (a takes the Hog 2.6 side "
                             "of each game, the real opponent's deck and deal; their recorded plays at their ticks)")
    parser.add_argument('--b-delay', default='none', choices=('live', 'none'))
    parser.add_argument('--a-lead', type=_lead_arg, default=0,
                        help="elixir_lead_ticks for a's mask (0 = FirstLight's), or auto: the console's rule "
                             "(delay measured from this side's plays, told to an extended model, lead from it)")
    parser.add_argument('--b-lead', type=_lead_arg, default=0)
    parser.add_argument('--matches', type=int, default=20)
    parser.add_argument('--opponent-has', help="with --real-decks: only real games whose opponent deck has one of "
                        "these card ids (comma-separated), e.g. 26000046 (Bandit)")
    parser.add_argument('--real-decks', action='store_true',
                        help="real games' decks and deals instead of Hog 2.6 mirrors: a on the Hog side, b (a model, "
                             "e.g. fl:general) on the other player's real deck")
    parser.add_argument('--a-extra-delay', type=int, default=0,
                        help="ticks added to a's taps (a slower pipeline); an extended model is told the true delay")
    parser.add_argument('--b-extra-delay', type=int, default=0)
    parser.add_argument('--target-delay', type=int,
                        help="live sides' plays land this many ticks after their moment (the console's held "
                             "landing, TARGET_DELAY), later only when the drawn pipeline time is longer; "
                             "a side with lead auto is told it")
    parser.add_argument('--frames', type=Path, default=CLAPHA / 'runs/conv-hog26')
    parser.add_argument('--out', type=Path, default=CLAPHA / 'runs/duels.jsonl')
    parser.add_argument('--port', type=int, default=26789)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--record', type=Path, help='save every match here in the conversion format (DAgger data)')
    parser.add_argument('--record-sides', choices=('a', 'b', 'both'), default='a',
                        help="whose side is training data (index.jsonl train_sides)")
    args = parser.parse_args(argv)
    import firstlight_bot as FLB
    from il.engine_convert import connect
    from il.speed import install
    install()           # identical results, less Python per decision (il/speed.py)
    native = connect(args.port)
    native.wait_ready(timeout=60.0)
    replay = args.b == 'replay'
    models = {'a': FLB.FirstLightRunner(args.a),
              'b': None if replay else FLB.FirstLightRunner(args.b)}
    rng = random.Random(args.seed)
    wins = {'a': 0, 'b': 0, 'draw': 0}
    overheads = {'a': [], 'b': []}          # each model's measured delays (lead auto), across matches
    if replay:
        matchups = [(config, forms, tag, hog, plays)
                    for config, forms, tag, hog, plays in replay_matchups(args.frames, args.matches, args.seed)]
    elif args.real_decks:
        # real games' decks and deals, both sides played by models: a on the Hog 2.6 side
        wanted = {int(card) for card in args.opponent_has.split(',')} if args.opponent_has else None
        matchups = [(config, forms, tag, hog, None) for config, forms, tag, hog, _plays
                    in replay_matchups(args.frames, args.matches, args.seed, opponent_has=wanted)]
    else:
        matchups = [(config, forms, tag, index % 2, None)          # sides swapped every match
                    for index, (config, forms, tag) in enumerate(hog26_matchups(args.frames, args.matches, args.seed))]
    for index, (config, forms, tag, a_side, script_plays) in enumerate(matchups):
        runners = {a_side: models['a'], 1 - a_side: models['b']}
        delays = {a_side: args.a_delay, 1 - a_side: args.b_delay}
        leads = {a_side: args.a_lead, 1 - a_side: args.b_lead}
        started = time.time()
        try:
            result = play_match(native, runners, delays, leads, config, forms, rng, f'{tag[:8]}-{index}',
                                record=args.record is not None,
                                extra_delay={a_side: args.a_extra_delay, 1 - a_side: args.b_extra_delay},
                                measured={a_side: overheads['a'], 1 - a_side: overheads['b']},
                                script={1 - a_side: script_plays} if script_plays is not None else None,
                                target_delay=args.target_delay)
        except Exception as error:  # noqa: BLE001  (one broken match must not end the set)
            print(f'match {index + 1} failed: {type(error).__name__}: {error}', flush=True)
            for runner in runners.values():
                try:
                    if runner is not None:
                        runner.end_battle()
                except Exception:  # noqa: BLE001
                    pass
            continue
        winner = result['winner']
        label = 'draw' if winner not in (0, 1) else ('a' if winner == a_side else 'b')
        wins[label] += 1
        row = {'a': args.a, 'a_delay': args.a_delay, 'b': args.b, 'b_delay': args.b_delay, 'deck_from': tag,
               'a_lead': args.a_lead, 'b_lead': args.b_lead, 'a_extra_delay': args.a_extra_delay,
               'b_extra_delay': args.b_extra_delay, 'target_delay': args.target_delay,
               'a_side': a_side, 'result': label, 'crowns': result['crowns'], 'end_tick': result['tick'],
               'tower_hp': result['tower_hp'],
               'counters': result['counters'], 'seconds': round(time.time() - started)}
        with args.out.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        if 'recording' in result:
            from il.frames import save_recorded
            header = result['recording']['header']
            header['played'] = {k: v for k, v in row.items() if k not in ('counters',)}
            save_recorded(args.record, header, result['recording']['frames'])
            sides = {'a': [a_side], 'b': [1 - a_side], 'both': [0, 1]}[args.record_sides]
            with (args.record / 'index.jsonl').open('a') as handle:
                handle.write(json.dumps({'tag': header['replay_tag'], 'ok': True, 'tower_hp_error': 0,
                                         'ended_at': header['ended_at'], 'end_tick': header['end_tick'],
                                         'train_sides': sides, 'played': header['played']}) + '\n')
        print(f"match {index + 1}: {label} (crowns {result['crowns']}, a on side {a_side}); "
              f"running score a {wins['a']} - b {wins['b']} ({wins['draw']} draws); {row['seconds']} s", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
