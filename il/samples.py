"""Imitation samples built by the live pipeline from engine conversions.

For one actor of one converted replay (il/frames.py), every five-tick turn is replayed through
the code the console runs live -- firstlight_obs.build and FirstLightRunner (teacher mode) --
from engine snapshots put in the reader's frame format, and FirstLight's tensorizer makes the
model input. The label at a decision tick is the actor's play that must be decided there to
land when it did (il/timeline.py timing), aligned to the tensorizer's candidates by FirstLight's
own build_expert_action_batch; the expert's action is then recorded as the previous action, as
the policy's own is live.

Reader frame from an engine snapshot (reader field <- engine field):
  entity  address, category <- nativeObjectId; kind 0 (no hitpoints: projectile or area
          effect) / 13 (crown tower) / 15 (unit); side <- owner; x, y; card_id <- cardId;
          data_id <- dataGlobalId; hp, max_hp; target <- targetEntityKey's object;
          has_attack, atk_stage, atk_timeline, atk_load, has_move, charge, deploy_remaining
          <- phaseRuntime (attack/movement validated, sequence stage, timeline, load remaining,
          classic charge progress, deploy remaining)
  own     elixir, hand, cycle, next, deck <- the plain observation ("state"); deck_form_flags
          <- the replay's form availability; evo_progress <- evolutionRuntime; abilities
          <- abilityRuntime, the controller named by the deck's hero (the reader names it by
          the hero unit's data id)
  opponent  exact elixir and ability controllers only: what the reader has for them.
Own hand and elixir are as the SCREEN shows them: the client takes a card out of the hand and
its cost off the bar at the tap, the game state ~1 s later when the command executes. Plays the
actor has sent that have not executed are applied to the engine's hand and elixir here, and the
console must do the same with its in-flight taps (see SPEC.md).

    ./py -m il.samples [--frames runs/conv-hog26] [--replays 20]    statistics, nothing saved
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))
import firstlight_obs as FLO          # noqa: E402  (puts the live FirstLight copy on sys.path)
import firstlight_bot as FLB          # noqa: E402

from il.frames import load_replay      # noqa: E402
from il.params import COMMAND_AGE_TICKS, DECISION_TICKS, FIRST_DECISION_TICK, REPLAY_TICK_AFTER_ISSUE, command_delay  # noqa: E402
from il.timeline import decision_tick, timeline_from_json  # noqa: E402

TOWER_IDS = range(5000000, 5000006)
HERO_FLAG, EVOLUTION_FLAG = 0x2, 0x1


def _address(native_object_id: int) -> str:
    """A stand-in heap address, stable for the object's life (the live code keys on addresses)."""
    return hex(0xB400000000000000 | int(native_object_id))


def reader_entities(frame: dict) -> list[dict]:
    entities = []
    for o in frame['objects']:
        native_id = int(o['nativeObjectId'])
        phase = o.get('phaseRuntime') or {}
        effect = o.get('hp') is None
        target = o.get('targetEntityKey')
        stage = phase.get('attackSequenceStage')
        charge = phase.get('classicChargeProgress')
        deploy = phase.get('deployRemainingMs')
        entities.append({
            'address': _address(native_id), 'category': native_id,
            'kind': 0 if effect else (13 if native_id in TOWER_IDS else 15),
            'side': int(o['owner']), 'x': int(o['x']), 'y': int(o['y']), 'card_id': int(o['cardId']),
            'data_id': int(o['dataGlobalId'] or 0) & 0xFFFFFFFF,
            'hp': 0 if effect else int(o['hp']), 'max_hp': 0 if effect else int(o['maxHp'] or 0),
            'target': _address(target[2]) if target else '0x0',
            'has_attack': int(bool(phase.get('attackValidated'))),
            'atk_stage': -1 if stage is None else int(stage),
            'atk_timeline': int(phase.get('attackTimelineMs') or 0),
            'atk_load': int(phase.get('loadRemainingMs') or 0),
            'has_move': int(bool(phase.get('movementValidated'))),
            'charge': -1 if charge is None else int(charge),
            'deploy_remaining': 0 if deploy is None else int(deploy)})
    return entities


def _card_cost(card_id: int) -> float | None:
    from native_runner.training.v4.factory import production_semantic_bundle
    spec = production_semantic_bundle().card_specs.get(int(card_id))
    return float(spec.elixir_cost) if spec is not None and spec.elixir_cost is not None else None


def hero_character(card_id: int) -> int | None:
    """The data id the reader names a hero controller by: the hero form's unit archetype."""
    for form, base in _hero_forms().items():
        if base == int(card_id):
            found = FLO.archetype_by_card().get(form)
            return int(found[0]) & 0xFFFFFFFF if found else None
    return None


_HERO_FORMS: dict[int, int] | None = None


def _hero_forms() -> dict[int, int]:
    global _HERO_FORMS
    if _HERO_FORMS is None:
        _HERO_FORMS = {int(c['hero_form_id']): int(c['card_id']) for c in FLO.CARDS.values()
                       if c.get('hero_form_id')}
    return _HERO_FORMS


_CARD_BY_ABILITY: dict[str, int] | None = None


def reader_abilities(lean_player: dict) -> list[dict]:
    """The reader's controller rows. The engine names each controller's action, which is
    FirstLight's ability id; its card's hero form gives the character id the reader reports.
    A champion's controller (no hero form) gets 0, which the live code skips, as it skips the
    champions the reader reports (no hero form for hero_card_by_character to join)."""
    global _CARD_BY_ABILITY
    if _CARD_BY_ABILITY is None:
        _CARD_BY_ABILITY = {ability_id: card for card, (ability_id, _spec) in FLO.ability_by_card().items()}
    rows = []
    for controller in lean_player.get('abilityRuntime') or ():
        card = _CARD_BY_ABILITY.get(str(controller.get('actionDataName') or ''))
        character = hero_character(card) if card is not None else None
        rows.append({'controller_slot': int(controller['controllerSlot']),
                     'charges': int(controller.get('remainingChargesRaw', -1)),
                     'button': int(controller.get('buttonState', 0)),
                     'cooldown_ms': int(controller.get('remainingCooldownMs') or 0),
                     'configured_ms': int(controller.get('configuredCooldownMs') or 0),
                     'character_id': character or 0})
    return rows


def reader_frame(frame: dict, actor: int, deck_forms: dict[int, list[int]], in_flight: list,
                 strict: bool = True) -> dict:
    """The frame the reader would give at this tick, own hand and elixir as the screen shows."""
    state = {p['owner']: p for p in frame['state']['players']}
    lean = {p['owner']: p for p in frame['players']}
    players = []
    for side in (0, 1):
        plain, rich = state[side], lean.get(side, {})
        deck = [d['cardId'] for d in sorted(plain['deck'], key=lambda d: d['deckSlot'])]
        forms = list(deck_forms[side])
        row = {'side': side, 'address': _address(0x7000000 + side), 'elixir_raw': int(plain['elixirRaw']),
               'refill_timer': 0, 'abilities': reader_abilities(rich)}
        if side == actor:
            hand = [-1, -1, -1, -1]
            for card in plain['hand']:
                hand[int(card['handIndex'])] = int(card['deckSlot'])
            cycle = [int(c['deckSlot']) for c in sorted(plain['cycle'], key=lambda c: c['cycleIndex'])]
            progress = [0] * len(deck)
            for evolution in rich.get('evolutionRuntime') or ():
                progress[int(evolution['deckSlot'])] = int(evolution.get('progress') or 0)
            row.update({'next_deck_index': cycle[0] if cycle else -1, 'hand_deck_indices': hand,
                        'cycle_deck_indices': cycle, 'deck_card_ids': deck, 'deck_form_flags': forms,
                        'evo_progress': progress})
            # as the screen shows it: drawn card in, sent cards out (the console does the same)
            row = FLO.screen_view(row, [p.card_id for p in in_flight if p.kind == 'card'], strict=strict)
        else:
            row.update({'next_deck_index': -1, 'hand_deck_indices': [-1, -1, -1, -1],
                        'cycle_deck_indices': [], 'deck_card_ids': [], 'deck_form_flags': [],
                        'evo_progress': []})
        players.append(row)
    return {'game_tick': int(frame['tick']), 'battle_active': True, 'players': players,
            'entities': reader_entities(frame)}


def executed_plays(timeline, tick: int, form_at) -> list[dict]:
    """The viewer's STATE['plays'] at `tick`: every command that has executed, both sides, dated
    as the viewer dates them, issue + 21 = its execute tick L + 1 (il/params: the engine's hand
    still holds the card in the snapshot at L; the viewer lists the play from L + 1)."""
    from native_runner.arena import cell_to_world
    rows = []
    for p in timeline.plays:
        if p.lands >= tick:
            continue
        x = y = None
        if p.grid is not None:
            x, y = cell_to_world(p.grid)
        rows.append({'tick': p.lands + 1, 'side': p.owner, 'card_id': p.card_id if p.kind == 'card' else 65535,
                     'form_code': form_at(p) if p.kind == 'card' else 0, 'kind': p.kind,
                     'x': x, 'y': y, 'seq': p.index, 'issue_tick': p.lands + 1 - COMMAND_AGE_TICKS})
    return rows


def revealed_cards(timeline, tick: int) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {0: [], 1: []}
    for p in timeline.plays:
        if p.kind == 'card' and p.lands < tick and p.card_id not in out[p.owner]:
            out[p.owner].append(p.card_id)
    return out


def actor_samples(header: dict, frames: list[dict], actor: int, stats: Counter, keep: bool = False,
                  delay: int | None = None, extras: bool = False) -> list:
    """Replay one actor's turns; returns (tick, frame batch, action sequence, label usable)."""
    import torch
    from native_runner.training.v4.expert import (ExpertActionAlignmentError, TimedExpertActionV4,
                                                  build_expert_action_batch, replay_expert_actions)
    from il.params import opponent_seen_tick
    timeline = timeline_from_json(header['timeline'])
    tag = timeline.replay_tag
    # delay None: the bot's (21 + measured overhead); 0 reproduces FirstLight's labels (execute at
    # the decision tick), for comparison
    delay = command_delay(tag, actor) if delay is None else int(delay)
    calibrated = header['calibrated']
    deck_forms = {side: list(timeline.form_availability[side]) for side in (0, 1)}
    own_plays = [p for p in timeline.plays if p.owner == actor]
    # the tap goes out (delay - 20) ticks after the decision plus the play's offset in its
    # window (0-4); the game checks elixir then, so the mask counts elixir as of the latest tap
    send_lead = delay - REPLAY_TICK_AFTER_ISSUE + DECISION_TICKS - 1 if delay >= REPLAY_TICK_AFTER_ISSUE else 0

    # the expert's actions, re-timed: a play landing at L is decided on the grid tick at or
    # before L - delay, and FirstLight's window contract reads its offset (0-4) from the tick
    # given here, so it is L - delay rather than L
    windows: dict[int, list] = {}
    for timed in replay_expert_actions(calibrated.replay):
        if timed.owner != actor:
            continue
        turn = decision_tick(timed.source_tick, delay)
        if turn < FIRST_DECISION_TICK:
            stats['label before first decision tick'] += 1
            continue
        windows.setdefault(turn, []).append(TimedExpertActionV4(
            source_tick=timed.source_tick - delay, action=timed.action, source_index=timed.source_index))

    evolution_ready: dict[int, bool] = {}

    def form_at(play) -> int:
        deck = list(timeline.decks[play.owner])
        flag = int(deck_forms[play.owner][deck.index(play.card_id)]) if play.card_id in deck else 0
        if flag & HERO_FLAG:
            return 2
        return 1 if flag & EVOLUTION_FLAG and evolution_ready.get((play.owner, play.card_id)) else 0

    runner = FLB.FirstLightRunner(None)
    battle = None
    samples = []
    for frame in frames:
        tick = int(frame['tick'])
        for player in frame['players']:
            for evolution in player.get('evolutionRuntime') or ():
                evolution_ready[(player['owner'], int(evolution['cardId']))] = bool(evolution.get('ready'))
        in_flight = sorted((p for p in own_plays if decision_tick(p.lands, delay) < tick <= p.lands),
                           key=lambda p: (p.lands, p.index))
        try:
            raw = reader_frame(frame, actor, deck_forms, in_flight)
        except ValueError:
            stats['screen hand: in-flight card not in the engine hand'] += 1
            return samples
        health = {'local_side': actor, 'visible_sides': [actor]}
        if battle is None:
            observation, battle = FLO.build(raw, health, episode_id=tag)
            me = raw['players'][actor]
            runner.start_battle(me['deck_card_ids'], None, actor, observation,
                                {p['side']: p['elixir_raw'] / 10000.0 for p in raw['players']},
                                our_forms=me['deck_form_flags'])
            runner.adopt_api_deck(timeline.decks[1 - actor])
        executed = executed_plays(timeline, tick, form_at)
        revealed = revealed_cards(timeline, tick)
        runner.register_plays(executed, revealed)
        opponent_player = raw['players'][1 - actor]
        runner.attribute_opponent_abilities(executed, opponent_player)
        seen = {actor: revealed[actor], 1 - actor: [c for c in revealed[1 - actor] if c in runner.opponent_seen]}
        me = raw['players'][actor]
        observation, battle = FLO.build(
            raw, health, episode_id=tag, battle=battle, revealed=seen, reserved=0.0, plays=executed,
            decks=runner.tracked_decks(),
            hand_forms=runner.hand_forms(me['deck_card_ids'], me['deck_form_flags'], me['evo_progress']),
            elixir_lead_ticks=send_lead)
        tensorizer = runner.session.tensorizer
        batch = tensorizer.tensorize(observation, validate=False)
        if tick < FIRST_DECISION_TICK:
            continue
        window = windows.get(tick, [])
        usable = True
        try:
            sequence = build_expert_action_batch(batch, (window,), decision_ticks_by_row=(tick,),
                                                 perspectives=(tensorizer.perspective,),
                                                 config=tensorizer.config, validate=False)
            stats['decision turns'] += 1
            stats['labels aligned'] += len(window)
        except ExpertActionAlignmentError as error:
            usable = False
            stats['decision turns'] += 1
            stats['labels NOT aligned'] += len(window)
            stats[f'not aligned: {_reason(error, window, observation)}'] += 1
            sequence = build_expert_action_batch(batch, ((),), decision_ticks_by_row=(tick,),
                                                 perspectives=(tensorizer.perspective,),
                                                 config=tensorizer.config, validate=False)
        tensorizer.record_action(sequence, batch, row=0, validate=False)
        if keep:
            storage = batch.to_storage('cpu', float_dtype=torch.float16)
            if extras:
                from il.extras import build_extras, extend_batch
                pending = [(p.card_id, form_at(p), 0, p.grid, p.lands + 1 - tick) for p in in_flight if p.kind == 'card']
                pending += [(p.card_id, form_at(p), 1, p.grid, p.lands + 1 - tick) for p in timeline.plays
                            if p.owner != actor and p.kind == 'card' and p.lands >= tick
                            and opponent_seen_tick(tag, p.index, p.lands) <= tick]
                opponent_raw = {q['owner']: q['elixirRaw'] for q in frame['state']['players']}[1 - actor]
                built = build_extras(tensorizer, pending, opponent_elixir=opponent_raw / 10000.0, delay=delay)
                storage = extend_batch(storage, built.to_storage('cpu', float_dtype=torch.float16))
            samples.append((tick, storage, sequence, usable))
    runner.end_battle()
    return samples


HOG26_CARDS = frozenset({26000021, 26000014, 27000000, 26000038, 26000030, 26000010, 28000000, 28000011})


def actor_sequence(header: dict, frames: list[dict], actor: int, stats: Counter, delay: int | None = None,
                   extras: bool = False):
    """FirstLight's ILSequenceV4 for one actor: every decision turn from tick 90, in order.

    Unusable labels (not offered by the live mask at the decision tick) keep their frame but
    leave the gate out of the loss, FirstLight's own rule. The value head is left out
    (value_loss_mask False): its return definition is FirstLight's reward, and RL comes later."""
    import torch
    from native_runner.training.v4.imitation import ILSequenceV4
    samples = actor_samples(header, frames, actor, stats, keep=True, delay=delay, extras=extras)
    if not samples:
        return None
    steps = len(samples)
    return ILSequenceV4(
        observations=tuple(storage for _tick, storage, _sequence, _usable in samples),
        actions=tuple(sequence for _tick, _storage, sequence, _usable in samples),
        episode_start=torch.tensor([[step == 0] for step in range(steps)], dtype=torch.bool),
        valid_mask=torch.ones(steps, 1, dtype=torch.bool),
        returns=torch.zeros(steps, 1, dtype=torch.float32),
        gate_loss_mask=torch.tensor([[usable] for _t, _s, _q, usable in samples], dtype=torch.bool),
        value_loss_mask=torch.zeros(steps, 1, dtype=torch.bool),
        sequence_id=f"{header['replay_tag']}:owner-{actor}")


def _reason(error, window, observation) -> str:
    text = str(error)
    mask = observation.action_mask
    for timed in window:
        action = timed.action
        if action.kind.value == 'play_card':
            slot = action.hand_slot
            reason = mask.reasons.get(str(slot)) if slot is not None else None
            if reason and reason != 'legal':
                return f'card slot {slot}: {reason}'
    if 'matched 0 legal' in text:
        return 'no legal candidate matched'
    return text.split(';')[0][:80]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=Path, default=CLAPHA / 'runs/conv-hog26')
    parser.add_argument('--replays', type=int, default=20)
    args = parser.parse_args(argv)
    stats: Counter = Counter()
    paths = sorted((args.frames / 'frames').glob('*/*.jsonl.zst'))[:args.replays]
    started = time.perf_counter()
    for path in paths:
        header, frames = load_replay(path)
        if header['fidelity']['tower_hp_error'] > 100 or header['ended_at'] is None:
            stats['replays skipped (fidelity)'] += 1
            continue
        for actor in (0, 1):
            t0 = time.perf_counter()
            actor_samples(header, frames, actor, stats)
            stats['actors'] += 1
            stats['seconds'] += time.perf_counter() - t0
    elapsed = time.perf_counter() - started
    for key, value in sorted(stats.items()):
        print(f'  {key}: {value:.1f}' if isinstance(value, float) else f'  {key}: {value}')
    print(f'{elapsed:.1f} s for {len(paths)} replays')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
