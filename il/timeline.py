"""Imitation samples from RoyaleAPI replays, with the decision timing of the live game.

FirstLight's prepare_collected_replay does the bookkeeping (card keys, forms, tower troops, a
legal starting deal); this module turns its action trace into what an actor knew, and what it
did, at each decision tick. Board state (units, towers, elixir) is not here: it comes from the
engine at the same tick, which is causal by construction.

One function, sample_at(), builds every input for a decision tick. Training, the audit and the
live console use it, so what the model sees in training is what it sees live.

Inputs at decision tick t for actor a, each stamped with the tick it became known:
  own_hand       the hand as a's screen shows it: the deal, minus every play a had already
                 issued (decided in an earlier window), each replaced by its draw
  own_pending    a's plays issued but not executed (decided < t <= lands): card, tile, ticks left
  opp_pending    opponent commands visible at t (lands - lead <= t <= lands)
  opp_history    opponent plays known at t, in order (landed, or visible as pending)
  opp_hand       deduced from the opponent's deck and history: exact once four plays are known
                 (the last four played sit in the queue; the other four are the hand)
  command_delay  ticks from decision to landing for this actor
Labels at t: a's plays whose decision tick is t, with their offset (0-4) inside the window.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from il.params import DECISION_TICKS, FIRST_DECISION_TICK, command_delay, opponent_lead

MIRROR_CARD_ID = 28000006


@dataclass(frozen=True)
class Play:
    owner: int
    kind: str                     # 'card' | 'ability'
    card_id: int | None           # the card played; None for an ability
    grid: tuple[int, int] | None
    lands: int                    # execution boundary (source_command_tick)
    index: int                    # source_index: stable id within the replay
    ability_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class Timeline:
    replay_tag: str
    decks: tuple[tuple[int, ...], tuple[int, ...]]
    form_availability: tuple[tuple[int, ...], tuple[int, ...]]
    tower_troops: tuple[int | None, int | None]
    deals: dict[int, tuple[tuple[int, ...], tuple[int, ...]]]   # owner -> (opening, queue)
    first_certain_play: dict[int, int]                          # owner -> play count
    plays: tuple[Play, ...]
    end_tick: int
    winner: int | None
    omit_from_opening: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())


@dataclass
class Sample:
    replay_tag: str
    actor: int
    tick: int
    command_delay: int
    labels: list[dict] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    known_at: dict[str, int] = field(default_factory=dict)      # input -> latest source tick
    hand_certain: bool = True


def build_timeline(prepared) -> Timeline:
    """From FirstLight's PreparedCollectedReplay."""
    replay = prepared.replay
    config = replay.episode_config
    tags = dict(config.tags)
    plays = []
    for operation in replay.operations:
        for action in operation.actions:
            kind = str(getattr(action.kind, 'value', action.kind))
            if kind not in ('play_card', 'activate_ability'):
                continue
            meta = dict(action.metadata)
            plays.append(Play(
                owner=int(action.owner),
                kind='card' if kind == 'play_card' else 'ability',
                card_id=int(action.card_id) if action.card_id is not None else None,
                grid=tuple(action.target_grid) if action.target_grid is not None else None,
                lands=int(meta['source_command_tick']),
                index=int(meta.get('source_index', len(plays))),
                ability_keys=tuple(meta.get('ability_source_keys') or ())))
    plays.sort(key=lambda p: (p.lands, p.index))
    first_certain = {int(b['owner']): int(b['first_invariant_play_count'])
                     for b in tags.get('policy_deal_invariance', ())}
    terminal = dict(replay.terminal or {})
    return Timeline(
        replay_tag=prepared.replay_tag,
        decks=(tuple(config.deck0), tuple(config.deck1)),
        form_availability=(tuple(tags.get('deck0_form_availability', ())),
                           tuple(tags.get('deck1_form_availability', ()))),
        tower_troops=(tags.get('tower_troop0_id'), tags.get('tower_troop1_id')),
        deals={owner: (tuple(prepared.opening_cards[owner]), tuple(prepared.queue_cards[owner]))
               for owner in (0, 1)},
        first_certain_play=first_certain,
        plays=tuple(plays),
        end_tick=int(tags.get('end_tick') or terminal.get('native_tick') or 0),
        winner=terminal.get('winner'),
        omit_from_opening=(tuple(tags.get('deck0_omit_from_starting_hand_ids', ())),
                           tuple(tags.get('deck1_omit_from_starting_hand_ids', ()))))


def decision_tick(lands: int, delay: int) -> int:
    """The grid tick on which the bot must decide a play that lands at `lands`."""
    wanted = lands - delay
    return FIRST_DECISION_TICK + ((wanted - FIRST_DECISION_TICK) // DECISION_TICKS) * DECISION_TICKS


def restricted(timeline: Timeline, actor: int, tick: int) -> Timeline:
    """Only what `actor` could know at `tick`: its own deal and the plays it had already
    decided, and the opponent commands visible by then. No opponent deal. The audit builds
    every sample from this too; any input that differs from the full-replay sample leaks."""
    delay = command_delay(timeline.replay_tag, actor)
    known = tuple(p for p in timeline.plays
                  if (p.owner == actor and decision_tick(p.lands, delay) < tick)
                  or (p.owner != actor and p.lands - opponent_lead(timeline.replay_tag, p.index) <= tick))
    return replace(timeline, plays=known, deals={actor: timeline.deals[actor]},
                   first_certain_play={}, winner=None, end_tick=0)


def _hand_after(deal: tuple[tuple[int, ...], tuple[int, ...]], cards: list[int]) -> tuple[list[int], list[int]]:
    """Hand (4 slots, in slot order) and queue after playing `cards` from `deal`."""
    hand, queue = list(deal[0]), list(deal[1])
    for card in cards:
        if card not in hand:
            raise ValueError(f'card {card} played but not in hand {hand}')
        slot = hand.index(card)
        hand[slot] = queue.pop(0)
        queue.append(card)
    return hand, queue


def opponent_hand(deck: tuple[int, ...], history: list[int], omit: tuple[int, ...] = ()) -> dict[str, Any]:
    """What the opponent's hand can be, from their deck and the plays seen so far."""
    if MIRROR_CARD_ID in deck:
        return {'exact': False, 'candidates': sorted(deck), 'reason': 'mirror'}
    in_queue = history[-4:]
    if len(history) >= 4:
        return {'exact': True, 'hand': sorted(set(deck) - set(in_queue)), 'queue': list(in_queue)}
    candidates = set(deck) - set(in_queue)
    if not history:
        candidates -= set(omit)    # a card barred from the opening hand is not in it yet
    return {'exact': False, 'candidates': sorted(candidates), 'queue_tail': list(in_queue)}


def sample_at(timeline: Timeline, actor: int, tick: int) -> Sample:
    """Everything the actor had at decision tick `tick`, and what it did (labels)."""
    tag = timeline.replay_tag
    delay = command_delay(tag, actor)
    opponent = 1 - actor
    sample = Sample(replay_tag=tag, actor=actor, tick=tick, command_delay=delay)

    own = [p for p in timeline.plays if p.owner == actor]
    issued = [p for p in own if decision_tick(p.lands, delay) < tick]
    window = [p for p in own if decision_tick(p.lands, delay) == tick]
    issued_cards = [p.card_id for p in issued if p.kind == 'card']
    hand, queue = _hand_after(timeline.deals[actor], issued_cards)
    sample.inputs['own_hand'] = hand
    sample.inputs['own_next'] = queue[0]
    sample.known_at['own_hand'] = max([0] + [decision_tick(p.lands, delay) for p in issued])
    # After four plays the queue is exactly those four cards, so the hand is the rest of the deck
    # whatever the opening was. Before that the replay does not say which of the unplayed cards
    # were in hand: the deal is one random choice that fits the plays (FirstLight's own
    # certainty flag only ever considers that one choice, so it is always "certain").
    sample.hand_certain = len(issued_cards) >= 4

    pending = [p for p in issued if p.lands >= tick]
    sample.inputs['own_pending'] = [(p.kind, p.card_id, p.grid, p.lands - tick) for p in pending]
    sample.known_at['own_pending'] = max([0] + [decision_tick(p.lands, delay) for p in pending])

    theirs = [p for p in timeline.plays if p.owner == opponent]
    visible = [(p, p.lands - opponent_lead(tag, p.index)) for p in theirs]
    visible = [(p, seen) for p, seen in visible if seen <= tick]
    opp_pending = [(p, seen) for p, seen in visible if p.lands >= tick]
    sample.inputs['opp_pending'] = [(p.kind, p.card_id, p.grid, p.lands - tick) for p, _ in opp_pending]
    sample.known_at['opp_pending'] = max([0] + [seen for _, seen in opp_pending])
    history = [p.card_id for p, _ in visible if p.kind == 'card']
    sample.inputs['opp_history'] = history
    sample.known_at['opp_history'] = max([0] + [seen for _, seen in visible])
    sample.inputs['opp_deck'] = sorted(timeline.decks[opponent])
    sample.inputs['opp_hand'] = opponent_hand(timeline.decks[opponent], history,
                                              timeline.omit_from_opening[opponent])
    sample.known_at['opp_hand'] = sample.known_at['opp_history']
    sample.inputs['tower_troops'] = timeline.tower_troops
    sample.inputs['command_delay'] = delay

    for p in window:
        sample.labels.append({'kind': p.kind, 'card_id': p.card_id, 'grid': p.grid,
                              'offset': p.lands - delay - tick, 'lands': p.lands,
                              'ability_keys': p.ability_keys})
    return sample


def decision_ticks(timeline: Timeline) -> range:
    end = timeline.end_tick or (timeline.plays[-1].lands if timeline.plays else FIRST_DECISION_TICK)
    return range(FIRST_DECISION_TICK, end + 1, DECISION_TICKS)
