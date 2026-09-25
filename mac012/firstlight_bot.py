"""Run a FirstLight V4 policy from our live reader, for the console.

No injected probe and no BattleEnv: the episode tensorizer is built from a hand-made
EpisodeConfigV1, and observations come from firstlight_obs.build(). Entity archetypes are
resolved offline from FirstLight's own shipped catalogs, which they have to be:
build_episode_tensorizer_v4 sets reject_unknown_public_semantics=True, so an entity it cannot
place raises rather than degrading.

The decision schedule is theirs, not ours (native_runner/training/v4/expert.py and
serve_policy.py):

  * turns sit on a fixed five-tick grid (POLICY_DECISION_TICKS = 5, so every tick divisible
    by five, since their first decision tick is 90);
  * ticks before FIRST_POLICY_DECISION_TICK = 90 are warm-up: the tensorizer is fed with
    observe() and no action is taken, exactly as PolicyService does;
  * once decisions start, every turn must be taken. decide() calls record_action internally
    and advances a recurrent state -- "Training feeds the selected semantic action into the
    next decision. Offline inference must do exactly the same on every five-tick turn."
    Skipping turns desynchronises that state from everything the policy learned;
  * a turn may return up to ModelConfigV4.max_micro_actions (2) actions, each with its own
    execute_offset_ticks inside the window. Both are real plays.

Actions are SAMPLED, not argmaxed. Every shipped checkpoint is a stochastic-rollout policy --
four are 'ppo-league-snapshot' carrying ppo_gate_temperature 0.2 and one is 'imitation' at
temperature 1.0 -- and load_policy_v4 restores those temperatures onto the model, where only
the sampling path reads them (model.py: gate_temperature=self.ppo_gate_temperature if sample
else 1.0).

This matters more than any other setting here. The act/wait gate is a distribution, and a
correct Clash Royale policy waits on roughly 95% of its turns: ~35 plays over ~700 decision
turns is the whole elixir budget of a three-minute match. Taking the argmax of a head that puts
95% on WAIT yields WAIT on every turn except the rare state where acting is outright more
likely than waiting -- so a greedy policy does almost literally nothing, which is exactly what
it did. Measured on a fixed board, the greedy gate opened only at ten elixir against eight
attackers with our towers at 20% health.

PolicySessionV4 defaults to sample=False, and evaluate.py defaults the same way, but both take
--sample; for an imitation policy argmax is especially wrong, since imitation learns the
expert's distribution and the expert waits most turns.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

FIRSTLIGHT = Path(os.environ.get('FIRSTLIGHT_ROOT')
                  or Path.home() / 'Documents/GitHub/FirstLight_CR')
if str(FIRSTLIGHT) not in sys.path:
    sys.path.insert(0, str(FIRSTLIGHT))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

CHECKPOINTS = {
    'fl:general': FIRSTLIGHT / 'checkpoints/General/checkpoint-step-00000460.pt',
    'fl:hog1': FIRSTLIGHT / 'checkpoints/2_6hog_expert/hog26-specialist1.pt',
    'fl:hog2': FIRSTLIGHT / 'checkpoints/2_6hog_expert/hog26-specialist2.pt',
    'fl:il': FIRSTLIGHT / 'checkpoints/IL/checkpoint-step-00029396.pt',
    'fl:active-il': FIRSTLIGHT / 'checkpoints/active IL/checkpoint-step-00000030.pt',
}


def FLO_ruleset_id() -> str:
    from firstlight_obs import ruleset_id
    return ruleset_id()


def available() -> list[str]:
    return [name for name, path in CHECKPOINTS.items() if path.is_file()]


class Move(tuple):
    """(kind, hand_slot, card_id, target_grid, offset_ticks), plus source_entity / ability_id
    for an ability activation -- the hero unit whose controller's button must be tapped."""
    source_entity = None
    ability_id = None


class FirstLightRunner:
    """One policy bound to one battle. Rebuild it when the battle changes."""

    def __init__(self, model: str, device: str = 'cpu', sample: bool = True):
        from native_runner.training.v4.policy_session import load_policy_v4
        from native_runner.training.v4.expert import (FIRST_POLICY_DECISION_TICK,
                                                      POLICY_DECISION_TICKS)
        self.model_name = model
        self.device = device
        self.decision_ticks = POLICY_DECISION_TICKS
        self.first_decision_tick = FIRST_POLICY_DECISION_TICK
        loaded = load_policy_v4(CHECKPOINTS[model], device=device)
        self.model = getattr(loaded, 'model', loaded)
        self.sample = bool(sample)
        self.session = None
        self.actor_owner = None
        self.opponent_source = 'unknown'
        self.opponent_seen: list[int] = []      # distinct opponent cards, first-seen order
        self.refused: list[tuple[int, int, str]] = []   # (side, card, reason), for the console
        self._processed: set = set()            # plays / reveals already registered

    def start_battle(self, our_deck, opponent_deck, actor_owner: int,
                     observation, initial_elixir, our_forms=None,
                     opponent_forms=None) -> None:
        """opponent_deck None means unknown: the episode is built on a stand-in, and the
        opponent's tracker deck starts EMPTY and is learned card by card (register_plays)."""
        known = opponent_deck is not None and len(opponent_deck) == 8
        if not known:
            opponent_deck, opponent_forms = our_deck, None
        from native_runner.contracts import EpisodeConfigV1, ObservationTier
        from native_runner.training.v4.factory import build_episode_tensorizer_v4
        from native_runner.training.v4.policy_session import PolicySessionV4
        deck0, deck1 = ((tuple(our_deck), tuple(opponent_deck)) if actor_owner == 0
                        else (tuple(opponent_deck), tuple(our_deck)))
        # deck<n>_form_availability is the tag build_episode_tensorizer_v4 reads to derive
        # deck roles and to seed the tracker's evolution cycles. Left out, it defaults to
        # all zeros -- "this deck has no evolutions and no hero" -- which is a false statement
        # about any deck that runs one. Our reader does expose per-slot form flags.
        forms0, forms1 = ((our_forms, opponent_forms) if actor_owner == 0
                          else (opponent_forms, our_forms))
        tags = {}
        for owner, flags in ((0, forms0), (1, forms1)):
            if flags and len(flags) == 8:
                tags[f'deck{owner}_form_availability'] = tuple(int(f) for f in flags)
        episode = EpisodeConfigV1(
            ruleset_id=FLO_ruleset_id(), deck0=deck0, deck1=deck1, seed=1,
            observation_tier=ObservationTier.FAIR,
            decision_hz=20.0 / self.decision_ticks,
            **({'tags': tags} if tags else {}))
        tensorizer = build_episode_tensorizer_v4(episode, actor_owner=actor_owner)
        self.session = PolicySessionV4(self.model, tensorizer, device=self.device,
                                       sample=self.sample)
        self.session.start_episode(observation, initial_elixir=initial_elixir)
        self.actor_owner = actor_owner
        self.opponent_seen, self.refused, self._processed = [], [], set()
        self.opponent_source = 'published' if known else 'learned'
        if not known:
            self._forget_opponent_deck()

    def _tracker(self):
        return getattr(getattr(self.session, 'tensorizer', None), 'tracker', None)

    def _forget_opponent_deck(self) -> None:
        """Empty the opponent's side of the tracker: the stand-in deck is not theirs.

        FirstLight's tracker is built for an engine that knows both decks. Left holding the
        stand-in, it would refuse every opponent card outside it -- which is how Training Camp
        and any friendly without the other console lost most of the opponent's plays. Its
        state is plain per-owner tables, so the opponent's can start empty and grow.
        """
        tracker, opponent = self._tracker(), 1 - self.actor_owner
        tracker.decks[opponent] = ()
        tracker._card_states[opponent] = {}
        tracker.evolution_cycle_required_by_owner[opponent] = {}
        for table in (tracker.ability_cost_by_owner_card,
                      tracker.ability_cooldown_ticks_by_owner_card,
                      tracker.ability_max_charges_by_owner_card):
            table[opponent] = {}

    def _register(self, owner: int, card_id: int) -> str | None:
        """Make the tracker able to take a play of card_id by owner. None, or why not."""
        from native_runner.training.tracking import AvailabilityIndex, PublicCardState
        from firstlight_obs import known_cards, MIRROR_CARD_ID
        from native_runner.training.v4.factory import production_semantic_bundle
        tracker = self._tracker()
        if card_id in tracker._card_states[owner]:
            return None
        if owner == self.actor_owner:
            return 'not in our own deck'
        if card_id == MIRROR_CARD_ID:
            return 'Mirror plays are not modelled yet'
        spec = production_semantic_bundle().card_specs.get(card_id)
        if spec is None or spec.elixir_cost is None or card_id not in known_cards():
            return "not in FirstLight's card catalog (newer than 15.535?)"
        if card_id not in self.opponent_seen and len(self.opponent_seen) >= 8:
            return (f'a ninth distinct card (already seen: '
                    f'{", ".join(map(str, self.opponent_seen))})')
        tracker.decks[owner] = tuple(tracker.decks[owner]) + (card_id,)
        tracker.card_costs.setdefault(card_id, float(spec.elixir_cost))
        tracker._card_states[owner][card_id] = PublicCardState(
            availability=AvailabilityIndex.UNKNOWN_INITIAL, cycle_distance=0,
            initial_state_known=False, evolution_cycle_required=0,
            evolution_cycle_remaining=0, evolution_ready=False)
        return None

    def _see(self, owner: int, card_id: int) -> str | None:
        """Register one card the opponent has shown; remember it in first-seen order."""
        reason = self._register(owner, card_id)
        if reason is None and owner != self.actor_owner and card_id not in self.opponent_seen:
            self.opponent_seen.append(card_id)
        return reason

    def register_plays(self, plays, revealed=None) -> list[tuple[int, int, str]]:
        """Register every card either player has shown, before the observation is built.

        Returns what could not be registered this call, as (side, card, reason), so the
        console can say so every time. Nothing is dropped silently any more.
        """
        if self.session is None:
            return []
        refused = []
        opponent = 1 - self.actor_owner
        for play in plays or ():
            if play.get('kind', 'card') != 'card':
                continue
            side, card_id = int(play['side']), int(play['card_id'])
            key = ('play', side, play.get('issue_tick'), play.get('seq'), card_id, play['tick'])
            if key in self._processed:
                continue
            self._processed.add(key)
            reason = self._see(side, card_id) if side == opponent else (
                self._register(side, card_id))
            if reason is not None:
                refused.append((side, card_id, reason))
        for card_id in (revealed or {}).get(opponent, ()):
            key = ('reveal', opponent, int(card_id))
            if key in self._processed:
                continue
            self._processed.add(key)
            reason = self._see(opponent, int(card_id))
            if reason is not None:
                refused.append((opponent, int(card_id), reason))
        self.refused.extend(refused)
        return refused

    def tracked_decks(self) -> dict[int, tuple[int, ...]]:
        """The cards the tracker can take a play of, per side (see play_events)."""
        tracker = self._tracker()
        if tracker is None:
            return {}
        return {owner: tuple(tracker._card_states[owner]) for owner in (0, 1)}

    def hand_forms(self, deck, form_flags, evo_progress=None) -> dict[int, int]:
        """card id -> the form our hand holds it in: 0 normal, 1 evolution, 2 hero.

        FirstLight's probe reads this per hand card (card_parameter & 0xF). We cannot call the
        game's selection builder from outside the process, so it is derived instead, from the
        same rules their BattleEnv applies: a hero-enabled deck slot (form flag bit 0x2) is
        always played as its hero form, and an evolution-enabled slot (bit 0x1) is evolved
        exactly when their own tracker, fed our executed plays, has counted its evolution
        cycles down. Sending 0 for everything told the policy an Evo Cannon was a plain Cannon
        and a Hero Musketeer a plain Musketeer, in the deck the Hog specialists trained on.
        """
        forms: dict[int, int] = {}
        if not deck or not form_flags or len(form_flags) != len(deck):
            return forms
        tracker = getattr(getattr(self.session, 'tensorizer', None), 'tracker', None)
        # The game's own per-deck-slot evolution progress (player+0x2e8), when the reader has it:
        # a slot is evolved exactly when progress has reached FirstLight's cycles required --
        # their probe's definition. The tracker count below is only the fallback.
        memory = list(evo_progress) if evo_progress and len(evo_progress) == len(deck) else None
        if memory is not None:
            from native_runner.training.v4.factory import production_semantic_bundle
            specs = production_semantic_bundle().card_specs
        for slot, (card_id, flag) in enumerate(zip(deck, form_flags)):
            flag = int(flag or 0)
            if flag & 0x2:
                forms[int(card_id)] = 2
            elif flag & 0x1 and memory is not None:
                spec = specs.get(int(card_id))
                evolution = getattr(spec, 'evolution', None) if spec is not None else None
                required = int(evolution.cycle_required) if evolution is not None else None
                forms[int(card_id)] = 1 if required and int(memory[slot]) >= required else 0
            elif flag & 0x1 and tracker is not None:
                try:
                    ready = tracker.card_state(self.actor_owner, int(card_id)).evolution_ready
                except (KeyError, RuntimeError, ValueError):
                    ready = False
                forms[int(card_id)] = 1 if ready else 0
        return forms

    def turn_tick(self, tick: int) -> int:
        """The five-tick decision turn this tick belongs to."""
        return int(tick) - int(tick) % self.decision_ticks

    def observe(self, observation) -> None:
        """Feed a warm-up turn without acting, as PolicyService's 'observe' op does."""
        if self.session is not None:
            self.session.tensorizer.tensorize(observation, validate=False)

    def decide(self, observation):
        """-> list of (kind, hand_slot, card_id, target_grid, offset_ticks), in play order.

        Every decoded action is returned. A turn can carry two, and dropping the second
        throws away a real play the policy asked for.
        """
        if self.session is None:
            return []
        decision = self.session.decide(observation)
        actions = getattr(getattr(decision, 'decoded', None), 'actions', ()) or ()
        moves = []
        for a in actions:
            move = Move((a.kind, a.hand_slot, a.card_id, a.target_grid,
                         int(getattr(a, 'execute_offset_ticks', 0) or 0)))
            move.source_entity = getattr(a, 'source_entity', None)
            move.ability_id = getattr(a, 'ability_id', None)
            moves.append(move)
        moves.sort(key=lambda move: move[4])
        return moves

    def end_battle(self) -> None:
        if self.session is not None:
            self.session.end_episode()
        self.session = None
