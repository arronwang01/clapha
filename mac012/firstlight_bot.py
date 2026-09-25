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

    def start_battle(self, our_deck, opponent_deck, actor_owner: int,
                     observation, initial_elixir, our_forms=None,
                     opponent_forms=None) -> None:
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

    def hand_forms(self, deck, form_flags) -> dict[int, int]:
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
        for card_id, flag in zip(deck, form_flags):
            flag = int(flag or 0)
            if flag & 0x2:
                forms[int(card_id)] = 2
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
        moves = [(a.kind, a.hand_slot, a.card_id, a.target_grid,
                  int(getattr(a, 'execute_offset_ticks', 0) or 0)) for a in actions]
        moves.sort(key=lambda move: move[4])
        return moves

    def end_battle(self) -> None:
        if self.session is not None:
            self.session.end_episode()
        self.session = None
