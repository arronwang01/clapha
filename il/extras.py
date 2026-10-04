"""Stage B inputs: what FirstLight's observation lacks, as a separate record and a gated head.

Inputs (one row per decision turn, the actor's perspective):
  pending   up to MAX_PENDING commands sent and not executed yet, both sides: all of the actor's
            own, and the opponent's that are visible in the actor's queue. Per command: the card
            (FirstLight card vocabulary), its form (0 normal, 1 evolution, 2 hero), owner (0 the
            actor, 1 the opponent), target tile in the actor's perspective, ticks until it executes.
            The screen already shows the actor's own sent cards gone from the hand and elixir
            (firstlight_obs.screen_view); this says where and when they land, and what the
            opponent has queued. It is the input FirstLight never had ("did it see its own troop").
            v2: also the ticks until it ARRIVES: a thrown spell or a tunnelling unit lands after a
            flight (il/flight.py), so a pending Goblin Barrel's goblins come ~1 s after it executes.
  abilities up to MAX_ABILITIES hero or champion controllers, both players (v2): the card (a hero's
            base card in form 2, a champion's own card), owner, and the controller's state as the
            game holds it -- phase (FirstLight's button enum), cooldown left, charges, and the ticks
            since it was last activated (AbilityClock). FirstLight's observation has only the
            actor's own heroes, and no champions at all.
  scalars   the opponent's exact elixir (the reader has it; FirstLight's FAIR observation only has
            the tracker's estimate), and the actor's command delay (decision -> replay tick).

The model: ExtendedPolicyV4 is FirstLight's UniversalCardPolicyV4 with one more summary. Pending
commands go through the model's own card encoder (a pending Hog Rider starts out meaning Hog
Rider), get owner, place and time, and are pooled by attention with the scene as the query; the
scalars through a small MLP. The summary enters the recurrent core's input and the policy context
through two linear gates initialised to zero, so an extended checkpoint computes exactly what its
base does until training moves the gates. The head is kept out of the model's state_dict: an
extended checkpoint is a strict FirstLight checkpoint (their tools and our console load it) with
the head in its `extra` payload; load_policy() here attaches it. A v1 head loads as v2 with the new
inputs switched off (zero weights), computing exactly what it did.

Built by build_extras() from plain lists, so training (il/samples.py: replay timeline + engine
snapshot) and live (console: in-flight taps + the reader's queue and opponent elixir) share it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch import Tensor, nn

from native_runner.training.v4.model import EncodedObservationV4, UniversalCardPolicyV4
from native_runner.training.v4.tensors import TensorRecordV4, UniversalSemanticBatchV4

MAX_PENDING = 8
MAX_ABILITIES = 4        # two controllers per player
SCALARS = 2              # opponent elixir / 10, command delay / 40
WHERE = 4                # tile x, tile y, ticks to execute, ticks to arrival
EXTRAS_VERSION = 'clapha-extras.v2'
EXTRAS_V1 = 'clapha-extras.v1'

# FirstLight's ability button enum -> phase (firstlight_obs._ABILITY_PHASE_BY_BUTTON, the same
# table); anything else is 'unknown'. 2 and 4 are the queueable (ready) states.
BUTTON_PHASE = {1: 'unavailable', 2: 'ready', 4: 'ready', 6: 'exhausted', 8: 'cooldown', 9: 'unavailable',
                10: 'casting', 11: 'unavailable', 12: 'unavailable', 13: 'unavailable'}
PHASES = ('unavailable', 'ready', 'casting', 'cooldown', 'exhausted', 'unknown')
ABILITY_STATE = len(PHASES) + 6
NEVER_ACTIVATED_TICKS = 400      # "since activation" saturates here (20 s)


@dataclass(slots=True)
class ExtrasV2(TensorRecordV4):
    pending_card: Tensor     # [B, K] long, FirstLight card vocab id (0 = none)
    pending_form: Tensor     # [B, K] long
    pending_owner: Tensor    # [B, K] long, 0 the actor / 1 the opponent
    pending_where: Tensor    # [B, K, WHERE] float: tile x / 17, tile y / 31 (actor's view), ticks to execute / 40,
    #                          ticks to arrival / 40
    pending_mask: Tensor     # [B, K] bool
    ability_card: Tensor     # [B, A] long: a hero's base card, a champion's card (vocab id)
    ability_form: Tensor     # [B, A] long: 2 hero, 0 champion
    ability_owner: Tensor    # [B, A] long
    ability_state: Tensor    # [B, A, ABILITY_STATE] float (ability_features)
    ability_mask: Tensor     # [B, A] bool
    scalars: Tensor          # [B, SCALARS] float


@dataclass(slots=True)
class ExtendedBatchV4(UniversalSemanticBatchV4):
    extras: ExtrasV2


@dataclass(slots=True)
class ExtendedEncodedV4(EncodedObservationV4):
    extras_summary: Tensor


def extend_batch(batch: UniversalSemanticBatchV4, extras: ExtrasV2) -> ExtendedBatchV4:
    return ExtendedBatchV4(**{item.name: getattr(batch, item.name) for item in fields(UniversalSemanticBatchV4)},
                           extras=extras)


def ability_features(button: int, remaining_ms: int, configured_ms: int, charges_raw: int,
                     ticks_since: int | None) -> list[float]:
    """One controller's state as the model reads it: phase one-hot (PHASES), cooldown left as a
    fraction and in 20 s units, charges (/5; -1 = unlimited, flagged), ticks since its last
    activation (/NEVER_ACTIVATED_TICKS, saturating; flagged when never seen activated)."""
    phase = [0.0] * len(PHASES)
    phase[PHASES.index(BUTTON_PHASE.get(int(button), 'unknown'))] = 1.0
    remaining = max(0, int(remaining_ms or 0))
    configured = max(0, int(configured_ms or 0))
    unlimited = int(charges_raw) == -1
    since = NEVER_ACTIVATED_TICKS if ticks_since is None else min(NEVER_ACTIVATED_TICKS, max(0, int(ticks_since)))
    return phase + [remaining / configured if configured else 0.0, min(1.5, remaining / 20000.0),
                    0.0 if unlimited else max(0, int(charges_raw)) / 5.0, float(unlimited),
                    since / NEVER_ACTIVATED_TICKS, float(ticks_since is None)]


def build_extras(tensorizer, pending, *, opponent_elixir: float, delay: int, abilities=()) -> ExtrasV2:
    """pending: (card_id, form, owner relative to the actor (0/1), native tile (x, y), ticks to
    execute[, ticks to arrival]), soonest first; more than MAX_PENDING keeps the soonest. Without
    an arrival it is the execute tick (a troop is on the board when its command executes).
    abilities: (card_id, form, owner relative to the actor, button, remaining_ms, configured_ms,
    charges_raw, ticks since activation or None), the actor's first; a card FirstLight's
    vocabulary lacks is left out."""
    rows = sorted(pending, key=lambda item: item[4])[:MAX_PENDING]
    card = torch.zeros(1, MAX_PENDING, dtype=torch.long)
    form = torch.zeros(1, MAX_PENDING, dtype=torch.long)
    owner = torch.zeros(1, MAX_PENDING, dtype=torch.long)
    where = torch.zeros(1, MAX_PENDING, WHERE, dtype=torch.float32)
    mask = torch.zeros(1, MAX_PENDING, dtype=torch.bool)
    for index, row in enumerate(rows):
        card_id, card_form, relative_owner, tile, ticks = row[:5]
        arrival = row[5] if len(row) > 5 and row[5] is not None else ticks
        card[0, index] = int(tensorizer.catalog.vocab_id(int(card_id)))
        form[0, index] = int(card_form or 0)
        owner[0, index] = int(relative_owner)
        if tile is not None:
            x, y = tensorizer.perspective._grid(tile)
            where[0, index, 0], where[0, index, 1] = x / 17.0, y / 31.0
        where[0, index, 2] = max(0, int(ticks)) / 40.0
        where[0, index, 3] = max(0, int(arrival)) / 40.0
        mask[0, index] = True
    a_card = torch.zeros(1, MAX_ABILITIES, dtype=torch.long)
    a_form = torch.zeros(1, MAX_ABILITIES, dtype=torch.long)
    a_owner = torch.zeros(1, MAX_ABILITIES, dtype=torch.long)
    a_state = torch.zeros(1, MAX_ABILITIES, ABILITY_STATE, dtype=torch.float32)
    a_mask = torch.zeros(1, MAX_ABILITIES, dtype=torch.bool)
    index = 0
    for card_id, card_form, relative_owner, button, remaining, configured, charges, since in abilities:
        if index == MAX_ABILITIES:
            break
        try:
            vocab = int(tensorizer.catalog.vocab_id(int(card_id)))
        except Exception:  # noqa: BLE001  (a card FirstLight never had)
            continue
        a_card[0, index], a_form[0, index], a_owner[0, index] = vocab, int(card_form or 0), int(relative_owner)
        a_state[0, index] = torch.tensor(ability_features(button, remaining, configured, charges, since))
        a_mask[0, index] = True
        index += 1
    scalars = torch.tensor([[float(opponent_elixir) / 10.0, float(delay) / 40.0]], dtype=torch.float32)
    return ExtrasV2(pending_card=card, pending_form=form, pending_owner=owner, pending_where=where,
                    pending_mask=mask, ability_card=a_card, ability_form=a_form, ability_owner=a_owner,
                    ability_state=a_state, ability_mask=a_mask, scalars=scalars)


class AbilityClock:
    """Ticks since each controller was last activated, from its state at the decision turns (the
    same granularity live, in il.duel and in training). Activated = it left a ready state, its
    charges dropped, or its cooldown restarted while cooling down (a cast's own cooldown starting
    after it is the same activation)."""

    def __init__(self) -> None:
        self.previous: dict = {}
        self.activated: dict = {}

    def update(self, key, tick: int, button: int, remaining_ms: int, charges_raw: int) -> int | None:
        before = self.previous.get(key)
        if before is not None:
            was = BUTTON_PHASE.get(before[0])
            if ((was == 'ready' and BUTTON_PHASE.get(int(button)) != 'ready')
                    or (before[2] >= 0 and 0 <= int(charges_raw) < before[2])
                    or (was == 'cooldown' and int(remaining_ms or 0) > before[1] + 500)):
                self.activated[key] = tick
        self.previous[key] = (int(button), int(remaining_ms or 0), int(charges_raw))
        return tick - self.activated[key] if key in self.activated else None


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class ExtrasHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        width = config.token_dim
        self.card = nn.Linear(config.card_semantic_dim, width)
        self.owner = nn.Embedding(2, width)
        self.where = _mlp(WHERE, 64, width)
        self.null = nn.Parameter(torch.zeros(1, 1, width))      # always attendable: empty queues pool to it
        self.token_norm = nn.LayerNorm(width)
        self.query = nn.Linear(config.token_dim, width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.scalars = _mlp(SCALARS, 64, width)
        self.out_norm = nn.LayerNorm(width)
        self.core_gate = nn.Linear(width, config.lstm_input_dim)
        self.policy_gate = nn.Linear(width, config.lstm_hidden_dim)
        # v2: hero and champion controllers, pooled on their own; the projection into the summary
        # starts at zero, so a head gains the input without its outputs changing
        self.ability_state = _mlp(ABILITY_STATE, 64, width)
        self.ability_null = nn.Parameter(torch.zeros(1, 1, width))
        self.ability_norm = nn.LayerNorm(width)
        self.ability_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.ability_out = nn.Linear(width, width)
        for gate in (self.core_gate, self.policy_gate, self.ability_out):
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def _pool(self, attention, null, tokens, mask, query):
        batch = tokens.shape[0]
        tokens = torch.cat((null.expand(batch, 1, -1).to(tokens.dtype), tokens), dim=1)
        ignore = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device), ~mask), dim=1)
        pooled, _weights = attention(query, tokens, tokens, key_padding_mask=ignore, need_weights=False)
        return pooled.squeeze(1)

    def summary(self, extras: ExtrasV2, scene_state: Tensor, card_encoder, profile_memory: Tensor) -> Tensor:
        query = self.query(scene_state).unsqueeze(1)
        cards = self.card(card_encoder(extras.pending_card, extras.pending_form, profile_memory))
        tokens = self.token_norm(cards + self.owner(extras.pending_owner) + self.where(extras.pending_where))
        pooled = self._pool(self.attention, self.null, tokens, extras.pending_mask, query)
        heroes = self.card(card_encoder(extras.ability_card, extras.ability_form, profile_memory))
        heroes = self.ability_norm(heroes + self.owner(extras.ability_owner) + self.ability_state(extras.ability_state))
        abilities = self._pool(self.ability_attention, self.ability_null, heroes, extras.ability_mask, query)
        return self.out_norm(pooled + self.scalars(extras.scalars) + self.ability_out(abilities))


class ExtendedPolicyV4(UniversalCardPolicyV4):
    """FirstLight's policy with the extras summary. Instances are made by attach_extras()."""

    def encode_observation(self, batch, *, validate: bool = True):
        encoded = UniversalCardPolicyV4.encode_observation(self, batch, validate=validate)
        head: ExtrasHead = self.__dict__['_extras_head']
        extras = getattr(batch, 'extras', None)
        if extras is None:
            # no extras (a caller that does not build them): the base model's computation, not the
            # trained gates' biases on an empty summary
            return encoded
        profile_memory = self.mechanic_profile_encoder(self.effect_encoder.all_effects())
        summary = head.summary(extras, encoded.scene_state, self.card_encoder, profile_memory)
        return ExtendedEncodedV4(**{item.name: getattr(encoded, item.name) for item in fields(EncodedObservationV4)},
                                 extras_summary=summary)

    def _core_input(self, encoded):
        base = UniversalCardPolicyV4._core_input(self, encoded)
        summary = getattr(encoded, 'extras_summary', None)
        return base if summary is None else base + self.__dict__['_extras_head'].core_gate(summary)

    def _policy_context(self, encoded, hidden, cell):
        summary = getattr(encoded, 'extras_summary', None)
        if summary is None:
            return UniversalCardPolicyV4._policy_context(self, encoded, hidden, cell)
        from native_runner.training.v4.model import PolicyContextV4, RecurrentPolicyStateV4
        normalized = self.core_output_norm(hidden)
        policy_context = self.policy_context_norm(
            normalized
            + self.scene_policy_skip(encoded.scene_state)
            + self.spatial_policy_skip(encoded.spatial_summary)
            + self.candidate_policy_skip(encoded.candidate_summary)
            + self.__dict__['_extras_head'].policy_gate(summary)
        )
        value_context = self.value_context_norm(
            normalized
            + self.scene_value_skip(encoded.scene_state)
            + self.spatial_value_skip(encoded.spatial_summary)
            + self.scalar_value_skip(encoded.scalar_summary)
            + self.event_value_skip(encoded.event_summary)
        )
        return PolicyContextV4(policy_context=policy_context, value_context=value_context, encoded=encoded,
                               next_state=RecurrentPolicyStateV4(hidden=hidden, cell=cell),
                               value=self.value_head(value_context).squeeze(-1))


def attach_extras(policy: UniversalCardPolicyV4, head: ExtrasHead | None = None) -> ExtrasHead:
    """Turn a loaded FirstLight policy into an ExtendedPolicyV4 (in place). The head is held
    outside the module tree, so the policy's state_dict stays a strict FirstLight one."""
    if head is None:
        head = ExtrasHead(policy.config)
    device = next(policy.parameters()).device
    head.to(device)
    policy.__class__ = ExtendedPolicyV4
    policy.__dict__['_extras_head'] = head
    return head


def extras_payload(head: ExtrasHead) -> dict:
    return {'extras_version': EXTRAS_VERSION, 'extras_head': {k: v.detach().cpu() for k, v in head.state_dict().items()}}


def load_policy(path: str | Path, device):
    """FirstLight's strict loader, then the extras head if the checkpoint carries one."""
    from native_runner.training.v4.policy_session import load_policy_v4
    loaded = load_policy_v4(path, device=device)
    policy = getattr(loaded, 'model', loaded)
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    extra = payload.get('extra') or {}
    if extra.get('extras_version') in (EXTRAS_VERSION, EXTRAS_V1):
        head = ExtrasHead(policy.config)
        head.load_state_dict(upgrade_head_state(extra['extras_head'], extra['extras_version'], head))
        attach_extras(policy, head)
    return policy


def upgrade_head_state(state: dict, version: str, head: ExtrasHead) -> dict:
    """A saved head's state for this version's head. v1 had three pending inputs and no ability
    branch: the arrival input gets a zero weight column and the ability branch its fresh values
    (its output projection is zero), so the upgraded head computes what the v1 head did."""
    if version == EXTRAS_VERSION:
        return state
    state = dict(state)
    weight = state['where.0.weight']
    state['where.0.weight'] = torch.cat((weight, torch.zeros(weight.shape[0], WHERE - weight.shape[1],
                                                             dtype=weight.dtype)), dim=1)
    fresh = head.state_dict()
    for name, value in fresh.items():
        if name.startswith('ability_'):
            state[name] = value.detach().clone()
    return state


def cache_static_encodings(model) -> None:
    """Inference only: FirstLight re-encodes its whole effect catalog and mechanic-profile table in
    every encode_observation (and the extras head asks for them again) -- ~60% of one decision on
    the Mac's CPU (duel profile: 109 of 184 s). Both depend on the weights alone, so without
    gradients the first result is kept and reused; with gradients enabled (training) the original
    computation runs every time. The cached tensors are the ones the first call produced, so the
    policy's outputs are unchanged bit for bit."""
    import torch
    effects, profiles = model.effect_encoder, model.mechanic_profile_encoder
    if getattr(effects, '_clapha_cached', False):
        return
    original_effects, original_profiles = effects.all_effects, profiles.forward
    kept: dict = {}

    def all_effects():
        if torch.is_grad_enabled():
            return original_effects()
        if 'effects' not in kept:
            kept['effects'] = original_effects()
        return kept['effects']

    def profile_forward(effect_memory):
        if torch.is_grad_enabled() or effect_memory is not kept.get('effects'):
            return original_profiles(effect_memory)
        if 'profiles' not in kept:
            kept['profiles'] = original_profiles(effect_memory)
        return kept['profiles']

    effects.all_effects = all_effects
    profiles.forward = profile_forward
    effects._clapha_cached = True


def install_session_hook(session) -> None:
    """Make a PolicySessionV4 feed extras to an extended model: before each decide(), set
    `session.next_extras` (build_extras); its tensorizer then returns the extended batch. The
    live console and il/duel.py use this; a session whose model has no head is left alone."""
    model = getattr(session, 'model', None)
    if model is None or '_extras_head' not in model.__dict__ or getattr(session, '_extras_hooked', False):
        return
    tensorizer = session.tensorizer
    original = tensorizer.tensorize

    def tensorize(observation, *, validate: bool = True):
        batch = original(observation, validate=validate)
        extras = getattr(session, 'next_extras', None)
        return batch if extras is None else extend_batch(batch, extras)

    tensorizer.tensorize = tensorize
    session.next_extras = None
    session._extras_hooked = True


def checkpoint_extra(path: str | Path) -> dict:
    """The `extra` payload of a checkpoint (recipe, extras head, ...); {} if it has none."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    return dict(payload.get('extra') or {})
