"""Stage B inputs: what FirstLight's observation lacks, as a separate record and a gated head.

Inputs (one row per decision turn, the actor's perspective):
  pending   up to MAX_PENDING commands sent and not executed yet, both sides: all of the actor's
            own, and the opponent's that are visible in the actor's queue. Per command: the card
            (FirstLight card vocabulary), its form (0 normal, 1 evolution, 2 hero), owner (0 the
            actor, 1 the opponent), target tile in the actor's perspective, ticks until it executes.
            The screen already shows the actor's own sent cards gone from the hand and elixir
            (firstlight_obs.screen_view); this says where and when they land, and what the
            opponent has queued. It is the input FirstLight never had ("did it see its own troop").
  scalars   the opponent's exact elixir (the reader has it; FirstLight's FAIR observation only has
            the tracker's estimate), and the actor's command delay (decision -> replay tick).

The model: ExtendedPolicyV4 is FirstLight's UniversalCardPolicyV4 with one more summary. Pending
commands go through the model's own card encoder (a pending Hog Rider starts out meaning Hog
Rider), get owner, place and time, and are pooled by attention with the scene as the query; the
scalars through a small MLP. The summary enters the recurrent core's input and the policy context
through two linear gates initialised to zero, so an extended checkpoint computes exactly what its
base does until training moves the gates. The head is kept out of the model's state_dict: an
extended checkpoint is a strict FirstLight checkpoint (their tools and our console load it) with
the head in its `extra` payload; load_policy() here attaches it.

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
SCALARS = 2              # opponent elixir / 10, command delay / 40
EXTRAS_VERSION = 'clapha-extras.v1'


@dataclass(slots=True)
class ExtrasV1(TensorRecordV4):
    pending_card: Tensor     # [B, K] long, FirstLight card vocab id (0 = none)
    pending_form: Tensor     # [B, K] long
    pending_owner: Tensor    # [B, K] long, 0 the actor / 1 the opponent
    pending_where: Tensor    # [B, K, 3] float: tile x / 17, tile y / 31 (actor's view), ticks to execute / 40
    pending_mask: Tensor     # [B, K] bool
    scalars: Tensor          # [B, SCALARS] float


@dataclass(slots=True)
class ExtendedBatchV4(UniversalSemanticBatchV4):
    extras: ExtrasV1


@dataclass(slots=True)
class ExtendedEncodedV4(EncodedObservationV4):
    extras_summary: Tensor


def extend_batch(batch: UniversalSemanticBatchV4, extras: ExtrasV1) -> ExtendedBatchV4:
    return ExtendedBatchV4(**{item.name: getattr(batch, item.name) for item in fields(UniversalSemanticBatchV4)},
                           extras=extras)


def build_extras(tensorizer, pending, *, opponent_elixir: float, delay: int) -> ExtrasV1:
    """pending: (card_id, form, owner relative to the actor (0/1), native tile (x, y), ticks to
    execute), soonest first; more than MAX_PENDING keeps the soonest."""
    rows = sorted(pending, key=lambda item: item[4])[:MAX_PENDING]
    card = torch.zeros(1, MAX_PENDING, dtype=torch.long)
    form = torch.zeros(1, MAX_PENDING, dtype=torch.long)
    owner = torch.zeros(1, MAX_PENDING, dtype=torch.long)
    where = torch.zeros(1, MAX_PENDING, 3, dtype=torch.float32)
    mask = torch.zeros(1, MAX_PENDING, dtype=torch.bool)
    for index, (card_id, card_form, relative_owner, tile, ticks) in enumerate(rows):
        card[0, index] = int(tensorizer.catalog.vocab_id(int(card_id)))
        form[0, index] = int(card_form or 0)
        owner[0, index] = int(relative_owner)
        if tile is not None:
            x, y = tensorizer.perspective._grid(tile)
            where[0, index, 0], where[0, index, 1] = x / 17.0, y / 31.0
        where[0, index, 2] = max(0, int(ticks)) / 40.0
        mask[0, index] = True
    scalars = torch.tensor([[float(opponent_elixir) / 10.0, float(delay) / 40.0]], dtype=torch.float32)
    return ExtrasV1(pending_card=card, pending_form=form, pending_owner=owner, pending_where=where,
                    pending_mask=mask, scalars=scalars)


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class ExtrasHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        width = config.token_dim
        self.card = nn.Linear(config.card_semantic_dim, width)
        self.owner = nn.Embedding(2, width)
        self.where = _mlp(3, 64, width)
        self.null = nn.Parameter(torch.zeros(1, 1, width))      # always attendable: empty queues pool to it
        self.token_norm = nn.LayerNorm(width)
        self.query = nn.Linear(config.token_dim, width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.scalars = _mlp(SCALARS, 64, width)
        self.out_norm = nn.LayerNorm(width)
        self.core_gate = nn.Linear(width, config.lstm_input_dim)
        self.policy_gate = nn.Linear(width, config.lstm_hidden_dim)
        for gate in (self.core_gate, self.policy_gate):
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def summary(self, extras: ExtrasV1, scene_state: Tensor, card_encoder, profile_memory: Tensor) -> Tensor:
        cards = self.card(card_encoder(extras.pending_card, extras.pending_form, profile_memory))
        tokens = self.token_norm(cards + self.owner(extras.pending_owner) + self.where(extras.pending_where))
        batch = tokens.shape[0]
        tokens = torch.cat((self.null.expand(batch, 1, -1).to(tokens.dtype), tokens), dim=1)
        ignore = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device), ~extras.pending_mask), dim=1)
        pooled, _weights = self.attention(self.query(scene_state).unsqueeze(1), tokens, tokens,
                                          key_padding_mask=ignore, need_weights=False)
        return self.out_norm(pooled.squeeze(1) + self.scalars(extras.scalars))


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
    if extra.get('extras_version') == EXTRAS_VERSION:
        head = ExtrasHead(policy.config)
        head.load_state_dict(extra['extras_head'])
        attach_extras(policy, head)
    return policy


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
