"""FirstLight's IL chunk evaluation with the recurrent core fused and the heads batched over time.

evaluate_recurrent_sequence (native_runner/training/v4/learning.py) encodes a [T,B] chunk in one
call, then runs T separate steps: an LSTMCell and the full action decoder on B rows each. At
B=8 that is ~1.75 s per update on an L4, nearly all of it launch and Python overhead. FirstLight's
own PPO path (UniversalCardPolicyV4.evaluate_encoded_action_sequence) avoids it: one nn.LSTM
kernel over the chunk, then the heads once over all T*B rows ("flattening the independent action
heads over T*B is exact"). It reports the head components summed over micro-actions, which the
imitation loss cannot use (the delay term smooths over the whole delay distribution), so this
module does the same thing with the model's dense decoder and returns exactly what
evaluate_recurrent_sequence returns: per-head components stacked [T,B,...], value, final state.

Requirement (checked): no episode starts inside the chunk after its first step. An IL lane is one
game side and only its first turn starts an episode; padded turns never do.
"""
from __future__ import annotations

from dataclasses import fields

import torch

from native_runner.training.v4.learning import RecurrentEvaluationV4
from native_runner.training.v4.tensors import (ActionEvaluationComponentsV4, RecurrentPolicyStateV4,
                                               concatenate_padded_tensor_records)


def evaluate_sequence_fused(model, observations, actions, episode_start, *, initial_state=None,
                            gate_temperature: float = 1.0, action_temperature: float = 1.0,
                            continue_temperature: float = 1.0, validate: bool = False) -> RecurrentEvaluationV4:
    time_steps = len(observations)
    batch_size = observations[0].batch_size
    if time_steps == 0 or len(actions) != time_steps:
        raise ValueError('observations and actions need the same nonzero length')
    if tuple(episode_start.shape) != (time_steps, batch_size) or episode_start.dtype != torch.bool:
        raise ValueError('episode_start must be bool [T,B]')
    if time_steps > 1 and bool(torch.any(episode_start[1:])):
        raise ValueError('fused evaluation needs no episode start after the first step of a chunk')
    device = observations[0].match_scalars.device
    state = initial_state or model.initial_state(batch_size, device=device)

    flat_batch = concatenate_padded_tensor_records(tuple(observations))
    flat_actions = concatenate_padded_tensor_records(tuple(actions))
    encoded = model.encode_observation(flat_batch, validate=validate)

    keep = (~episode_start[0]).unsqueeze(-1).to(state.hidden.dtype)
    hidden = (state.hidden * keep).unsqueeze(0)
    cell = (state.cell * keep).unsqueeze(0)
    core_input = model._core_input(encoded).reshape(time_steps, batch_size, -1)
    lstm = model.lstm_core
    # the same kernel call as FirstLight's evaluate_encoded_action_sequence (nn.LSTM with the
    # LSTMCell's weights: one layer, biases, no dropout, not bidirectional, time-major)
    recurrent, final_hidden, final_cell = torch._VF.lstm(  # type: ignore[attr-defined]
        core_input, (hidden, cell), (lstm.weight_ih, lstm.weight_hh, lstm.bias_ih, lstm.bias_hh),
        True, 1, 0.0, model.training, False, False)
    recurrent_flat = recurrent.flatten(0, 1)
    # the decoder reads each step's hidden state, never the intermediate cell states
    context = model._policy_context(encoded, recurrent_flat, torch.zeros_like(recurrent_flat))
    output = model._decode(flat_batch, context, sample=False, forced_actions=flat_actions,
                           gate_temperature=gate_temperature, action_temperature=action_temperature,
                           continue_temperature=continue_temperature, validate=validate)
    if output.action_components is None:
        raise RuntimeError('V4 action evaluation did not return head components')

    def per_step(value):
        return value.reshape(time_steps, batch_size, *value.shape[1:])

    components = ActionEvaluationComponentsV4(**{item.name: per_step(getattr(output.action_components, item.name))
                                                 for item in fields(ActionEvaluationComponentsV4)})
    return RecurrentEvaluationV4(log_prob=per_step(output.log_prob), entropy=per_step(output.entropy),
                                 value=per_step(output.value), components=components,
                                 final_state=RecurrentPolicyStateV4(hidden=final_hidden[0], cell=final_cell[0]))
