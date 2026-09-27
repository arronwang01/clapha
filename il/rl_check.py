"""Collector/learner agreement: at the weights that played a lane, the learner's evaluation of the
lane's actions must reproduce the log-probs and values the collector recorded (PPO's ratio starts
at 1). Run it on lanes from any collector (local, the inference server eager or CUDA-graphed).

    ./py -m il.rl_check LANE_DIR --init CHECKPOINT [--steps 256] [--device cpu]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))


def main(argv: list[str]) -> int:
    import torch
    import firstlight_bot  # noqa: F401  (puts FirstLight's native_runner on the path)
    from il.fused import evaluate_forced, fused_context
    from il.rl_learn import lane_sequences
    from il.train import batch_sequences, load_policy
    from native_runner.training.v4.imitation import slice_il_sequence
    parser = argparse.ArgumentParser()
    parser.add_argument('lanes', type=Path)
    parser.add_argument('--init', required=True, help='the weights that played the lanes (update 0: the run start)')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--lane', type=int, default=0, help='which lane (sorted by name)')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    lane = sorted(args.lanes.glob('*.lane'))[args.lane]
    policy = load_policy(args.init, device)
    policy.eval()
    sequences, extras = lane_sequences([lane], 0.999, 0.95)
    batch = batch_sequences(sequences)
    steps = min(args.steps, batch.time_steps)
    gate_t, action_t, continue_t = extras[0]['temperatures']
    state = policy.initial_state(1, device=device)
    log_probs, values = [], []
    with torch.no_grad():
        for start in range(0, steps, 32):
            chunk = slice_il_sequence(batch, start, min(steps, start + 32)).to(device)
            context = fused_context(policy, chunk.observations, chunk.episode_start, initial_state=state)
            evaluation = evaluate_forced(policy, context, chunk.actions, gate_temperature=gate_t,
                                         action_temperature=action_t, continue_temperature=continue_t)
            log_probs.append(evaluation.log_prob.float().reshape(-1).cpu())
            values.append(evaluation.value.float().reshape(-1).cpu())
            state = evaluation.final_state
    log_prob, value = torch.cat(log_probs), torch.cat(values)
    old_log_prob, old_value = extras[0]['old_log_prob'][:steps], extras[0]['old_value'][:steps]
    ratio = torch.exp(log_prob - old_log_prob)
    print(f'{lane.name}: {steps} steps, temperatures {extras[0]["temperatures"]}')
    print(f'log-prob diff: mean {float((log_prob - old_log_prob).abs().mean()):.6f} '
          f'max {float((log_prob - old_log_prob).abs().max()):.6f}')
    print(f'ratio: min {float(ratio.min()):.4f} max {float(ratio.max()):.4f}; '
          f'{int(((ratio - 1).abs() > 0.2).sum())} of {steps} outside the clip')
    print(f'value diff: mean {float((value - old_value).abs().mean()):.6f} max {float((value - old_value).abs().max()):.6f}')
    # a GPU server and a CPU check round differently (~1e-3 in log-prob): what matters is that PPO's
    # ratio starts at 1, far inside its 0.2 clip
    good = float((ratio - 1).abs().max()) < 0.01 and float((value - old_value).abs().max()) < 0.01
    print('AGREE' if good else 'DISAGREE')
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
