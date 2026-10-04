"""What one decision costs by batch size: our extended model's rollout sampling (FirstLight's
sample_for_ppo_rollout, what il/rl.py collects with) on real mid-game observations from RL lanes.

Per batch size: the host-to-device copy of the batch, the forward with sampling, and the cost per
decision (both divided by the batch). This decides the collector's shape: how many games' decisions
one forward should carry.

    ./py -m il.bench_forward LANE_DIR [--device cuda|cpu] [--batches 1 4 16 64] [--threads N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))


def main(argv: list[str]) -> int:
    import torch
    import firstlight_bot  # noqa: F401  (puts FirstLight's native_runner on the path)
    from il.rl import load_lane
    from il.train import batch_sequences, load_policy
    from native_runner.training.v4.imitation import ILSequenceV4
    parser = argparse.ArgumentParser()
    parser.add_argument('lanes', type=Path, help='a folder of il.rl lanes (their observations are the inputs)')
    parser.add_argument('--init', default=str(CLAPHA / 'runs/pc/distill-v2-final.pt'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--threads', type=int, default=0, help='torch CPU threads (0: default)')
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    from il.extras import cache_static_encodings
    model = load_policy(args.init, device)
    model.eval()
    cache_static_encodings(model)       # as the collector's runner does (weight-only encodings kept)
    lane = load_lane(sorted(args.lanes.glob('*.lane'))[0])
    pool = list(zip(lane['observations'], lane['actions']))[200:]      # past the opening: a full board

    def sync() -> None:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    print(f'{device} threads {torch.get_num_threads()}', flush=True)
    for size in args.batches:
        rows = [pool[(index * 37) % len(pool)] for index in range(size)]
        sequences = [ILSequenceV4(observations=(observation,), actions=(action,),
                                  episode_start=torch.zeros(1, 1, dtype=torch.bool),
                                  valid_mask=torch.ones(1, 1, dtype=torch.bool), returns=torch.zeros(1, 1))
                     for observation, action in rows]
        host = batch_sequences(sequences).observations[0]
        with torch.inference_mode():
            state = model.initial_state(size, device=device)
            start = torch.zeros(size, dtype=torch.bool, device=device)
            for _ in range(3):
                model.sample_for_ppo_rollout(host.to_model_input(device), state, episode_start=start, validate=False)
            sync()
            began = time.perf_counter()
            for _ in range(args.repeats):
                batch = host.to_model_input(device)
            sync()
            copied = time.perf_counter()
            for _ in range(args.repeats):
                model.sample_for_ppo_rollout(batch, state, episode_start=start, validate=False)
            sync()
            done = time.perf_counter()
        copy_ms = (copied - began) / args.repeats * 1000
        forward_ms = (done - copied) / args.repeats * 1000
        print(f'batch {size:3}: copy {copy_ms:6.1f} ms  forward {forward_ms:6.1f} ms  '
              f'per decision {(copy_ms + forward_ms) / size:6.2f} ms', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
