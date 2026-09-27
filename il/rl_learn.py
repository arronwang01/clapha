"""PPO for our model on self-play lanes (il/rl.py collect writes them).

FirstLight's update rule -- clipped probability ratio, clipped value error, entropy bonus, GAE
(native_runner.training.v4.learning) -- applied to our model (the extended policy: pending cards,
arrival, exact elixir, hero states) on games played through our live path, with our reward (il/rl.py).
Each lane is one side of one game, evaluated from its first decision in recurrent chunks through
il/fused.py at the temperatures it was sampled with; the optimizer steps after every chunk, as the
imitation trainer does.

  value warm-up  the first --value-warmup updates move only the value estimate: FirstLight's value
                 head learned its own shaped reward, and advantages are only as good as it is
  KL anchor      --kl-coef * KL(policy || starting policy), estimated on the sampled actions, keeps the
                 early updates near what the model knows (it knows much more than self-play shows)

Loop: wait for --lanes new lanes in the games folder, update, write policy-NNNN.pt and latest.pt
(the collectors reload latest.pt), repeat.

    ./py -m il.rl_learn --init CKPT --games runs/rl/<run>/games --out runs/rl/<run>
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))

VALUE_PREFIXES = ('value_head', 'scene_value_skip', 'spatial_value_skip', 'scalar_value_skip', 'event_value_skip',
                  'value_context_norm')


def lane_sequences(paths, gamma: float, gae_lambda: float):
    """ILSequenceV4 per lane (returns = GAE returns) and, per lane, old log-probs, old values and
    advantages [T] plus the sampling temperatures."""
    import torch
    from il.rl import load_lane
    from native_runner.training.v4.imitation import ILSequenceV4
    from native_runner.training.v4.learning import generalized_advantage_estimate
    sequences, extras = [], []
    for path in paths:
        lane = load_lane(path)
        steps = len(lane['observations'])
        rewards = torch.tensor(lane['reward'], dtype=torch.float32).reshape(steps, 1)
        values = torch.tensor(lane['value'], dtype=torch.float32).reshape(steps, 1)
        next_values = torch.cat((values[1:], torch.zeros(1, 1)))
        terminated = torch.zeros(steps, 1, dtype=torch.bool)
        terminated[-1] = True
        advantages, returns = generalized_advantage_estimate(
            rewards, values, next_values, terminated, torch.zeros_like(terminated), gamma=gamma, gae_lambda=gae_lambda)
        sequences.append(ILSequenceV4(
            observations=tuple(lane['observations']), actions=tuple(lane['actions']),
            episode_start=torch.tensor([[step == 0] for step in range(steps)], dtype=torch.bool),
            valid_mask=torch.ones(steps, 1, dtype=torch.bool), returns=returns))
        extras.append({'old_log_prob': torch.tensor(lane['log_prob'], dtype=torch.float32),
                       'old_value': values.reshape(-1), 'advantage': advantages.reshape(-1),
                       'temperatures': tuple(lane['temperatures']), 'return': float(sum(lane['reward'])),
                       'winner': lane['winner'], 'side': lane['side']})
    return sequences, extras


def _padded(rows: list, name: str, steps: int):
    import torch
    out = torch.zeros(steps, len(rows))
    for column, row in enumerate(rows):
        out[:len(row[name]), column] = row[name]
    return out


def update(policy, reference, optimizer, parameters, sequences, extras, args, device, value_only: bool) -> dict:
    """One pass of PPO over these lanes (minibatches of whole lanes, chunked in time)."""
    import torch
    from il.fused import evaluate_forced, fused_context
    from il.train import batch_sequences
    from native_runner.training.v4.imitation import slice_il_sequence
    temperatures = {row['temperatures'] for row in extras}
    if len(temperatures) != 1:
        raise ValueError(f'lanes sampled at different temperatures: {temperatures}')
    gate_t, action_t, continue_t = temperatures.pop()
    order = list(range(len(sequences)))
    stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'kl_ref': 0.0, 'clip_fraction': 0.0,
             'approx_kl': 0.0, 'chunks': 0}
    for _epoch in range(args.epochs):
        random.shuffle(order)
        for first in range(0, len(order), args.minibatch):
            picked = order[first:first + args.minibatch]
            batch = batch_sequences([sequences[i] for i in picked])
            rows = [extras[i] for i in picked]
            steps = batch.time_steps
            old_log_prob = _padded(rows, 'old_log_prob', steps)
            old_value = _padded(rows, 'old_value', steps)
            advantage = _padded(rows, 'advantage', steps)
            valid = batch.valid_mask.float()
            mean = (advantage * valid).sum() / valid.sum()
            spread = (((advantage - mean) ** 2) * valid).sum() / valid.sum()
            advantage = (advantage - mean) / (spread.sqrt() + 1e-8)
            state = policy.initial_state(batch.batch_size, device=device)
            reference_state = reference.initial_state(batch.batch_size, device=device)
            for start in range(0, steps, args.time_steps):
                stop = min(steps, start + args.time_steps)
                chunk = slice_il_sequence(batch, start, stop).to(device)
                mask = chunk.valid_mask.float()
                count = mask.sum().clamp_min(1.0)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    context = fused_context(policy, chunk.observations, chunk.episode_start, initial_state=state)
                    evaluation = evaluate_forced(policy, context, chunk.actions, gate_temperature=gate_t,
                                                 action_temperature=action_t, continue_temperature=continue_t)
                    with torch.no_grad():
                        reference_context = fused_context(reference, chunk.observations, chunk.episode_start,
                                                          initial_state=reference_state)
                        reference_eval = evaluate_forced(reference, reference_context, chunk.actions,
                                                         gate_temperature=gate_t, action_temperature=action_t,
                                                         continue_temperature=continue_t)
                log_prob = evaluation.log_prob.float()
                value = evaluation.value.float()
                returns = chunk.returns.float()
                old_lp = old_log_prob[start:stop].to(device)
                old_v = old_value[start:stop].to(device)
                adv = advantage[start:stop].to(device)
                ratio = torch.exp(log_prob - old_lp)
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                policy_loss = -(torch.minimum(ratio * adv, clipped * adv) * mask).sum() / count
                value_clipped = old_v + torch.clamp(value - old_v, -args.value_clip, args.value_clip)
                value_loss = (torch.maximum((value - returns) ** 2, (value_clipped - returns) ** 2) * mask).sum() / count
                entropy = (evaluation.entropy.float() * mask).sum() / count
                log_ratio_ref = reference_eval.log_prob.float() - log_prob
                kl_ref = ((torch.exp(log_ratio_ref) - 1.0 - log_ratio_ref) * mask).sum() / count
                if value_only:
                    loss = args.value_coef * value_loss
                else:
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy + args.kl_coef * kl_ref
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                state = evaluation.final_state.detach()
                reference_state = reference_eval.final_state.detach()
                with torch.no_grad():
                    stats['policy_loss'] += float(policy_loss)
                    stats['value_loss'] += float(value_loss)
                    stats['entropy'] += float(entropy)
                    stats['kl_ref'] += float(kl_ref)
                    stats['clip_fraction'] += float((((ratio - 1.0).abs() > args.clip).float() * mask).sum() / count)
                    stats['approx_kl'] += float((((ratio - 1.0) - torch.log(ratio)) * mask).sum() / count)
                    stats['chunks'] += 1
    chunks = max(1, stats.pop('chunks'))
    return {name: round(total / chunks, 5) for name, total in stats.items()}


def _version(path: Path) -> int:
    """The update number of the weights that played a lane (il/rl.py names it ...-vN.lane)."""
    stem = path.stem
    return int(stem.rsplit('-v', 1)[1]) if '-v' in stem else 0


def save_policy(path: Path, policy, head, update_index: int, args) -> None:
    from native_runner.training.v4.checkpoint import save_actor_critic_checkpoint
    from il.extras import extras_payload
    temporary = path.with_suffix('.part')
    save_actor_critic_checkpoint(temporary, policy, optimizer=None, update_step=update_index, training_stage='ppo',
                                 gamma_per_decision=args.gamma,
                                 extra={'recipe': 'clapha il.rl_learn: self-play, live path, our reward',
                                        'init': args.init, 'update': update_index, **extras_payload(head)})
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            # Windows: a collector is reading the old latest.pt right now
            time.sleep(0.25)
    raise RuntimeError(f'could not replace {path}')


def main(argv: list[str]) -> int:
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', required=True, help='the starting policy (our extended checkpoint)')
    parser.add_argument('--games', required=True, type=Path, help='the collectors\' lane folder')
    parser.add_argument('--out', required=True, type=Path)
    # 32 lanes of 2 per minibatch: next to the other user's job on the PC, 64 lanes took 8.7 GB of RAM
    # and 8 per minibatch ran the GPU out of memory (2026-09-27)
    parser.add_argument('--lanes', type=int, default=32, help='new lanes per update')
    parser.add_argument('--max-age', type=int, default=2, help='use lanes from at most this many updates back')
    parser.add_argument('--updates', type=int, default=0, help='stop after this many (0: run until stopped)')
    parser.add_argument('--value-warmup', type=int, default=4)
    # one pass: on the shared GPU the learner is the slow part and games are plentiful (2026-09-27)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--minibatch', type=int, default=2, help='lanes per minibatch')
    parser.add_argument('--time-steps', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--value-clip', type=float, default=0.2)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.001)
    parser.add_argument('--kl-coef', type=float, default=0.1)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--once', action='store_true', help='one update on the lanes present, then stop (tests)')
    parser.add_argument('--keep-lanes', action='store_true',
                        help='keep lanes once used (by default they are deleted: ~4 MB per side per game)')
    parser.add_argument('--save-every', type=int, default=10,
                        help='keep policy-NNNN.pt every N updates (the league\'s snapshots are every 10)')
    parser.add_argument('--min-free-gb', type=float, default=5.0,
                        help='the GPU is shared: update only when this much GPU memory is free (0: always)')
    args = parser.parse_args(argv)
    import firstlight_bot  # noqa: F401  (puts FirstLight's native_runner on the path)
    from il.extras import checkpoint_extra
    from il.train import load_policy
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    # a restarted run continues from its latest update; the KL anchor stays --init
    latest = args.out / 'latest.pt'
    resumed = int(checkpoint_extra(latest).get('update', 0)) if latest.is_file() else 0
    policy = load_policy(str(latest) if resumed else args.init, device)
    reference = load_policy(args.init, device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    head = policy.__dict__.get('_extras_head')
    reference_head = reference.__dict__.get('_extras_head')
    if head is None or reference_head is None:
        raise SystemExit('--init must be one of our extended checkpoints (il/extras.py)')
    for parameter in reference_head.parameters():
        parameter.requires_grad_(False)
    policy.train()
    head.train()
    everything = list(policy.parameters()) + list(head.parameters())
    value_parameters = [p for name, p in policy.named_parameters() if name.startswith(VALUE_PREFIXES)]
    log = (args.out / 'learn.jsonl').open('a')
    seen: set[str] = set()
    update_index = resumed
    if resumed:
        print(f'resuming at update {resumed} from {latest}', flush=True)
    optimizer, optimizing = None, None
    while not args.updates or update_index < args.updates:
        fresh = sorted(p for p in args.games.glob('*.lane') if p.name not in seen)
        if len(fresh) < args.lanes and not (args.once and fresh):
            time.sleep(30)
            continue
        picked = fresh[:args.lanes]
        seen.update(p.name for p in picked)
        stale = [p for p in picked if update_index - _version(p) > args.max_age]
        picked = [p for p in picked if update_index - _version(p) <= args.max_age]
        if not args.keep_lanes:
            for path in stale:
                path.unlink(missing_ok=True)
        if not picked:
            continue
        # the 4080 is shared (another user's jobs): take turns -- wait while it is busy, give the memory
        # back after every update
        waited = 0
        while device.type == 'cuda' and args.min_free_gb > 0:
            free, _total = torch.cuda.mem_get_info(device)
            if free >= args.min_free_gb * 2 ** 30:
                break
            if waited % 600 == 0:
                print(f'waiting for GPU memory: {free / 2 ** 30:.1f} GB free, need {args.min_free_gb:.1f}', flush=True)
            time.sleep(30)
            waited += 30
        started = time.time()
        sequences, extras = lane_sequences(picked, args.gamma, args.gae_lambda)
        value_only = update_index < args.value_warmup
        if optimizing != ('value' if value_only else 'all'):
            optimizing = 'value' if value_only else 'all'
            optimizer = (torch.optim.AdamW(value_parameters, lr=args.learning_rate * 10) if value_only
                         else torch.optim.AdamW(everything, lr=args.learning_rate))
        parameters = value_parameters if value_only else everything
        stats = None
        for attempt in range(6):
            try:
                stats = update(policy, reference, optimizer, parameters, sequences, extras, args, device, value_only)
                break
            except torch.OutOfMemoryError:
                pass
            # the shared GPU filled up (the other user's job grew): let go of everything, wait, try again
            # (outside the except block, so the traceback no longer holds the update's tensors)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f'GPU out of memory during update {update_index + 1}: try {attempt + 2} of 6 in 60 s', flush=True)
            time.sleep(60)
        if stats is None:
            del sequences
            for path in picked:
                path.unlink(missing_ok=True)
            continue
        update_index += 1
        del sequences
        if not args.keep_lanes and not args.once:
            for path in picked:
                path.unlink(missing_ok=True)
        wins = sum(1 for row in extras if row['winner'] == row['side'])
        row = {'update': update_index, 'value_only': value_only, 'lanes': len(picked), 'seconds': round(time.time() - started),
               'waited_s': waited,
               'mean_return': round(sum(r['return'] for r in extras) / len(extras), 4), 'lane_wins': wins, **stats}
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + '\n')
        log.flush()
        if args.save_every and update_index % args.save_every == 0:
            # the league's snapshots (il/rl.py --snapshot-every) and the run's history; ~50 MB each
            save_policy(args.out / f'policy-{update_index:04d}.pt', policy, head, update_index, args)
        save_policy(args.out / 'latest.pt', policy, head, update_index, args)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if args.once:
            break
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
