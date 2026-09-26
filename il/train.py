"""Imitation fine-tuning of a FirstLight V4 checkpoint on live-code samples (il/samples.py).

FirstLight's own recipe (train_imitation_cache.py), on one GPU, with the data made on the fly:
  - one sequence per (replay, Hog 2.6 side): every decision turn from tick 90, built by the live
    code from the engine conversion, labels at the bot's decision tick (il/SPEC.md);
  - batches of sequences cut into 32-turn chunks; the recurrent state is carried from chunk to
    chunk (detached), as theirs;
  - their imitation_loss (gate with act weighted 8x, card, target, timing with neighbour
    smoothing); the value head is left out (its returns are FirstLight's reward; RL comes later);
    AdamW 3e-5, weight decay 1e-2, gradient clip 1.0, bf16 autocast on CUDA.
DataLoader workers tensorize replays in parallel (~7 s of CPU per game side), so there is no
multi-hundred-GB tensor cache: the compact conversion (~170 KB per game) is the dataset.

    python -m il.train --frames runs/conv-hog26 --init fl:hog2 --out runs/train-hog26 [--epochs 1]
    python -m il.train ... --smoke      two short sequences, two updates: checks the code path only
Replays whose engine replay is not exact (tower HP off by more than 100, or ended early) are left
out; 2% of the rest, by replay tag, are held out for validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
if str(CLAPHA) not in sys.path:
    sys.path.insert(0, str(CLAPHA))


def select_units(frames_dir: Path, *, max_tower_error: int = 100) -> list[tuple[str, int]]:
    """(replay file, actor) for every Hog 2.6 side of an exactly replayed game."""
    from il.samples import HOG26_CARDS
    units = []
    for line in (frames_dir / 'index.jsonl').read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue            # a line cut off by copying the index while the conversion writes it
        if not record.get('ok') or record['tower_hp_error'] > max_tower_error:
            continue
        if record.get('ended_at') is not None and record['end_tick'] - record['ended_at'] > 150:
            continue
        path = frames_dir / 'frames' / record['tag'][:2] / f"{record['tag']}.jsonl.zst"
        if path.exists():
            units.append((str(path), record['tag']))
    selected = []
    from il.frames import load_replay
    for path, tag in units:
        header, _frames = load_replay(Path(path), with_replay=False)
        for actor, deck in enumerate(header['timeline']['decks']):
            if HOG26_CARDS <= set(deck):
                selected.append((path, actor))
    return selected


def held_out(path: str) -> bool:
    return int(hashlib.sha256(Path(path).name.encode()).hexdigest()[:8], 16) % 50 == 0


class ReplaySequences:
    """torch Dataset: one ILSequenceV4 per (replay, actor), made by the live code."""

    def __init__(self, units, max_turns: int | None = None, extras: bool = False):
        self.units = list(units)
        self.max_turns = max_turns
        self.extras = extras

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int):
        from il.frames import load_replay
        from il.samples import actor_sequence
        from native_runner.training.v4.imitation import slice_il_sequence
        path, actor = self.units[index]
        stats: Counter = Counter()
        try:
            header, frames = load_replay(Path(path))
            sequence = actor_sequence(header, frames, actor, stats, extras=self.extras)
        except Exception as error:  # noqa: BLE001  (one bad replay must not stop training)
            return {'error': f'{Path(path).name} actor {actor}: {type(error).__name__}: {error}'[:300]}
        if sequence is not None and self.max_turns and sequence.time_steps > self.max_turns:
            sequence = slice_il_sequence(sequence, 0, self.max_turns)
        return {'sequence': sequence, 'stats': dict(stats)}


def _keep(items):
    return items


def _rows(collection) -> int:
    from dataclasses import fields
    for item in fields(collection):
        value = getattr(collection, item.name)
        if getattr(value, 'ndim', 0) >= 2:
            return int(value.shape[1])
    return 0


def batch_sequences(sequences):
    """FirstLight's padded collation, after padding every turn's variable collections (active
    effects, relation edges, candidates) to the batch's largest, as their cache loader does."""
    from native_runner.training.v4.cache import _pad_sequence_dynamic_observations
    from native_runner.training.v4.imitation import collate_padded_il_sequences
    counts = {name: max(_rows(getattr(observation, name)) for sequence in sequences
                        for observation in sequence.observations)
              for name in ('active_effects', 'relation_edges', 'candidates')}
    padded = [_pad_sequence_dynamic_observations(sequence, active_effect_count=counts['active_effects'],
                                                 relation_edge_count=counts['relation_edges'],
                                                 candidate_count=counts['candidates'])
              for sequence in sequences]
    return collate_padded_il_sequences(padded)


def load_policy(init: str, device):
    """A FirstLight checkpoint by name (firstlight_bot.CHECKPOINTS) or path; an extended one
    (il/extras.py) comes back with its head attached."""
    import firstlight_bot as FLB
    from il.extras import load_policy as load_extended
    return load_extended(FLB.CHECKPOINTS.get(init, Path(init)), device)


def evaluate(policy, sequences, device, time_steps: int) -> dict:
    """Loss and gate/card agreement on held-out sequences, no gradient."""
    import torch
    from native_runner.training.v4.imitation import imitation_loss, slice_il_sequence
    from native_runner.training.v4.learning import evaluate_recurrent_sequence
    totals, counts = Counter(), Counter()
    policy.eval()
    with torch.no_grad():
        batch = batch_sequences(sequences)
        state = policy.initial_state(batch.batch_size, device=device)
        for start in range(0, batch.time_steps, time_steps):
            chunk = slice_il_sequence(batch, start, min(batch.time_steps, start + time_steps)).to(device)
            evaluation = evaluate_recurrent_sequence(
                policy, chunk.observations, chunk.actions, chunk.episode_start, initial_state=state,
                gate_temperature=1.0, action_temperature=1.0, continue_temperature=1.0, validate=False,
                preencode_observations=True)
            losses = imitation_loss(chunk, evaluation)
            for head in ('gate', 'candidate', 'target', 'delay'):
                count = float(getattr(losses, f'{head}_count'))
                totals[head] += float(getattr(losses, head)) * count
                counts[head] += count
            state = evaluation.final_state
    policy.train()
    return {head: round(totals[head] / counts[head], 4) if counts[head] else None for head in totals}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=Path, default=CLAPHA / 'runs/conv-hog26')
    parser.add_argument('--init', default='fl:hog2')
    parser.add_argument('--out', type=Path, default=CLAPHA / 'runs/train-hog26')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=8, help='sequences per update group')
    parser.add_argument('--time-steps', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--save-every', type=int, default=200, help='update groups between checkpoints')
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--extras', action='store_true', help='Stage B inputs: pending commands, opponent elixir, delay')
    parser.add_argument('--head-lr-mult', type=float, default=10.0, help='learning rate multiplier for the new head')
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader
    import il.samples  # noqa: F401  (puts the live code and its FirstLight copy on sys.path)
    from native_runner.training.v4.checkpoint import save_actor_critic_checkpoint
    from native_runner.training.v4.imitation import imitation_loss, slice_il_sequence
    from native_runner.training.v4.learning import evaluate_recurrent_sequence

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not args.smoke:
        raise SystemExit('training needs CUDA (the Mac is for --smoke checks only)')
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / 'train-metrics.jsonl').open('a')

    units = select_units(args.frames)
    train_units = [u for u in units if not held_out(u[0])]
    val_units = [u for u in units if held_out(u[0])]
    if args.smoke:
        train_units, val_units, args.batch, args.workers = train_units[:2], val_units[:1] or train_units[:1], 2, 0
    print(f'{len(units)} Hog 2.6 sides: {len(train_units)} train, {len(val_units)} validation; device {device}',
          flush=True)

    policy = load_policy(args.init, device)
    head = None
    if args.extras:
        from il.extras import attach_extras
        head = policy.__dict__.get('_extras_head') or attach_extras(policy)
        head.train()
    policy.train()
    groups = [{'params': list(policy.parameters())}]
    if head is not None:
        groups.append({'params': list(head.parameters()), 'lr': args.learning_rate * args.head_lr_mult})
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    all_parameters = [p for group in groups for p in group['params']]

    def payload(**values) -> dict:
        from il.extras import extras_payload
        return {**values, **(extras_payload(head) if head is not None else {})}
    autocast = device.type == 'cuda'
    max_turns = 64 if args.smoke else None
    val_items = [ReplaySequences(val_units[:32], max_turns, args.extras)[i] for i in range(min(32, len(val_units)))]
    val_sequences = [item['sequence'] for item in val_items if item.get('sequence') is not None]
    if val_sequences:
        row = {'event': 'validation', 'update': 0, **evaluate(policy, val_sequences, device, args.time_steps)}
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + '\n')

    update, started = 0, time.time()
    for epoch in range(args.epochs):
        order = list(train_units)
        random.Random(args.seed + epoch).shuffle(order)
        loader = DataLoader(ReplaySequences(order, max_turns, args.extras), batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=_keep, persistent_workers=False,
                            prefetch_factor=2 if args.workers else None)
        for group_index, items in enumerate(loader):
            for item in items:
                if 'error' in item:
                    print('skipped', item['error'], flush=True)
            sequences = [item['sequence'] for item in items if item.get('sequence') is not None]
            if not sequences:
                continue
            batch = batch_sequences(sequences)
            state = policy.initial_state(batch.batch_size, device=device)
            metrics = Counter()
            for start in range(0, batch.time_steps, args.time_steps):
                chunk = slice_il_sequence(batch, start, min(batch.time_steps, start + args.time_steps)).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                    evaluation = evaluate_recurrent_sequence(
                        policy, chunk.observations, chunk.actions, chunk.episode_start, initial_state=state,
                        gate_temperature=1.0, action_temperature=1.0, continue_temperature=1.0,
                        validate=False, preencode_observations=True)
                    losses = imitation_loss(chunk, evaluation)
                if not torch.isfinite(losses.total):
                    raise FloatingPointError(f'non-finite loss at epoch {epoch} group {group_index} turn {start}')
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(all_parameters, args.max_grad_norm)
                optimizer.step()
                state = evaluation.final_state.detach()
                update += 1
                for name in ('gate', 'candidate', 'target', 'delay'):
                    count = float(getattr(losses, f'{name}_count').detach())
                    metrics[name] += float(getattr(losses, name).detach()) * count
                    metrics[f'{name}_count'] += count
                if args.smoke and update >= 2:
                    break
            row = {'event': 'group', 'epoch': epoch, 'group': group_index, 'update': update,
                   'sequences': len(sequences), 'turns': batch.time_steps, 'elapsed_s': round(time.time() - started),
                   **{h: round(metrics[h] / metrics[f'{h}_count'], 4) for h in ('gate', 'candidate', 'target', 'delay')
                      if metrics[f'{h}_count']}}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + '\n')
            log.flush()
            if args.smoke or (group_index + 1) % args.save_every == 0:
                path = args.out / f'checkpoint-{update:08d}.pt'
                save_actor_critic_checkpoint(path, policy, optimizer=optimizer, update_step=update,
                                             training_stage='imitation', gamma_per_decision=0.997,
                                             extra=payload(init=args.init, epoch=epoch, group=group_index,
                                                           recipe='clapha il.train: live-code samples, bot timing',
                                                           extras=args.extras))
                if val_sequences:
                    row = {'event': 'validation', 'update': update,
                           **evaluate(policy, val_sequences, device, args.time_steps)}
                    print(json.dumps(row), flush=True)
                    log.write(json.dumps(row) + '\n')
            if args.smoke:
                return 0
    path = args.out / f'checkpoint-{update:08d}.pt'
    save_actor_critic_checkpoint(path, policy, optimizer=optimizer, update_step=update, training_stage='imitation',
                                 gamma_per_decision=0.997, extra=payload(init=args.init, epochs=args.epochs,
                                                                         recipe='clapha il.train', extras=args.extras))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
