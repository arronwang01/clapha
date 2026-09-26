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
    """(replay file, actor) for every Hog 2.6 side of an exactly replayed game. Cached in
    units.json next to the index (reading every header takes minutes), keyed by the index size."""
    index_text = (frames_dir / 'index.jsonl').read_text()
    cache = frames_dir / 'units.json'
    key = [len(index_text), max_tower_error]
    if cache.exists():
        try:
            saved = json.loads(cache.read_text())
            if saved.get('key') == key:
                return [(str(frames_dir / relative), int(actor)) for relative, actor in saved['units']]
        except (ValueError, KeyError):
            pass
    selected = _select_units(frames_dir, index_text, max_tower_error)
    try:
        cache.write_text(json.dumps({'key': key, 'units': [[str(Path(path).relative_to(frames_dir)), actor]
                                                           for path, actor in selected]}))
    except OSError:
        pass
    return selected


def _select_units(frames_dir: Path, index_text: str, max_tower_error: int) -> list[tuple[str, int]]:
    from il.samples import HOG26_CARDS
    units = []
    for line in index_text.splitlines():
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
            units.append((str(path), record.get('train_sides')))
    selected = []
    from il.frames import load_header
    for path, sides in units:
        header = load_header(Path(path))
        for actor, deck in enumerate(header['timeline']['decks']):
            # a played match (il/duel.py --record) lists the sides to learn from (the student's)
            if HOG26_CARDS <= set(deck) and (sides is None or actor in sides):
                selected.append((path, actor))
    return selected


def held_out(path: str) -> bool:
    return int(hashlib.sha256(Path(path).name.encode()).hexdigest()[:8], 16) % 50 == 0


class ReplaySequences:
    """torch Dataset: one ILSequenceV4 per (replay, actor), made by the live code; with the
    teacher's labels on the same turns (il/teacher.py) when `teacher` is the frames directory."""

    def __init__(self, units, max_turns: int | None = None, extras: bool = False, teacher: Path | None = None):
        self.units = list(units)
        self.max_turns = max_turns
        self.extras = extras
        self.teacher = teacher

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int):
        from il.frames import load_replay
        from il.samples import actor_sequence
        from native_runner.training.v4.imitation import slice_il_sequence
        path, actor = self.units[index]
        stats: Counter = Counter()
        labels = None
        try:
            header, frames = load_replay(Path(path))
            if self.teacher is None:
                sequence = actor_sequence(header, frames, actor, stats, extras=self.extras)
            else:
                from il.params import command_delay
                from il.teacher import label_path, load_labels, student_labels
                sequence, ticks = actor_sequence(header, frames, actor, stats, extras=self.extras, with_ticks=True)
                if sequence is not None:
                    teacher = load_labels(label_path(self.teacher, path, actor))
                    labels = student_labels(sequence, ticks, teacher, command_delay(header['replay_tag'], actor),
                                            stats)
        except Exception as error:  # noqa: BLE001  (one bad replay must not stop training)
            return {'error': f'{Path(path).name} actor {actor}: {type(error).__name__}: {error}'[:300]}
        if sequence is not None and self.max_turns and sequence.time_steps > self.max_turns:
            sequence = slice_il_sequence(sequence, 0, self.max_turns)
            if labels is not None:
                labels = labels.slice(0, self.max_turns)
        item = {'sequence': sequence, 'stats': dict(stats)}
        if labels is not None:
            item['teacher'] = labels
        return item


def _keep(items):
    return items


def _collate(items):
    """Runs in the DataLoader worker: pad and collate the group there, and hand it over as one
    packed buffer (il/pack.py). A batch holds ~10^5 small tensors; passed as tensors, each crossed
    processes through its own shared-memory handle, which on Windows stalled a run for hours
    without one batch arriving, and plain pickling costs seconds per batch on both sides.
    The sequences are dropped as soon as they are collated, to keep the worker's peak down.
    -> (packed batch or None, errors, sequence count)."""
    from il.pack import pack
    errors = [item['error'] for item in items if 'error' in item]
    good = [item for item in items if item.get('sequence') is not None]
    count = len(good)
    if not good:
        return None, errors, 0
    steps = max(item['sequence'].time_steps for item in good)
    labels = [item.pop('teacher') for item in good if 'teacher' in item]
    batch = batch_sequences([item.pop('sequence') for item in good], consume=True)
    if labels:
        from il.teacher import collate_labels
        if len(labels) != count:
            raise ValueError('teacher labels missing for part of a batch')
        packed = pack((batch, collate_labels(labels, steps)))
    else:
        packed = pack(batch)
    del batch
    return packed, errors, count


def merged_groups(loader, merge: int):
    """Join `merge` worker groups into one update batch in the training process: workers stay at
    small groups (a worker's memory peaks at ~2 GB per 8 sides while it collates, and they all
    peak together), updates still see merge x as many sides. Joining re-pads and collates the
    groups exactly as one bigger group would be (checked: identical losses and label counts).
    Yields (last worker group index, batch, teacher labels or None, sequence count)."""
    from il.pack import unpack
    parts, teachers, count, group_index = [], [], 0, -1
    for group_index, (packed, errors, sequence_count) in enumerate(loader):
        for error in errors:
            print('skipped', error, flush=True)
        if packed is None:
            continue
        item = unpack(packed)
        del packed
        batch, teacher = item if isinstance(item, tuple) else (item, None)
        parts.append(batch)
        teachers.append(teacher)
        count += sequence_count
        if len(parts) >= merge:
            yield (group_index, *_join(parts, teachers), count)
            parts, teachers, count = [], [], 0
    if parts:
        yield (group_index, *_join(parts, teachers), count)


def _join(parts: list, teachers: list):
    if len(parts) == 1:
        return parts[0], teachers[0]
    steps = max(part.time_steps for part in parts)
    teacher = None
    if teachers[0] is not None:
        from il.teacher import collate_labels
        teacher = collate_labels(teachers, steps)
    return batch_sequences(parts, consume=True), teacher


def _one_thread(_worker: int) -> None:
    """Each DataLoader worker tensorizes on one CPU thread: thirty workers each opening a
    full-width torch thread pool would oversubscribe the machine."""
    import torch
    torch.set_num_threads(1)


def _rows(collection) -> int:
    from dataclasses import fields
    for item in fields(collection):
        value = getattr(collection, item.name)
        if getattr(value, 'ndim', 0) >= 2:
            return int(value.shape[1])
    return 0


def batch_sequences(sequences, consume: bool = False):
    """FirstLight's padded collation, after padding every turn's variable collections (active
    effects, relation edges, candidates) to the batch's largest, as their cache loader does.
    consume=True empties the given list while padding, so each original is freed once padded."""
    from native_runner.training.v4.cache import _pad_sequence_dynamic_observations
    from native_runner.training.v4.imitation import collate_padded_il_sequences
    counts = {name: max(_rows(getattr(observation, name)) for sequence in sequences
                        for observation in sequence.observations)
              for name in ('active_effects', 'relation_edges', 'candidates')}

    def pad(sequence):
        return _pad_sequence_dynamic_observations(sequence, active_effect_count=counts['active_effects'],
                                                  relation_edge_count=counts['relation_edges'],
                                                  candidate_count=counts['candidates'])
    if consume:
        padded = []
        while sequences:
            padded.append(pad(sequences.pop(0)))
    else:
        padded = [pad(sequence) for sequence in sequences]
    return collate_padded_il_sequences(padded)


def load_policy(init: str, device):
    """A FirstLight checkpoint by name (firstlight_bot.CHECKPOINTS) or path; an extended one
    (il/extras.py) comes back with its head attached."""
    import firstlight_bot as FLB
    from il.extras import load_policy as load_extended
    return load_extended(FLB.CHECKPOINTS.get(init, Path(init)), device)


def evaluate_chunk(policy, chunk, state, fused: bool):
    """One [T,B] chunk teacher-forced: il/fused.py (one LSTM kernel, heads batched over T*B;
    checked equal to FirstLight's step-by-step evaluation) or their evaluate_recurrent_sequence."""
    if fused:
        from il.fused import evaluate_sequence_fused
        return evaluate_sequence_fused(policy, chunk.observations, chunk.actions, chunk.episode_start,
                                       initial_state=state)
    from native_runner.training.v4.learning import evaluate_recurrent_sequence
    return evaluate_recurrent_sequence(policy, chunk.observations, chunk.actions, chunk.episode_start,
                                       initial_state=state, gate_temperature=1.0, action_temperature=1.0,
                                       continue_temperature=1.0, validate=False, preencode_observations=True)


def evaluate(policy, batch, device, time_steps: int, fused: bool = True, teacher=None) -> dict:
    """Held-out losses per head, no gradient. batch: the collated validation sequences; teacher:
    their teacher labels (il/teacher.py), reported as t_gate, t_candidate, t_target, t_delay."""
    import torch
    from native_runner.training.v4.imitation import imitation_loss, slice_il_sequence
    totals, counts = Counter(), Counter()
    policy.eval()
    with torch.no_grad():
        state = policy.initial_state(batch.batch_size, device=device)
        for start in range(0, batch.time_steps, time_steps):
            stop = min(batch.time_steps, start + time_steps)
            chunk = slice_il_sequence(batch, start, stop).to(device)
            if teacher is None:
                evaluation = evaluate_chunk(policy, chunk, state, fused)
            else:
                from il.fused import evaluate_forced, fused_context
                from il.teacher import teacher_loss
                context = fused_context(policy, chunk.observations, chunk.episode_start, initial_state=state)
                evaluation = evaluate_forced(policy, context, chunk.actions)
                labels = teacher.slice(start, stop).to(device)
                taught = teacher_loss(labels, evaluate_forced(policy, context, labels.actions))
                _add_teacher_metrics(totals, counts, taught)
            losses = imitation_loss(chunk, evaluation)
            for head in ('gate', 'candidate', 'target', 'delay'):
                count = float(getattr(losses, f'{head}_count'))
                totals[head] += float(getattr(losses, head)) * count
                counts[head] += count
            state = evaluation.final_state
    policy.train()
    return {head: round(totals[head] / counts[head], 4) if counts[head] else None for head in totals}


def _add_teacher_metrics(totals: Counter, counts: Counter, taught: dict) -> None:
    gate_count, cond_weight = float(taught['gate_count']), float(taught['cond_weight'])
    totals['t_gate'] += float(taught['gate'].detach()) * gate_count
    counts['t_gate'] += gate_count
    for head in ('candidate', 'target', 'delay'):
        totals[f't_{head}'] += float(taught[head].detach()) * cond_weight
        counts[f't_{head}'] += cond_weight


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=Path, default=CLAPHA / 'runs/conv-hog26')
    parser.add_argument('--init', default='fl:hog2')
    parser.add_argument('--out', type=Path, default=CLAPHA / 'runs/train-hog26')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=8, help='sequences per update group')
    parser.add_argument('--time-steps', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--skip-sides', type=int, default=0,
                        help='continuing a run: skip this many sides of the first epoch (the same seeded order)')
    parser.add_argument('--merge', type=int, default=1,
                        help='worker groups joined per update batch (updates see batch x merge sides)')
    parser.add_argument('--prefetch', type=int, default=1,
                        help='batches in flight per worker (each ~0.6 GB, plus ~2 GB peak while one is built)')
    parser.add_argument('--val-sides', type=int, default=32, help='held-out game sides evaluated at each checkpoint')
    parser.add_argument('--step-eval', action='store_true',
                        help="FirstLight's step-by-step chunk evaluation instead of il/fused.py (same numbers, slower)")
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--save-every', type=int, default=200, help='update batches between checkpoints')
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--extras', action='store_true', help='Stage B inputs: pending commands, opponent elixir, delay')
    parser.add_argument('--head-lr-mult', type=float, default=10.0, help='learning rate multiplier for the new head')
    parser.add_argument('--limit-units', type=int, default=0, help='train on the first N sides only (tests)')
    parser.add_argument('--max-updates', type=int, default=0, help='stop after N updates (tests)')
    parser.add_argument('--teacher', action='store_true',
                        help='self-distillation: add the no-delay teacher labels in <frames>/teacher (il/teacher.py)')
    parser.add_argument('--teacher-weight', type=float, default=1.0)
    parser.add_argument('--human-weight', type=float, default=1.0, help='weight of the replay (human) labels')
    args = parser.parse_args(argv)
    if args.teacher and args.step_eval:
        raise SystemExit('--teacher needs the fused evaluation (drop --step-eval)')

    import torch
    from torch.utils.data import DataLoader
    import il.samples  # noqa: F401  (puts the live code and its FirstLight copy on sys.path)
    from native_runner.training.v4.checkpoint import save_actor_critic_checkpoint
    from native_runner.training.v4.imitation import imitation_loss, slice_il_sequence

    if sys.platform.startswith('linux'):
        # a batch holds ~10^5 small tensors; one file descriptor each exceeds the default limit
        torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not (args.smoke or args.max_updates):
        raise SystemExit('training needs CUDA (the Mac is for --smoke and --max-updates checks only)')
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / 'train-metrics.jsonl').open('a')

    units = select_units(args.frames)
    if args.teacher:
        from il.teacher import label_path
        labelled = [u for u in units if label_path(args.frames, u[0], u[1]).exists()]
        print(f'teacher labels for {len(labelled)} of {len(units)} sides', flush=True)
        units = labelled
    teacher_dir = args.frames if args.teacher else None
    train_units = [u for u in units if not held_out(u[0])]
    val_units = [u for u in units if held_out(u[0])]
    if args.smoke:
        train_units, val_units, args.batch, args.workers = train_units[:2], val_units[:1] or train_units[:1], 2, 0
    if args.limit_units:
        train_units, val_units = train_units[:args.limit_units], val_units[:2]
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
    val_data = ReplaySequences(val_units[:args.val_sides], max_turns, args.extras, teacher_dir)
    val_items = [item for item in (val_data[i] for i in range(len(val_data))) if item.get('sequence') is not None]
    val_batch, val_teacher = None, None
    if val_items:
        steps = max(item['sequence'].time_steps for item in val_items)
        if teacher_dir is not None:
            from il.teacher import collate_labels
            val_teacher = collate_labels([item.pop('teacher') for item in val_items], steps)
        val_batch = batch_sequences([item.pop('sequence') for item in val_items], consume=True)
    if val_batch is not None:
        row = {'event': 'validation', 'update': 0, 'sides': val_batch.batch_size,
               **evaluate(policy, val_batch, device, args.time_steps, not args.step_eval, val_teacher)}
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + '\n')

    update, started = 0, time.time()
    for epoch in range(args.epochs):
        order = list(train_units)
        random.Random(args.seed + epoch).shuffle(order)
        if epoch == 0 and args.skip_sides:
            order = order[args.skip_sides:]
            print(f'continuing: {args.skip_sides} sides of epoch 0 already trained, {len(order)} to go', flush=True)
        loader = DataLoader(ReplaySequences(order, max_turns, args.extras, teacher_dir), batch_size=args.batch,
                            shuffle=False,
                            num_workers=args.workers, collate_fn=_collate, persistent_workers=False,
                            worker_init_fn=_one_thread,
                            prefetch_factor=args.prefetch if args.workers else None)
        for step_index, (group_index, batch, teacher, sequence_count) in enumerate(merged_groups(loader, args.merge)):
            state = policy.initial_state(batch.batch_size, device=device)
            metrics, metrics_counts = Counter(), Counter()
            for start in range(0, batch.time_steps, args.time_steps):
                stop = min(batch.time_steps, start + args.time_steps)
                chunk = slice_il_sequence(batch, start, stop).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                    if teacher is None:
                        evaluation = evaluate_chunk(policy, chunk, state, not args.step_eval)
                        losses = imitation_loss(chunk, evaluation)
                        total = args.human_weight * losses.total
                    else:
                        from il.fused import evaluate_forced, fused_context
                        from il.teacher import teacher_loss
                        context = fused_context(policy, chunk.observations, chunk.episode_start, initial_state=state)
                        evaluation = evaluate_forced(policy, context, chunk.actions)
                        losses = imitation_loss(chunk, evaluation)
                        labels = teacher.slice(start, stop).to(device)
                        taught = teacher_loss(labels, evaluate_forced(policy, context, labels.actions))
                        total = args.human_weight * losses.total + args.teacher_weight * taught['total']
                        _add_teacher_metrics(metrics, metrics_counts, taught)
                if not torch.isfinite(total):
                    raise FloatingPointError(f'non-finite loss at epoch {epoch} group {group_index} turn {start}')
                total.backward()
                torch.nn.utils.clip_grad_norm_(all_parameters, args.max_grad_norm)
                optimizer.step()
                state = evaluation.final_state.detach()
                update += 1
                for name in ('gate', 'candidate', 'target', 'delay'):
                    count = float(getattr(losses, f'{name}_count').detach())
                    metrics[name] += float(getattr(losses, name).detach()) * count
                    metrics[f'{name}_count'] += count
                if (args.smoke and update >= 2) or (args.max_updates and update >= args.max_updates):
                    break
            row = {'event': 'group', 'epoch': epoch, 'batch': step_index, 'group': group_index, 'update': update,
                   'sequences': sequence_count, 'turns': batch.time_steps, 'elapsed_s': round(time.time() - started),
                   **{h: round(metrics[h] / metrics[f'{h}_count'], 4) for h in ('gate', 'candidate', 'target', 'delay')
                      if metrics[f'{h}_count']},
                   **{h: round(metrics[h] / metrics_counts[h], 4) for h in ('t_gate', 't_candidate', 't_target', 't_delay')
                      if metrics_counts[h]}}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + '\n')
            log.flush()
            if args.smoke or (step_index + 1) % args.save_every == 0:
                path = args.out / f'checkpoint-{update:08d}.pt'
                save_actor_critic_checkpoint(path, policy, optimizer=optimizer, update_step=update,
                                             training_stage='imitation', gamma_per_decision=0.997,
                                             extra=payload(init=args.init, epoch=epoch, group=group_index,
                                                           recipe='clapha il.train: live-code samples, bot timing',
                                                           extras=args.extras, teacher=args.teacher,
                                                           teacher_weight=args.teacher_weight,
                                                           human_weight=args.human_weight))
                if val_batch is not None:
                    row = {'event': 'validation', 'update': update,
                           **evaluate(policy, val_batch, device, args.time_steps, not args.step_eval, val_teacher)}
                    print(json.dumps(row), flush=True)
                    log.write(json.dumps(row) + '\n')
            if args.smoke or (args.max_updates and update >= args.max_updates):
                return 0
    path = args.out / f'checkpoint-{update:08d}.pt'
    save_actor_critic_checkpoint(path, policy, optimizer=optimizer, update_step=update, training_stage='imitation',
                                 gamma_per_decision=0.997, extra=payload(init=args.init, epochs=args.epochs,
                                                                         recipe='clapha il.train', extras=args.extras))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
