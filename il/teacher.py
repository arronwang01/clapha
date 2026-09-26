"""Self-distillation labels: what no-delay fl:hog2 would do, moved 25 ticks earlier.

Why: FirstLight's models play right in their sandbox (a play lands 1-5 ticks after the decision)
and late in the real game (~25 ticks: 21 of command age plus ~4 of ours); fl:hog2 with the real
delay lost 0-17 to itself without it. The student plays with the real delay, so at turn k it must
pick the play that lands when the no-delay teacher, at turn k + 5 (25 ticks later), would play.
It sees the board at k plus the pending commands of both sides (Stage B inputs); the teacher sees
those commands landed at k + 5. That pairing is what teaches the student to use them.

Two phases:
  label (main() here, on a GPU machine): the teacher runs over every Hog 2.6 game side in
    FirstLight's own timing (actor_sequence delay=0: hand, elixir and board as the engine has
    them, plays recorded at their execution turn). Per turn it keeps p_act, the gate's probability
    of acting at temperature 1 (the live decode acts when it is above 0.5), and its greedy action
    given that it acts: card identity, cell and delay bin for up to two micro actions. One small
    .npz per side under <frames>/teacher/.
  train (il/train.py --teacher): student_labels() maps each student turn at tick t to the
    teacher's labels at t + 25 on the student's own candidates, matched by card identity rather
    than hand slot (the hand has moved on by then), with the delay bin re-timed for this side's
    delay; teacher_loss() is the binary cross-entropy of the student's act probability against
    p_act, plus the card / cell / delay terms weighted by p_act.

    python -m il.teacher --frames runs/conv-hog26 --teacher fl:hog2 [--workers 14 --batch 16]
Resumable: sides that already have a label file are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
if str(CLAPHA) not in sys.path:
    sys.path.insert(0, str(CLAPHA))

LABELS_VERSION = 'clapha-teacher-labels.v1'
SHIFT_TICKS = 25            # our delays are 23-27 ticks: five 5-tick decision turns; bins take the rest
MAX_MICRO = 2
KEY_FIELDS = ('variant', 'native_visible_card_id', 'effective_card_vocab_id', 'effective_form',
              'ability_vocab_id', 'native_source_entity')


def label_path(frames_dir: Path, replay_file: str, actor: int) -> Path:
    tag = Path(replay_file).name.split('.')[0]
    return Path(frames_dir) / 'teacher' / tag[:2] / f'{tag}-{actor}.npz'


# ----------------------------------------------------------------------------- phase 1: label

class TeacherViews:
    """torch Dataset: one side's turns in FirstLight's own timing, with each turn's tick."""

    def __init__(self, units):
        self.units = list(units)

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int):
        from il.frames import load_replay
        from il.samples import actor_sequence
        path, actor = self.units[index]
        stats: Counter = Counter()
        try:
            header, frames = load_replay(Path(path))
            sequence, ticks = actor_sequence(header, frames, actor, stats, delay=0, extras=False, with_ticks=True)
        except Exception as error:  # noqa: BLE001  (one bad replay must not stop the job)
            return {'error': f'{Path(path).name} actor {actor}: {type(error).__name__}: {error}'[:300]}
        if sequence is None:
            return {'error': f'{Path(path).name} actor {actor}: no turns'}
        return {'sequence': sequence, 'ticks': ticks, 'unit': (path, actor)}


def _collate_views(items):
    """In the worker: collate and pack (il/pack.py). -> (packed batch or None, [(unit, ticks)], errors)."""
    from il.pack import pack
    from il.train import batch_sequences
    errors = [item['error'] for item in items if 'error' in item]
    good = [item for item in items if item.get('sequence') is not None]
    meta = [(item['unit'], item['ticks']) for item in good]
    if not good:
        return None, meta, errors
    batch = batch_sequences([item.pop('sequence') for item in good], consume=True)
    return pack(batch), meta, errors


def teacher_outputs(model, batch, device, time_steps: int = 32) -> dict:
    """The teacher over a collated [T,B] batch in its own timing -> per turn and lane: p_act
    [T,B]; count [T,B] (micro actions of its greedy act-given-act choice, 0 if it cannot act);
    keys [T,B,2,F] (KEY_FIELDS of each chosen candidate); cell [T,B,2]; bin [T,B,2]."""
    import torch
    from native_runner.training.v4.decoding import ShadowCandidateLegality
    from native_runner.training.v4.imitation import slice_il_sequence
    from native_runner.training.v4.tensors import GATE_ACT, GATE_WAIT
    from il.fused import fused_context
    steps, lanes = batch.time_steps, batch.batch_size
    out = {'p_act': torch.zeros(steps, lanes),
           'count': torch.zeros(steps, lanes, dtype=torch.long),
           'keys': torch.full((steps, lanes, MAX_MICRO, len(KEY_FIELDS)), -1, dtype=torch.long),
           'cell': torch.full((steps, lanes, MAX_MICRO), -1, dtype=torch.long),
           'bin': torch.full((steps, lanes, MAX_MICRO), -1, dtype=torch.long)}
    state = model.initial_state(lanes, device=device)
    autocast = device.type == 'cuda'
    with torch.no_grad():
        for start in range(0, steps, time_steps):
            stop = min(steps, start + time_steps)
            chunk = slice_il_sequence(batch, start, stop).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                fused = fused_context(model, chunk.observations, chunk.episode_start, initial_state=state)
                flat = fused.flat_batch
                legal = ShadowCandidateLegality(flat.candidates, model.config).candidate_mask().any(dim=-1)
                gate = torch.where(legal, torch.full_like(legal, GATE_ACT, dtype=torch.long),
                                   torch.full_like(legal, GATE_WAIT, dtype=torch.long))
                # FirstLight's own "argmax downstream heads for rows whose gate is already ACT"
                # (act_after_preselected_act), over every row of the chunk at once
                decoded = model._decode(flat, fused.context, sample=False, forced_actions=None,
                                        gate_temperature=1.0, action_temperature=1.0, continue_temperature=1.0,
                                        validate=False, preselected_gate=gate)
            log_p = decoded.action_components.gate_log_prob.float()
            actions = decoded.actions
            p_act = torch.where(legal, log_p.exp(), torch.zeros_like(log_p))
            used = torch.arange(MAX_MICRO, device=device)[None, :] < actions.micro_action_count[:, None]
            rows = actions.candidate_index.clamp_min(0)
            keys = torch.stack([getattr(flat.candidates, name).gather(1, rows) for name in KEY_FIELDS], dim=-1)
            keys = torch.where(used[..., None], keys, torch.full_like(keys, -1))
            out['p_act'][start:stop] = fused.per_step(p_act).cpu()
            out['count'][start:stop] = fused.per_step(actions.micro_action_count).cpu()
            out['keys'][start:stop] = fused.per_step(keys).cpu()
            out['cell'][start:stop] = fused.per_step(torch.where(used, actions.target_cell, -1)).cpu()
            out['bin'][start:stop] = fused.per_step(torch.where(used, actions.delay_offset_bin, -1)).cpu()
            state = fused.final_state
    return out


def save_labels(path: Path, ticks, outputs: dict, lane: int, teacher: str) -> None:
    import numpy as np
    steps = len(ticks)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, version=np.array(LABELS_VERSION), teacher=np.array(teacher),
                            ticks=np.asarray(ticks, dtype=np.int32),
                            p_act=outputs['p_act'][:steps, lane].numpy().astype(np.float32),
                            count=outputs['count'][:steps, lane].numpy().astype(np.int8),
                            keys=outputs['keys'][:steps, lane].numpy().astype(np.int64),
                            cell=outputs['cell'][:steps, lane].numpy().astype(np.int16),
                            bin=outputs['bin'][:steps, lane].numpy().astype(np.int8))
    temporary.replace(path)


def load_labels(path: Path) -> dict:
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        if str(data['version']) != LABELS_VERSION:
            raise ValueError(f'{path.name}: labels {data["version"]}, expected {LABELS_VERSION}')
        return {name: data[name] for name in ('ticks', 'p_act', 'count', 'keys', 'cell', 'bin')} | \
               {'teacher': str(data['teacher'])}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=Path, default=CLAPHA / 'runs/conv-hog26')
    parser.add_argument('--teacher', default='fl:hog2')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch', type=int, default=16, help='game sides per teacher pass')
    parser.add_argument('--time-steps', type=int, default=32)
    parser.add_argument('--limit-units', type=int, default=0)
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader
    import il.samples  # noqa: F401  (live code and its FirstLight copy on sys.path)
    from il.pack import unpack
    from il.train import _one_thread, load_policy, select_units
    if sys.platform.startswith('linux'):
        torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    units = select_units(args.frames)
    todo = [unit for unit in units if not label_path(args.frames, unit[0], unit[1]).exists()]
    already = len(units) - len(todo)
    if args.limit_units:
        todo = todo[:args.limit_units]
    print(f'{len(units)} Hog 2.6 sides, {already} already labelled, labelling {len(todo)}; '
          f'teacher {args.teacher}; device {device}', flush=True)
    model = load_policy(args.teacher, device)
    model.eval()
    loader = DataLoader(TeacherViews(todo), batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=_collate_views, worker_init_fn=_one_thread,
                        prefetch_factor=1 if args.workers else None)
    started, done, turns, acting = time.time(), 0, 0, 0
    for group, (packed, meta, errors) in enumerate(loader):
        for error in errors:
            print('skipped', error, flush=True)
        if packed is None:
            continue
        batch = unpack(packed)
        del packed
        outputs = teacher_outputs(model, batch, device, args.time_steps)
        for lane, ((path, actor), ticks) in enumerate(meta):
            save_labels(label_path(args.frames, path, actor), ticks, outputs, lane, args.teacher)
            turns += len(ticks)
            acting += int((outputs['p_act'][:len(ticks), lane] > 0.5).sum())
        done += len(meta)
        if group % 10 == 0 or done == len(todo):
            elapsed = time.time() - started
            print(json.dumps({'event': 'labelled', 'sides': done, 'of': len(todo), 'turns': turns,
                              'teacher_acts_per_100_turns': round(100 * acting / max(1, turns), 2),
                              'elapsed_s': round(elapsed), 'sides_per_min': round(60 * done / max(1e-9, elapsed), 1)}),
                  flush=True)
    return 0


# ----------------------------------------------------------------------------- phase 2: train

@dataclass
class TeacherLabels:
    """Teacher targets on the student's turns, for one lane ([T,1]) or a collated batch ([T,B])."""
    actions: tuple            # ActionSequenceV4 per step: the teacher's choice on the student's candidates
    p_target: object          # [T,B] float: the teacher's act probability (0 where its play is not open to the student)
    gate_weight: object       # [T,B] float: 1 where the teacher has a turn 25 ticks later
    cond_weight: object       # [T,B] float: p_target where the play is matched (card / cell / delay terms)

    @property
    def time_steps(self) -> int:
        return len(self.actions)

    def slice(self, start: int, stop: int) -> 'TeacherLabels':
        return TeacherLabels(actions=self.actions[start:stop], p_target=self.p_target[start:stop],
                             gate_weight=self.gate_weight[start:stop], cond_weight=self.cond_weight[start:stop])

    def to(self, device) -> 'TeacherLabels':
        return TeacherLabels(actions=tuple(item.to(device) for item in self.actions),
                             p_target=self.p_target.to(device), gate_weight=self.gate_weight.to(device),
                             cond_weight=self.cond_weight.to(device))


def _match(candidates, key) -> int | None:
    import torch
    fields = torch.stack([getattr(candidates, name)[0].to(torch.long) for name in KEY_FIELDS], dim=-1)
    wanted = torch.as_tensor(key, dtype=torch.long)
    hits = ((fields == wanted[None, :]).all(dim=-1) & candidates.mask[0]).nonzero(as_tuple=False).flatten()
    return int(hits[0]) if hits.numel() else None


def student_labels(sequence, ticks, labels: dict, delay: int, stats: Counter) -> TeacherLabels:
    """The teacher's labels at tick t + 25 on the student's turn at tick t (one lane)."""
    import torch
    from native_runner.training.v4.tensors import GATE_ACT, GATE_WAIT, TARGET_GRID, ActionSequenceV4
    index = {int(tick): position for position, tick in enumerate(labels['ticks'])}
    actions, p_target, gate_weight, cond_weight = [], [], [], []
    for observation, tick in zip(sequence.observations, ticks):
        candidates = observation.candidates
        rows, uids, cells, bins = [-1] * MAX_MICRO, [-1] * MAX_MICRO, [-1] * MAX_MICRO, [-1] * MAX_MICRO
        matched, p, has_turn = 0, 0.0, 0.0
        position = index.get(int(tick) + SHIFT_TICKS)
        if position is None:
            stats['teacher: no turn 25 ticks later'] += 1
        else:
            has_turn = 1.0
            p_act = float(labels['p_act'][position])
            for step in range(int(labels['count'][position])):
                row = _match(candidates, labels['keys'][position][step])
                if row is None:
                    stats['teacher: card not open to the student'] += 1
                    break
                cell = int(labels['cell'][position][step])
                if int(candidates.target_mode[0, row]) == TARGET_GRID:
                    if cell < 0 or not bool(candidates.placement[0, row].reshape(-1)[cell]):
                        stats['teacher: cell not open to the student'] += 1
                        break
                else:
                    cell = -1
                # same landing tick: t + delay + b = (t + 25) + b'  ->  b = b' + 25 - delay
                shifted = min(4, max(0, int(labels['bin'][position][step]) + SHIFT_TICKS - int(delay)))
                if step:
                    shifted = max(shifted, bins[0])      # FirstLight keeps micro actions in order
                rows[step], uids[step], cells[step], bins[step] = row, int(candidates.uid[0, row]), cell, shifted
                matched += 1
            if matched:
                p = p_act
                stats['teacher: play matched'] += 1
            elif p_act > 0.5:
                stats['teacher: would act, play not open (target: wait)'] += 1
        actions.append(ActionSequenceV4(
            gate=torch.tensor([GATE_ACT if matched else GATE_WAIT], dtype=torch.long),
            micro_action_count=torch.tensor([matched], dtype=torch.long),
            candidate_index=torch.tensor([rows], dtype=torch.long),
            candidate_uid=torch.tensor([uids], dtype=torch.long),
            target_cell=torch.tensor([cells], dtype=torch.long),
            delay_offset_bin=torch.tensor([bins], dtype=torch.long)))
        p_target.append(p)
        gate_weight.append(has_turn)
        cond_weight.append(p if matched else 0.0)

    def column(values):
        return torch.tensor(values, dtype=torch.float32)[:, None]
    return TeacherLabels(actions=tuple(actions), p_target=column(p_target), gate_weight=column(gate_weight),
                         cond_weight=column(cond_weight))


def collate_labels(lanes: list[TeacherLabels], time_steps: int) -> TeacherLabels:
    """Lanes side by side over the batch's time_steps (FirstLight pads a short sequence by repeating
    its last turn with the masks cleared; the repeats here carry zero weights)."""
    import torch
    from native_runner.training.v4.tensors import concatenate_padded_tensor_records
    actions = tuple(concatenate_padded_tensor_records(tuple(lane.actions[min(step, lane.time_steps - 1)]
                                                            for lane in lanes))
                    for step in range(time_steps))

    def column(name):
        return torch.cat([torch.cat((value, value.new_zeros(time_steps - value.shape[0], 1)))
                          for value in (getattr(lane, name) for lane in lanes)], dim=1)
    return TeacherLabels(actions=actions, p_target=column('p_target'), gate_weight=column('gate_weight'),
                         cond_weight=column('cond_weight'))


def teacher_loss(labels: TeacherLabels, evaluation, *, delay_neighbor_weight: float = 0.2) -> dict:
    """evaluation: the student teacher-forced on labels.actions (il/fused.evaluate_forced)."""
    import torch
    from native_runner.training.v4.imitation import _delay_neighbor_smoothed_nll
    from native_runner.training.v4.tensors import GATE_ACT
    components = evaluation.components
    forced = components.gate_log_prob.float().clamp(max=0.0)                 # log p(forced gate)
    act_forced = torch.stack([item.gate for item in labels.actions]) == GATE_ACT
    other = torch.log(-torch.expm1(forced.clamp(max=-1e-7)))                 # log(1 - p(forced gate))
    log_act = torch.where(act_forced, forced, other).clamp(min=-30.0)
    log_wait = torch.where(act_forced, other, forced).clamp(min=-30.0)
    p = labels.p_target.to(forced.dtype)
    cross_entropy = -(p * log_act + (1.0 - p) * log_wait)
    gate_weight = labels.gate_weight.to(forced.dtype)
    gate = (cross_entropy * gate_weight).sum() / gate_weight.sum().clamp_min(1.0)

    counts = torch.stack([item.micro_action_count for item in labels.actions])
    micro = (torch.arange(MAX_MICRO, device=counts.device)[None, None, :] < counts[..., None]).to(forced.dtype)
    cells = torch.stack([item.target_cell for item in labels.actions])
    bins = torch.stack([item.delay_offset_bin for item in labels.actions])
    grid = micro * (cells >= 0).to(forced.dtype)
    weight = labels.cond_weight.to(forced.dtype)
    total_weight = weight.sum().clamp_min(1e-6)
    candidate = ((-(components.candidate_log_prob.float() * micro).sum(-1)) * weight).sum() / total_weight
    target = ((-(components.target_log_prob.float() * grid).sum(-1)) * weight).sum() / total_weight
    delay_nll = _delay_neighbor_smoothed_nll(components.delay_offset_log_probs.float(),
                                             components.delay_offset_legal_mask, bins,
                                             neighbor_weight=delay_neighbor_weight)
    delay = (((delay_nll * micro).sum(-1)) * weight).sum() / total_weight
    return {'total': gate + candidate + target + delay, 'gate': gate, 'candidate': candidate, 'target': target,
            'delay': delay, 'gate_count': gate_weight.sum(), 'cond_weight': labels.cond_weight.sum()}


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
