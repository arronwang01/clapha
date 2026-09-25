"""Audit imitation samples before anything trains on them.

For a spread of IL_Replay battles, every decision tick of both actors:
  leak        the sample built from the full replay must equal the sample built from the replay
              cut down to what the actor could know at that tick (restricted()). Any input that
              differs is using information from the future or from the opponent's hidden state.
  stamps      every input's known_at <= the decision tick.
  in_hand     every labelled card is in the actor's hand when it is chosen (labels in one window
              applied in order: the second play of a Hog + Ice Spirit is drawn by the first).
  pending     a card issued and not yet landed is not also in the hand.
  offset      each label lands 0-4 ticks into its window.
  opp_hand    where the deduced opponent hand is exact and no issued opponent command is still
              invisible, it equals their true hand from the deal (checks the cycle logic).
Counts: labels that fall before the first decision tick (lost), hand-certain share.

    ./py -m il.audit [--replays 300] [--root IL_Replay]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'ref-firstlight'))
sys.path.insert(0, str(CLAPHA))

from il.params import COMMAND_AGE_TICKS, FIRST_DECISION_TICK, command_delay, opponent_lead  # noqa: E402
from il.timeline import (_hand_after, build_timeline, decision_tick, decision_ticks,  # noqa: E402
                         restricted, sample_at)

DEFAULT_ROOT = Path.home() / 'Documents/GitHub/cr-engine-extraction/IL_Replay'


def replay_rows(root: Path, count: int):
    """`count` replays spread evenly over every parquet part."""
    import pyarrow.parquet as pq
    parts = sorted(glob.glob(str(root / 'replays' / '*.parquet')))
    per_part = max(1, count // len(parts))
    for part in parts:
        table = pq.read_table(part, columns=['payload_json'])
        step = max(1, table.num_rows // per_part)
        for index in range(0, table.num_rows, step)[:per_part]:
            yield json.loads(table.slice(index, 1).to_pylist()[0]['payload_json'])


def audit_timeline(timeline, stats: collections.Counter, failures: list) -> None:
    tag = timeline.replay_tag

    def fail(kind, actor, tick, detail):
        stats[f'FAIL {kind}'] += 1
        if len(failures) < 40:
            failures.append((kind, tag[:8], actor, tick, str(detail)[:200]))

    for actor in (0, 1):
        delay = command_delay(tag, actor)
        own = [p for p in timeline.plays if p.owner == actor]
        stats['labels'] += len(own)
        stats['labels lost before first decision tick'] += sum(
            decision_tick(p.lands, delay) < FIRST_DECISION_TICK for p in own)
        for tick in decision_ticks(timeline):
            sample = sample_at(timeline, actor, tick)
            stats['samples'] += 1
            stats['samples hand-certain'] += sample.hand_certain
            stats['samples with own pending'] += bool(sample.inputs['own_pending'])
            stats['samples with opponent pending'] += bool(sample.inputs['opp_pending'])
            stats['samples acting'] += bool(sample.labels)
            cut = sample_at(restricted(timeline, actor, tick), actor, tick)
            for key, value in sample.inputs.items():
                if cut.inputs.get(key) != value:
                    fail('leak', actor, tick, f'{key}: full {value} vs knowable {cut.inputs.get(key)}')
            for key, known in sample.known_at.items():
                if known > tick:
                    fail('stamp', actor, tick, f'{key} known at {known}')
            hand = list(sample.inputs['own_hand'])
            queue_next = [sample.inputs['own_next']]
            cards_so_far = [p.card_id for p in own
                            if decision_tick(p.lands, delay) < tick and p.kind == 'card']
            for label in sample.labels:
                if not 0 <= label['offset'] <= 4:
                    fail('offset', actor, tick, label)
                if label['kind'] != 'card':
                    continue
                if label['card_id'] not in hand:
                    fail('in_hand', actor, tick, f"{label['card_id']} not in {hand}")
                    break
                cards_so_far.append(label['card_id'])
                hand, _ = _hand_after(timeline.deals[actor], cards_so_far)
            for kind, card, _grid, _left in sample.inputs['own_pending']:
                if kind == 'card' and card in sample.inputs['own_hand']:
                    fail('pending', actor, tick, f'{card} pending and in hand')
            opp = sample.inputs['opp_hand']
            if opp.get('exact'):
                opponent = 1 - actor
                issued = [p for p in timeline.plays if p.owner == opponent and p.kind == 'card'
                          and p.lands - COMMAND_AGE_TICKS <= tick]
                hidden = [p for p in issued if p.lands - opponent_lead(tag, p.index) > tick]
                if not hidden:
                    true_hand, _ = _hand_after(timeline.deals[opponent], [p.card_id for p in issued])
                    stats['opponent hands checked'] += 1
                    if sorted(true_hand) != opp['hand']:
                        fail('opp_hand', actor, tick, f"deduced {opp['hand']} true {sorted(true_hand)}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--replays', type=int, default=300)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    from native_runner.royaleapi_replay import prepare_collected_replay
    stats: collections.Counter = collections.Counter()
    failures: list = []
    for payload in replay_rows(args.root, args.replays):
        try:
            timeline = build_timeline(prepare_collected_replay(payload))
        except Exception as error:  # noqa: BLE001
            stats[f'prepare rejected: {type(error).__name__}'] += 1
            continue
        stats['replays'] += 1
        audit_timeline(timeline, stats, failures)
    for key, value in sorted(stats.items()):
        print(f'{key:45} {value}')
    samples = max(1, stats['samples'])
    print(f"\nacting share {stats['samples acting'] / samples:.1%}, hand-certain "
          f"{stats['samples hand-certain'] / samples:.1%}, labels lost early "
          f"{stats['labels lost before first decision tick'] / max(1, stats['labels']):.2%}")
    for failure in failures:
        print('  ', failure)
    bad = sum(v for k, v in stats.items() if k.startswith('FAIL'))
    print('AUDIT', 'PASSED' if not bad else f'FAILED ({bad})')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
