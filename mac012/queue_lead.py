"""How early each player's commands are visible in our command queue, from a recorded session.

The game executes a command 21 ticks after its issue tick, for both players. Our client sees
the opponent's command some time after they issued it (their round trip plus the server's
relay), and our own a few ticks after our tap. The difference between "first seen in our
queue" and "executes" is how much warning the queue gives -- i.e. how far ahead of the board
a bot reading the queue knows a play is coming. This measures it from the viewer's
queue.jsonl (artifacts/viewer-sessions/<session>/queue.jsonl).

    python3 mac012/queue_lead.py [SESSION_DIR]      (default: the newest session)
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viewer as V  # noqa: E402

COMMAND_AGE_TICKS = 21


def main() -> int:
    sessions = sorted((V.CLAPHA / 'artifacts' / 'viewer-sessions').glob('*/queue.jsonl'))
    path = Path(sys.argv[1]) / 'queue.jsonl' if len(sys.argv) > 1 else (sessions[-1] if sessions else None)
    if path is None or not path.is_file():
        print('no queue.jsonl found; pass a session directory')
        return 1
    first_seen: dict[tuple, tuple[int, int | None, int]] = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        tick = row.get('tick_0x60')
        if tick is None:
            continue
        for entry in (row.get('queue') or {}).get('entries', []):
            if entry.get('card_id', 0) <= 0 or not isinstance(entry.get('issue_tick'), int):
                continue
            key = (row.get('battle'), entry.get('account_lo'), entry.get('seq'),
                   entry['issue_tick'], entry['card_id'])
            if key not in first_seen:
                first_seen[key] = (int(tick), V.entry_side(entry, row.get('accounts')),
                                   entry['issue_tick'])
    by_side: dict = {}
    for (_battle, _account, _seq, _issue, card), (seen, side, issue) in first_seen.items():
        lead = issue + COMMAND_AGE_TICKS - seen          # ticks of warning before it executes
        delay = seen - issue                             # ticks from issue to our client
        by_side.setdefault(side, []).append((lead, delay, card))
    print(f'{path}')
    for side, rows in sorted(by_side.items(), key=lambda item: str(item[0])):
        leads = sorted(r[0] for r in rows)
        delays = sorted(r[1] for r in rows)
        q = lambda values, p: values[min(len(values) - 1, int(p * len(values)))]  # noqa: E731
        print(f'side {side}: {len(rows)} commands | visible before execution: median '
              f'{statistics.median(leads):.0f} ticks ({statistics.median(leads) * 50:.0f} ms), '
              f'p10 {q(leads, 0.1)}, p90 {q(leads, 0.9)} | issue -> our queue: median '
              f'{statistics.median(delays):.0f} ticks')
    print('A lead of 21 would mean seen at issue; 0 would mean seen only as it executes. '
          'Sampling adds up to one queue interval of error.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
