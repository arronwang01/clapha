"""Replay every recorded battle through FLO.build and report when battle_result() first calls
it, whether the call ever flickers, and the verdict -- the console stops tapping on this, so a
premature or unstable result would cost real turns.

    python3 mac012/test_battle_result.py [SESSION ...]

Every recorded session takes over an hour (it builds the full observation for each of ~300k
frames); name sessions to check just those. Not while a console is playing -- it competes for CPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import firstlight_obs as FLO  # noqa: E402

SESSIONS = Path(__file__).resolve().parents[1] / 'artifacts' / 'viewer-sessions'


def battles(path: Path):
    """(battle id, [(frame, health)]) per battle in a session, in order."""
    current, rows = None, []
    for line in path.open():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        frame, health = record.get('frame') or {}, record.get('health') or {}
        if not frame.get('battle_active') or not frame.get('coherent'):
            continue
        battle = (frame.get('chain') or {}).get('battle')
        if battle != current:
            if rows:
                yield current, rows
            current, rows = battle, []
        rows.append((frame, health))
    if rows:
        yield current, rows


def check(rows) -> dict:
    side = next((h.get('local_side') for _, h in rows if h.get('local_side') in (0, 1)), 1)
    state, first, flips, last = None, None, 0, None
    previous = None
    for frame, health in rows:
        _, state = FLO.build(frame, {**health, 'local_side': side}, '1', battle=state)
        result = FLO.battle_result(state, frame['game_tick'])
        if (result is None) != (previous is None) and previous is not None:
            flips += 1
        if result is not None and first is None:
            first = (frame['game_tick'], result)
        previous, last = result, (frame['game_tick'], result)
    return {'side': side, 'ticks': (rows[0][0]['game_tick'], rows[-1][0]['game_tick']),
            'first': first, 'final': last, 'withdrawn': flips}


def main(argv: list[str]) -> int:
    sessions = [SESSIONS / name for name in argv] or sorted(SESSIONS.iterdir())
    bad = 0
    for session in sessions:
        frames = session / 'frames.jsonl'
        if not frames.exists():
            continue
        for battle, rows in battles(frames):
            if len(rows) < 200:
                continue
            report = check(rows)
            first = report['first']
            verdict = 'undecided' if first is None else (
                f'decided at t={first[0] / 20:.1f}s: '
                + ('draw/tiebreak' if first[1][0] is None else
                   ('WIN' if first[1][0] == report['side'] else 'LOSS')) + f' ({first[1][1]})')
            ended = report['ticks'][1]
            # Decided more than ~6 s before the recording's clock stopped would be premature.
            premature = first is not None and ended - first[0] > 120 and ended < 6100
            bad += premature or report['withdrawn'] > 0
            print(f'{session.name} side {report["side"]} ticks {report["ticks"][0]}..{ended}: '
                  f'{verdict}' + (f'  WITHDRAWN {report["withdrawn"]}x' if report['withdrawn'] else '')
                  + ('  PREMATURE?' if premature else ''))
    print('OK' if not bad else f'{bad} suspicious')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
