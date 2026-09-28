"""What an RL run is doing, from its records (runs/rl/<run>/games/games.jsonl, learn.jsonl).

Per stretch of games (by the learner's weights version): games per minute, the learner's score per
league opponent (the anchor row is the running score against where we started), its reward pieces
(princess damage dealt / taken, King activations by its own spell), its habits (Log / Ice Golem
targets, seconds at full elixir), and what the learner's updates did (losses, entropy, KL to the
start, clip fraction).

    ./py -m il.rl_report runs/rl/<run> [--every 10]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue            # a line being written
    return rows


def _score(games: list[dict]) -> str:
    decided = [g for g in games if g.get('learner_won') is not None]
    if not decided:
        return '-'
    wins = sum(1 for g in decided if g['learner_won'])
    rate = wins / len(decided)
    spread = 1.96 * math.sqrt(rate * (1 - rate) / len(decided)) if len(decided) > 1 else 0.0
    return f'{wins}-{len(decided) - wins} ({100 * rate:.0f}% +-{100 * spread:.0f})'


def report(run: Path, every: int) -> str:
    games = _rows(run / 'games' / 'games.jsonl')
    updates = _rows(run / 'learn.jsonl')
    out = [f'# {run.name}: {len(games)} games, {len(updates)} updates']
    if games:
        span = (games[-1]['time'] - games[0]['time']) / 60 if len(games) > 1 else 0
        recent = [g for g in games if g['time'] >= games[-1]['time'] - 600]
        out.append(f'{len(games) / span:.1f} games/min overall, {len(recent) / 10:.1f} in the last 10 min'
                   if span else '')
    by_stretch: dict[int, list[dict]] = defaultdict(list)
    for game in games:
        by_stretch[int(game.get('learner_version', 0)) // every * every].append(game)
    out.append('')
    out.append('| updates | games | vs self | vs snapshots | vs v2 (anchor) | vs hog2 no-delay | vs General (real decks) |'
               ' dealt | taken | King activ. | full elixir s | Log on nothing | Ice Golem on building-only |')
    out.append('|' + '---|' * 13)
    for start in sorted(by_stretch):
        stretch = by_stretch[start]
        leagues = defaultdict(list)
        for game in stretch:
            leagues[game['league']].append(game)
        sides = [g['sides']['learner'] for g in stretch if 'sides' in g]
        n = max(1, len(sides))
        habits = Counter()
        for side in sides:
            habits.update(side.get('habits', {}))
        logs = sum(v for k, v in habits.items() if k.startswith('Log'))
        golems = sum(v for k, v in habits.items() if k.startswith('Ice Golem'))
        out.append(
            f"| {start}-{start + every - 1} | {len(stretch)} | {_score(leagues['self'])} | {_score(leagues['snap'])} | "
            f"{_score(leagues['anchor'])} | {_score(leagues['hog2'])} | {_score(leagues['general'])} | "
            f"{sum(s.get('dealt', 0) for s in sides) / n:.2f} | {sum(s.get('taken', 0) for s in sides) / n:.2f} | "
            f"{100 * sum(1 for s in sides if s.get('king_activation') is not None) / n:.0f}% | "
            f"{sum(s.get('full_elixir_s', 0) for s in sides) / n:.0f} | "
            f"{100 * habits.get('Log: nothing', 0) / max(1, logs):.0f}% of {logs} | "
            f"{100 * habits.get('Ice Golem: building-targeters only', 0) / max(1, golems):.0f}% of {golems} |")
    if updates:
        out.append('')
        out.append('| update | value only | lanes | s | mean return | policy loss | value loss | entropy | KL to start | clip frac | steps | grad agree |')
        out.append('|' + '---|' * 12)
        for row in updates[-12:]:
            out.append(f"| {row['update']} | {row['value_only']} | {row['lanes']} | {row['seconds']} | "
                       f"{row['mean_return']:+.3f} | {row['policy_loss']:+.4f} | {row['value_loss']:.4f} | "
                       f"{row['entropy']:.3f} | {row['kl_ref']:.4f} | {row['clip_fraction']:.3f} | "
                       f"{row.get('optimizer_steps', '-')} | "
                       f"{'-' if row.get('grad_cos') is None else format(row['grad_cos'], '+.3f')} |")
    if any(row.get('grad_cos') is not None for row in updates):
        out.append('')
        out.append('grad agree: cosine between the gradients of the two halves of an update\'s games. Near 0: the '
                   'update is noise; clearly above 0: the games agree on a direction.')
    return '\n'.join(out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--every', type=int, default=10, help='updates per row')
    args = parser.parse_args(argv)
    print(report(args.run, args.every))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
