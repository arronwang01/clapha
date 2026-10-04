"""Run src/runtime_probe.c during a battle and summarise what moved.

Finds, on this build, where the game keeps (a) hero/champion ability controllers -- cooldown,
charges, button state -- and (b) per-deck-slot evolution progress. FirstLight's model reads
both; our reader does not yet, so the bot cannot use hero abilities and its evolution state
is derived rather than read.

Play normally while it runs: deploy the hero (Hero Musketeer), use its ability at least once,
and cycle your evolution cards (Skeletons, Cannon) through at least one evolution.

    python3 mac012/run_runtime_probe.py [SECONDS]      (default 150)

Writes build/runtime_probe.jsonl and prints the summary to paste back.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_profile import ADB, CLAPHA, MANAGER_RVA, ROOT_CONTEXT_OFFSET, SERIAL, apply  # type: ignore  # noqa: E402

apply()
from native_core.mumu_live_protocol import verify_runtime  # noqa: E402


def summarize(rows) -> str:
    lines: list[str] = []
    controllers: dict = {}
    vectors: dict = {}
    for row in rows:
        for player in row.get('players') or []:
            if not player:
                continue
            slot = player['slot']
            for c in player.get('controllers', []):
                key = (slot, c['player_off'])
                seen = controllers.setdefault(key, {k: set() for k in c if k != 'player_off'})
                for k, v in c.items():
                    if k != 'player_off':
                        seen[k].add(v)
            for v in player.get('vectors8', []):
                key = (slot, v['player_off'])
                history = vectors.setdefault(key, [])
                if not history or history[-1][1] != v['values']:
                    history.append((row.get('tick'), v['values']))
    lines.append('controllers (player slot, offset): distinct values seen')
    for key, seen in sorted(controllers.items()):
        lines.append(f'  {key}: ' + ', '.join(f'{k}={sorted(v)[:8]}' for k, v in seen.items()))
    lines.append('8-slot int vectors (player slot, offset): value changes over the battle')
    for key, history in sorted(vectors.items()):
        if len(history) > 1:
            lines.append(f'  {key}: ' + ' -> '.join(f'@{t}:{v}' for t, v in history[:12]))
    lines.append('  (vectors that never changed are omitted)')
    return '\n'.join(lines)


def main() -> int:
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    runtime = verify_runtime(ADB, SERIAL)
    command = (f'/data/local/tmp/runtime_probe {runtime["pid"]} {hex(MANAGER_RVA)} '
               f'{hex(ROOT_CONTEXT_OFFSET)} {seconds * 4} 250')
    out = CLAPHA / 'build' / 'runtime_probe.jsonl'
    out.parent.mkdir(exist_ok=True)
    print(f'probing for {seconds} s -- deploy the hero, use its ability, cycle evolutions ...')
    rows = []
    with out.open('w') as log, subprocess.Popen([str(ADB), '-s', SERIAL, 'shell', command],
                                                stdout=subprocess.PIPE, text=True) as process:
        started = time.time()
        for line in process.stdout:
            log.write(line)
            if line.startswith('{'):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
            if time.time() - started > seconds + 10:
                process.kill()
                break
    print(f'{len(rows)} samples written to {out}')
    print(summarize(rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
