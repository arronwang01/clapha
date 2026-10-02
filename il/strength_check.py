"""Is our model weaker than training says, and was training's FirstLight weaker than the real one?
Four measurements on the Mac engine (CR_4k), one game at a time. Whenever a live match runs on either
console the game in progress is dropped and CR_4k is shut down until the match is over (the emulator's
load lags live play), then the same game is played again.

  replica   our model vs hog2 as training fed it (our pipeline, no delay), the specialist's deck:
            il.duel --specialist --together (training's 66% at update 520)
  plain     the same with no evolution or hero on either side (the deck the user owns)
  pipeline  hog2 as training fed it vs hog2 in its own environment, both no delay: 50% if our pipeline
            costs it nothing (il.vs_firstlight --ours fl:hog2 --ours-delay none)
  full      our model vs hog2 in its own environment at full strength (il.vs_firstlight)

replica, pipeline and full play the same deals (seeds 2001..., our side by seed), so games pair up.
Rows: runs/strength-check.jsonl; recordings in runs/rl/check-<test> (Training games, Watch in Null's).

    ./py -m il.strength_check [--tests replica,plain,pipeline,full] [--games 8]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
OUT = CLAPHA / 'runs' / 'strength-check.jsonl'
OURS = 'clapha:p3'
DUEL = ['--a-delay', 'live', '--a-lead', 'auto', '--target-delay', '26', '--b', 'fl:hog2', '--b-delay', 'none',
        '--b-lead', '0', '--specialist', '--together']
TESTS = {
    'replica': ('duel', 2001, ['--a', OURS] + DUEL),
    'plain': ('duel', 3001, ['--a', OURS] + DUEL + ['--plain']),
    'pipeline': ('fl', 2001, ['--ours', 'fl:hog2', '--ours-delay', 'none']),
    'full': ('fl', 2001, ['--ours', OURS]),
}


def say(text: str) -> None:
    print(f'{time.strftime("%H:%M:%S")} {text}', flush=True)


NO_PAUSE = False                 # --no-pause: the user plays by hand, no bot to lag


def live_match() -> bool:
    """Either console's reader sees a battle (the camp loop's own test: live_candidate / warming_up)."""
    if NO_PAUSE:
        return False
    for port in (8777, 8778):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/state', timeout=3) as response:
                state = json.load(response)
        except Exception:  # noqa: BLE001  (console not running)
            continue
        status = (state.get('reader') or {}).get('status', state.get('status'))
        if status in ('live_candidate', 'warming_up'):
            return True
    return False


def wait_for_quiet() -> None:
    import il.watch_nulls as W
    if not live_match():
        return
    say('a live match is running: CR_4k off until it is over')
    W.shutdown()
    quiet_since = None
    while True:
        time.sleep(10)
        if live_match():
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.time()
        elif time.time() - quiet_since > 120:
            say('no live match for 2 min: going on')
            return


def engine_for(kind: str) -> None:
    """duel: our run-lean probe (il.duel steps with it); fl: FirstLight's attested probe (its environment)."""
    import il.watch_nulls as W
    W.prepare_engine()                    # CR_4k up, FirstLight's probe, a fresh engine
    if kind == 'fl':
        return
    adb, serial = W.ADB, W.cr4k()
    lib = W._adb(serial, 'shell', f'ls -d /data/app/*/{W.PACKAGE}*/lib/arm64')
    subprocess.run((adb, '-s', serial, 'push', str(CLAPHA / 'runs/libcrprobe_run.so'), '/data/local/tmp/libcrprobe.run.so'),
                   capture_output=True, check=True)
    W._adb(serial, 'shell', f'am force-stop {W.PACKAGE}; cp /data/local/tmp/libcrprobe.run.so {lib}/libcrprobe.so; '
                            f'chown system:system {lib}/libcrprobe.so; chmod 755 {lib}/libcrprobe.so; '
                            f'restorecon {lib}/libcrprobe.so 2>/dev/null; true')
    import os
    env = dict(os.environ, CR_ADB=adb, CR_ADB_SERIAL=serial, CR_CONTROL_PORT=str(W.PORT))
    subprocess.run(('/bin/bash', str(W.PORT_DIR / 'start_offline.sh')), cwd=str(W.PORT_DIR), env=env,
                   capture_output=True, check=True, timeout=420)


def one_game(test: str, seed: int) -> dict | None:
    """Play one game; None if a live match interrupted it."""
    kind, _first, extra = TESTS[test]
    if kind == 'duel':
        out = CLAPHA / 'runs' / f'strength-check-{test}.duel.jsonl'
        command = [sys.executable, '-m', 'il.duel', *extra, '--matches', '1', '--seed', str(seed), '--out', str(out),
                   '--record', str(CLAPHA / 'runs' / 'rl' / f'check-{test}' / 'recordings')]
    else:
        command = [sys.executable, '-m', 'il.vs_firstlight', *extra, '--games', '1', '--seed', str(seed),
                   '--run', f'check-{test}']
    started = time.time()
    process = subprocess.Popen(command, cwd=str(CLAPHA), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines: list[str] = []
    import threading
    reader = threading.Thread(target=lambda: lines.extend(process.stdout), daemon=True)
    reader.start()
    while process.poll() is None:
        time.sleep(5)
        if live_match():
            process.kill()
            say(f'{test} seed {seed}: dropped, a live match started')
            return None
    reader.join(timeout=10)
    if kind == 'duel':
        rows = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
        row = next((r for r in reversed(rows) if r.get('deck_from') == f'spec{seed:08d}'), None)
        if row is None:
            say(f'{test} seed {seed}: no result -- ' + ''.join(lines[-3:]).strip()[-300:])
            return {'test': test, 'seed': seed, 'result': 'error'}
        result = {'a': 'ours', 'b': 'theirs'}.get(row['result'], 'draw')
        crowns, end_tick = row['crowns'], row['end_tick']
    else:
        rows = [json.loads(line) for line in lines if line.startswith('{"game"')]
        if not rows:
            say(f'{test} seed {seed}: no result -- ' + ''.join(lines[-3:]).strip()[-300:])
            return {'test': test, 'seed': seed, 'result': 'error'}
        result, crowns, end_tick = rows[-1]['winner'], rows[-1]['crowns'], rows[-1]['end_tick']
    return {'test': test, 'seed': seed, 'result': result, 'crowns': crowns, 'end_tick': end_tick,
            'seconds': round(time.time() - started), 'time': round(time.time())}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests', default='replica,plain,pipeline,full')
    parser.add_argument('--games', type=int, default=8)
    parser.add_argument('--no-pause', action='store_true', help='keep going while a live match runs')
    args = parser.parse_args(argv)
    global NO_PAUSE
    NO_PAUSE = args.no_pause
    tests = [t.strip() for t in args.tests.split(',') if t.strip()]
    done = {(r['test'], r['seed']) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    current_kind = None
    kinds = list(dict.fromkeys(TESTS[t][0] for t in tests))       # one probe swap per kind, in the order asked
    for test in sorted(tests, key=lambda t: kinds.index(TESTS[t][0])):
        kind, first, _extra = TESTS[test]
        games = min(args.games, 6) if test == 'pipeline' else args.games
        score = {'ours': 0, 'theirs': 0, 'draw': 0}
        for seed in range(first, first + games):
            if (test, seed) in done:
                continue
            while True:
                wait_for_quiet()
                if current_kind != kind:
                    say(f'engine for {kind}')
                    engine_for(kind)
                    current_kind = kind
                row = one_game(test, seed)
                if row is None:
                    current_kind = None
                    continue
                break
            with OUT.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            if row['result'] in score:
                score[row['result']] += 1
            say(f"{test} seed {seed}: {row['result']} {row.get('crowns')} ({row.get('seconds')} s) -- "
                f"{test} so far: ours {score['ours']}, theirs {score['theirs']}, draws {score['draw']}")
    say('all done')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
