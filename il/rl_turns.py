"""Keep an RL run going on the shared 4080 PC, unattended for days: our run only while nobody else
uses the GPU, brought back when a part of it breaks, the engines (or their VM) restarted when they
are down, the disk kept in bounds, and a plain-text status for people to read.

Turns. The PC's GPU is shared with another user's jobs (a YOLOv8 training holding ~3-13 GB and up
to ~95% of the GPU while it runs). The user's rule (2026-09-27): when it is not training, it is our
turn. Every minute this reads Windows' per-process GPU memory counters and sums what processes
other than ours hold; it starts the run (tools/windows/rl.ps1) once others have held under
--others-gb for --calm minutes, and stops it within a minute of someone else taking memory. The
learner continues from latest.pt. The YOLO job then trained for days on end at ~67% of the GPU in
2.7 GB, and the user chose to share (2026-09-27 evening): start-training.cmd passes --others-gb 7, so
our run goes on next to it and backs off only when other processes hold more than 7 GB (our run
needs up to ~8 of the 16). Their job cycles: ~18 min training at ~3 GB, ~5 min at ~12 GB (its
validation, presumably); our run stops for the 12 GB stretch and starts again 2 min (--calm 2) after.

Health, while the run is ours. A part of it ended (the inference server, the learner, a collector:
a collector ends itself after 5 failed games in a row), or no game finished and no update was made
for --stall minutes: stop the run, ask every engine port for its status, restart the engines
(D:\\crtrain\\engine\\stage6c.ps1) when fewer than 3 in 4 answer -- or the VM and the engines
(stage7.ps1, which also puts the offline firewall back and checks it) when the VM itself does not
answer, e.g. after a reboot -- and start again on the engines that answer. At most one restart per
--restart-gap minutes.

Guard. pilot1 got worse than its start for a day before anyone looked. Now, once the policy moves
(after --guard-after updates, the value warm-up), the learner's last --guard-games games against
the fixed v2 are checked every 10 minutes: a score under --guard-below (42% of 150 is ~2 standard
errors under 50%) stops the run and writes runs/rl/<run>/HALT with the reason. Nothing restarts
while HALT exists; delete it to go on.

Disk. Under 30 GB free: recordings beyond the newest 200 and policy files other than every 50th
and the league's latest snapshots are deleted; under 8 GB the run stops until there is room.

Records in runs/rl/<run>/: turns.log (a line per minute and every action), status.txt (rewritten
every 10 minutes: GPU share, games, updates, restarts, il.rl_report's tables), launch.log and
engines.log (the scripts' output). One keeper per run (a lock file): start-training.cmd starts it
and is safe to run twice; stop-training.cmd stops it and the run.

    D:\\crtrain\\py312\\python.exe -m il.rl_turns --run pilot1 --engines 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
LAUNCHER = CLAPHA.parent / 'rl.ps1'          # the code update puts it next to clapha/
ENGINE_SCRIPTS = Path(r'D:\crtrain\engine')  # stage6c.ps1 (engines), stage7.ps1 (VM, firewall, engines)
ADB = Path(r'D:\crtrain\platform-tools\adb.exe')
SERIAL = '127.0.0.1:16416'                   # the engine VM, crengine12
BASE_PORT = 26789


def gpu_by_process() -> dict[int, int]:
    """pid -> dedicated GPU memory in MiB (Windows 'GPU Process Memory' counters; nvidia-smi shows
    N/A per process under WDDM)."""
    command = ("(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples"
               " | ForEach-Object { $_.InstanceName + ' ' + [int64]$_.CookedValue }")
    out = subprocess.run(['powershell', '-NoProfile', '-Command', command], capture_output=True, text=True,
                         timeout=120, stdin=subprocess.DEVNULL).stdout
    usage: dict[int, int] = {}
    for line in out.splitlines():
        name, _, value = line.strip().rpartition(' ')
        match = re.search(r'pid_(\d+)', name)
        if match and value.isdigit():
            pid = int(match.group(1))
            usage[pid] = usage.get(pid, 0) + int(value) // 2 ** 20
    return usage


def process_names() -> dict[int, str]:
    out = subprocess.run(['tasklist', '/fo', 'csv', '/nh'], capture_output=True, text=True, timeout=60,
                         stdin=subprocess.DEVNULL).stdout
    names = {}
    for line in out.splitlines():
        parts = [part.strip('"') for part in line.split('","')]
        if len(parts) > 1 and parts[1].isdigit():
            names[int(parts[1])] = parts[0].strip('"')
    return names


def ours(run_dir: Path) -> set[int]:
    pids = run_dir / 'pids.txt'
    return {int(line) for line in pids.read_text().split() if line.isdigit()} if pids.is_file() else set()


def powershell(script: Path, *arguments: str, timeout: int, log: Path) -> str:
    """Run a PowerShell script to its end and return what it printed. The output goes through a
    file, not a pipe: the processes a script starts inherit its handles and would hold a pipe open."""
    offset = log.stat().st_size if log.is_file() else 0
    with log.open('ab') as out:
        out.write(f"--- {time.strftime('%m-%d %H:%M:%S')} {script.name} {' '.join(arguments)}\n".encode())
        out.flush()
        try:
            subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script), *arguments],
                           stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, timeout=timeout,
                           cwd=str(script.parent))
        except subprocess.TimeoutExpired:
            out.write(f'did not finish in {timeout // 60} min\n'.encode())
    return log.read_bytes()[offset:].decode('utf-8', 'replace')


def _oneline(text: str, lines: int = 4) -> str:
    kept = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith('--- ')]
    return ' | '.join(kept[-lines:])


def engine_up(port: int, timeout: float = 3.0) -> bool:
    """The engine behind `port` answers engine-status (asked only while no collector is connected)."""
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.sendall(b'engine-status\n')
            reply = b''
            while b'\n' not in reply and (chunk := connection.recv(65536)):
                reply += chunk
        return bool(json.loads(reply.split(b'\n')[0] or b'{}').get('ok'))
    except (OSError, ValueError):
        return False


def vm_up() -> bool:
    try:
        subprocess.run([str(ADB), 'connect', SERIAL], capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
        out = subprocess.run([str(ADB), '-s', SERIAL, 'shell', 'echo', 'up'], capture_output=True, text=True,
                             timeout=30, stdin=subprocess.DEVNULL).stdout
        return out.strip() == 'up'
    except (OSError, subprocess.TimeoutExpired):
        return False


def ensure_engines(count: int, run_dir: Path, say) -> list[int]:
    """The engine ports that answer, after restarting the engines (or their VM) if too few do."""
    ports = [BASE_PORT + i for i in range(count)]
    deadline = time.time() + 90
    while True:
        # right after a stop an engine can still be serving its killed collector's connection
        up = [port for port in ports if engine_up(port)]
        if len(up) == count or time.time() > deadline:
            break
        time.sleep(10)
    if 4 * len(up) >= 3 * count:
        return up
    log = run_dir / 'engines.log'
    if vm_up():
        say(f'{len(up)} of {count} engines answer: restarting the engines (stage6c, ~1 min each)')
        powershell(ENGINE_SCRIPTS / 'stage6c.ps1', '-Count', str(count), timeout=150 * 60, log=log)
        status = Path(r'D:\crtrain\status6.txt')
    else:
        say('the engine VM does not answer: restarting it, its firewall and the engines (stage7)')
        powershell(ENGINE_SCRIPTS / 'stage7.ps1', timeout=180 * 60, log=log)
        status = Path(r'D:\crtrain\status7.txt')
    tail = _oneline(status.read_text(errors='replace'), 2) if status.is_file() else ''
    up = [port for port in ports if engine_up(port)]
    say(f'engines: {len(up)} of {count} answer ({tail})')
    return up


def _progress(run_dir: Path) -> tuple[int, int]:
    """Grows whenever a game finishes or the learner updates."""
    sizes = []
    for path in (run_dir / 'games' / 'games.jsonl', run_dir / 'learn.jsonl'):
        sizes.append(path.stat().st_size if path.is_file() else 0)
    return tuple(sizes)


def free_gb(run_dir: Path) -> float:
    return shutil.disk_usage(run_dir).free / 2 ** 30


def thin(run_dir: Path, say, snapshot_every: int = 10, snapshots: int = 5) -> None:
    """Make room: recordings beyond the newest 200; policy files other than every 50th and the
    league's latest snapshots (il/rl.py _snapshots)."""
    freed = 0
    recordings = sorted((run_dir / 'recordings').rglob('*.jsonl.zst'), key=lambda path: path.stat().st_mtime)
    for path in recordings[:-200]:
        freed += path.stat().st_size
        path.unlink(missing_ok=True)
    numbered = []
    for path in run_dir.glob('policy-*.pt'):
        number = path.stem.split('-')[-1]
        if number.isdigit():
            numbered.append((int(number), path))
    league = {number for number, _path in sorted(n for n in numbered if n[0] % snapshot_every == 0)[-snapshots:]}
    for number, path in numbered:
        if number % 50 and number not in league:
            freed += path.stat().st_size
            path.unlink(missing_ok=True)
    if freed:
        say(f'disk: freed {freed / 2 ** 30:.1f} GB (old recordings, policy files)')


def _score(games: list[dict]) -> str:
    if not games:
        return '-'
    wins = sum(1 for g in games if g['learner_won'])
    rate = wins / len(games)
    return f'{100 * rate:.0f}% +-{196 * math.sqrt(rate * (1 - rate) / len(games)):.0f} ({wins}-{len(games) - wins})'


def _decided(run_dir: Path, league: str, after: int = 0) -> list[dict]:
    from il.rl_report import _rows
    return [g for g in _rows(run_dir / 'games' / 'games.jsonl') if g.get('league') == league
            and g.get('learner_won') is not None and int(g.get('learner_version', 0)) >= after]


def guard(run_dir: Path, after: int, games: int, below: float) -> str | None:
    """The reason to stop, when the moving policy is clearly worse than the fixed start."""
    recent = _decided(run_dir, 'anchor', after)[-games:]
    if len(recent) < games:
        return None
    rate = sum(1 for g in recent if g['learner_won']) / len(recent)
    return (f'worse than the start: {100 * rate:.0f}% over its last {len(recent)} games against v2'
            if rate < below else None)


def write_status(run_dir: Path, now_line: str, restarts: list[str]) -> None:
    from il.rl_report import _rows, report
    games = _rows(run_dir / 'games' / 'games.jsonl')
    updates = _rows(run_dir / 'learn.jsonl')
    readings = [line for line in (run_dir / 'turns.log').read_text(errors='replace').splitlines()
                if re.match(r'^\d\d-\d\d \d\d:\d\d:\d\d  (RUN |idle)', line)]
    day = readings[-1440:]
    ours_day = sum(1 for line in day if line[16:20] == 'RUN ')
    ours_all = sum(1 for line in readings if line[16:20] == 'RUN ')
    hour = [g for g in games if g.get('time', 0) >= time.time() - 3600]
    every = max(10, 10 * math.ceil(len(updates) / 250))
    text = [f'{run_dir.name}, written {time.strftime("%Y-%m-%d %H:%M")}',
            f'now: {now_line}',
            f'GPU ours: {ours_day / 60:.1f} h of the last {len(day) / 60:.1f} h; {ours_all / 60:.1f} h in all',
            f'games: {len(games)} ({len(hour)} in the last hour); updates: {len(updates)}',
            f'restarts: {len(restarts)}' + (f' (last: {restarts[-1]})' if restarts else ''),
            f'disk: {free_gb(run_dir):.0f} GB free',
            *([f'HALTED: {(run_dir / "HALT").read_text().strip()} (delete HALT to go on)']
              if (run_dir / 'HALT').is_file() else []),
            '',
            'Running scores, the first 300 games against each fixed opponent and the latest 300:']
    anchors = {g.get('opponent') for g in _decided(run_dir, 'anchor')}
    anchor_name = ('v2 (the start)' if not anchors or any('distill-v2' in str(a) for a in anchors)
                   else f"the anchor, {Path(str(next(iter(anchors)))).stem}")
    for league, name in (('anchor', anchor_name), ('hog2', 'hog2 no-delay'), ('general', 'General, real decks')):
        decided = _decided(run_dir, league)
        text.append(f'  vs {name}: first {_score(decided[:300])}  latest {_score(decided[-300:])}')
    text += ['',
            'Read the tables: "vs v2 (anchor)" is the running score against the starting model -- above',
            '50% and rising means the games are making it better; hog2 and General are the fixed',
            'benchmarks. dealt / taken: princess towers per game. King activ.: games where its own',
            'spell woke the King Tower (the reward charges it).',
            '',
            report(run_dir, every)]
    temporary = run_dir / 'status.part'
    temporary.write_text('\n'.join(text) + '\n')
    temporary.replace(run_dir / 'status.txt')


def _single(run_dir: Path):
    """The lock that makes this the only keeper of the run (released when the process ends)."""
    handle = (run_dir / 'keeper.lock').open('a+')
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--engines', type=int, default=16)
    parser.add_argument('--league', default='self=2/snap=1/anchor=2/hog2=2/general=3')
    parser.add_argument('--init', help="the checkpoint a new run starts from, and its 'anchor' opponent (rl.ps1 -Init; "
                                       'default: rl.ps1\'s, v2)')
    parser.add_argument('--others-gb', type=float, default=1.5,
                        help="other processes holding more than this = someone else's turn")
    parser.add_argument('--calm', type=int, default=3, help='minutes free of others before we start')
    parser.add_argument('--every', type=int, default=60, help='seconds between readings')
    parser.add_argument('--stall', type=int, default=20, help='minutes without a game or an update = stuck')
    parser.add_argument('--restart-gap', type=int, default=10, help='minutes between restarts, at least')
    parser.add_argument('--guard-after', type=int, default=5, help='weights versions before this are not judged')
    parser.add_argument('--guard-games', type=int, default=150)
    parser.add_argument('--guard-below', type=float, default=0.42)
    args = parser.parse_args(argv)
    run_dir = CLAPHA / 'runs' / 'rl' / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = _single(run_dir)
    if lock is None:
        print(f'a keeper already looks after {args.run}')
        return 0
    sys.stderr = (run_dir / 'turns.err').open('a', buffering=1)
    (run_dir / 'turns.pid').write_text(str(os.getpid()))
    log = (run_dir / 'turns.log').open('a')

    def say(line: str) -> None:
        log.write(f"{time.strftime('%m-%d %H:%M:%S')}  {line}\n")
        log.flush()

    def launcher(*arguments: str) -> str:
        return powershell(LAUNCHER, '-Run', args.run, *arguments, timeout=1800, log=run_dir / 'launch.log')

    parts: dict[int, str] = {}
    restarts: list[str] = []
    progress, progress_at = _progress(run_dir), time.time()
    last_restart = 0.0

    def start() -> None:
        nonlocal parts, progress, progress_at
        up = ensure_engines(args.engines, run_dir, say)
        if not up:
            say('no engine answers: not starting (trying again later)')
            return
        output = launcher('-Engines', str(len(up)), '-Ports', '/'.join(map(str, up)), '-League', args.league,
                          *(('-Init', args.init) if args.init else ()))
        parts = {int(pid): name for name, pid in re.findall(r'(\S+) pid (\d+)', output)}
        progress, progress_at = _progress(run_dir), time.time()
        say(f'started on {len(up)} engines: {_oneline(output, 3)}')

    say(f'keeper started: run {args.run}, {args.engines} engines, league {args.league}')
    if os.name == 'nt' and (run_dir / 'pids.txt').is_file():
        import ctypes
        uptime = ctypes.windll.kernel32.GetTickCount64
        uptime.restype = ctypes.c_uint64
        if (run_dir / 'pids.txt').stat().st_mtime < time.time() - uptime() / 1000:
            # from before the last boot: those ids may belong to other processes now
            say(f'pids.txt is from before the last boot: {_oneline(launcher("-Stop"), 1)}')
    calm, next_status, now_line = 0, 0.0, 'starting'
    while True:
        try:
            usage, names = gpu_by_process(), process_names()
            mine = ours(run_dir)
            running = bool(mine) and any(pid in names for pid in mine)
            # ours: the run's processes and the engine VM (MuMu renders nothing but holds a little)
            own = sum(mib for pid, mib in usage.items()
                      if pid in mine or names.get(pid, '').lower().startswith('mumu'))
            others = {pid: mib for pid, mib in usage.items()
                      if pid not in mine and not names.get(pid, '').lower().startswith('mumu') and mib >= 64}
            other_mib = sum(others.values())
            top = ', '.join(f'{names.get(pid, pid)} {mib}' for pid, mib in sorted(others.items(), key=lambda kv: -kv[1])[:3])
            busy = other_mib > args.others_gb * 1024
            calm = 0 if busy else calm + 1
            say(f"{'RUN ' if running else 'idle'} ours {own} MiB, others {other_mib} MiB ({top})")
            now_line = (f'RUN (ours {own} MiB)' if running else
                        f'idle: others hold {other_mib} MiB ({top})' if busy else f'idle: GPU free for {calm} min')
            disk = free_gb(run_dir)
            if disk < 30:
                thin(run_dir, say)
                disk = free_gb(run_dir)
            full = disk < 8
            halt = run_dir / 'HALT'
            if not halt.is_file() and time.time() >= next_status:
                reason = guard(run_dir, args.guard_after, args.guard_games, args.guard_below)
                if reason:
                    halt.write_text(reason + '\n')
                    say(f'HALT: {reason}')
            held = halt.is_file()
            if running and (busy or full or held):
                why = ('others took the GPU' if busy else f'disk nearly full ({disk:.1f} GB free)' if full
                       else 'halted (see HALT)')
                say(f'{why}: stopping -> {_oneline(launcher("-Stop"), 1)}')
            elif running:
                if _progress(run_dir) != progress:
                    progress, progress_at = _progress(run_dir), time.time()
                dead = sorted(pid for pid in mine if pid not in names)
                stalled = time.time() - progress_at > args.stall * 60
                if (dead or stalled) and time.time() - last_restart > args.restart_gap * 60:
                    why = (f"ended: {', '.join(parts.get(pid, str(pid)) for pid in dead)}" if dead
                           else f'no game and no update for {args.stall} min')
                    restarts.append(f"{time.strftime('%m-%d %H:%M')} {why}")
                    say(f'{why}: restarting the run -> {_oneline(launcher("-Stop"), 1)}')
                    last_restart = time.time()
                    start()
            elif calm >= args.calm and not full and not held:
                if mine:
                    launcher('-Stop')           # stale pids from an earlier stop
                say(f'GPU free for {calm} min: starting')
                start()
                calm = 0
            if time.time() >= next_status:
                write_status(run_dir, now_line, restarts)
                next_status = time.time() + 600
        except Exception as error:  # noqa: BLE001  (a reading or a script can fail; try again next minute)
            say(f'error: {type(error).__name__}: {error}')
        time.sleep(args.every)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
