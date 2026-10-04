"""Bring the training games an RL run keeps (il/rl.py --save-every) from the PC to the Mac.

On the PC (tools/windows/pack-recordings.cmd):

    python -m il.recordings pack [--all] [--models]

puts every run's recordings (runs/rl/<run>/recordings) that were not packed before into ONE zip,
clapha-recordings.zip next to the clapha folder, as <run>/<tag>.jsonl.zst. One file moves through
ToDesk far faster than hundreds of small ones; .zst does not compress further, so it is stored as
is. --all packs everything again (a lost zip). --models adds each run's newest kept model
(policy-NNNN.pt, ~50 MB) as <run>/<run>-u<N>.pt, for playing it live in the console.

On the Mac (the Clapha app's Training games -> Import, or by hand):

    ./py -m il.recordings import PATH... [--remove]

takes zips, folders or single recordings from anywhere and files each game under
runs/rl/<run>/recordings/frames/<tag[:2]>/<tag>.jsonl.zst, where il.game_viewer reads it, and each
model (<run>-u<N>.pt) under runs/pc, where the console lists it (firstlight_bot: clapha:<run>-u<N>). Games
already there are skipped; a file cut short in transfer is reported and left out. The run comes
from the recording (played.run, newer recordings), from the zip's or folder's path (runs/rl/<run>,
a pack's <run>/), or from what the game names (ex1's frozen target, a snapshot's run folder); a
folder whose games name one run is that run's. Otherwise 'unsorted'.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
RUNS = CLAPHA / 'runs' / 'rl'
SUFFIX = '.jsonl.zst'
SKIP = {'recordings', 'frames', 'runs', 'rl', 'inbox'}
MODEL = re.compile(r'[A-Za-z0-9_.-]+-u\d+\.pt')
MODELS = CLAPHA / 'runs' / 'pc'


def _check(data: bytes) -> dict:
    """The header of a whole recording; raises if the file is cut short (it must decompress)."""
    import zstandard
    body = zstandard.ZstdDecompressor().decompress(data, max_output_size=1 << 31)
    header = json.loads(body[:body.index(b'\n')] if b'\n' in body else body)
    if not isinstance(header, dict) or 'timeline' not in header:
        raise ValueError('not a recorded game')
    return header


def _run_in_path(parts: tuple[str, ...], packed: bool) -> str | None:
    """runs/rl/<run>/... anywhere in the path, or a pack's top folder (<run>/<tag>.jsonl.zst)."""
    for i, part in enumerate(parts[:-2]):
        if part == 'rl' and parts[i + 1] not in SKIP:
            return parts[i + 1]
    if packed and len(parts) == 2 and parts[0] not in SKIP:
        return parts[0]
    return None


def _run_named(header: dict) -> str | None:
    played = header.get('played') or {}
    if played.get('run'):
        return str(played['run'])
    who = str(played.get('b', '')).replace('\\', '/')
    match = re.search(r'rl/([A-Za-z0-9_.-]+)/policy-\d+', who)
    if match:
        return match.group(1)
    match = re.search(r'([A-Za-z0-9_.-]+)-target\.pt$', who)       # an exploiter's frozen opponent
    if match:
        return match.group(1)
    return None


def _entries(path: Path):
    """(name, the run its path names, a reader of its bytes) for every recording in a zip, folder or file."""
    if path.is_dir():
        for file in sorted(path.rglob('*' + SUFFIX)):
            yield file.name, _run_in_path(file.resolve().parts, False), file.read_bytes
        for archive in sorted(path.rglob('*.zip')):
            yield from _entries(archive)
    elif path.suffix.lower() == '.zip':
        archive = zipfile.ZipFile(path)
        for info in archive.infolist():
            if info.filename.endswith(SUFFIX) and not info.is_dir():
                yield Path(info.filename).name, _run_in_path(tuple(Path(info.filename).parts), True), \
                    (lambda info=info: archive.read(info))
    elif path.name.endswith(SUFFIX):
        yield path.name, _run_in_path(path.resolve().parts, False), path.read_bytes


def _models(path: Path):
    """(name, reader) for every model a pack carries (<run>-u<N>.pt), in a zip, folder or file."""
    if path.is_dir():
        for file in sorted(path.rglob('*.pt')):
            if MODEL.fullmatch(file.name):
                yield file.name, file.read_bytes
        for archive in sorted(path.rglob('*.zip')):
            yield from _models(archive)
    elif path.suffix.lower() == '.zip':
        archive = zipfile.ZipFile(path)
        for info in archive.infolist():
            if MODEL.fullmatch(Path(info.filename).name):
                yield Path(info.filename).name, (lambda info=info: archive.read(info))
    elif MODEL.fullmatch(path.name):
        yield path.name, path.read_bytes


def import_paths(paths: list[Path], default_run: str | None = None) -> dict:
    found, problems, models = [], [], []
    for source in paths:
        if not source.exists():
            problems.append(f'{source}: not found')
            continue
        batch = []
        try:
            entries = list(_entries(source))
        except zipfile.BadZipFile:
            problems.append(f'{source.name}: not a whole zip (still transferring?), left out')
            continue
        for name, path_run, read in entries:
            try:
                data = read()
                header = _check(data)
            except Exception as error:  # noqa: BLE001  (cut short in transfer, or not a recording)
                problems.append(f'{name}: cut short or unreadable ({type(error).__name__}), left out')
                continue
            run = (header.get('played') or {}).get('run') or path_run or _run_named(header)
            batch.append([run, name, data, header])
        try:
            for name, read in _models(source):
                target = MODELS / name
                data = read()
                if not (target.exists() and target.stat().st_size == len(data)):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix('.part')
                    temporary.write_bytes(data)
                    os.replace(temporary, target)
                    models.append(name[:-3])
        except (zipfile.BadZipFile, OSError) as error:
            problems.append(f'{source.name}: a model could not be read ({type(error).__name__}), left out')
        # a folder or zip from one run: games that name no run belong to the one the others name
        named = {run for run, *_ in batch if run}
        for item in batch:
            if not item[0]:
                item[0] = next(iter(named)) if len(named) == 1 else (default_run or 'unsorted')
        found += batch
    added, already = {}, {}
    for run, name, data, header in found:
        tag = name[:-len(SUFFIX)]
        target = RUNS / run / 'recordings' / 'frames' / tag[:2] / name
        if target.exists() and target.stat().st_size == len(data):
            already[run] = already.get(run, 0) + 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.part')
        temporary.write_bytes(data)
        os.replace(temporary, target)
        added[run] = added.get(run, 0) + 1
    return {'added': added, 'already': already, 'problems': problems, 'models': models}


def pack(args) -> int:
    """The PC side: recordings not packed before, every run, into one stored zip."""
    out = Path(args.out)
    manifest = RUNS / 'packed-recordings.txt'
    done = set() if args.all or not manifest.exists() else set(manifest.read_text().split())
    runs = [run_dir for run_dir in sorted(RUNS.iterdir()) if run_dir.is_dir()]
    files = [(run_dir.name, file, f'{run_dir.name}/{file.name}') for run_dir in runs
             for file in sorted((run_dir / 'recordings').rglob('*' + SUFFIX))]
    fresh = [item for item in files if item[2] not in done]
    models = []
    if args.models:
        for run_dir in runs:
            kept = sorted(run_dir.glob('policy-[0-9]*.pt'), key=lambda f: int(re.sub(r'\D', '', f.stem) or 0))
            if kept:
                name = f'{run_dir.name}-u{int(re.sub(r"[^0-9]", "", kept[-1].stem))}.pt'
                if f'{run_dir.name}/{name}' not in done:
                    models.append((run_dir.name, kept[-1], f'{run_dir.name}/{name}'))
    if not fresh and not models:
        print(f'nothing new to pack ({len(files)} games packed before; --all packs them again)')
        return 0
    temporary = out.with_name(out.name + '.part')
    counts: dict[str, int] = {}
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
        for run, file, name in fresh + models:
            archive.write(file, name)
            if name.endswith(SUFFIX):
                counts[run] = counts.get(run, 0) + 1
    os.replace(temporary, out)
    with manifest.open('a') as handle:
        handle.writelines(f'{name}\n' for _run, _file, name in fresh + models)
    size = out.stat().st_size / 1e6
    print(f'{len(fresh)} games ({", ".join(f"{run} {n}" for run, n in counts.items()) or "none new"})'
          + (f' and the models {", ".join(Path(name).stem for *_, name in models)}' if models else '')
          + f', {size:.1f} MB -> {out}')
    print('Transfer that one file to the Mac, then Clapha -> Training games -> Import.'
          + ('' if args.models else ' (--models also brings each run\'s newest model.)'))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack', help='PC: new recordings of every run into one zip')
    p.add_argument('--out', default=str(CLAPHA.parent / 'clapha-recordings.zip'))
    p.add_argument('--all', action='store_true', help='pack every recording again, not only new ones')
    p.add_argument('--models', action='store_true', help="also each run's newest kept model (policy-NNNN.pt)")
    i = commands.add_parser('import', help='Mac: file zips / folders / recordings under runs/rl/<run>')
    i.add_argument('paths', nargs='+', type=Path)
    i.add_argument('--run', help="the run for games that name none (default: 'unsorted')")
    i.add_argument('--remove', action='store_true', help='delete the given paths after importing (an inbox copy)')
    args = parser.parse_args(argv)
    if args.command == 'pack':
        return pack(args)
    started = time.time()
    result = import_paths(args.paths, args.run)
    for line in result['problems']:
        print(line)
    runs = sorted(set(result['added']) | set(result['already']))
    summary = '; '.join(f"{run}: {result['added'].get(run, 0)} new"
                        + (f" ({result['already'][run]} already here)" if result['already'].get(run) else '')
                        for run in runs) or 'no recordings found'
    if result['models']:
        summary += '; models for the console: ' + ', '.join(f'clapha:{name}' for name in result['models'])
    print(f'imported in {time.time() - started:.1f} s -- {summary}')
    if args.remove:
        for path in args.paths:
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
    return 0 if runs or result['models'] or not result['problems'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
