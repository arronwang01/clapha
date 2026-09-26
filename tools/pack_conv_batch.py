"""Zip the converted replays not sent yet (plus the current index) for the training PC.

    ./py tools/pack_conv_batch.py [--out build/train-pack]   -> OUT/conv-hog26-NNN.zip
Paths inside the zip are relative to the clapha folder (runs/conv-hog26/...), so tools/windows/
data.cmd unpacks it in place. Sent files are listed in OUT/sent.txt; a replay whose index line is
not written yet is left for the next batch.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'build/train-pack')
    parser.add_argument('--frames', type=Path, default=ROOT / 'runs/conv-hog26')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sent_path = args.out / 'sent.txt'
    sent = set(sent_path.read_text().split()) if sent_path.exists() else set()
    lines = (args.frames / 'index.jsonl').read_text().splitlines()
    complete = lines if lines and lines[-1].strip() else lines[:-1]
    indexed = set()
    for line in complete:
        try:
            indexed.add(json.loads(line)['tag'])
        except (ValueError, KeyError):
            continue
    new = [path for path in sorted((args.frames / 'frames').glob('*/*.jsonl.zst'))
           if path.name.split('.')[0] in indexed and path.name not in sent]
    number = len(list(args.out.glob('conv-hog26-*.zip'))) + 1
    target = args.out / f'conv-hog26-{number:03d}.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_STORED) as bundle:
        for path in new:
            bundle.write(path, path.relative_to(ROOT).as_posix())
        bundle.writestr('runs/conv-hog26/index.jsonl', '\n'.join(complete) + '\n')
        for name in ('selection.json', 'meta.json'):
            if (args.frames / name).exists():
                bundle.write(args.frames / name, f'runs/conv-hog26/{name}')
    with sent_path.open('a') as handle:
        handle.write(''.join(f'{path.name}\n' for path in new))
    print(f'{target}: {len(new)} replays, {target.stat().st_size / 1e6:.0f} MB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
