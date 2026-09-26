"""Saved engine conversions: one file per replay, board snapshots at every decision tick.

    DIR/frames/<tag[:2]>/<tag>.jsonl.zst
        line 1: header -- timeline (il.timeline JSON), FirstLight's calibrated replay (pickle),
                fidelity against the recording, plays that could not be queued
        then:   one bare lean snapshot per line, ticks 5, 10, 15, ... (the state after that tick,
                before any play queued at it); DIR/meta.json holds the provenance and capability
                tables bare snapshots leave out

Written by il/engine_convert.py; read by training with load_replay(). Unpickling the calibrated
replay needs ~/Documents/GitHub/FirstLight_CR on sys.path (the copy matching the engine).
"""
from __future__ import annotations

import base64
import copyreg
import os
import pickle
import types
from pathlib import Path

FRAME_FORMAT = 'clapha-engine-frames.v1'


def _mappingproxy(mapping: dict) -> types.MappingProxyType:
    return types.MappingProxyType(mapping)


# FirstLight's replay objects hold read-only mappings, which pickle does not take by default.
copyreg.pickle(types.MappingProxyType, lambda proxy: (_mappingproxy, (dict(proxy),)))


def frame_path(out: Path, tag: str) -> Path:
    return out / 'frames' / tag[:2] / f'{tag}.jsonl.zst'


def save_replay(out: Path, result: dict, every: int) -> int:
    """Header line, then one snapshot per line; zstd. Written aside and renamed into place."""
    import orjson
    import zstandard
    from il.timeline import build_timeline, timeline_to_json
    calibrated = result['calibrated']
    header = {'format': FRAME_FORMAT, 'replay_tag': result['replay_tag'], 'end_tick': result['end_tick'],
              'ended_at': result['ended_at'], 'every': every, 'frames': len(result['frames']),
              'first_tick': result['frames'][0]['tick'] if result['frames'] else None,
              'commands': result['commands'], 'queued': result['queued'], 'failures': result['failures'],
              'fidelity': result['fidelity'], 'timeline': timeline_to_json(build_timeline(calibrated)),
              'calibrated_pickle': base64.b64encode(pickle.dumps(calibrated, protocol=5)).decode()}
    body = b'\n'.join([orjson.dumps(header, option=orjson.OPT_NON_STR_KEYS)]
                     + [orjson.dumps(frame) for frame in result['frames']])
    data = zstandard.ZstdCompressor(level=9).compress(body)
    path = frame_path(out, result['replay_tag'])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return len(data)


def load_replay(path: Path, with_replay: bool = True) -> tuple[dict, list[dict]]:
    """(header, snapshots) of one saved replay; header['calibrated'] is FirstLight's calibrated
    replay (needs FirstLight's native_runner importable; with_replay=False skips it)."""
    import orjson
    import zstandard
    lines = zstandard.ZstdDecompressor().decompress(path.read_bytes(), max_output_size=1 << 31).split(b'\n')
    header = orjson.loads(lines[0])
    encoded = header.pop('calibrated_pickle')
    if with_replay:
        header['calibrated'] = pickle.loads(base64.b64decode(encoded))
    return header, [orjson.loads(line) for line in lines[1:]]
