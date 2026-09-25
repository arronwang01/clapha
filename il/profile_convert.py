"""Where does replay conversion spend its time? Same FirstLight calls as cache_builder, one process.

Phases per batch: prepare (pure Python), calibrate (engine checks the chosen deal, per replay),
produce (battles played back in the engine with N resident slots). The produce phase is run
under cProfile so waiting on the engine and Python work show up separately.

    ./py -m il.profile_convert [--slots 8] [--replays 9] [--dataset runs/il-mini/dataset]
Run from ~/Documents/GitHub/FirstLight_CR's code (the copy matching the installed engine).
"""
from __future__ import annotations

import argparse
import cProfile
import glob
import io
import json
import pstats
import sys
import time
from pathlib import Path

FIRSTLIGHT = Path.home() / 'Documents/GitHub/FirstLight_CR'
sys.path.insert(0, str(FIRSTLIGHT))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--replays', type=int, default=9)
    parser.add_argument('--dataset', type=Path, default=Path('runs/il-mini/dataset'))
    parser.add_argument('--port', type=int, default=26789)
    parser.add_argument('--capture', default='zlib-json', choices=('zlib-json', 'json', 'compact'))
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq
    import torch
    from native_runner.cr_native_env import NativeClashEnv
    from native_runner.royaleapi_replay import calibrate_collected_replay_deal, prepare_collected_replay
    from native_runner.training.v4.cache_builder import _headless_calibration_probe
    from native_runner.training.v4.expert import replay_expert_actions, require_accepted_timeline, screen_expert_timeline
    from native_runner.training.v4.producer import produce_il_replay_batch
    from native_runner.training.v4.factory import build_episode_tensorizers_v4, build_resident_collected_replay_batch_v4

    torch.set_num_threads(1)
    rows = []
    for part in sorted(glob.glob(str(args.dataset / 'replays' / '*.parquet'))):
        rows += [json.loads(r['payload_json']) for r in pq.read_table(part, columns=['payload_json']).to_pylist()]
    rows = rows[:args.replays]

    native = NativeClashEnv(host='127.0.0.1', port=args.port, timeout=120.0)
    native.wait_ready(timeout=120.0)
    native.stop_resident_mode()
    probe = _headless_calibration_probe(native)
    timings = {'prepare': 0.0, 'calibrate': 0.0, 'tensorizers': 0.0}
    prepared = []
    for payload in rows:
        t0 = time.perf_counter()
        p = prepare_collected_replay(payload)
        t1 = time.perf_counter()
        c = calibrate_collected_replay_deal(p, probe)
        t2 = time.perf_counter()
        require_accepted_timeline(screen_expert_timeline(replay_expert_actions(c.replay)), replay_id=c.replay_tag)
        tz = build_episode_tensorizers_v4(c.replay.episode_config)
        t3 = time.perf_counter()
        timings['prepare'] += t1 - t0
        print(f'  calibrate {p.replay_tag[:8]}: {t2 - t1:.1f} s', flush=True)
        timings['calibrate'] += t2 - t1
        timings['tensorizers'] += t3 - t2
        prepared.append((c, tz))

    slots = min(args.slots, len(prepared))
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    environments, coordinator = build_resident_collected_replay_batch_v4(
        tuple(c for c, _ in prepared[:slots]), env_ids=tuple(range(slots)), host='127.0.0.1', port=args.port,
        timeout=120.0, capture_mode=args.capture)
    results = produce_il_replay_batch(
        environments, tuple(c for c, _ in prepared[:slots]), tuple(t for _, t in prepared[:slots]),
        gamma_per_decision=0.997, batch_coordinator=coordinator,
        replacement_prepared_replays=tuple(c for c, _ in prepared[slots:]),
        replacement_tensorizers_by_replay=tuple(t for _, t in prepared[slots:]),
        max_frames=None, validate_tensors=False)
    profiler.disable()
    timings['produce'] = time.perf_counter() - t0
    for environment in environments:
        try:
            environment.native.close()
        except Exception:
            pass

    try:
        perf = native._raw_request('multi-perf')
        print('engine hot-path counters:', json.dumps(perf)[:1500])
    except Exception as error:  # noqa: BLE001
        print('engine counters unavailable:', error)
    n = len(prepared)
    print(f'{n} replays, {slots} resident slots, capture {args.capture}')
    for phase, seconds in timings.items():
        print(f'  {phase:12} {seconds:8.1f} s total   {seconds / n:6.2f} s per replay')
    total = sum(timings.values())
    print(f'  {"TOTAL":12} {total:8.1f} s total   {total / n:6.2f} s per replay')
    for r in results:
        print(f'  {r.replay_tag[:8]} completed={r.completed} winner_matches={r.winner_matches} '
              f'crowns_match={r.crowns_match} executed {r.executed_expert_action_count}/{r.expert_action_count} '
              f'end {r.simulated_end_tick}/{r.source_end_tick} untrusted_from={r.first_untrusted_tick} '
              f'{r.failure_reason or ""}')
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('tottime').print_stats(18)
    print(stream.getvalue()[:6000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
