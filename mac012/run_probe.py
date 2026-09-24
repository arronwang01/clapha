"""Read-only probe entry point for this Mac/012 setup.

Applies the Mac/012 profile, then hands over to the author's native_core.mumu_live_probe
unchanged. Captures a screenshot first: this session is limited to Training Camp, and the
binding's game-mode field is unverified, so the mode is confirmed by looking at the screen
before any reading starts.

Usage:
    python3 mac012/run_probe.py --seconds 15            # writes screenshot, then stops
    python3 mac012/run_probe.py --seconds 15 --confirmed-training-camp
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_profile import ADB, CLAPHA, LOG_ROOT, SERIAL, apply  # type: ignore  # noqa: E402


def screenshot(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as handle:
        subprocess.run([str(ADB), '-s', SERIAL, 'exec-out', 'screencap', '-p'],
                       stdout=handle, check=True, timeout=30)
    return destination


def main(argv: list[str]) -> int:
    confirmed = '--confirmed-training-camp' in argv
    argv = [a for a in argv if a != '--confirmed-training-camp']
    settings = apply()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    shot = screenshot(LOG_ROOT / 'mode-checks' / f'{stamp}.png')
    print(json.dumps({'profile': settings, 'screenshot': str(shot),
                      'training_camp_confirmed': confirmed}, indent=2))
    if not confirmed:
        print('\nScreenshot written. Confirm a Royal Trainer opponent is on screen, '
              'then re-run with --confirmed-training-camp.', file=sys.stderr)
        return 3

    from native_core import mumu_live_probe  # noqa: E402  (after apply())
    if '--output' not in argv:
        argv += ['--output', str(LOG_ROOT / 'probes')]
    if '--serial' not in argv:
        argv += ['--serial', SERIAL]
    if '--adb' not in argv:
        argv += ['--adb', str(ADB)]
    if '--reader' not in argv:
        argv += ['--reader', str(CLAPHA / 'build' / 'live_sampler_tbi')]
    return mumu_live_probe.main(argv)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
