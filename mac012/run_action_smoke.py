"""Bounded deployment smoke test for this Mac/012 setup.

Same Training Camp gate as run_probe.py, then hands over to the author's
native_core.mumu_live_action_smoke unchanged: at most a few plays, each one requiring a
receipt (the chosen hand slot rotated AND elixir went down) before another is sent, and
no blind retries. Without --execute it only reports the action it would have taken.

Usage:
    python3 mac012/run_action_smoke.py --seconds 120                      # observe only
    python3 mac012/run_action_smoke.py --seconds 120 --confirmed-training-camp --execute
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_profile import ADB, CLAPHA, LOG_ROOT, SERIAL, apply  # type: ignore  # noqa: E402
from run_probe import screenshot  # type: ignore  # noqa: E402


def main(argv: list[str]) -> int:
    confirmed = '--confirmed-training-camp' in argv
    argv = [a for a in argv if a != '--confirmed-training-camp']
    executing = '--execute' in argv
    settings = apply()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    shot = screenshot(LOG_ROOT / 'mode-checks' / f'{stamp}.png')
    print(json.dumps({'profile': settings, 'screenshot': str(shot),
                      'execute': executing, 'training_camp_confirmed': confirmed}, indent=2))
    if executing and not confirmed:
        print('\nTaps require --confirmed-training-camp; screenshot written for review.',
              file=sys.stderr)
        return 3

    import layout_fix  # noqa: F401  (side-0 x mirror fix)
    from native_core import mumu_live_action_smoke  # noqa: E402  (after apply())
    for flag, value in (('--output', str(LOG_ROOT / 'action-smoke')),
                        ('--serial', SERIAL), ('--adb', str(ADB)),
                        ('--reader', str(CLAPHA / 'build' / 'live_sampler_tbi'))):
        if flag not in argv:
            argv += [flag, value]
    return mumu_live_action_smoke.main(argv)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
