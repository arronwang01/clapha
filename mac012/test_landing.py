"""Offline checks for the overlay's landing tracker and screen geometry.

    PYTHONPATH=<cr-native-sandbox> python3 mac012/test_landing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import landing as L  # noqa: E402

OPP = 1
failed = []


def check(name, ok):
    print(('ok   ' if ok else 'FAIL ') + name)
    if not ok:
        failed.append(name)


def play(card, tick=100, x=9000, y=8000, seq=1):
    return {'kind': 'card', 'side': OPP, 'card_id': card, 'tick': tick, 'x': x, 'y': y,
            'issue_tick': tick - 21, 'seq': seq, 'form_code': 0}


def run(card, entities_at, until=200):
    tracker, alive = L.LandingTracker(), {}
    for tick in range(95, until):
        plays = [play(card)] if tick >= 100 else []
        active = tracker.update('b', tick, OPP, plays, entities_at(tick))
        alive[tick] = bool(active)
    return alive


# Fireball: projectile appears at 102 at their king tower, flies, gone at 120.
fireball = run(28000000, lambda t: [{'address': '0xp', 'side': OPP, 'kind': 0, 'x': 9000,
                                     'y': 29000 - (t - 102) * 1000}] if 102 <= t < 120 else [])
check('Fireball range shown while the projectile flies', all(fireball[t] for t in range(100, 120)))
check('Fireball range gone once the projectile is gone', not fireball[120])

# Zap: no projectile -> instant.
zap = run(28000008, lambda t: [])
check('instant spell (Zap) is not tracked after it executes', not any(zap[t] for t in range(100, 200)))

# Miner: surfaces at the target at 130 (it appeared at 101 far away and moved).
miner = run(26000032, lambda t: [{'address': '0xm', 'side': OPP, 'kind': 12, 'x': 9000,
                                  'y': 29000 - max(0, t - 101) * 700 if t < 130 else 8000}]
            if t >= 101 else [])
check('Miner marker stays while it travels', all(miner[t] for t in range(100, 128)))
check('Miner marker ends when it surfaces at the target', not miner[131])

# Goblin Barrel: goblins pop out at the target at 125; an old unit already standing there
# (seen before the play) must not end the marker early.
old_unit = {'address': '0xold', 'side': OPP, 'kind': 12, 'x': 9100, 'y': 8100}
barrel = run(28000004, lambda t: [old_unit] + ([{'address': '0xg', 'side': OPP, 'kind': 12,
                                                 'x': 8800, 'y': 8300}] if t >= 125 else []))
check('an old unit at the target does not end the Barrel marker', all(barrel[t] for t in range(100, 125)))
check('Barrel marker ends when its goblins appear', not barrel[126])

# Drill that never arrives: 7 s timeout.
drill = run(27000013, lambda t: [], until=260)
check('Drill marker times out after 7 s', drill[239] and not drill[241])

# Geometry: the continuous mapping equals the verified tap point at every tile centre.
try:
    sys.path.insert(0, '/home/user/imax9d/cr-native-sandbox')
    import native_core.mumu_live_actions as A  # noqa: E402
    import layout_fix  # noqa: E402,F401
    import importlib
    src = (Path(__file__).resolve().parent / 'console.py').read_text()
    start = src.index('def native_to_screen(')
    end = src.index('\ndef ', start + 10)
    namespace: dict = {}
    exec(src[start:end], namespace)
    layout = A.ScreenLayout.from_size(1440, 2560)
    worst = 0.0
    for side in (0, 1):
        for row in range(32):
            for col in range(18):
                cx, cy = col * 1000 + 500, row * 1000 + 500
                nx, ny = (18000 - cx, 32000 - cy) if side == 1 else (cx, cy)
                sx, sy = namespace['native_to_screen'](nx, ny, side, layout)
                tx, ty = layout.deployment_point(row * 18 + col, side)
                worst = max(worst, abs(sx - tx), abs(sy - ty))
    check(f'screen mapping = tap geometry at all 1152 tile centres (worst {worst:.2f} px)',
          worst <= 0.51)
except ImportError as error:
    print(f'skip geometry check: {error}')

print('FAIL' if failed else 'OK')
raise SystemExit(1 if failed else 0)
