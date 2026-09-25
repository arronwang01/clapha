"""No-touch screen calibration check for this 1440x2560 MuMu.

Maps stationary towers read from memory onto a screenshot using the author's
ScreenLayout, and draws markers so the mapping can be checked by eye before any
tap is ever sent. Sends no input.

Native arena units: 18 columns x 1000 on x (0..18000), 32 rows x 1000 on y (0..32000).
Local player here is side 1 (high y), so canonical (actor) space is the 180-degree
rotation of native, which is what ScreenLayout.deployment_point expects.

Usage: python3 mac012/calibrate.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_profile import ADB, CLAPHA, LOG_ROOT, SERIAL, apply  # type: ignore  # noqa: E402

apply()
from native_core.mumu_live_actions import ScreenLayout  # noqa: E402
from native_core.mumu_live_protocol import start_reader, verify_runtime  # noqa: E402

NATIVE_W, NATIVE_H = 18000, 32000
CATALOG = CLAPHA / 'live_card_catalog.json'


def card_names() -> dict[int, str]:
    if not CATALOG.is_file():
        return {}
    data = json.loads(CATALOG.read_text())
    return {row['card_id']: f"{row['display_name']} ({row['elixir']})"
            for row in data.get('cards', [])}


def native_to_screen(layout: ScreenLayout, x: int, y: int, local_side: int) -> tuple[int, int]:
    """Inverse of deployment_point, in continuous native units instead of cells."""
    cx, cy = (NATIVE_W - x, NATIVE_H - y) if local_side == 1 else (x, y)
    x_fraction = cx / NATIVE_W
    if local_side == 1:  # camera preserves native X; actor canonicalization mirrors it
        x_fraction = 1 - x_fraction
    return (round(layout.arena_left + x_fraction * (layout.arena_right - layout.arena_left)),
            round(layout.arena_bottom - (cy / NATIVE_H) * (layout.arena_bottom - layout.arena_top)))


def capture_frame(pid: int) -> dict:
    process = start_reader(ADB, SERIAL, pid, interval_ms=100, max_frames=6)
    raw, _ = process.communicate(timeout=30)
    frames = [json.loads(line) for line in raw.splitlines()
              if line.startswith('{') and '"mumu_live_frame"' in line]
    live = [f for f in frames if f.get('battle_active') and f.get('coherent') and f.get('entities')]
    if not live:
        raise SystemExit('no live battle frame; start a Training Camp match first')
    return live[-1]


def main() -> int:
    from PIL import Image, ImageDraw

    runtime = verify_runtime(ADB, SERIAL)
    shot = LOG_ROOT / 'calibration' / 'screen.png'
    shot.parent.mkdir(parents=True, exist_ok=True)
    with shot.open('wb') as handle:
        subprocess.run([str(ADB), '-s', SERIAL, 'exec-out', 'screencap', '-p'],
                       stdout=handle, check=True, timeout=30)
    frame = capture_frame(runtime['pid'])

    image = Image.open(shot).convert('RGB')
    layout = ScreenLayout.from_size(image.width, image.height)
    local_side = next((p['side'] for p in frame['players']
                       if p['hand_deck_indices'][0] != -1), 1)
    names = card_names()
    draw = ImageDraw.Draw(image)
    draw.rectangle([layout.arena_left, layout.arena_top, layout.arena_right, layout.arena_bottom],
                   outline=(0, 255, 255), width=4)
    rows = []
    for entity in frame['entities']:
        sx, sy = native_to_screen(layout, entity['x'], entity['y'], local_side)
        stationary = entity['card_id'] == -1 and entity.get('kind') != 0
        colour = (255, 0, 0) if stationary else (255, 255, 0)
        radius = 26 if stationary else 16
        draw.ellipse([sx - radius, sy - radius, sx + radius, sy + radius], outline=colour, width=6)
        draw.line([sx - radius, sy, sx + radius, sy], fill=colour, width=3)
        draw.line([sx, sy - radius, sx, sy + radius], fill=colour, width=3)
        rows.append({'native': [entity['x'], entity['y']], 'screen': [sx, sy],
                     'side': entity['side'], 'card_id': entity['card_id'],
                     'card': names.get(entity['card_id'], 'tower' if stationary else '?'),
                     'hp': entity['hp'], 'tower': stationary})
    for slot in range(4):
        hx, hy = layout.hand_point(slot)
        draw.ellipse([hx - 30, hy - 30, hx + 30, hy + 30], outline=(0, 255, 0), width=6)
    out = LOG_ROOT / 'calibration' / 'overlay.png'
    image.save(out)
    print(json.dumps({'local_side': local_side, 'display': [image.width, image.height],
                      'arena_box': [layout.arena_left, layout.arena_top,
                                    layout.arena_right, layout.arena_bottom],
                      'hand_points': [layout.hand_point(s) for s in range(4)],
                      'overlay': str(out), 'entities': rows}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
