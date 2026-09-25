"""Resident touch injection: one adb shell kept open to /data/local/tmp/fast_tap.

The console used to call native_core's send_card_taps, which runs
    adb shell "input tap X Y; sleep 0.05; input tap X Y"
per play: a new adb client on the Mac, a new shell on the device, two `input` launches (each
starts a Java app_process) and a fixed 50 ms sleep, all while the decision loop waits. Here the
process is started once; a play is one line written to its stdin, and the loop does not wait
for the gesture to finish. fast_tap writes multi-touch events straight to the touchscreen
device -- the ordinary Android input path, nothing in the game process is touched.

Gesture shape is configurable because what the client accepts has to be measured
(mac012/tap_bench.py), not assumed:
    CR_TAP_MODE   place (tap card, tap tile) | drag (one gesture)      default place
    CR_TAP_GAP_MS gap between the two taps / steps of a drag            default 8 / 3
    CR_TAP_HOLD_MS hold per tap / ms per drag step                      default 16 / 10
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import deque

REMOTE = '/data/local/tmp/fast_tap'


def touch_device(adb, serial) -> tuple[str, int | None, int | None]:
    """(event path, max x, max y) of the multi-touch screen, from `getevent -pl`."""
    out = subprocess.run([str(adb), '-s', serial, 'shell', 'getevent -pl'],
                         capture_output=True, text=True, timeout=10).stdout
    for block in re.split(r'\nadd device \d+: ', '\n' + out):
        path = block.split('\n', 1)[0].strip()
        x = re.search(r'ABS_MT_POSITION_X\s*:.*?max (\d+)', block)
        y = re.search(r'ABS_MT_POSITION_Y\s*:.*?max (\d+)', block)
        if path.startswith('/dev/input/') and x and y:
            return path, int(x.group(1)), int(y.group(1))
    return '/dev/input/event1', None, None


class Tapper:
    def __init__(self, adb, serial, width: int, height: int):
        self.mode = os.environ.get('CR_TAP_MODE', 'place')
        drag = self.mode == 'drag'
        # place gap 8 hold 16: 4/4 accepted, 41 ms (tap_bench, 2026-09-25). gap 0 was also
        # 4/4 at 33 ms; the 8 ms is margin for a slow frame, until more trials say otherwise.
        self.gap = int(os.environ.get('CR_TAP_GAP_MS', '3' if drag else '8'))
        self.hold = int(os.environ.get('CR_TAP_HOLD_MS', '10' if drag else '16'))
        path, max_x, max_y = touch_device(adb, serial)
        # MuMu reports the panel in screen pixels; scale in case another device does not.
        self.scale = ((max_x + 1) / width if max_x else 1.0,
                      (max_y + 1) / height if max_y else 1.0)
        self.path = path
        self.process = subprocess.Popen(
            [str(adb), '-s', serial, 'shell', f'{REMOTE} {path}'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = self.process.stdout.readline()
        if '"ready"' not in ready:
            self.process.kill()
            raise RuntimeError(f'fast_tap did not start on {path}: {ready.strip()!r}')
        # An older fast_tap answers "bad command" to placeh/drag and would drop every play.
        self.process.stdin.write('version\n')
        self.process.stdin.flush()
        version = self.process.stdout.readline()
        if '"version"' not in version:
            self.process.kill()
            raise RuntimeError('fast_tap on the device is out of date (rebuild: start-consoles.sh)')
        self.sent: deque[float] = deque()
        self.timings: deque[dict] = deque(maxlen=200)   # {'gesture_ms', 'ack_ms'}
        self.lock = threading.Lock()
        threading.Thread(target=self._acks, daemon=True).start()

    def describe(self) -> str:
        return (f'fast_tap on {self.path}, {self.mode} gap {self.gap} ms hold {self.hold} ms')

    def _acks(self) -> None:
        for line in self.process.stdout:
            if not line.startswith('{'):
                continue
            with self.lock:
                sent = self.sent.popleft() if self.sent else None
            try:
                body = json.loads(line)
            except ValueError:
                continue
            self.timings.append({'gesture_ms': body.get('us', 0) / 1000.0,
                                 'ack_ms': (time.time() - sent) * 1000.0 if sent else None})

    def alive(self) -> bool:
        return self.process.poll() is None

    def play(self, hand_point, target_point) -> float:
        """Queue one card placement; returns the time the command was written."""
        (x0, y0), (x1, y1) = hand_point, target_point
        sx, sy = self.scale
        x0, x1 = round(x0 * sx), round(x1 * sx)
        y0, y1 = round(y0 * sy), round(y1 * sy)
        verb = 'drag' if self.mode == 'drag' else 'placeh'
        line = f'{verb} {x0} {y0} {x1} {y1} {self.gap} {self.hold}\n'
        written = time.time()
        with self.lock:
            self.sent.append(written)
        self.process.stdin.write(line)
        self.process.stdin.flush()
        return written

    def tap(self, point) -> float:
        """One tap (a hero ability button); returns the time the command was written."""
        x, y = point
        sx, sy = self.scale
        line = f'tap {round(x * sx)} {round(y * sy)}\n'
        written = time.time()
        with self.lock:
            self.sent.append(written)
        self.process.stdin.write(line)
        self.process.stdin.flush()
        return written

    def close(self) -> None:
        try:
            self.process.stdin.write('quit\n')
            self.process.stdin.flush()
        except (OSError, ValueError):
            pass
        self.process.kill()
