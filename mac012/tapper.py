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
        # place gap 8 hold 16: 4/4 accepted, 41 ms (tap_bench, 2026-09-25). What the client takes is measured in
        # docs/GAME_INTEGRATION.md (mac012/tap_probe.py, 2026-10-03): the touch length and the gap inside a
        # gesture do not matter (2-34 ms holds, 0-33 ms gaps: 30 of 31 taken), so they stay as they were; a
        # gesture that begins less than ~5 ms after the one before it ended is lost about one time in three
        # (10 of 17 pairs whole, against 18 of 19 from 5 ms up), so gestures are written `spacing` ms apart.
        self.gap = int(os.environ.get('CR_TAP_GAP_MS', '3' if drag else '8'))
        self.hold = int(os.environ.get('CR_TAP_HOLD_MS', '10' if drag else '16'))
        self.spacing = int(os.environ.get('CR_TAP_SPACING_MS', '20'))
        self.free_at = 0.0          # when the last gesture written will have ended, plus the spacing
        self.write_lock = threading.Lock()
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
        try:
            self.version = int(json.loads(version).get('version', 2))      # 3: `at US <gesture>`
        except ValueError:
            self.version = 2
        self.sent: deque[float] = deque()
        self.timings: deque[dict] = deque(maxlen=200)   # {'gesture_ms', 'ack_ms', 'n'}
        self.count = 0          # commands written; the n-th ack answers the n-th command
        self.lock = threading.Lock()
        threading.Thread(target=self._acks, daemon=True).start()

    def describe(self) -> str:
        return (f'fast_tap v{self.version} on {self.path}, {self.mode} gap {self.gap} ms hold {self.hold} ms, '
                f'{self.spacing} ms between gestures')

    def _acks(self) -> None:
        answered = 0
        for line in self.process.stdout:
            if not line.startswith('{'):
                continue
            with self.lock:
                sent = self.sent.popleft() if self.sent else None
            answered += 1
            try:
                body = json.loads(line)
            except ValueError:
                continue
            self.timings.append({'gesture_ms': body.get('us', 0) / 1000.0,
                                 'ack_ms': (time.time() - sent) * 1000.0 if sent else None, 'n': answered})

    def timing_of(self, n: int) -> dict | None:
        """The ack of the n-th command written, once it has come back."""
        return next((t for t in reversed(self.timings) if t.get('n') == n), None)

    def alive(self) -> bool:
        return self.process.poll() is None

    def play(self, hand_point, target_point, at_us: int | None = None, wait_ms: float = 0.0) -> float:
        """Queue one card placement; returns the time the command was written.
        at_us: the device's monotonic time (the reader's sample_monotonic_us clock) the first touch goes down at,
        `wait_ms` from now: fast_tap holds the gesture until then, so it lands on the device's clock and not on
        ours (a card just dealt can be touched from 100 ms after its deal, not before)."""
        (x0, y0), (x1, y1) = hand_point, target_point
        sx, sy = self.scale
        x0, x1 = round(x0 * sx), round(x1 * sx)
        y0, y1 = round(y0 * sy), round(y1 * sy)
        verb = 'drag' if self.mode == 'drag' else 'placeh'
        line = f'{verb} {x0} {y0} {x1} {y1} {self.gap} {self.hold}\n'
        if at_us is not None and self.version >= 3:
            line = f'at {int(at_us)} {line}'
        duration = (self.gap + 1) * self.hold if self.mode == 'drag' else 2 * self.hold + self.gap
        return self._send(line, duration + max(0.0, wait_ms))

    def _write(self, line: str) -> None:
        with self.write_lock:
            try:
                self.process.stdin.write(line)
                self.process.stdin.flush()
            except (OSError, ValueError):
                pass

    def _send(self, line: str, duration_ms: float) -> float:
        """Write a gesture now, or once the previous one has ended and `spacing` ms have passed;
        returns the time it is written."""
        with self.lock:
            now = time.time()
            at = max(now, self.free_at)
            self.free_at = at + (duration_ms + self.spacing) / 1000.0
            self.sent.append(at)
            self.count += 1
        if at <= now:
            self._write(line)
        else:
            threading.Timer(at - now, self._write, (line,)).start()
        return at

    def tap(self, point) -> float:
        """One tap (a hero ability button); returns the time the command was written."""
        x, y = point
        sx, sy = self.scale
        line = f'tap {round(x * sx)} {round(y * sy)}\n'
        return self._send(line, 34)

    def close(self) -> None:
        try:
            self.process.stdin.write('quit\n')
            self.process.stdin.flush()
        except (OSError, ValueError):
            pass
        self.process.kill()
