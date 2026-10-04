"""What the client does with a card tap, measured tap by tap on the device's own clock.

Run in a battle with both bots off (a friendly between the user's own accounts, or Training Camp; 7x elixir
gives the most trials). Reads the game's memory every READER_MS (20 ms, the reader's minimum: the hand, the tick) and its command queue
every QUEUE_MS, with their own reader processes next to the console's, and sends touches through
src/fast_tap.c version 3, which can start a gesture at a given monotonic time and reports when each touch
went down and up -- the same clock as the reader's sample_monotonic_us. So every touch is placed against what
the game's memory held at that moment, to ~READER_MS.

    CR_MUMU_SERIAL=127.0.0.1:26656 ./py mac012/tap_probe.py deal [pairs shape select ...] [--seconds N]

Experiments (each a loop of trials until the battle or --seconds ends; several named: they alternate):
  deal    play the card in a slot, then tap that slot again for the card the game will deal into it, at a
          chosen time around the deal: which taps does the client take?
  pairs   two cards already in the hand, the second gesture starting a chosen time after the first ends
  shape   one card already in the hand, with a chosen touch length and gap between the card and tile touch
  select  the card touch and the tile touch sent apart: does a selected card stay selected; a second card
          touched before the tile; a tile touched with nothing selected
  elixir  two cards we can afford one of but not both, tapped a moment apart: which elixir does the client count?
  illegal a troop asked onto the other half, the river, a bridge, our King Tower: where does it go?
  build   a building asked next to a tower: taken there, moved, or refused?
  early   the slot touched before its next card is dealt, a tile touched after: was the card selected in advance?
  occupied a building asked on the tile where our last one still stands, then two plain plays
  burst   the whole hand played in one go from a full bar (uses the kept cards too): whose elixir does the client count?
A probe that meets a battle in its first seconds first plays one card at each of a few early ticks (start).
Everything is written to build/tap_probe/<time>.jsonl: every hand change, every command of ours in the
queue, every touch, every trial. mac012/tap_probe_report.py reads it.

Cards kept in the hand and never played (so the other side's towers are left standing and the battle lasts):
KEEP below.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viewer as V  # noqa: E402
import layout_fix  # noqa: F401,E402  (side 0's screen x, as the console maps it)
import scope_gate  # noqa: E402
from mac_profile import ADB, SERIAL, MANAGER_RVA, ROOT_CONTEXT_OFFSET  # type: ignore  # noqa: E402
from native_core.mumu_live_actions import ScreenLayout  # noqa: E402
from native_core.mumu_live_protocol import adb_run  # noqa: E402
from tapper import touch_device  # noqa: E402

READER_MS = int(os.environ.get('CR_PROBE_READER_MS', '20'))
QUEUE_MS = int(os.environ.get('CR_PROBE_QUEUE_MS', '20'))
REMOTE = '/data/local/tmp/fast_tap_v3'
TICK_US = 50_000
COMMAND_AGE = 21
X_TILES = 18
KEEP = {26000021, 26000014}          # Hog Rider, Musketeer: the cards that would take towers
OUT = Path(__file__).resolve().parents[1] / 'build' / 'tap_probe'


class Probe:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.side: int | None = None
        self.me: dict | None = None
        self.tick = -1
        self.us = 0                      # device clock of the latest sample
        self.host_at_us = 0.0            # host time that sample arrived
        self.active = False
        self.can_control = False
        self.first_us: dict[int, int] = {}       # tick -> the first sample that showed it
        self.hand_log: list[dict] = []           # every change of the hand
        self.commands: list[dict] = []           # our commands as the queue showed them
        self.seen_commands: set = set()
        self.accounts = None
        self.acks: list[dict] = []
        self.asked: list[dict] = []
        self.sent = 0
        self.battle = None
        OUT.mkdir(parents=True, exist_ok=True)
        self.path = OUT / (time.strftime('%Y%m%dT%H%M%S') + '.jsonl')
        self.out = self.path.open('a', encoding='utf-8')
        sizes = re.findall(r'(\d+)x(\d+)', adb_run(ADB, SERIAL, 'shell', 'wm size'))
        self.width, self.height = map(int, sizes[-1])
        self.layout = ScreenLayout.from_size(self.width, self.height)
        path, _mx, _my = touch_device(ADB, SERIAL)
        self.tap_process = subprocess.Popen([str(ADB), '-s', SERIAL, 'shell', f'{REMOTE} {path}'],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = self.tap_process.stdout.readline()
        if '"ready"' not in ready:
            raise SystemExit(f'{REMOTE} did not start: {ready!r} (build and push src/fast_tap.c)')
        threading.Thread(target=self._acks, daemon=True).start()
        threading.Thread(target=self._frames, daemon=True).start()
        threading.Thread(target=self._queue, daemon=True).start()

    # ---- recording --------------------------------------------------------------------------
    def write(self, kind: str, **row) -> None:
        with self.lock:
            self.out.write(json.dumps({'k': kind, 'host': round(time.time(), 4), **row}) + '\n')
            self.out.flush()

    def _acks(self) -> None:
        for line in self.tap_process.stdout:
            if not line.startswith('{'):
                continue
            try:
                body = json.loads(line)
            except ValueError:
                continue
            body['n'] = len(self.acks) + 1
            self.acks.append(body)
            self.write('ack', **body)

    def _frames(self) -> None:
        from native_core.mumu_live_protocol import BattleClockGuard, install_reader, start_reader, verify_runtime
        guard = BattleClockGuard()
        while True:
            try:
                runtime = verify_runtime(ADB, SERIAL)
                install_reader(ADB, SERIAL, V.READER)
                process = start_reader(ADB, SERIAL, runtime['pid'], interval_ms=READER_MS, max_frames=0)
                last_hand, last_us = None, 0
                for line in process.stdout:
                    if '"mumu_live_frame"' not in line:
                        continue
                    host = time.time()
                    frame = json.loads(line)
                    us = int(frame['sample_monotonic_us'])
                    health = guard.observe(frame, now=us / 1_000_000)
                    side = health.get('local_side')
                    active = bool(frame.get('battle_active')) and side in (0, 1)
                    me = next((p for p in frame.get('players') or () if p.get('side') == side), None) if active else None
                    tick = int(frame.get('game_tick') or 0)
                    battle = (frame.get('chain') or {}).get('battle')
                    with self.lock:
                        if battle != self.battle:
                            self.battle, self.first_us = battle, {}
                            last_hand = None
                        self.side, self.me, self.tick, self.us, self.host_at_us = side, me, tick, us, host
                        self.active, self.can_control = active, bool(health.get('can_control'))
                        new_tick = active and tick not in self.first_us
                        if new_tick:
                            self.first_us[tick] = us
                    if new_tick:
                        self.write('tick', tick=tick, us=us, elixir=me.get('elixir_raw') if me else None)
                    if me is not None:
                        hand = (tuple(me.get('hand_deck_indices') or ()), tuple(me.get('cycle_deck_indices') or ()))
                        if hand != last_hand:
                            row = {'us': us, 'prev_us': last_us, 'tick': tick, 'hand': hand[0], 'cycle': hand[1],
                                   'elixir': me.get('elixir_raw')}
                            self.hand_log.append(row)
                            self.write('hand', **row)
                            last_hand = hand
                    last_us = us
            except Exception as error:  # noqa: BLE001  (the game is in a menu or restarting)
                self.write('reader_error', error=f'{type(error).__name__}: {error}')
                time.sleep(2)

    def _queue(self) -> None:
        from native_core.mumu_live_protocol import verify_runtime
        while True:
            try:
                runtime = verify_runtime(ADB, SERIAL)
                command = (f'/data/local/tmp/queue_probe {runtime["pid"]} {hex(MANAGER_RVA)} '
                           f'{hex(ROOT_CONTEXT_OFFSET)} 0 {QUEUE_MS}')
                process = subprocess.Popen([str(ADB), '-s', SERIAL, 'shell', command],
                                           stdout=subprocess.PIPE, text=True, bufsize=1)
                for line in process.stdout:
                    if not line.startswith('{'):
                        continue
                    host = time.time()
                    row = json.loads(line)
                    self.accounts = row.get('accounts')
                    for entry in (row.get('queue') or {}).get('entries', []):
                        if int(entry.get('card_id') or 0) <= 0 or not isinstance(entry.get('issue_tick'), int):
                            continue
                        key = (entry.get('account_lo'), entry.get('seq'), entry['issue_tick'], entry['card_id'])
                        if key in self.seen_commands:
                            continue
                        self.seen_commands.add(key)
                        if V.entry_side(entry, self.accounts) != self.side:
                            continue
                        with self.lock:
                            us = self.us + int((host - self.host_at_us) * 1e6)
                        found = {'card': V.card_identity(entry['card_id'])[0], 'issue': entry['issue_tick'],
                                 'seq': entry.get('seq'), 'x': entry.get('x'), 'y': entry.get('y'),
                                 'seen_tick': row.get('tick_0x60'), 'seen_us': us}
                        self.commands.append(found)
                        self.write('cmd', **found)
                time.sleep(1)           # the tool ended (refused its arguments, or the game closed)
            except Exception as error:  # noqa: BLE001
                self.write('queue_error', error=f'{type(error).__name__}: {error}')
                time.sleep(2)

    # ---- the game as read ---------------------------------------------------------------------
    def snapshot(self):
        with self.lock:
            return self.side, self.me, self.tick, self.us, self.host_at_us

    def now_us(self) -> int:
        """The device's clock now, from the latest sample and the host time since it arrived."""
        with self.lock:
            return self.us + int((time.time() - self.host_at_us) * 1e6)

    def tick_us(self, tick: int) -> int:
        """When a tick begins on the device's clock: the earliest any recent tick was seen, less its ticks."""
        with self.lock:
            recent = [(t, u) for t, u in self.first_us.items() if self.tick - 80 <= t <= self.tick]
        base = min(u - t * TICK_US for t, u in recent)
        return base + tick * TICK_US

    def cards(self):
        """(hand card ids by position (None: empty), the cycle's card ids, elixir)."""
        _side, me, _tick, _us, _host = self.snapshot()
        deck = me['deck_card_ids']
        ident = lambda i: V.card_identity(deck[i])[0] if 0 <= i < len(deck) else None  # noqa: E731
        return ([ident(i) for i in me['hand_deck_indices']], [ident(i) for i in me['cycle_deck_indices']],
                me['elixir_raw'] / 10000.0)

    @staticmethod
    def cost(card) -> float:
        return float(V.CARDS.get(card, {}).get('elixir') or 9)

    @staticmethod
    def name(card) -> str:
        return str(V.CARDS.get(card, {}).get('name', card))

    @staticmethod
    def kind(card) -> str:
        return str(V.CARDS.get(card, {}).get('type', '?'))

    # ---- touches ------------------------------------------------------------------------------
    def send(self, line: str) -> int:
        self.sent += 1
        self.tap_process.stdin.write(line + '\n')
        self.tap_process.stdin.flush()
        self.write('send', n=self.sent, line=line)
        return self.sent

    def ack(self, n: int, timeout: float = 3.0) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(self.acks) >= n:
                return self.acks[n - 1]
            time.sleep(0.005)
        return None

    def hand_xy(self, position: int) -> tuple[int, int]:
        x, y = self.layout.hand_point(position)
        return round(x), round(y)

    def tile_xy(self, column: int, row: int) -> tuple[int, int]:
        """A tile in our own view (row 0: our back line) on the screen."""
        x, y = self.layout.deployment_point(row * X_TILES + column, self.side)
        return round(x), round(y)

    def place_line(self, position: int, tile, gap: int = 8, hold: int = 16, at: int | None = None) -> str:
        (x0, y0), (x1, y1) = self.hand_xy(position), self.tile_xy(*tile)
        self.asked.append({'n': self.sent + 1, 'position': position, 'tile': list(tile), 'native': list(native_xy(self, tile))})
        self.write('asked', **self.asked[-1])
        return (f'at {at} ' if at else '') + f'placeh {x0} {y0} {x1} {y1} {gap} {hold}'

    def command_of(self, card: int, after_tick: int, timeout: float = 0.7) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            found = next((c for c in reversed(self.commands) if c['card'] == card and c['issue'] >= after_tick), None)
            if found:
                return found
            time.sleep(0.005)
        return None

    def wait_tick(self, tick: int, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline and self.active and self.tick < tick:
            time.sleep(0.005)

    def settle(self, need: float, timeout: float = 12.0) -> bool:
        """Every slot dealt, none of our commands still on its way, and the elixir there."""
        deadline = time.time() + timeout
        while time.time() < deadline and self.active:
            hand, _cycle, elixir = self.cards()
            pending = [c for c in self.commands if c['issue'] + COMMAND_AGE + 2 > self.tick]
            if all(c is not None for c in hand) and not pending and elixir >= min(10.0, need):
                return True
            time.sleep(0.02)
        return False


# tiles in our own view (column, row from our back line). Troops go in front of the princess towers (which stand
# on columns 2-4 and 13-15, rows 5-7), buildings in the middle, three tiles square each and clear of one another
# and of the troops' tiles, so no trial's placement is one the client has to move or refuse (the build and illegal
# experiments ask for those on purpose).
TROOP_TILES = [(2, 10), (15, 10), (2, 12), (15, 12), (1, 9), (16, 9), (3, 13), (14, 13)]
BUILDING_TILES = [(5, 9), (12, 9), (5, 13), (12, 13), (8, 11), (9, 7)]
SPELL_TILES = [(8, 14), (9, 14), (6, 14), (11, 14)]


class Tiles:
    def __init__(self) -> None:
        self.count = {'troop': 0, 'building': 0, 'spell': 0}

    def next(self, kind: str):
        pool = {'building': BUILDING_TILES, 'spell': SPELL_TILES}.get(kind, TROOP_TILES)
        key = kind if kind in self.count else 'troop'
        self.count[key] += 1
        return pool[self.count[key] % len(pool)]


def playable(probe: Probe, card) -> bool:
    return card is not None and card not in KEEP


def trial_deal(probe: Probe, tiles: Tiles, delta_ms: int, gap: int = 8, hold: int = 16) -> dict | None:
    """Play a slot's card, then tap that slot for the card dealt into it, delta_ms from the deal's tick."""
    hand, cycle, _elixir = probe.cards()
    slots = [p for p, c in enumerate(hand) if playable(probe, c)]
    if not slots or not cycle or not playable(probe, cycle[0]):
        return trial_cycle(probe, tiles)             # the next card is one we keep: let it come in, and move on
    position = random.choice(slots)
    first, second = hand[position], cycle[0]
    if not probe.settle(probe.cost(first) + probe.cost(second) + 0.3):
        return None
    hand, cycle, elixir = probe.cards()
    if hand[position] != first or not cycle or cycle[0] != second:
        return None
    tick0 = probe.tick
    n1 = probe.send(probe.place_line(position, tiles.next(probe.kind(first)), gap, hold))
    played = probe.command_of(first, tick0 - 1)
    if played is None:
        row = {'trial': 'deal', 'ok': False, 'why': 'the first card was not taken', 'first': probe.name(first),
               'position': position, 'ack1': probe.ack(n1)}
        probe.write('trial', **row)
        probe.wait_tick(probe.tick + 30)
        return row
    deal_tick = played['issue'] + COMMAND_AGE
    at = probe.tick_us(deal_tick) + delta_ms * 1000
    n2 = probe.send(probe.place_line(position, tiles.next(probe.kind(second)), gap, hold, at=at))
    probe.wait_tick(deal_tick + 16)
    ack2 = probe.ack(n2)
    taken = next((c for c in probe.commands if c['card'] == second and c['issue'] > played['issue']), None)
    row = {'trial': 'deal', 'ok': True, 'delta_ms': delta_ms, 'position': position, 'first': probe.name(first),
           'second': probe.name(second), 'first_issue': played['issue'], 'deal_tick': deal_tick,
           'deal_us': probe.tick_us(deal_tick), 'ack1': probe.ack(n1), 'ack2': ack2, 'gap': gap, 'hold': hold,
           'taken': taken is not None, 'second_issue': taken and taken['issue'], 'elixir': elixir}
    probe.write('trial', **row)
    return row


def trial_cycle(probe: Probe, tiles: Tiles) -> dict | None:
    """One plain play of a card long in the hand (to move the cycle on)."""
    hand, _cycle, _elixir = probe.cards()
    slots = [p for p, c in enumerate(hand) if playable(probe, c)]
    if not slots:
        return None
    plain = [p for p in slots if probe.kind(hand[p]) != 'building']
    position = random.choice(plain or slots)
    card = hand[position]
    if not probe.settle(probe.cost(card) + 0.3):
        return None
    tick0 = probe.tick
    n = probe.send(probe.place_line(position, tiles.next(probe.kind(card))))
    played = probe.command_of(card, tick0 - 1)
    row = {'trial': 'cycle', 'card': probe.name(card), 'position': position, 'taken': played is not None,
           'issue': played and played['issue'], 'tick': tick0, 'ack': probe.ack(n)}
    probe.write('trial', **row)
    probe.wait_tick(tick0 + 28)
    return row


def two_slots(probe: Probe, min_cost: float = 0.0):
    """Two slots holding troops or spells we may play (not a building: where one may stand is the illegal
    experiment's question, and it would blur these)."""
    hand, _cycle, _elixir = probe.cards()
    slots = [p for p, c in enumerate(hand) if playable(probe, c) and probe.kind(c) != 'building'
             and probe.cost(c) >= min_cost]
    return (hand, random.sample(slots, 2)) if len(slots) >= 2 else (hand, None)


def trial_pairs(probe: Probe, tiles: Tiles, spacing_ms: int, gap: int = 8, hold: int = 16) -> dict | None:
    """Two cards long in the hand; the second gesture starts spacing_ms after the first has ended."""
    hand, pair = two_slots(probe)
    if pair is None:
        return trial_cycle(probe, tiles)
    a, b = pair
    if not probe.settle(probe.cost(hand[a]) + probe.cost(hand[b]) + 0.3):
        return None
    hand2, _cycle, _elixir = probe.cards()
    if hand2 != hand:
        return None
    tick0 = probe.tick
    start = probe.now_us() + 60_000
    length = (2 * hold + gap) * 1000
    n1 = probe.send(probe.place_line(a, tiles.next(probe.kind(hand[a])), gap, hold, at=start))
    n2 = probe.send(probe.place_line(b, tiles.next(probe.kind(hand[b])), gap, hold, at=start + length + spacing_ms * 1000))
    first = probe.command_of(hand[a], tick0 - 1)
    second = probe.command_of(hand[b], tick0 - 1, timeout=0.7)
    row = {'trial': 'pairs', 'spacing_ms': spacing_ms, 'gap': gap, 'hold': hold, 'cards': [probe.name(hand[a]), probe.name(hand[b])],
           'positions': [a, b], 'taken': [first is not None, second is not None],
           'issues': [first and first['issue'], second and second['issue']], 'tick': tick0,
           'ack1': probe.ack(n1), 'ack2': probe.ack(n2)}
    probe.write('trial', **row)
    probe.wait_tick(tick0 + 30)
    return row


def trial_shape(probe: Probe, tiles: Tiles, gap: int, hold: int) -> dict | None:
    """One card long in the hand, with the touch length and the gap between its two touches given."""
    hand, _cycle, _elixir = probe.cards()
    slots = [p for p, c in enumerate(hand) if playable(probe, c) and probe.kind(c) != 'building']
    if not slots:
        return trial_cycle(probe, tiles)
    position = random.choice(slots)
    card = hand[position]
    if not probe.settle(probe.cost(card) + 0.3):
        return None
    tick0 = probe.tick
    n = probe.send(probe.place_line(position, tiles.next(probe.kind(card)), gap, hold, at=probe.now_us() + 50_000))
    played = probe.command_of(card, tick0 - 1)
    row = {'trial': 'shape', 'gap': gap, 'hold': hold, 'card': probe.name(card), 'position': position,
           'taken': played is not None, 'issue': played and played['issue'], 'tick': tick0, 'ack': probe.ack(n)}
    probe.write('trial', **row)
    probe.wait_tick(tick0 + 28)
    return row


def trial_select(probe: Probe, tiles: Tiles, case: str, hold: int = 16) -> dict | None:
    """The two touches of a play sent apart.
      wait N    the card touched, the tile N ms later: does the selection last?
      second    card A touched, then card B, then a tile: which one goes down?
      tile      a tile touched with no card touched first
      twice     the same card touched twice, then a tile: is the second touch a deselect?"""
    hand, pair = two_slots(probe)
    if pair is None:
        return trial_cycle(probe, tiles)
    a, b = pair
    if not probe.settle(probe.cost(hand[a]) + probe.cost(hand[b]) + 0.3):
        return None
    tick0 = probe.tick
    start = probe.now_us() + 60_000
    tile = tiles.next(probe.kind(hand[a]) if probe.kind(hand[a]) == probe.kind(hand[b]) else 'spell')
    if probe.kind(hand[a]) != probe.kind(hand[b]):
        tile = tiles.next('troop')                  # a troop tile is legal for a spell and a building too
    (ax, ay), (bx, by), (tx, ty) = probe.hand_xy(a), probe.hand_xy(b), probe.tile_xy(*tile)
    sends = []
    if case.startswith('wait'):
        wait = int(case.split()[1])
        sends = [f'at {start} taph {ax} {ay} {hold}', f'at {start + (hold + wait) * 1000} taph {tx} {ty} {hold}']
    elif case == 'second':
        sends = [f'at {start} taph {ax} {ay} {hold}', f'at {start + 120_000} taph {bx} {by} {hold}',
                 f'at {start + 240_000} taph {tx} {ty} {hold}']
    elif case == 'tile':
        sends = [f'at {start} taph {tx} {ty} {hold}']
    elif case == 'twice':
        sends = [f'at {start} taph {ax} {ay} {hold}', f'at {start + 120_000} taph {ax} {ay} {hold}',
                 f'at {start + 240_000} taph {tx} {ty} {hold}']
    numbers = [probe.send(line) for line in sends]
    time.sleep(0.25 + (int(case.split()[1]) / 1000 if case.startswith('wait') else 0.3))
    first = probe.command_of(hand[a], tick0 - 1, timeout=0.5)
    second = probe.command_of(hand[b], tick0 - 1, timeout=0.05)
    row = {'trial': 'select', 'case': case, 'cards': [probe.name(hand[a]), probe.name(hand[b])], 'positions': [a, b],
           'taken': [first is not None, second is not None], 'issues': [first and first['issue'], second and second['issue']],
           'tick': tick0, 'acks': [probe.ack(n) for n in numbers]}
    probe.write('trial', **row)
    probe.wait_tick(tick0 + 34)
    # a card left selected would be put down by the next trial's tile: clear it by playing it properly
    return row


def trial_elixir(probe: Probe, tiles: Tiles, apart_ms: int = 120) -> dict | None:
    """Two cards we can afford one of but not both, the second tapped apart_ms after the first: does the
    client count the first card's cost at once (the screen's elixir) or only when it executes (the game's)?"""
    hand, pair = two_slots(probe, min_cost=2.0)
    if pair is None:
        return trial_cycle(probe, tiles)
    a, b = pair
    ca, cb = probe.cost(hand[a]), probe.cost(hand[b])
    deadline = time.time() + 10.0
    while time.time() < deadline and probe.active:
        now_hand, _cycle, elixir = probe.cards()
        pending = [c for c in probe.commands if c['issue'] + COMMAND_AGE + 2 > probe.tick]
        if now_hand != hand:
            return None
        if not pending and max(ca, cb) + 0.1 <= elixir <= ca + cb - 0.9:
            break
        if not pending and elixir > ca + cb - 0.9:
            return trial_cycle(probe, tiles)         # too much elixir: spend some and come back
        time.sleep(0.01)
    else:
        return None
    tick0 = probe.tick
    start = probe.now_us() + 40_000
    n1 = probe.send(probe.place_line(a, tiles.next(probe.kind(hand[a])), at=start))
    n2 = probe.send(probe.place_line(b, tiles.next(probe.kind(hand[b])), at=start + (40 + apart_ms) * 1000))
    first = probe.command_of(hand[a], tick0 - 1)
    second = probe.command_of(hand[b], tick0 - 1, timeout=0.5)
    row = {'trial': 'elixir', 'apart_ms': apart_ms, 'cards': [probe.name(hand[a]), probe.name(hand[b])], 'costs': [ca, cb],
           'elixir': elixir, 'taken': [first is not None, second is not None],
           'issues': [first and first['issue'], second and second['issue']], 'tick': tick0,
           'ack1': probe.ack(n1), 'ack2': probe.ack(n2)}
    probe.write('trial', **row)
    probe.wait_tick(tick0 + 34)
    return row


def native_xy(probe: Probe, tile) -> tuple[int, int]:
    """The centre of a tile of our own view in the game's own coordinates (side 1 sees the board turned)."""
    column, row = tile
    if probe.side == 1:
        column, row = X_TILES - 1 - column, 31 - row
    return column * 1000 + 500, row * 1000 + 500


def trial_illegal(probe: Probe, tiles: Tiles, case: str) -> dict | None:
    """A card asked onto a tile it may not go: where does the client put it, if anywhere?
      enemy     a troop on the other side's half          river    a troop on the river
      tower     a troop on our own King Tower             bridge   a troop on a bridge tile
      spell     a spell on the other side's half (legal: the reference)"""
    hand, _cycle, _elixir = probe.cards()
    want = 'spell' if case == 'spell' else 'troop'
    slots = [p for p, c in enumerate(hand) if playable(probe, c) and probe.kind(c) == want]
    if not slots:
        return trial_cycle(probe, tiles)
    position = random.choice(slots)
    card = hand[position]
    if not probe.settle(probe.cost(card) + 0.3):
        return None
    tile = {'enemy': (4, 22), 'river': (6, 15), 'tower': (8, 2), 'bridge': (3, 15), 'spell': (4, 24)}[case]
    tick0 = probe.tick
    n = probe.send(probe.place_line(position, tile, at=probe.now_us() + 40_000))
    played = probe.command_of(card, tick0 - 1)
    row = {'trial': 'illegal', 'case': case, 'card': probe.name(card), 'tile': tile, 'asked': native_xy(probe, tile),
           'taken': played is not None, 'got': played and [played['x'], played['y']], 'tick': tick0, 'ack': probe.ack(n)}
    probe.write('trial', **row)
    probe.wait_tick(tick0 + 30)
    if played is None:
        # the card may have been left selected: put it down properly so the next trial starts clean
        trial_cycle(probe, tiles)
    return row


def trial_build(probe: Probe, tiles: Tiles, tile) -> dict | None:
    """A building asked on a tile next to a tower (or far from one, for reference): taken there, moved, or refused?"""
    hand, _cycle, _elixir = probe.cards()
    slots = [p for p, c in enumerate(hand) if playable(probe, c) and probe.kind(c) == 'building']
    if not slots:
        return trial_cycle(probe, tiles)
    position = slots[0]
    card = hand[position]
    if not probe.settle(probe.cost(card) + 0.3):
        return None
    tick0 = probe.tick
    n = probe.send(probe.place_line(position, tile, at=probe.now_us() + 40_000))
    played = probe.command_of(card, tick0 - 1)
    row = {'trial': 'build', 'card': probe.name(card), 'tile': list(tile), 'asked': native_xy(probe, tile),
           'taken': played is not None, 'got': played and [played['x'], played['y']], 'tick': tick0, 'ack': probe.ack(n)}
    probe.write('trial', **row)
    probe.wait_tick(tick0 + 30)
    if played is None:
        trial_cycle(probe, tiles)           # it may be left selected: put a card down properly before the next trial
    return row


def trial_early(probe: Probe, tiles: Tiles, before_ms: int) -> dict | None:
    """Play a slot's card; touch that slot before_ms before the next card is dealt into it (no tile); touch a tile
    300 ms after the deal. Is the card that arrived put down -- did the early touch select it in advance?"""
    hand, cycle, _elixir = probe.cards()
    slots = [p for p, c in enumerate(hand) if playable(probe, c) and probe.kind(c) != 'building']
    if not slots or not cycle or not playable(probe, cycle[0]) or probe.kind(cycle[0]) == 'building':
        return trial_cycle(probe, tiles)
    position = random.choice(slots)
    first, second = hand[position], cycle[0]
    if not probe.settle(probe.cost(first) + probe.cost(second) + 0.3):
        return None
    tick0 = probe.tick
    n1 = probe.send(probe.place_line(position, tiles.next(probe.kind(first))))
    played = probe.command_of(first, tick0 - 1)
    if played is None:
        return None
    deal_us = probe.tick_us(played['issue'] + COMMAND_AGE)
    (sx, sy), tile = probe.hand_xy(position), tiles.next(probe.kind(second))
    tx, ty = probe.tile_xy(*tile)
    n2 = probe.send(f'at {deal_us - before_ms * 1000} taph {sx} {sy} 16')
    n3 = probe.send(f'at {deal_us + 300_000} taph {tx} {ty} 16')
    probe.wait_tick(played['issue'] + COMMAND_AGE + 16)
    taken = next((c for c in probe.commands if c['card'] == second and c['issue'] > played['issue']), None)
    row = {'trial': 'early', 'before_ms': before_ms, 'position': position, 'first': probe.name(first), 'second': probe.name(second),
           'first_issue': played['issue'], 'taken': taken is not None, 'second_issue': taken and taken['issue'],
           'tile': list(tile), 'acks': [probe.ack(n) for n in (n1, n2, n3)]}
    probe.write('trial', **row)
    probe.wait_tick(probe.tick + 12)
    if taken is None:
        trial_cycle(probe, tiles)
    return row


def trial_start(probe: Probe, tiles: Tiles, ticks=(70, 78, 86, 94, 102)) -> dict | None:
    """Right after a battle begins: one play at each of a few ticks. From which tick does the client take a play?"""
    if probe.tick > ticks[0] - 12:
        return None
    hand, _cycle, _elixir = probe.cards()
    slots = [p for p, c in enumerate(hand) if c is not None and probe.kind(c) != 'building']
    slots = sorted(slots, key=lambda p: hand[p] in KEEP)[:len(ticks)]      # the kept cards last, if needed at all
    numbers, plays = [], []
    tick0 = probe.tick
    for position, tick in zip(slots, ticks):
        numbers.append(probe.send(probe.place_line(position, tiles.next(probe.kind(hand[position])), at=probe.tick_us(tick))))
        plays.append((position, hand[position], tick))
    probe.wait_tick(ticks[-1] + 14)
    found = [next((c for c in probe.commands if c['card'] == card and c['issue'] >= tick0), None) for _p, card, _t in plays]
    row = {'trial': 'start', 'ticks': [t for _p, _c, t in plays], 'cards': [probe.name(c) for _p, c, _t in plays],
           'taken': [f is not None for f in found], 'issues': [f and f['issue'] for f in found],
           'acks': [probe.ack(n) for n in numbers]}
    probe.write('trial', **row)
    probe.wait_tick(ticks[-1] + 40)
    return row


STATE = {'building_tile': None, 'building_tick': -10 ** 9}


def trial_occupied(probe: Probe, tiles: Tiles) -> dict | None:
    """A building asked on the very tile where our last one still stands; then two plain plays, to see whether a
    refusal leaves anything behind that costs the plays after it."""
    hand, _cycle, _elixir = probe.cards()
    slots = [p for p, c in enumerate(hand) if playable(probe, c) and probe.kind(c) == 'building']
    if not slots:
        return trial_cycle(probe, tiles)
    position = slots[0]
    card = hand[position]
    if not probe.settle(probe.cost(card) + 0.3):
        return None
    standing = probe.tick - STATE['building_tick'] < 400 and STATE['building_tile'] is not None     # a Cannon lives 30 s
    tile = STATE['building_tile'] if standing else (8, 9)
    tick0 = probe.tick
    n = probe.send(probe.place_line(position, tile, at=probe.now_us() + 40_000))
    played = probe.command_of(card, tick0 - 1)
    row = {'trial': 'occupied', 'card': probe.name(card), 'tile': list(tile), 'asked': native_xy(probe, tile),
           'on_a_standing_building': standing, 'taken': played is not None, 'got': played and [played['x'], played['y']],
           'tick': tick0, 'ack': probe.ack(n), 'after': []}
    if played is not None and not standing:
        STATE['building_tile'], STATE['building_tick'] = tile, played['issue'] + COMMAND_AGE
    probe.wait_tick(tick0 + 12)
    if played is None:
        # what the next plays do while the refused building may still be selected: two plain ones, 0.6 s apart
        for _ in range(2):
            hand, _cycle, _elixir = probe.cards()
            plain = [p for p, c in enumerate(hand) if playable(probe, c) and probe.kind(c) != 'building']
            if not plain:
                break
            other = random.choice(plain)
            tick1 = probe.tick
            m = probe.send(probe.place_line(other, tiles.next(probe.kind(hand[other])), at=probe.now_us() + 40_000))
            took = probe.command_of(hand[other], tick1 - 1, timeout=0.6)
            row['after'].append({'card': probe.name(hand[other]), 'taken': took is not None, 'ack': probe.ack(m)})
            probe.wait_tick(tick1 + 26)
    probe.write('trial', **row)
    probe.wait_tick(probe.tick + 16)
    return row


def trial_burst(probe: Probe, tiles: Tiles, apart_ms: int = 30) -> dict | None:
    """Every card in the hand played in one go from a full bar, the dearest last: the game has debited none of them
    yet when the last is tapped. If the last is refused, the client counts what the screen shows (each cost off at
    its tap); if taken, what the game holds."""
    hand, _cycle, _elixir = probe.cards()
    if any(c is None for c in hand) or any(probe.kind(c) == 'building' for c in hand):
        return trial_cycle(probe, tiles)
    order = sorted(range(4), key=lambda p: probe.cost(hand[p]))
    total = sum(probe.cost(c) for c in hand)
    if total < 11.0 or total - probe.cost(hand[order[-1]]) > 9.5:
        return trial_cycle(probe, tiles)             # the first three must fit in ten, the fourth must not
    if not probe.settle(10.0):
        return None
    tick0 = probe.tick
    start = probe.now_us() + 50_000
    numbers = [probe.send(probe.place_line(p, tiles.next(probe.kind(hand[p])), at=start + i * (40 + apart_ms) * 1000))
               for i, p in enumerate(order)]
    time.sleep(0.9)
    found = [next((c for c in probe.commands if c['card'] == hand[p] and c['issue'] >= tick0 - 1), None) for p in order]
    row = {'trial': 'burst', 'apart_ms': apart_ms, 'cards': [probe.name(hand[p]) for p in order],
           'costs': [probe.cost(hand[p]) for p in order], 'taken': [f is not None for f in found],
           'issues': [f and f['issue'] for f in found], 'tick': tick0, 'acks': [probe.ack(n) for n in numbers]}
    probe.write('trial', **row)
    probe.wait_tick(tick0 + 50)
    return row


DEAL_DELTAS = [-300, -200, -150, -100, -75, -50, -25, 0, 25, 50, 75, 100, 150, 200]
DEAL_HOLDS = [16]               # the touch length of the deal experiment's second tap (--deal-holds 2,48)
ILLEGALS = ['enemy', 'river', 'tower', 'bridge', 'spell']
BUILDS = [(3, 8), (14, 8), (3, 9), (14, 9), (8, 5), (8, 6), (3, 4), (5, 9)]
EARLIES = [400, 200, 100, 50]
PAIR_SPACINGS = [0, 4, 8, 12, 16, 24, 33, 50, 66, 100]
SHAPES = [(8, 16), (8, 8), (8, 4), (8, 2), (0, 16), (0, 8), (4, 8), (16, 16), (33, 16), (8, 34), (0, 2), (4, 4)]
SELECTS = ['wait 0', 'wait 50', 'wait 200', 'wait 600', 'wait 1500', 'second', 'tile', 'twice']


def main(argv: list[str]) -> int:
    seconds = 10_000.0
    if '--seconds' in argv:
        index = argv.index('--seconds')
        seconds = float(argv[index + 1])
        argv = argv[:index] + argv[index + 2:]
    # --deltas / --spacings / --shapes / --selects: the values to try instead of the lists above
    #   --deltas 60,80,100   --spacings 0,8,16   --shapes 8x16,0x8 (gap x hold)   --selects "wait 50,second"
    for flag, name in (('--deltas', 'DEAL_DELTAS'), ('--spacings', 'PAIR_SPACINGS'), ('--shapes', 'SHAPES'),
                       ('--selects', 'SELECTS'), ('--deal-holds', 'DEAL_HOLDS')):
        if flag in argv:
            index = argv.index(flag)
            text = argv[index + 1]
            argv = argv[:index] + argv[index + 2:]
            if name == 'SHAPES':
                globals()[name] = [tuple(int(v) for v in item.split('x')) for item in text.split(',')]
            elif name == 'SELECTS':
                globals()[name] = [item.strip() for item in text.split(',')]
            else:
                globals()[name] = [int(v) for v in text.split(',')]
    experiments = argv or ['deal']
    probe = Probe()
    print(f'writing {probe.path}', flush=True)
    print('waiting for a battle ...', flush=True)
    while not (probe.active and probe.can_control and probe.me and len(probe.first_us) > 20):
        time.sleep(0.2)
    while probe.accounts is None:
        time.sleep(0.1)
    allowed, reason = scope_gate.check(probe.accounts, probe.side)
    print(('scope: ' if allowed else 'REFUSING: ') + reason, flush=True)
    if not allowed:
        return 1
    hand, cycle, elixir = probe.cards()
    clock = probe.ack(probe.send('now'))
    print(f'side {probe.side}, tick {probe.tick}, hand {[probe.name(c) for c in hand]}, next {[probe.name(c) for c in cycle]}, '
          f'elixir {elixir:.1f}; device clock {clock and clock.get("now")} vs reader {probe.now_us()}', flush=True)
    probe.write('start', side=probe.side, tick=probe.tick, experiments=experiments, reader_ms=READER_MS, queue_ms=QUEUE_MS,
                deck=[probe.name(V.card_identity(c)[0]) for c in probe.me['deck_card_ids']], clock=clock, layout_fix=True,
                reader_now=probe.now_us(), screen=[probe.width, probe.height])
    tiles = Tiles()
    if probe.tick < 55:
        row = trial_start(probe, tiles)
        if row:
            print(f"start t={probe.tick / 20:5.1f}s {dict(ticks=row['ticks'], cards=row['cards'], taken=row['taken'], issues=row['issues'])}", flush=True)
    probe.wait_tick(130, timeout=10)
    plans = {'deal': deque(random.sample(DEAL_DELTAS, len(DEAL_DELTAS))), 'pairs': deque(random.sample(PAIR_SPACINGS, len(PAIR_SPACINGS))),
             'shape': deque(random.sample(SHAPES, len(SHAPES))), 'select': deque(SELECTS),
             'elixir': deque([120, 60, 200]), 'illegal': deque(ILLEGALS), 'build': deque(BUILDS), 'early': deque(EARLIES),
             'occupied': deque([0]), 'burst': deque([30, 60])}
    started, done = time.time(), 0
    while probe.active and time.time() - started < seconds:
        for name in experiments:
            if not probe.active:
                break
            value = plans[name][0]
            plans[name].rotate(-1)
            if name == 'deal':
                row = trial_deal(probe, tiles, value, hold=random.choice(DEAL_HOLDS))
            elif name == 'pairs':
                row = trial_pairs(probe, tiles, value)
            elif name == 'shape':
                row = trial_shape(probe, tiles, *value)
            elif name == 'elixir':
                row = trial_elixir(probe, tiles, value)
            elif name == 'illegal':
                row = trial_illegal(probe, tiles, value)
            elif name == 'build':
                row = trial_build(probe, tiles, tuple(value))
            elif name == 'early':
                row = trial_early(probe, tiles, value)
            elif name == 'occupied':
                row = trial_occupied(probe, tiles)
            elif name == 'burst':
                row = trial_burst(probe, tiles, value)
            else:
                row = trial_select(probe, tiles, value)
            if row is None:
                time.sleep(0.2)
                continue
            done += 1
            brief = {k: v for k, v in row.items() if k in ('trial', 'delta_ms', 'spacing_ms', 'gap', 'hold', 'case', 'taken',
                                                              'first', 'second', 'card', 'cards', 'why', 'elixir', 'apart_ms',
                                                              'asked', 'got', 'tile', 'before_ms', 'on_a_standing_building',
                                                              'costs')}
            if row.get('after'):
                brief['after'] = [(a['card'], a['taken']) for a in row['after']]
            print(f'{done:3d} t={probe.tick / 20:5.1f}s {brief}', flush=True)
    print(f'{done} trials; battle {"over" if not probe.active else "still on"}; {probe.path}', flush=True)
    probe.send('quit')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
