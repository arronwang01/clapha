"""Bot console: one page that shows the live board and lets a model play your side.

Read-only except for the bot's ordinary Android taps, which require arming in the UI.
Scope: Training Camp, or a friendly battle against your own account.

    python3 mac012/console.py      ->  http://127.0.0.1:8777
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viewer as V  # noqa: E402  (applies the profile, provides pumps + frame state)
import layout_fix  # noqa: F401,E402  (side-0 x mirror fix)
import model_adapter as MA  # noqa: E402
import scope_gate  # noqa: E402
import firstlight_bot as FLB  # noqa: E402
import firstlight_obs as FLO  # noqa: E402
import tapper as TAP  # noqa: E402
import opponent_intel as INTEL  # noqa: E402
from cycle_tracker import Tracker, opponent_deck  # noqa: E402
from mac_profile import ADB, SERIAL  # type: ignore  # noqa: E402
from native_core.mumu_live_actions import ScreenLayout, send_card_taps  # noqa: E402
from native_core.mumu_live_protocol import adb_run  # noqa: E402

PORT = int(os.environ.get('CR_CONSOLE_PORT', '8777'))
LOG_FILE = Path(__file__).resolve().parents[1] / 'build' / f'bot_{PORT}.log'
X_TILES, Y_TILES = 18, 32
# The game consumes a queued command this many ticks after its issue tick (measured 22 ticks
# queue -> unit on every play; FirstLight's COMMAND_CONSUMPTION_STEPS = 21).
COMMAND_AGE_TICKS = 21
# A tap that never reaches the queue (missed, or refused by the client) is given up on.
IN_FLIGHT_SECONDS = 3.0
# How long a play chosen against elixir the client has not credited yet may wait for it.
DEFER_SECONDS = 0.5

# The deck the Hog 2.6 specialists trained on (FirstLight checkpoints/README.md), with the
# forms their interface sets explicitly: 1 = evolution, 2 = hero.
HOG26_DECK = {26000021: 0,   # Hog Rider
              26000014: 2,   # Musketeer (hero)
              27000000: 1,   # Cannon (evolution)
              28000000: 0,   # Fireball
              28000011: 0,   # The Log
              26000010: 1,   # Skeletons (evolution)
              26000038: 0,   # Ice Golem
              26000030: 0}   # Ice Spirit


def pending_commands(queue: list, accounts, tick: int) -> list[dict]:
    """Every command in the queue, both sides, for the board's ghost markers.

    A command executes COMMAND_AGE_TICKS after its issue tick; until then it is only a
    promise -- which is why the app draws it apart from real units (dashed, with a countdown).
    Opponent commands reach us ~7 ticks after issue, so they show ~0.7 s before landing.
    """
    out = []
    for entry in queue or ():
        issue = entry.get('issue_tick')
        if not isinstance(issue, int) or int(entry.get('card_id') or 0) <= 0:
            continue
        card_id, form, kind = V.card_identity(entry['card_id'])
        side = V.entry_side(entry, accounts)
        remaining = issue + COMMAND_AGE_TICKS - int(tick)
        if side is None or remaining < 0:
            continue
        out.append({'x': entry.get('x'), 'y': entry.get('y'), 'side': int(side),
                    'card_id': card_id, 'form': form, 'kind': kind,
                    'name': ('ability' if kind == 'ability' else
                             V.CARDS.get(card_id, {}).get('name', str(card_id))),
                    'remaining_ticks': remaining})
    return out


def screen_cell(column: int, row: int, side: int) -> int:
    """FirstLight's native tile -> the cell ScreenLayout.deployment_point expects.

    deployment_point takes a CANONICAL cell: the local player's own view, row 0 at their own
    back line (verified in Training Camp as side 1: canonical (8500, 9500) landed native
    (9500, 22500)). FirstLight decodes to NATIVE tiles, which for side 1 is that view rotated
    180 degrees. Passing the native cell straight through sent every side-1 play to the
    point-mirrored tile: own-half troops were aimed into the enemy half (the client snaps them
    to the nearest legal tile) and spells landed on the wrong lane.
    """
    if side == 1:
        column, row = X_TILES - 1 - column, Y_TILES - 1 - row
    return row * X_TILES + column


def opponent_deck_file(account, *, fresh_after: float):
    """(deck, forms, note) published by the other console for this account, or (None, None, '')."""
    if account is None:
        return None, None, ''
    path = V.DECKS / f'{account}.json'
    try:
        body = json.loads(path.read_text())
    except (OSError, ValueError):
        return None, None, ''
    deck = body.get('deck') or []
    if len(deck) != 8 or body.get('written', 0) < fresh_after:
        return None, None, ''
    names = ', '.join(str(V.CARDS.get(c, {}).get('name', c)) for c in deck)
    return list(deck), body.get('forms'), f'opponent deck from their console: {names}'


class OutcomeLatch:
    """Acts on a battle result only once it has held for half a second -- one misread frame
    must not end a battle for us -- and resumes play if the result goes away again."""
    HOLD_S = 0.5

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.result, self.since, self.over = None, None, False

    def update(self, result, now: float) -> str | None:
        """'over' the moment the battle counts as decided, 'withdrawn' if a decided result
        goes away, otherwise None. A decided battle is recorded once."""
        if result is None:
            self.since = None
            if self.over:
                self.over, self.result = False, None
                return 'withdrawn'
            return None
        self.since = self.since if self.since is not None else now
        if not self.over and now - self.since >= self.HOLD_S:
            self.result, self.over = result, True
            return 'over'
        return None


class Bot:
    """Mirrors the user's bots.py loop: decide every STEP_TICKS, one play in flight at a
    time, never decide while our own command is still queued."""

    def __init__(self):
        self.model = None
        self.running = False
        self.armed = False
        self.status = 'off'
        self.plays = 0
        self.last_play = ''
        self.last_result = ''
        self._last_line, self._repeats, self._first_stamp = None, 0, ''
        self.log: list[str] = []
        self.thread = None
        self.layout = None
        self.gate = 'unchecked'
        self.gate_ok = False
        self.decoding = 'auto'          # auto | sampled | greedy
        self.decoding_used = None
        self.tracker = None
        self.opp_deck = None
        self.seen_plays = 0
        self.deduced = ''
        self.rtt: list[int] = []   # our taps: game ticks from tap to the command's issue tick
        self.tapper = None
        self.opponent_intel = None

    def note(self, line: str) -> None:
        """Log a line on the page and to a file.

        A message identical to the previous one is folded into a repeat count instead of
        appended again, so one error firing every turn cannot push every other message off the
        page. Everything also goes to build/bot_<port>.log, so nothing is lost when it scrolls.
        """
        stamp = time.strftime("%H:%M:%S")
        if self.log and self._last_line == line:
            self._repeats += 1
            self.log[-1] = f'{self._first_stamp}-{stamp}  {line}  (x{self._repeats + 1})'
        else:
            self._last_line, self._repeats, self._first_stamp = line, 0, stamp
            self.log.append(f'{stamp}  {line}')
            del self.log[:-200]
        try:
            with open(LOG_FILE, 'a', encoding='utf-8') as handle:
                handle.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {line}\n')
        except OSError:
            pass

    def start(self, model: str, armed: bool) -> str:
        if self.running:
            return 'already running'
        if model not in MA.MODELS and model not in FLB.CHECKPOINTS:
            return f'unknown model {model}'
        self.model, self.armed, self.plays = model, armed, 0
        self.running = True
        self.status = f'{model}: loading'
        self.log = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return 'started'

    def stop(self) -> str:
        self.running = False
        self.status = 'off'
        thread = getattr(self, 'thread', None)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)     # the loop checks `running` at least every 0.3 s
        return 'stopped'

    EVO_FILE = Path(__file__).resolve().parents[1] / 'build' / 'evo_cycles.json'

    @property
    def evo_required(self) -> dict:
        if not hasattr(self, '_evo_required'):
            try:
                self._evo_required = {int(k): int(v) for k, v in
                                      json.loads(self.EVO_FILE.read_text()).items()}
            except (OSError, ValueError):
                self._evo_required = {}
        return self._evo_required

    def _measure_evolutions(self, deck: list, me: dict, battle, tick: int) -> None:
        """The game's own evolution requirement per card: a deck slot's progress counter rises
        by one per normal play and falls to 0 on the evolved play (device: Skeletons
        0 -> 1 -> 2 -> 0), so a fall from k to 0 means the card needs k. Remembered across
        runs; used instead of FirstLight's 15.535 cycle counts, which differ for many cards.

        Only a fall inside one battle counts: every counter starts the next battle at 0, and
        comparing across that boundary read "needs 1" for Skeletons and Cannon at the start of
        the 2026-09-25 09:34 friendly (the previous battle had ended with both at 1)."""
        progress = me.get('evo_progress') or []
        previous = getattr(self, '_last_progress', None)
        self._last_progress = (battle, int(tick), list(deck), list(progress))
        if (not previous or previous[0] != battle or not 0 <= int(tick) - previous[1] <= 100
                or previous[2] != list(deck) or len(progress) != len(deck)):
            return
        for slot, (before, now) in enumerate(zip(previous[3], progress)):
            if before > 0 and now == 0:
                card = int(deck[slot])
                known = self.evo_required.get(card)
                if known != before:
                    self.evo_required[card] = int(before)
                    self.EVO_FILE.parent.mkdir(parents=True, exist_ok=True)
                    self.EVO_FILE.write_text(json.dumps({str(k): v for k, v in
                                                         sorted(self.evo_required.items())}))
                    self.note(f'evolution requirement measured: '
                              f'{V.CARDS.get(card, {}).get("name", card)} needs {before} plays'
                              + (f' (was {known})' if known else ''))

    OTHER_CONSOLES = (8777, 8778)

    def _decide_sampling(self, accounts, side) -> bool:
        """Sampled or greedy decoding for this battle, the way FirstLight runs each setup:
        against a human or a bot their run_offline_match samples; a model against a model is
        their offline_duel, which decodes greedily. 'auto' tells them apart by asking the other
        console whether it is running a model against this one's account."""
        if self.decoding in ('sampled', 'greedy'):
            choice = self.decoding
            why = 'set by hand'
        else:
            opponent = next((a['lo'] for a in (accounts or []) if a and a.get('side') == 1 - side), None)
            duel = False
            for port in self.OTHER_CONSOLES:
                if port == PORT or opponent is None:
                    continue
                try:
                    import urllib.request
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/state?log=1', timeout=1.0) as reply:
                        other = json.loads(reply.read())
                    theirs = (other.get('bot') or {})
                    other_account = None
                    for path in V.DECKS.glob('*.json'):
                        body = json.loads(path.read_text())
                        if body.get('account_lo') == opponent:
                            other_account = opponent
                    duel = bool(theirs.get('running')) and other_account is not None
                except Exception:  # noqa: BLE001  - the other console may simply be off
                    continue
            choice = 'greedy' if duel else 'sampled'
            why = ('model against model (your other device is running one): FirstLight\'s '
                   'offline_duel decodes greedily' if duel else
                   'against a human or a bot: FirstLight\'s run_offline_match samples')
        self.decoding_used = choice
        self.note(f'decoding: {choice.upper()} - {why}')
        return choice == 'sampled'

    def set_mode(self, mode: str, model: str | None) -> str:
        """One control instead of start/stop + an 'armed' box: off, watch (decides, never
        taps) or play (taps). Watch <-> play on the same model only flips taps, so the policy's
        episode and recurrent state carry on; a different model restarts it."""
        if mode not in ('off', 'watch', 'play'):
            return f'unknown mode {mode}'
        if mode == 'off':
            return self.stop()
        model = model or self.model
        if not model:
            return 'pick a model'
        if self.running and model == self.model:
            self.armed = mode == 'play'
            self.note(f'mode -> {mode.upper()}' + (' (taps on)' if self.armed else
                                                   ' (decides, no taps)'))
            return 'ok'
        if self.running:
            self.stop()
        return self.start(model, mode == 'play')

    def _settle_in_flight(self, in_flight: list[dict], queue: list, executed: list,
                          local_account, side: int, tick: int) -> list[dict]:
        """Match our taps to their queue entries, time the round trip, and drop plays that
        have executed (issue + 21, when the client debits the elixir) or never arrived.

        A command the 100 ms queue sampler never caught still shows up in the executed list,
        so that is searched too; otherwise its elixir would stay reserved until the timeout."""
        claimed = {(f.get('issue_tick'), f.get('seq')) for f in in_flight}
        own_executed = [dict(p, account_lo=local_account) for p in executed
                        if p.get('side') == side and p.get('kind', 'card') == 'card']
        for flight in in_flight:
            if flight.get('issue_tick') is not None:
                continue
            for entry in [*queue, *own_executed]:
                key = (entry.get('issue_tick'), entry.get('seq'))
                if (key in claimed or entry.get('card_id') != flight['card']
                        or not isinstance(entry.get('issue_tick'), int)
                        or entry['issue_tick'] < flight['tap_tick'] - 2
                        or (local_account is not None
                            and entry.get('account_lo') != local_account)):
                    continue
                flight['issue_tick'], flight['seq'] = key
                claimed.add(key)
                self.rtt.append(max(0, entry['issue_tick'] - flight['tap_tick']))
                del self.rtt[:-50]
                self._report_latency(flight)
                self._report_placement(flight, entry)
                break
        now = time.time()
        return [f for f in in_flight
                if (f.get('issue_tick') is None and now - f['tap_time'] <= IN_FLIGHT_SECONDS)
                or (f.get('issue_tick') is not None
                    and tick < f['issue_tick'] + COMMAND_AGE_TICKS)]

    def _report_placement(self, flight: dict, entry: dict) -> None:
        """Where the game put the command, against where the policy asked, both native.
        A building or troop lands on the tile; anything over a tile off is a mapping bug."""
        target = flight.get('target')
        if not target or entry.get('x') is None or entry.get('y') is None:
            return
        dx, dy = (entry['x'] - target[0]) / 1000.0, (entry['y'] - target[1]) / 1000.0
        off = (dx * dx + dy * dy) ** 0.5
        name = V.CARDS.get(flight['card'], {}).get('name', flight['card'])
        self.note(f'placement {name}: asked native ({target[0]}, {target[1]}), game got '
                  f'({entry["x"]}, {entry["y"]}), off {off:.1f} tiles'
                  + ('  <-- MISPLACED' if off > 1.5 else ''))

    def _lookup_opponent(self, accounts, side: int, battle=None) -> None:
        """Opponent's tag and currently equipped deck from the official API, in the
        background (never delays a decision). A prior, not ground truth: logged and saved to
        build/opponent_decks/, while the tracker still learns their real deck card by card.
        Skipped for Training Camp (the trainer has no account) and without an API token."""
        opponent = next((a for a in (accounts or []) if a and a.get('side') == 1 - side), None)
        if not opponent or int(opponent.get('lo') or 0) <= 0 or INTEL.token() is None:
            return

        def work():
            info = INTEL.lookup(int(opponent.get('hi') or 0), int(opponent['lo']))
            if info['error']:
                self.note(f'opponent {info["tag"]}: deck lookup failed - {info["error"][:120]}')
                return
            cards = ', '.join(f'{c["name"]}{" (evo)" if c["evolution_level"] else ""}'
                              for c in info['deck'])
            self.note(f'opponent {info["tag"]} {info["name"]} ({info["trophies"]} trophies), '
                      f'equipped deck per API: {cards}')
            self.opponent_intel = info
            # Kept with the match recording (same session folder as frames.jsonl /
            # queue.jsonl), so training data carries what the API said about each opponent.
            try:
                V.SESSION.mkdir(parents=True, exist_ok=True)
                with open(V.SESSION / 'opponent_intel.jsonl', 'a', encoding='utf-8') as handle:
                    handle.write(json.dumps({'battle': battle, 'side': 1 - side,
                                             'looked_up': time.time(), **info}) + '\n')
            except OSError:
                pass

        threading.Thread(target=work, daemon=True).start()

    def _report_queue_oddities(self, executed: list, handled: set) -> None:
        """Say, once per play, about queue entries that cannot become a card play."""
        for play in executed:
            kind = play.get('kind', 'card')
            if kind == 'card':
                continue
            key = (play.get('issue_tick'), play.get('seq'), play.get('raw_card_id'))
            if key in handled:
                continue
            handled.add(key)
            raw = play.get('raw_card_id', play.get('card_id'))
            if kind == 'unknown':
                self.note(f'queue: card id {raw} is not in this build\'s catalog '
                          f'(game update?) - play not registered')
            elif kind == 'unattributed':
                self.note(f'queue: a play of {V.CARDS.get(play.get("card_id"), {}).get("name", raw)}'
                          f' could not be attributed to a side - not registered')
            elif kind == 'ability' and play.get('ability_card') is None:
                self.note(f't={play["tick"]/20:5.1f}s  side {play["side"]} activated an ability')

    def _report_latency(self, flight: dict) -> None:
        """One line per play: where the time went between the frame the policy saw and the
        command reaching the game. Everything after that (issue + 21 ticks to execute) is the
        game's own command delay, the same for a human."""
        timing = flight.get('timing') or {}
        gesture = ''
        if self.tapper is not None and self.tapper.timings:
            last = self.tapper.timings[-1]
            gesture = f', gesture {last["gesture_ms"]:.0f} ms'
        ticks = flight['issue_tick'] - flight['tap_tick']
        self.note(f'latency {V.CARDS.get(flight["card"], {}).get("name", flight["card"])}: '
                  f'frame age {timing.get("frame_age_ms", 0):.0f} ms, '
                  f'turn wait {timing.get("turn_wait_ms", 0):.0f} ms, '
                  f'inference {timing.get("inference_ms", 0):.0f} ms, '
                  f'tap sent {timing.get("send_ms", 0):.0f} ms after decision{gesture}, '
                  f'tap -> issued {ticks} ticks ({ticks * 50} ms), then 21 ticks to land')

    def _try_play(self, move: dict, me: dict, deck: list, reserved: float,
                  in_flight: list[dict], accounts, side: int, frame: dict) -> bool:
        """Tap one play. True when it is done with (tapped, or refused for good); False when it
        should wait for elixir.

        The card is found by identity in the hand as memory holds it now, not by the policy's
        slot, so a hand that changed between the frame and the tap cannot tap the wrong card.
        """
        card = move['card']
        name = V.CARDS.get(card, {}).get('name', str(card))
        if any(f['card'] == card for f in in_flight):
            return True        # already on its way; the policy's state includes it
        positions = [pos for pos, index in enumerate(me['hand_deck_indices'])
                     if 0 <= index < len(deck) and deck[index] == card]
        if not positions:
            return False       # not in the hand (yet): the hand changed since the frame
        position = positions[0]
        cost = float(V.CARDS.get(card, {}).get('elixir') or 0)
        if me['elixir_raw'] / 10000.0 - reserved < cost - 1e-6:
            return False       # the client cannot place it yet
        cell = screen_cell(move['column'], move['row'], side)
        allowed, reason = scope_gate.check(accounts, side)
        if reason != self.gate:
            self.gate, self.gate_ok = reason, allowed
            self.note(('scope: ' if allowed else 'SCOPE BLOCK: ') + reason)
        if not self.armed or not allowed:
            why = ' [dry run]' if not self.armed else ' [BLOCKED by scope gate]'
            self.note(f't={frame["game_tick"]/20:5.1f}s  would play {name} '
                      f'slot {position} at row {move["row"]} col {move["column"]}{why}')
            return True
        try:
            if self.tapper is not None and self.tapper.alive():
                sent = self.tapper.play(self.layout.hand_point(position),
                                        self.layout.deployment_point(cell, side))
            else:
                send_card_taps(ADB, SERIAL, self.layout, position, cell, side=side)
                sent = time.time()
        except Exception as error:  # noqa: BLE001
            self.note(f'tap failed: {error}')
            return True
        timing = dict(move.get('timing') or {})
        if 'decided' in timing:
            timing['send_ms'] = (sent - timing['decided']) * 1000.0
        in_flight.append({'slot': me['hand_deck_indices'][position], 'card': card,
                          'cost': cost, 'tap_tick': int(frame['game_tick']),
                          'tap_time': sent, 'timing': timing,
                          'target': (move['column'] * 1000 + 500, move['row'] * 1000 + 500)})
        self.plays += 1
        self.last_play = f'{name} at row {move["row"]} col {move["column"]}'
        waited = time.time() - move['since']
        self.note(f't={frame["game_tick"]/20:5.1f}s  {name:<14} row {move["row"]:2} '
                  f'col {move["column"]:2}' + (f'  (waited {waited:.1f}s for elixir)'
                                               if waited > 0.15 else ''))
        return True

    # Hero ability buttons on the 1440x2560 device: centres ~(140, 1955) and ~(1300, 1955), just
    # above the hand. Which side a button uses is a game setting, so it differs per account: one
    # screenshot (2026-09-24) had a single hero on the LEFT, the 2026-09-25 friendly had it on the
    # RIGHT (and taps on the left did nothing). The side is therefore learned per account: a tap
    # that leaves the button Ready with the same charges for 2.5 s missed, and the other side is
    # used from then on (build/ability_buttons.json).
    ABILITY_BUTTONS = {'left': (140, 1955), 'right': (1300, 1955)}
    ABILITY_DEFAULTS = {'single': 'right', 'dual_1': 'left', 'dual_2': 'right'}
    ABILITY_SIDES = Path(__file__).resolve().parents[1] / 'build' / 'ability_buttons.json'

    def _ability_sides(self) -> dict:
        try:
            return json.loads(self.ABILITY_SIDES.read_text())
        except (OSError, ValueError):
            return {}

    def _ability_side(self, controller_slot: int, me: dict, account) -> tuple[str, str]:
        bound = [a for a in me.get('abilities') or () if int(a.get('character_id') or 0)]
        key = 'single' if len(bound) < 2 else f'dual_{controller_slot}'
        learned = self._ability_sides().get(str(account), {})
        return key, learned.get(key, self.ABILITY_DEFAULTS.get(key, 'right'))

    def _ability_point(self, side_name: str) -> tuple[int, int]:
        x, y = self.ABILITY_BUTTONS[side_name]
        return (round(x * self.layout.width / 1440), round(y * self.layout.height / 2560))

    def _ability_missed(self, flight: dict) -> None:
        """The button stayed Ready with its charges: the tap was on the wrong side."""
        other = 'left' if flight['side'] == 'right' else 'right'
        sides = self._ability_sides()
        sides.setdefault(str(flight['account']), {})[flight['side_key']] = other
        try:
            self.ABILITY_SIDES.write_text(json.dumps(sides, indent=1) + '\n')
        except OSError:
            pass
        self.note(f'ability tap on the {flight["side"].upper()} did nothing (button still Ready '
                  f'after 2.5 s) - this account has it on the {other.upper()}; using that now')

    def _try_ability(self, move, observation, me: dict, in_flight: list[dict], accounts,
                     side: int, frame: dict) -> None:
        """Tap the hero ability the policy chose: its source unit -> the controller that owns it
        -> that controller's button. Gated exactly like card plays (arming, scope gate)."""
        source = getattr(move, 'source_entity', None)
        player = next((p for p in observation.players if p.owner == side), None)
        state = next((a for a in (player.ability_runtime_states if player else ())
                      if a.source_entity == source), None)
        if state is None:
            self.note(f'ability for unit {source} has no controller in this frame - not tapped')
            return
        slot = int(state.attributes.get('controller_slot', 1))
        if any(a['slot'] == slot for a in in_flight):
            return
        name = state.ability_id
        allowed, reason = scope_gate.check(accounts, side)
        if reason != self.gate:
            self.gate, self.gate_ok = reason, allowed
            self.note(('scope: ' if allowed else 'SCOPE BLOCK: ') + reason)
        if not self.armed or not allowed:
            why = ' [dry run]' if not self.armed else ' [BLOCKED by scope gate]'
            self.note(f't={frame["game_tick"]/20:5.1f}s  would use {name}{why}')
            return
        account = next((a['lo'] for a in (accounts or []) if a and a.get('side') == side), None)
        side_key, side_name = self._ability_side(slot, me, account)
        point = self._ability_point(side_name)
        try:
            if self.tapper is not None and self.tapper.alive():
                self.tapper.tap(point)
            else:
                adb_run(ADB, SERIAL, 'shell', 'input', 'tap', str(point[0]), str(point[1]))
        except Exception as error:  # noqa: BLE001
            self.note(f'ability tap failed: {error}')
            return
        charges = next((int(a.get('charges', -1)) for a in me.get('abilities') or ()
                        if int(a.get('controller_slot', 0)) == slot), -1)
        in_flight.append({'slot': slot, 'source': source, 'cost': float(state.elixir_cost or 0),
                          'time': time.time(), 'charges': charges, 'account': account,
                          'side': side_name, 'side_key': side_key})
        self.plays += 1
        self.last_play = name
        self.note(f't={frame["game_tick"]/20:5.1f}s  ABILITY {name} (controller {slot}, '
                  f'button at {point[0]},{point[1]})')

    RESULTS = Path(__file__).resolve().parents[1] / 'build' / 'results.jsonl'

    def _record_result(self, result, side: int, tick: int, runner, accounts) -> None:
        """Say who won, and keep one line per battle in build/results.jsonl."""
        winner, reason = result
        verdict = 'no winner yet (tiebreak)' if winner is None else \
            ('YOU WON' if winner == side else 'you lost')
        self.last_result = f'{verdict} - {reason}'
        self.note(f'battle over at t={tick / 20:.1f}s: {self.last_result}. {self.plays} plays; '
                  f'no more taps this battle')
        opponent = next((a['lo'] for a in (accounts or []) if a and a.get('side') == 1 - side),
                        None)
        entry = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'port': PORT,
                 'serial': SERIAL, 'model': self.model, 'sampled': bool(runner.sample),
                 'side': side, 'won': None if winner is None else winner == side,
                 'reason': reason, 'tick': tick, 'plays': self.plays,
                 'opponent_account': opponent}
        try:
            with self.RESULTS.open('a') as handle:
                handle.write(json.dumps(entry) + '\n')
        except OSError:
            pass

    def _check_deck_fit(self, deck, forms) -> None:
        """The Hog specialists were trained on one deck with fixed forms. Say so when the
        deck differs: out of that deck they are a different, weaker model."""
        if self.model not in ('fl:hog1', 'fl:hog2'):
            return
        flags = {int(c): int(f or 0) for c, f in zip(deck or [], forms or [0] * 8)}
        missing = [V.CARDS.get(c, {}).get('name', c) for c in HOG26_DECK if c not in flags]
        wrong_form = [V.CARDS.get(c, {}).get('name', c) for c, form in HOG26_DECK.items()
                      if c in flags and form and not flags[c] & form]
        if missing or wrong_form:
            self.note('DECK MISMATCH for the Hog specialist - trained on Hog, Hero Musketeer, '
                      'Evo Cannon, Fireball, Log, Evo Skeletons, Ice Golem, Ice Spirit. '
                      + (f'missing: {", ".join(map(str, missing))}. ' if missing else '')
                      + (f'form not equipped: {", ".join(map(str, wrong_form))}.'
                         if wrong_form else ''))
        else:
            self.note('deck matches the Hog specialist training deck')

    def _hand_position(self, hand_indices: list[int], deck_slot: int) -> int | None:
        return hand_indices.index(deck_slot) if deck_slot in hand_indices else None

    def _run_firstlight(self) -> None:
        """FirstLight V4: FAIR-tier observations built from our reader, decisions at 4 Hz."""
        import re
        try:
            runner = FLB.FirstLightRunner(self.model)
        except Exception as error:  # noqa: BLE001
            self.status = f'load failed: {error}'
            self.running = False
            return
        sizes = re.findall(r'(\d+)x(\d+)', adb_run(ADB, SERIAL, 'shell', 'wm size'))
        self.layout = ScreenLayout.from_size(*map(int, sizes[-1]))
        if self.armed:
            try:
                if self.tapper is None or not self.tapper.alive():
                    self.tapper = TAP.Tapper(ADB, SERIAL, *map(int, sizes[-1]))
                self.note(f'taps: {self.tapper.describe()}')
            except Exception as error:  # noqa: BLE001
                self.tapper = None
                self.note(f'fast_tap unavailable ({error}); falling back to adb input tap')
        self.note(f'{self.model} loaded (FirstLight V4, FAIR tier, '
                  f'{20.0/runner.decision_ticks:.0f} Hz); '
                  f'{"ARMED - will tap" if self.armed else "dry run - no taps"}')
        # Pay the model's cold start now, not on the first decision of the match, then take the
        # model and FirstLight's catalogs out of the garbage collector's scans: a full pass over
        # them stalled the decision loop past a turn (live: a 638 ms turn late in a match).
        try:
            warm_ms = runner.warm_up()
            import gc
            gc.collect()
            gc.freeze()
            self.note(f'model warmed up in {warm_ms:.0f} ms; long-lived objects frozen out of GC')
        except Exception as error:  # noqa: BLE001
            self.note(f'warm-up failed (first decision will be slow): {error}')
        battle = None
        fl_battle = None
        latch = OutcomeLatch()
        reported: set[int] = set()
        reported_untracked: set[tuple[int, int]] = set()
        pending_battle, pending_since = None, 0.0
        handled_queue: set = set()
        last_turn = -10 ** 9
        in_flight: list[dict] = []
        deferred: list[dict] = []
        reported_abilities: set[int] = set()
        abilities_in_flight: list[dict] = []
        while self.running:
            with V.LOCK:
                frame, health = V.STATE['frame'], V.STATE['health']
                frame_time = V.STATE['updated'] or time.time()
                reader_error = V.STATE['error']
                queue = list(V.STATE['queue'])
                accounts = V.STATE['accounts']
                revealed = V.STATE['revealed']
            if not frame or not health or not frame.get('battle_active'):
                # A failed reader and an idle game are not the same thing: say which,
                # so a device whose reader never attached stops reading as 'no battle'.
                self.status = (f'{self.model}: reader not attached - {reader_error}'
                               if reader_error else f'{self.model}: waiting for a battle')
                time.sleep(0.3)
                continue
            side = health.get('local_side')
            if side not in (0, 1) or not health.get('can_control'):
                self.status = f'{self.model}: {health.get("status")}'
                time.sleep(0.2)
                continue
            me = next((p for p in frame['players'] if p['side'] == side), None)
            # For up to ~0.5 s after a play the played slot reads -1 while the next card is drawn
            # (the card has already moved to the end of the cycle, so hand + cycle still make the
            # deck). FirstLight's env keeps deciding through that with the slot simply not
            # playable; skipping the turn instead left the policy's recurrent state behind the
            # game on every such draw (the "missed decision turn" lines).
            if not me or len(me['hand_deck_indices']) != 4 or not any(
                    h >= 0 for h in me['hand_deck_indices']):
                time.sleep(0.05)
                continue
            our_deck = me.get('deck_card_ids') or []
            if len(our_deck) != 8:
                time.sleep(0.2)
                continue
            self._measure_evolutions(our_deck, me, frame['chain']['battle'], frame['game_tick'])

            if frame['chain']['battle'] != battle:
                # The episode config needs both decks, and FirstLight's tracker needs the
                # opponent's real one: it refuses a public play of a card outside it. It is
                # not readable from this client mid-battle, but in these friendlies the other
                # account is the user's own and its console publishes its deck (viewer.
                # publish_deck). Warm-up is observe-only anyway, so wait for that file up to
                # tick 60 rather than start the episode on a stand-in.
                if pending_battle != frame['chain']['battle']:
                    pending_battle, pending_since = frame['chain']['battle'], time.time()
                opponent_account = next((a['lo'] for a in (accounts or [])
                                         if a and a.get('side') == 1 - side), None)
                opponent, opponent_forms, deck_note = opponent_deck_file(
                    opponent_account, fresh_after=pending_since - 5)
                if (opponent is None and opponent_account is not None
                        and frame['game_tick'] < 60):
                    self.status = f'{self.model}: waiting for the opponent deck'
                    time.sleep(0.05)
                    continue
                # No fallback to an older publication: a stale file is someone's previous
                # deck, and its wrong cards were refused by the tracker all match. Unknown is
                # better -- the runner learns the deck from what the opponent plays.
                if opponent is None:
                    deck_note = ('opponent deck not published - learning it from their plays '
                                 '(Training Camp, or the other console is not running)')
                self.note(deck_note)
                self._lookup_opponent(accounts, side, frame['chain']['battle'])
                observation, fl_battle = FLO.build(
                    frame, health, episode_id=str(frame['chain']['battle']))
                try:
                    runner.end_battle()
                    # The tracker is seeded with exact initial elixir for BOTH owners, which
                    # is what BattleEnv passes and what the tensorizer demands of a
                    # tracker-backed episode. Read, not assumed: a battle joined late does not
                    # start at five.
                    runner.sample = self._decide_sampling(accounts, side)
                    runner.start_battle(our_deck, opponent, side, observation,
                                        {p['side']: p['elixir_raw'] / 10000.0
                                         for p in frame['players']},
                                        our_forms=me.get('deck_form_flags'),
                                        opponent_forms=opponent_forms)
                except Exception as error:  # noqa: BLE001
                    self.note(f'episode start failed: {error}')
                    time.sleep(1.0)
                    continue
                battle = frame['chain']['battle']
                handled_queue: set = set()
                last_turn, in_flight, deferred = -10 ** 9, [], []
                abilities_in_flight.clear()
                latch.reset()
                self.plays = 0
                self.note(f'new battle, you are side {side}; '
                          f'warm-up until tick {runner.first_decision_tick}')
                self._check_deck_fit(our_deck, me.get('deck_form_flags'))

            local_account = next((a['lo'] for a in (accounts or [])
                                  if a and a.get('side') == side), None)
            deck = me['deck_card_ids']
            with V.LOCK:
                executed = list(V.STATE['plays'])
            in_flight = self._settle_in_flight(in_flight, queue, executed, local_account,
                                               side, frame['game_tick'])
            reserved = sum(f['cost'] for f in in_flight)

            # A tap chosen a moment before the client credits the elixir (frame age, rounding)
            # waits here, briefly, until the client can actually place it.
            deferred = [d for d in deferred if time.time() - d['since'] <= DEFER_SECONDS
                        and not latch.over]
            for move in list(deferred):
                if self._try_play(move, me, deck, reserved, in_flight, accounts, side, frame):
                    deferred.remove(move)
                    reserved = sum(f['cost'] for f in in_flight)

            # One turn per five-tick window, and never a skipped one. decide() advances a
            # recurrent state and feeds its own chosen action into the next turn, so the
            # schedule belongs to the policy: our tap bookkeeping may suppress a tap, but it
            # must not suppress a turn. Ticks before their first decision tick are warm-up,
            # fed to the tensorizer without acting, as PolicyService's 'observe' op does.
            turn = runner.turn_tick(frame['game_tick'])
            if turn <= last_turn:
                time.sleep(0.02)
                continue
            skipped = (turn - last_turn) // runner.decision_ticks - 1 if last_turn > 0 else 0
            last_turn = turn
            turn_began = time.time()
            # Ticks this frame is past the start of its decision turn: time the policy could
            # have acted but the five-tick grid it was trained on made it wait.
            turn_wait_ms = (frame['game_tick'] - turn) * 50.0

            try:
                # Every card either side has shown is registered with the tracker first, so
                # the opponent's deck grows as they play; anything refused is said, per play.
                revealed_cards = {
                    index: [V.card_identity(c)[0] for c in cards
                            if V.card_identity(c)[2] == 'card']
                    for index, cards in enumerate(revealed or [[], []])}
                for owner, card_id, reason in runner.register_plays(executed, revealed_cards):
                    who = 'opponent' if owner != side else 'own'
                    self.note(f'{who} {V.CARDS.get(card_id, {}).get("name", card_id)} NOT '
                              f'registered: {reason}')
                opponent_player = next((p for p in frame['players']
                                        if p.get('side') == 1 - side), None)
                for line in runner.attribute_opponent_abilities(executed, opponent_player):
                    self.note(line)
                self._report_queue_oddities(executed, handled_queue)
                seen = {side: revealed_cards.get(side, []),
                        1 - side: [c for c in revealed_cards.get(1 - side, [])
                                   if c in runner.opponent_seen]}
                observation, fl_battle = FLO.build(
                    frame, health, episode_id=str(battle), battle=fl_battle, revealed=seen,
                    reserved=reserved + sum(a['cost'] for a in abilities_in_flight),
                    plays=executed, decks=runner.tracked_decks(),
                    hand_forms=runner.hand_forms(deck, me.get('deck_form_flags'),
                                                 me.get('evo_progress'), self.evo_required),
                    pending_ability_sources=tuple(a['source'] for a in abilities_in_flight),
                    evo_required=self.evo_required)
                for character in sorted(fl_battle.unresolved_abilities - reported_abilities):
                    reported_abilities.add(character)
                    self.note(f'hero controller character {character} has no single FirstLight '
                              f'ability - not offered to the policy')
                for card_id in sorted(set(fl_battle.unresolved) - reported):
                    reported.add(card_id)
                    base = FLO.base_card(card_id)
                    name = V.CARDS.get(base, {}).get('name', card_id)
                    if V.CARDS.get(base, {}).get('type') == 'spell':
                        self.note(f'{name}: spell effect on the board is not shown to the '
                                  f'model (no FirstLight archetype for spell objects yet)')
                    else:
                        self.note(f'UNIT {name} ({card_id}) is not shown to the model: '
                                  f'FirstLight has no archetype for it (newer card?)')
                for side_card in sorted(fl_battle.untracked_plays - reported_untracked):
                    reported_untracked.add(side_card)
                    who = 'opponent' if side_card[0] != side else 'own'
                    self.note(f'{who} play of '
                              f'{V.CARDS.get(side_card[1], {}).get("name", side_card[1])} not given '
                              f'to the tracker (Mirror, or no FirstLight card spec)')
                for card_id in sorted(fl_battle.form_fallbacks - reported):
                    reported.add(card_id)
                    base = FLO.base_card(card_id)
                    self.note(f'{V.CARDS.get(base, {}).get("name", base)} form {card_id} is '
                              f'unknown to FirstLight - shown to the model as the base unit')
                # Once the battle is decided the client keeps its clock running for a few
                # seconds and accepts taps it never executes (2026-09-25: four taps after a
                # sudden-death win). Stop acting then; a result that does not hold for half a
                # second is not acted on, and one that is withdrawn resumes play.
                event = latch.update(FLO.battle_result(fl_battle, frame['game_tick']),
                                     time.time())
                if event == 'over':
                    self._record_result(latch.result, side, frame['game_tick'], runner,
                                        accounts)
                elif event == 'withdrawn':
                    self.note('battle result withdrawn (a tower read as down is back) - '
                              'playing on')
                if latch.over:
                    self.status = f'{self.model}: battle over - {self.last_result}'
                    continue
                if turn < runner.first_decision_tick:
                    runner.observe(observation)
                    self.status = (f'{self.model}: warm-up '
                                   f'{turn}/{runner.first_decision_tick}')
                    continue
                started = time.time()
                moves = runner.decide(observation)
                decided = time.time()
                timing_now = {'prep_ms': (started - turn_began) * 1000.0,
                              'decide_ms': (decided - started) * 1000.0,
                              'frame_age_ms': (turn_began - frame_time) * 1000.0}
            except Exception as error:  # noqa: BLE001
                self.note(f'decide failed: {error}')
                self.status = f'{self.model}: DECIDE FAILING - {str(error)[:80]}'
                continue
            if skipped > 0:
                # Where the time went, so the cause is in the log rather than guessed at: the
                # previous turn's own cost, how long the loop was away between turns, and how
                # old the frame was when this turn began.
                prev = getattr(self, '_prev_turn_timing', {}) or {}
                self.note(f'missed {skipped} decision turn(s) at tick {turn} - the policy state '
                          f'is behind the game. previous turn: prep '
                          f'{prev.get("prep_ms", 0):.0f} ms, decide {prev.get("decide_ms", 0):.0f} '
                          f'ms, after {prev.get("after_ms", 0):.0f} ms; away '
                          f'{(turn_began - prev.get("ended", turn_began)) * 1000:.0f} ms; '
                          f'this frame {timing_now["frame_age_ms"]:.0f} ms old')
            self.status = (f'{self.model} playing - {me["elixir_raw"]/10000:.1f} elixir - '
                           f'{self.plays} plays - opponent deck {len(runner.opponent_seen)}/8 '
                           f'seen ({runner.opponent_source})')

            # A turn can carry two plays. Take them in the order the policy asked for; the
            # second is a real play (Hog + Ice Spirit is one decision), not a duplicate.
            # An ability stays "in flight" until its button leaves Ready (the command, like a
            # card, takes ~21 ticks to execute) or 2.5 s pass; until then it is not re-offered.
            live_buttons = {int(a.get('controller_slot', 0)): int(a.get('button', 0))
                            for a in me.get('abilities') or ()}
            live_charges = {int(a.get('controller_slot', 0)): int(a.get('charges', -1))
                            for a in me.get('abilities') or ()}
            for a in abilities_in_flight:
                if (time.time() - a['time'] >= 2.5 and live_buttons.get(a['slot']) in (2, 4)
                        and live_charges.get(a['slot']) == a['charges']):
                    self._ability_missed(a)
            abilities_in_flight[:] = [
                a for a in abilities_in_flight
                if time.time() - a['time'] < 2.5 and live_buttons.get(a['slot']) in (2, 4)]
            for move in moves:
                if str(getattr(move[0], 'value', move[0])) == 'activate_ability':
                    self._try_ability(move, observation, me, abilities_in_flight, accounts,
                                      side, frame)
            for kind, _hand_slot, card_id, target_grid, _offset in moves:
                if str(getattr(kind, 'value', kind)) != 'play_card' or target_grid is None:
                    continue
                # FirstLight decodes to a NATIVE grid point [x, y] (perspective already
                # undone), so x is the column and y is the row. Reading it as (row, col)
                # transposed every placement the model asked for.
                column, row = int(target_grid[0]), int(target_grid[1])
                if not (0 <= column < X_TILES and 0 <= row < Y_TILES):
                    continue
                move = {'card': int(card_id), 'row': row, 'column': column,
                        'since': decided,
                        'timing': {'decided': decided,
                                   'inference_ms': (decided - started) * 1000.0,
                                   'frame_age_ms': (started - frame_time) * 1000.0,
                                   'turn_wait_ms': turn_wait_ms}}
                if not self._try_play(move, me, deck, reserved, in_flight, accounts, side,
                                      frame):
                    deferred = [d for d in deferred if d['card'] != move['card']] + [move]
                reserved = sum(f['cost'] for f in in_flight)
            ended = time.time()
            self._prev_turn_timing = {**timing_now, 'after_ms': (ended - decided) * 1000.0,
                                      'ended': ended}
        runner.end_battle()
        self.status = 'off'


    def _run(self) -> None:
        if self.model in FLB.CHECKPOINTS:
            self._run_firstlight()
            return
        try:
            engine, adapter, calibrated = MA.build(self.model)
        except Exception as error:  # noqa: BLE001
            self.status = f'load failed: {error}'
            self.running = False
            return
        import re
        sizes = re.findall(r'(\d+)x(\d+)', adb_run(ADB, SERIAL, 'shell', 'wm size'))
        self.layout = ScreenLayout.from_size(*map(int, sizes[-1]))
        self.note(f'{self.model} loaded, threshold {adapter.threshold:.4f}'
                  f'{"" if calibrated else " (uncalibrated)"}; '
                  f'{"ARMED - will tap" if self.armed else "dry run - no taps"}')
        battle = None
        last_tick = -10 ** 9
        in_flight = None
        while self.running:
            with V.LOCK:
                frame, health = V.STATE['frame'], V.STATE['health']
                reader_error = V.STATE['error']
                queue = list(V.STATE['queue'])
                accounts = V.STATE['accounts']
                revealed = V.STATE['revealed']
            if not frame or not health or not frame.get('battle_active'):
                # A failed reader and an idle game are not the same thing: say which,
                # so a device whose reader never attached stops reading as 'no battle'.
                self.status = (f'{self.model}: reader not attached - {reader_error}'
                               if reader_error else f'{self.model}: waiting for a battle')
                time.sleep(0.3)
                continue
            side = health.get('local_side')
            if side not in (0, 1) or not health.get('can_control'):
                self.status = f'{self.model}: {health.get("status")}'
                time.sleep(0.2)
                continue
            if frame['chain']['battle'] != battle:
                battle = frame['chain']['battle']
                adapter.events, adapter._prev_units = [], {}
                last_tick, in_flight = -10 ** 9, None
                self.plays = 0
                self.tracker, self.opp_deck, self.seen_plays = None, None, 0
                self.deduced = ''
                self.note(f'new battle, you are side {side}')
            local_account_early = next((a['lo'] for a in (accounts or [])
                                        if a and a.get('side') == side), None)
            # Every command either side has sent is an event the policy reads.
            for entry in queue:
                key = (entry['account_lo'], entry['seq'])
                if entry['card_id'] > 0 and key not in getattr(self, '_seen', set()):
                    self._seen = getattr(self, '_seen', set())
                    self._seen.add(key)
                    who = side if (entry['account_lo'] == local_account_early) else 1 - side
                    adapter.note_play(entry['issue_tick'] / 20.0, who, entry['card_id'],
                                      entry['x'] / 1000.0, entry['y'] / 1000.0)
            local_account = next((a['lo'] for a in (accounts or [])
                                  if a and a.get('side') == side), None)
            def is_ours(entry):
                return (entry['account_lo'] == local_account if local_account is not None
                        else entry['account_lo'] >= 0)
            # Our own card still shows in hand until it resolves (~0.85 s behind); acting now
            # would tap the wrong card.
            if any(is_ours(e) and e['card_id'] > 0 for e in queue):
                time.sleep(0.05)
                continue
            me = next((p for p in frame['players'] if p['side'] == side), None)
            if not me:
                time.sleep(0.1)
                continue
            hand = me['hand_deck_indices']
            if len(hand) != 4 or any(h < 0 for h in hand) or me['next_deck_index'] < 0:
                time.sleep(0.05)
                continue
            if in_flight:
                slot, since = in_flight
                if slot not in hand or time.time() - since > 2.5:
                    in_flight = None
                else:
                    time.sleep(0.05)
                    continue
            if frame['game_tick'] - last_tick < engine.STEP_TICKS:
                time.sleep(0.03)
                continue
            last_tick = frame['game_tick']
            # Deduce the opponent's hand from their deck + the cards they have played.
            # Their deck is only readable in friendlies; without it there is no deduction and
            # the model gets unknown tokens, exactly as before.
            deduced = None
            opp_revealed = (revealed or [[], []])[1 - side]
            if self.tracker is None and opp_revealed:
                deck = opponent_deck(str(frame['pid']), opp_revealed)
                if deck:
                    self.opp_deck = deck
                    self.tracker = Tracker(deck)
                    self.seen_plays = 0
                    self.note('opponent deck: ' +
                              ', '.join(V.CARDS.get(c, {}).get('name', str(c)) for c in deck))
            if self.tracker is not None:
                for card in opp_revealed[self.seen_plays:]:
                    self.tracker.observe(card)
                    self.seen_plays += 1
                hand, certain = self.tracker.hand()
                if certain:
                    slots = sorted(next(iter({tuple(sorted(o[:4]))
                                              for o in self.tracker.states})))
                    deduced = {1 - side: (list(slots), self.tracker.states[0][4])}
                self.deduced = ('certain: ' if certain else 'likely: ') + ', '.join(map(str, hand))
            state = MA.to_state(frame, side, deduced=deduced)
            MA.decks(adapter, engine, state, frame, side, opponent_deck=self.opp_deck)
            try:
                move = adapter.decide(state, side)
            except Exception as error:  # noqa: BLE001
                self.note(f'decide failed: {error}')
                time.sleep(0.2)
                continue
            self.status = (f'{self.model} playing · {me["elixir_raw"]/10000:.1f} elixir · '
                           f'{self.plays} plays')
            if move is None:
                time.sleep(0.05)
                continue
            deck_slot, card_id, ex, ey, urge = move
            name = V.CARDS.get(card_id, {}).get('name', str(card_id))
            position = self._hand_position(hand, deck_slot)
            if position is None:
                time.sleep(0.05)
                continue
            cell = int(ey) * X_TILES + int(ex)
            if not 0 <= cell < 576:
                continue
            # Scope gate: re-checked every decision, not just at start.
            allowed, reason = scope_gate.check(accounts, side)
            if reason != self.gate:
                self.gate, self.gate_ok = reason, allowed
                self.note(('scope: ' if allowed else 'SCOPE BLOCK: ') + reason)
            if not self.armed or not allowed:
                why = '' if self.armed else ' [dry run]'
                if self.armed and not allowed:
                    why = ' [BLOCKED by scope gate]'
                self.note(f't={frame["game_tick"]/20:5.1f}s  would play {name} '
                          f'at ({ex:.1f},{ey:.1f}) urge {urge:.2f}{why}')
                time.sleep(0.2)
                continue
            try:
                send_card_taps(ADB, SERIAL, self.layout, position, cell, side=side)
            except Exception as error:  # noqa: BLE001
                self.note(f'tap failed: {error}')
                time.sleep(0.2)
                continue
            in_flight = (deck_slot, time.time())
            self.plays += 1
            self.last_play = f'{name} at ({ex:.1f}, {ey:.1f})'
            self.note(f't={frame["game_tick"]/20:5.1f}s  {name:<14} tile ({ex:4.1f},{ey:4.1f})  '
                      f'urge {urge:.2f}')
            time.sleep(0.15)
        self.status = 'off'


BOT = Bot()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload: bytes, kind: str) -> None:
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        route = urlparse(self.path)
        if route.path == '/api/bot':
            query = parse_qs(route.query)
            action = (query.get('action') or [''])[0]
            if action == 'start':
                result = BOT.start((query.get('model') or ['100k'])[0],
                                   (query.get('armed') or ['0'])[0] == '1')
            elif action == 'stop':
                result = BOT.stop()
            else:
                result = 'unknown action'
            self._send(json.dumps({'result': result}).encode(), 'application/json')
            return
        if route.path == '/api/decoding':
            value = (parse_qs(route.query).get('value') or [''])[0]
            if value in ('auto', 'sampled', 'greedy'):
                BOT.decoding = value
                BOT.note(f'decoding preference -> {value} (applies from the next battle)')
                result = 'ok'
            else:
                result = f'unknown decoding {value}'
            self._send(json.dumps({'result': result}).encode(), 'application/json')
            return
        if route.path == '/api/mode':
            query = parse_qs(route.query)
            result = BOT.set_mode((query.get('mode') or [''])[0],
                                  (query.get('model') or [None])[0])
            self._send(json.dumps({'result': result}).encode(), 'application/json')
            return
        if route.path == '/state':
            with V.LOCK:
                frame, health = V.STATE['frame'], V.STATE['health']
                reader_error = V.STATE['error']
                age = time.time() - V.STATE['updated'] if V.STATE['updated'] else None
                queue, gap = list(V.STATE['queue']), V.STATE['gap']
                accounts = V.STATE['accounts']
            bot = {'running': BOT.running, 'armed': BOT.armed, 'model': BOT.model,
                   'status': BOT.status, 'plays': BOT.plays, 'last_play': BOT.last_play,
                   'log': BOT.log[-int((parse_qs(route.query).get('log') or ['14'])[0]):],
                   'models': list(MA.MODELS) + FLB.available(),
                   'mode': ('off' if not BOT.running else 'play' if BOT.armed else 'watch'),
                   'serial': SERIAL, 'port': PORT,
                   'decoding': BOT.decoding, 'decoding_used': BOT.decoding_used,
                   'gate': BOT.gate, 'gate_ok': BOT.gate_ok, 'deduced': BOT.deduced,
                   'reader_error': reader_error}
            if frame and health and frame.get('battle_active'):
                pending = [{'x': e['x'], 'y': e['y'], 'card_id': e['card_id'],
                            'name': V.CARDS.get(e['card_id'], {}).get('name', str(e['card_id'])),
                            'issue_tick': e['issue_tick'], 'seq': e['seq']}
                           for e in queue if V.mine(e)]
                revealed = {}
                for player in frame['players']:
                    revealed[player['side']] = []
                body = {'ok': True, 'age': age, 'pending': pending, 'lag_ticks': gap,
                        'pending_commands': pending_commands(queue, accounts,
                                                             frame.get('game_tick') or 0),
                        'session': V.SESSION.name, 'bot': bot, 'revealed': revealed,
                        **V.to_state(frame, health)}
            else:
                body = {'ok': False, 'age': age, 'bot': bot,
                        'status': (health or {}).get('status', 'no_battle')}
            self._send(json.dumps(body).encode(), 'application/json')
            return
        self._send(PAGE.encode(), 'text/html; charset=utf-8')


PAGE = V.PAGE.replace('</div>\n</div>\n<script>', """</div>
  <div class="panel" style="min-width:300px"><h2>bot</h2>
    <div id="botPick"></div>
    <label style="display:flex;gap:8px;align-items:center;margin:10px 0">
      <input type="checkbox" id="armed"><span>arm taps (Training Camp / own account only)</span>
    </label>

    <div style="display:flex;gap:8px">
      <button id="botStart" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--line);
        background:#2b3446;color:var(--ink);cursor:pointer">start</button>
      <button id="botStop" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--line);
        background:#2b3446;color:var(--ink);cursor:pointer">stop</button>
    </div>
    <div class="row" style="margin-top:10px"><span>status</span><span id="botStatus">off</span></div>
    <div class="row"><span>last play</span><span id="botLast">—</span></div>
    <div class="row"><span>scope</span><span id="botGate">unchecked</span></div>
    <div class="row"><span>their hand</span><span id="botDeduced">—</span></div>
    <pre id="botLog" style="margin-top:10px;max-height:220px;overflow:auto;font-size:11px;
      background:#151922;border:1px solid var(--line);border-radius:8px;padding:8px;
      white-space:pre-wrap"></pre>
  </div>
</div>
<script>
let botModel = '100k';
function renderBot(b) {
  const pick = document.getElementById('botPick');
  if (!pick.dataset.done && b.models) {
    pick.innerHTML = b.models.map(m =>
      `<label style="display:inline-flex;gap:6px;margin-right:12px;align-items:center">
         <input type="radio" name="bm" value="${m}" ${m===botModel?'checked':''}><span>${m}</span></label>`).join('');
    pick.dataset.done = '1';
    pick.querySelectorAll('input').forEach(i => i.onchange = () => botModel = i.value);
  }
  document.getElementById('botStatus').textContent = b.status + (b.armed ? ' · armed' : ' · dry run');
  document.getElementById('botLast').textContent = b.last_play || '—';
  document.getElementById('botLog').textContent = (b.log || []).join('\\n');
}
document.getElementById('botStart').onclick = () =>
  fetch(`/api/bot?action=start&model=${botModel}&armed=${document.getElementById('armed').checked?1:0}`);
document.getElementById('botStop').onclick = () => fetch('/api/bot?action=stop');
""").replace("setInterval(tick, 150); tick();",
             "setInterval(tick, 150); tick();").replace(
    "  draw(s);\n  const body = document.getElementById('statusBody');",
    "  draw(s);\n  if (s.bot) renderBot(s.bot);\n  const body = document.getElementById('statusBody');").replace(
    "    body.innerHTML = `<span class=\"bad\">${s.status || 'waiting'}</span>`",
    "    if (s.bot) renderBot(s.bot);\n    body.innerHTML = `<span class=\"bad\">${s.status || 'waiting'}</span>`")


def main() -> int:
    threading.Thread(target=V.pump, daemon=True).start()
    threading.Thread(target=V.pump_queue, daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    print(f'console on http://127.0.0.1:{PORT}  (bot taps require arming; ctrl-c to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        BOT.stop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
