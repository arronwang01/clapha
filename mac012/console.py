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
from cycle_tracker import Tracker, opponent_deck  # noqa: E402
from mac_profile import ADB, SERIAL  # type: ignore  # noqa: E402
from native_core.mumu_live_actions import ScreenLayout, send_card_taps  # noqa: E402
from native_core.mumu_live_protocol import adb_run  # noqa: E402

PORT = int(os.environ.get('CR_CONSOLE_PORT', '8777'))
LOG_FILE = Path(__file__).resolve().parents[1] / 'build' / f'bot_{PORT}.log'
X_TILES, Y_TILES = 18, 32


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
        self._last_line, self._repeats, self._first_stamp = None, 0, ''
        self.log: list[str] = []
        self.thread = None
        self.layout = None
        self.gate = 'unchecked'
        self.gate_ok = False
        self.tracker = None
        self.opp_deck = None
        self.seen_plays = 0
        self.deduced = ''

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
        return 'stopped'

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
        self.note(f'{self.model} loaded (FirstLight V4, FAIR tier, '
                  f'{20.0/runner.decision_ticks:.0f} Hz); '
                  f'{"ARMED - will tap" if self.armed else "dry run - no taps"}')
        battle = None
        fl_battle = None
        reported: set[int] = set()
        reported_plays: set[tuple[int, int]] = set()
        pending_battle, pending_since = None, 0.0
        episode_decks: dict = {}
        last_turn = -10 ** 9
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
            me = next((p for p in frame['players'] if p['side'] == side), None)
            if not me or len(me['hand_deck_indices']) != 4 or any(h < 0 for h in me['hand_deck_indices']):
                time.sleep(0.05)
                continue
            our_deck = me.get('deck_card_ids') or []
            if len(our_deck) != 8:
                time.sleep(0.2)
                continue

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
                if opponent is None and frame['game_tick'] >= 60:
                    # nothing fresh: an earlier publication beats a stand-in
                    opponent, opponent_forms, deck_note = opponent_deck_file(
                        opponent_account, fresh_after=0)
                if opponent is None and frame['game_tick'] < 60:
                    self.status = f'{self.model}: waiting for the opponent deck'
                    time.sleep(0.05)
                    continue
                opponent_known = opponent is not None
                if not opponent_known:
                    opponent, opponent_forms = our_deck, None
                    deck_note = ('opponent deck unknown - stand-in used; opponent plays will not '
                                 'reach the tracker (run both consoles for friendlies)')
                self.note(deck_note)
                seen = {index: list(cards) for index, cards
                        in enumerate(revealed or [[], []])}
                observation, fl_battle = FLO.build(
                    frame, health, episode_id=str(frame['chain']['battle']),
                    revealed=seen)
                try:
                    runner.end_battle()
                    # The tracker is seeded with exact initial elixir for BOTH owners, which
                    # is what BattleEnv passes and what the tensorizer demands of a
                    # tracker-backed episode. Read, not assumed: a battle joined late does not
                    # start at five.
                    runner.start_battle(our_deck, opponent, side, observation,
                                        {p['side']: p['elixir_raw'] / 10000.0
                                         for p in frame['players']},
                                        our_forms=me.get('deck_form_flags'),
                                        opponent_forms=opponent_forms)
                except Exception as error:  # noqa: BLE001
                    self.note(f'episode start failed: {error}')
                    time.sleep(1.0)
                    continue
                episode_decks = {side: tuple(our_deck),
                                 1 - side: tuple(opponent) if opponent_known else ()}
                battle = frame['chain']['battle']
                last_turn, in_flight = -10 ** 9, None
                self.plays = 0
                self.note(f'new battle, you are side {side}; '
                          f'warm-up until tick {runner.first_decision_tick}')

            # One turn per five-tick window, and never a skipped one. decide() advances a
            # recurrent state and feeds its own chosen action into the next turn, so the
            # schedule belongs to the policy: our tap bookkeeping may suppress a tap, but it
            # must not suppress a turn. Ticks before their first decision tick are warm-up,
            # fed to the tensorizer without acting, as PolicyService's 'observe' op does.
            local_account = next((a['lo'] for a in (accounts or [])
                                  if a and a.get('side') == side), None)
            turn = runner.turn_tick(frame['game_tick'])
            if turn <= last_turn:
                time.sleep(0.02)
                continue
            skipped = (turn - last_turn) // runner.decision_ticks - 1 if last_turn > 0 else 0
            last_turn = turn

            try:
                seen = {index: list(cards) for index, cards
                        in enumerate(revealed or [[], []])}
                reserved = 0.0
                if in_flight and time.time() - in_flight[1] <= 2.5:
                    reserved = float(in_flight[2])
                with V.LOCK:
                    plays = list(V.STATE['plays'])
                observation, fl_battle = FLO.build(frame, health, episode_id=str(battle),
                                                   battle=fl_battle, revealed=seen,
                                                   reserved=reserved, plays=plays,
                                                   decks=episode_decks)
                for side_card in sorted(fl_battle.untracked_plays - reported_plays):
                    reported_plays.add(side_card)
                    who = 'opponent' if side_card[0] != side else 'own'
                    self.note(f'{who} play of {V.CARDS.get(side_card[1], {}).get("name", side_card[1])}'
                              f' not in the episode deck - not given to the tracker')
                for card_id in sorted(set(fl_battle.unresolved) - reported):
                    reported.add(card_id)
                    name = V.CARDS.get(card_id, {}).get('name', card_id)
                    self.note(f'{name} has no entity archetype - that entity is left out '
                              f'(spell effects; units are unaffected)')
                if turn < runner.first_decision_tick:
                    runner.observe(observation)
                    self.status = (f'{self.model}: warm-up '
                                   f'{turn}/{runner.first_decision_tick}')
                    continue
                moves = runner.decide(observation)
            except Exception as error:  # noqa: BLE001
                self.note(f'decide failed: {error}')
                self.status = f'{self.model}: DECIDE FAILING - {str(error)[:80]}'
                continue
            if skipped > 0:
                self.note(f'missed {skipped} decision turn(s) at tick {turn} - '
                          f'the policy state is behind the game')
            self.status = (f'{self.model} playing - {me["elixir_raw"]/10000:.1f} elixir - '
                           f'{self.plays} plays')

            # A turn can carry two plays. Take them in the order the policy asked for.
            for kind, hand_slot, card_id, target_grid, _offset in moves:
                if str(getattr(kind, 'value', kind)) != 'play_card' or target_grid is None:
                    continue
                name = V.CARDS.get(card_id, {}).get('name', str(card_id))
                # FirstLight decodes to a NATIVE grid point [x, y] (perspective already
                # undone), so x is the column and y is the row. Reading it as (row, col)
                # transposed every placement the model asked for.
                column, row = int(target_grid[0]), int(target_grid[1])
                cell = row * X_TILES + column
                position = hand_slot if isinstance(hand_slot, int) else None
                if position is None or not 0 <= position < 4 or not 0 <= cell < 576:
                    continue

                allowed, reason = scope_gate.check(accounts, side)
                if reason != self.gate:
                    self.gate, self.gate_ok = reason, allowed
                    self.note(('scope: ' if allowed else 'SCOPE BLOCK: ') + reason)
                if not self.armed or not allowed:
                    why = ' [dry run]' if not self.armed else ' [BLOCKED by scope gate]'
                    self.note(f't={frame["game_tick"]/20:5.1f}s  would play {name} '
                              f'slot {position} at row {row} col {column}{why}')
                    continue
                # The client needs ~20 ticks to accept a play. Suppress a duplicate tap for
                # the same slot while one is outstanding -- but the turn above was still
                # taken, so the policy's state stays aligned.
                if in_flight:
                    slot, since = in_flight[0], in_flight[1]
                    if slot in me['hand_deck_indices'] and time.time() - since <= 2.5:
                        continue
                    in_flight = None
                ours_pending = [e for e in queue if e['card_id'] > 0 and
                                (e['account_lo'] == local_account
                                 if local_account is not None else e['account_lo'] >= 0)]
                if ours_pending:
                    continue
                try:
                    send_card_taps(ADB, SERIAL, self.layout, position, cell, side=side)
                except Exception as error:  # noqa: BLE001
                    self.note(f'tap failed: {error}')
                    continue
                in_flight = (me['hand_deck_indices'][position], time.time(),
                             float(V.CARDS.get(card_id, {}).get('elixir') or 0))
                self.plays += 1
                self.last_play = f'{name} at row {row} col {column}'
                self.note(f't={frame["game_tick"]/20:5.1f}s  {name:<14} row {row:2} col {column:2}')
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
        if route.path == '/state':
            with V.LOCK:
                frame, health = V.STATE['frame'], V.STATE['health']
                reader_error = V.STATE['error']
                age = time.time() - V.STATE['updated'] if V.STATE['updated'] else None
                queue, gap = list(V.STATE['queue']), V.STATE['gap']
            bot = {'running': BOT.running, 'armed': BOT.armed, 'model': BOT.model,
                   'status': BOT.status, 'plays': BOT.plays, 'last_play': BOT.last_play,
                   'log': BOT.log[-14:], 'models': list(MA.MODELS) + FLB.available(),
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
