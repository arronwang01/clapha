"""Live board viewer: reads frames from the sampler and serves them to a browser.

Read-only. Sends no touches. Open http://127.0.0.1:8777 while a battle is running.

The local player is drawn at the bottom, matching the phone camera:
x is used as-is when the local player is side 1, mirrored when side 0 (per the
prior project's note that owner-0 games come out mirrored otherwise).
"""
from __future__ import annotations

import json
import sys
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_profile import ADB, CLAPHA, READER, SERIAL, apply  # type: ignore  # noqa: E402

apply()
from native_core.mumu_live_protocol import (BattleClockGuard, install_reader,  # noqa: E402
                                            start_reader, verify_runtime)

PORT = 8777
CATALOG = CLAPHA / 'live_card_catalog.json'
STATE: dict = {'frame': None, 'health': None, 'error': None, 'updated': 0.0,
               'queue': [], 'gap': None, 'queue_updated': 0.0, 'accounts': None, 'revealed': None,
               'plays': [], 'plays_battle': None}
DECKS = CLAPHA / 'build' / 'decks'
LOCK = threading.Lock()
SESSION = CLAPHA / 'artifacts' / 'viewer-sessions' / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())

# In Training Camp the trainer's commands carry account_lo -1 / seq -1, so a non-negative
# account is ours. In a friendly battle against another account this no longer holds --
# read the local side's avatar account id before relying on it there.
def mine(entry: dict) -> bool:
    return entry.get('account_lo', -1) >= 0


def catalog() -> dict[int, dict]:
    if not CATALOG.is_file():
        return {}
    data = json.loads(CATALOG.read_text())
    return {row['card_id']: {'name': row['display_name'], 'elixir': row['elixir'],
                             'type': row['type']} for row in data.get('cards', [])}


CARDS = catalog()


def to_state(frame: dict, health: dict) -> dict:
    """One frame in the shape the viewer draws (close to the prior project's to_state)."""
    players = []
    local_side = health.get('local_side')
    for player in frame.get('players', []):
        deck = player.get('deck_card_ids') or []
        def card(index):
            if not (isinstance(index, int) and 0 <= index < len(deck)):
                return None
            cid = deck[index]
            info = CARDS.get(cid, {})
            return {'card_id': cid, 'name': info.get('name', str(cid)),
                    'elixir': info.get('elixir'), 'type': info.get('type')}
        players.append({'side': player['side'], 'elixir': player['elixir_raw'] / 10000,
                        'revealed': [],  # filled by the snapshot poller
                        'hand': [card(i) for i in player.get('hand_deck_indices', [])],
                        'next': card(player.get('next_deck_index')),
                        'hand_indices': player.get('hand_deck_indices', [])})
    entities = []
    for entity in frame.get('entities', []):
        info = CARDS.get(entity['card_id'], {})
        tower = entity['card_id'] == -1
        entities.append({'x': entity['x'], 'y': entity['y'], 'side': entity['side'],
                         'hp': entity['hp'], 'max_hp': entity['max_hp'],
                         'card_id': entity['card_id'], 'level': entity['level'],
                         'name': 'King' if tower and entity['kind'] == 12 else
                                 ('Tower' if tower else info.get('name', str(entity['card_id']))),
                         'tower': tower})
    return {'tick': frame.get('game_tick'), 'players': players, 'entities': entities,
            'local_side': local_side, 'status': health.get('status'),
            'coherent': frame.get('coherent'), 'read_us': frame.get('read_us'),
            'entity_count': frame.get('decoded_entity_count')}


def pump() -> None:
    guard = BattleClockGuard()
    SESSION.mkdir(parents=True, exist_ok=True)
    log = (SESSION / 'frames.jsonl').open('a', encoding='utf-8')
    while True:
        try:
            runtime = verify_runtime(ADB, SERIAL)
            # start_reader executes /data/local/tmp/mumu-live-reader-v2, which is not one of
            # the names the launcher pushes. Without this a device that never had the reader
            # installed by hand runs nothing, emits no frames, and looks exactly like a device
            # that is merely between battles. install_reader is the repo's own installer: it
            # checks the SHA and refuses to overwrite a reader another observer is using.
            install_reader(ADB, SERIAL, READER)
            process = start_reader(ADB, SERIAL, runtime['pid'], interval_ms=100, max_frames=0)
            for line in process.stdout:
                if '"mumu_live_frame"' not in line:
                    continue
                frame = json.loads(line)
                health = guard.observe(frame, now=frame['sample_monotonic_us'] / 1_000_000)
                with LOCK:
                    STATE['frame'] = frame
                    STATE['health'] = health
                    STATE['error'] = None
                    STATE['updated'] = time.time()
                if frame.get('battle_active') and health.get('local_side') in (0, 1):
                    publish_deck(frame, health)
                if frame.get('battle_active'):
                    log.write(json.dumps({'frame': frame, 'health': health}) + '\n')
                    log.flush()
        except Exception as error:  # keep serving; the game may be in a menu or restarting
            with LOCK:
                STATE['error'] = f'{type(error).__name__}: {error}'
            # A TCP adb device drops out on its own (sleep, emulator restart) and nothing else
            # brings it back, so the console would sit on "no battle" until relaunched.
            if ':' in SERIAL:
                try:
                    subprocess.run([str(ADB), 'connect', SERIAL], capture_output=True,
                                   timeout=10, check=False)
                except Exception:  # noqa: BLE001  - reconnecting is best effort
                    pass
            time.sleep(2)


def executed_plays(row: dict, entries: list, pending: dict) -> list[dict]:
    """Card plays that executed since the previous queue sample, from both players.

    A command sits in the queue from its issue until the simulation consumes it (~21 ticks),
    and consumption IS the play executing -- FirstLight's env emits action_executed at that
    same moment. So an entry that was present and is now gone is one executed play: owner
    from its account (the queue probe maps accounts to sides; a trainer's commands carry
    account -1 and belong to whichever side is not a real account), card and position from
    the entry, tick = the simulation tick at which it was first seen missing.
    """
    tick = row.get('tick_0x60')
    if tick is None:
        return []
    sides = {a['lo']: a['side'] for a in (row.get('accounts') or []) if a}
    current = {(e.get('account_lo'), e.get('seq'), e.get('issue_tick'), e.get('card_id')): e
               for e in entries if e.get('card_id', 0) > 0}
    played = []
    for key, entry in list(pending.items()):
        if key in current:
            continue
        del pending[key]
        side = sides.get(entry.get('account_lo'))
        if side is None:
            # A trainer's commands carry no real account: they belong to the side that is not
            # the one real account present.
            real = [s for lo, s in sides.items() if lo is not None and lo > 0]
            side = 1 - real[0] if len(real) == 1 else None
        if side is None:
            continue
        card_id = int(entry['card_id'])
        # A champion's ability activation also passes through the command queue, with no
        # card behind it (id 65535). It is not a card play and must not move the card cycle.
        kind = 'card' if 25000000 <= card_id < 30000000 else 'ability'
        # The game consumes a command a fixed 21 ticks after issue (measured: queued -> unit
        # 22 ticks on every play; FirstLight's COMMAND_CONSUMPTION_STEPS = 21). Dating the play
        # from its issue tick is exact; dating it from when we noticed the entry gone depends
        # on sampling and can be late.
        issue = entry.get('issue_tick')
        executed = int(issue) + 21 if isinstance(issue, int) and issue + 21 <= int(tick) \
            else int(tick)
        played.append({'tick': executed, 'side': int(side), 'card_id': card_id, 'kind': kind,
                       'x': entry.get('x'), 'y': entry.get('y'), 'seq': entry.get('seq'),
                       'issue_tick': entry.get('issue_tick')})
    for key, entry in current.items():
        pending.setdefault(key, entry)
    return played


def publish_deck(frame: dict, health: dict) -> None:
    """Write this device's own deck under its account id, for the other console to read.

    Both accounts in these friendlies are the user's own. FirstLight's environment gives its
    public tracker both full decks from the start; the opponent's is not readable from this
    client mid-battle, but the other device reads it as ITS own deck, so the two consoles
    simply hand them over.
    """
    side = health.get('local_side')
    accounts = STATE.get('accounts') or []
    account = next((a['lo'] for a in accounts if a and a.get('side') == side), None)
    me = next((p for p in frame.get('players', []) if p.get('side') == side), None)
    if account is None or not me or len(me.get('deck_card_ids') or []) != 8:
        return
    DECKS.mkdir(parents=True, exist_ok=True)
    target = DECKS / f'{account}.json'
    body = {'account_lo': account, 'deck': me['deck_card_ids'],
            'forms': me.get('deck_form_flags'), 'battle': (frame.get('chain') or {}).get('battle'),
            'written': time.time()}
    try:
        previous = json.loads(target.read_text()) if target.is_file() else {}
    except (OSError, ValueError):
        previous = {}
    if previous.get('deck') != body['deck'] or previous.get('battle') != body['battle']:
        target.write_text(json.dumps(body))


def pump_queue() -> None:
    """Command queue: our own plays appear here ~0.6-1.0 s before the unit exists."""
    from mac_profile import MANAGER_RVA, ROOT_CONTEXT_OFFSET  # type: ignore
    SESSION.mkdir(parents=True, exist_ok=True)
    log = (SESSION / 'queue.jsonl').open('a', encoding='utf-8')
    while True:
        try:
            runtime = verify_runtime(ADB, SERIAL)
            command = (f'/data/local/tmp/queue_probe {runtime["pid"]} '
                       f'{hex(MANAGER_RVA)} {hex(ROOT_CONTEXT_OFFSET)} 0 100')
            process = subprocess.Popen([str(ADB), '-s', SERIAL, 'shell', command],
                                       stdout=subprocess.PIPE, text=True, bufsize=1)
            pending: dict = {}
            for line in process.stdout:
                if not line.startswith('{'):
                    continue
                row = json.loads(line)
                entries = row.get('queue', {}).get('entries', [])
                plays = executed_plays(row, entries, pending)
                with LOCK:
                    if STATE['plays_battle'] != row.get('battle'):
                        STATE['plays'], STATE['plays_battle'] = [], row.get('battle')
                    STATE['plays'].extend(plays)
                    STATE['queue'] = entries
                    STATE['accounts'] = row.get('accounts')
                    STATE['revealed'] = row.get('revealed')
                    STATE['gap'] = row.get('gap')
                    STATE['queue_updated'] = time.time()
                # Also keep the first empty sample after a non-empty one: that is where the
                # last queued command was consumed, i.e. when that play executed.
                if entries or plays:
                    log.write(line)
                    log.flush()
        except Exception:
            time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def do_GET(self):
        if self.path.startswith('/state'):
            with LOCK:
                frame, health, error = STATE['frame'], STATE['health'], STATE['error']
                age = time.time() - STATE['updated'] if STATE['updated'] else None
            with LOCK:
                queue, gap = STATE['queue'], STATE['gap']
            if frame and health and frame.get('battle_active'):
                pending = [{'x': e['x'], 'y': e['y'], 'card_id': e['card_id'],
                            'name': CARDS.get(e['card_id'], {}).get('name', str(e['card_id'])),
                            'issue_tick': e['issue_tick'], 'seq': e['seq']}
                           for e in queue if mine(e)]
                body = {'ok': True, 'age': age, 'pending': pending, 'lag_ticks': gap,
                        'session': SESSION.name, **to_state(frame, health)}
            else:
                body = {'ok': False, 'age': age, 'error': error,
                        'status': (health or {}).get('status', 'no_battle')}
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        payload = PAGE.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>CR live view</title>
<style>
 :root { --bg:#12151a; --panel:#1b1f27; --ink:#e7ecf3; --dim:#8b95a5;
         --me:#4ea3ff; --them:#ff5f56; --line:#2a303b; }
 body { margin:0; background:var(--bg); color:var(--ink);
        font:14px/1.45 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif; }
 .wrap { display:flex; gap:18px; padding:18px; align-items:flex-start; flex-wrap:wrap; }
 canvas { background:#243018; border-radius:10px; border:1px solid var(--line); }
 .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:14px 16px; min-width:260px; }
 h2 { font-size:13px; letter-spacing:.08em; text-transform:uppercase; color:var(--dim);
      margin:0 0 10px; font-weight:600; }
 .row { display:flex; justify-content:space-between; gap:12px; padding:3px 0; }
 .row span:last-child { color:var(--dim); font-variant-numeric:tabular-nums; }
 .hand { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:6px; }
 .card { background:#222836; border:1px solid var(--line); border-radius:8px;
         padding:8px 6px; text-align:center; font-size:12px; }
 .card b { display:block; font-size:12px; font-weight:600; }
 .cost { color:#c98bff; font-variant-numeric:tabular-nums; }
 .bar { height:8px; background:#2a303b; border-radius:5px; overflow:hidden; margin-top:6px; }
 .bar i { display:block; height:100%; background:linear-gradient(90deg,#b061ff,#e06bff); }
 .bad { color:#ffb454; }
 .mine { color:var(--me); } .theirs { color:var(--them); }
</style></head><body>
<div class="wrap">
  <canvas id="board" width="396" height="704"></canvas>
  <div>
    <div class="panel" id="status"><h2>state</h2><div id="statusBody">connecting…</div></div>
    <div class="panel" style="margin-top:14px"><h2>you</h2>
      <div id="meElixir"></div><div class="bar"><i id="meBar" style="width:0%"></i></div>
      <div class="hand" id="meHand"></div>
      <div class="row" style="margin-top:8px"><span>next</span><span id="meNext">—</span></div>
    </div>
    <div class="panel" style="margin-top:14px"><h2>trainer</h2>
      <div id="themElixir"></div><div class="bar"><i id="themBar" style="width:0%"></i></div>
      <div class="row" style="margin-top:6px"><span>hand</span><span>hidden (never sent)</span></div>
    </div>
  </div>
</div>
<script>
const NW = 18000, NH = 32000;
const cv = document.getElementById('board'), cx = cv.getContext('2d');

function place(e, localSide) {
  // canonical: local player at the bottom
  const cxn = localSide === 1 ? NW - e.x : e.x;
  const cyn = localSide === 1 ? NH - e.y : e.y;
  // camera keeps native x for side 1, mirrors it for side 0
  const fx = 1 - cxn / NW;   // camera mirrors canonical x for BOTH sides (measured)
  return { px: fx * cv.width, py: cv.height * (1 - cyn / NH) };
}

function drawArena() {
  cx.fillStyle = '#3f6b2a'; cx.fillRect(0, 0, cv.width, cv.height);
  cx.fillStyle = '#39602511';
  for (let r = 0; r < 32; r++) for (let c = 0; c < 18; c++)
    if ((r + c) % 2) { cx.fillStyle = 'rgba(255,255,255,.035)';
      cx.fillRect(c * cv.width / 18, r * cv.height / 32, cv.width / 18, cv.height / 32); }
  const ry = cv.height / 2 - cv.height / 32, rh = cv.height / 16;
  cx.fillStyle = '#2f7fb5'; cx.fillRect(0, ry, cv.width, rh);
  cx.fillStyle = '#8a6a3f';
  cx.fillRect(cv.width * .12, ry, cv.width * .1, rh);
  cx.fillRect(cv.width * .78, ry, cv.width * .1, rh);
}

function draw(s) {
  drawArena();
  if (!s.ok) return;
  const ls = s.local_side == null ? 1 : s.local_side;
  for (const p of (s.pending || [])) {     // our own plays, seen before they land
    const { px, py } = place(p, ls);
    cx.save(); cx.setLineDash([4, 3]); cx.lineWidth = 2; cx.strokeStyle = '#ffd166';
    cx.beginPath(); cx.arc(px, py, 13, 0, 7); cx.stroke(); cx.restore();
    cx.fillStyle = '#ffd166'; cx.font = '10px ui-sans-serif'; cx.textAlign = 'center';
    cx.fillText(p.name + ' →', px, py - 18);
  }
  for (const e of s.entities) {
    const { px, py } = place(e, ls);
    const mine = e.side === ls;
    const col = mine ? '#4ea3ff' : '#ff5f56';
    const r = e.tower ? 15 : 8;
    cx.beginPath(); cx.arc(px, py, r, 0, 7); cx.fillStyle = col + 'cc'; cx.fill();
    cx.lineWidth = 2; cx.strokeStyle = '#0006'; cx.stroke();
    if (e.max_hp > 0) {
      const w = e.tower ? 40 : 22, f = Math.max(0, Math.min(1, e.hp / e.max_hp));
      cx.fillStyle = '#0008'; cx.fillRect(px - w / 2, py - r - 9, w, 4);
      cx.fillStyle = mine ? '#6fe27a' : '#ff8d84';
      cx.fillRect(px - w / 2, py - r - 9, w * f, 4);
    }
    if (!e.tower) {
      cx.fillStyle = '#fff'; cx.font = '10px ui-sans-serif';
      cx.textAlign = 'center'; cx.fillText(e.name, px, py + r + 11);
    }
  }
}

function card(c) {
  if (!c) return '<div class="card"><b>—</b></div>';
  return `<div class="card"><b>${c.name}</b><span class="cost">${c.elixir ?? '?'}</span></div>`;
}

async function tick() {
  let s;
  try { s = await (await fetch('/state')).json(); }
  catch (e) { document.getElementById('statusBody').innerHTML =
    '<span class="bad">viewer offline</span>'; return; }
  draw(s);
  const body = document.getElementById('statusBody');
  if (!s.ok) {
    body.innerHTML = `<span class="bad">${s.status || 'waiting'}</span>` +
      (s.error ? `<div class="row"><span>error</span><span>${s.error}</span></div>` : '') +
      `<div class="row"><span>start a battle</span><span></span></div>`;
    return;
  }
  body.innerHTML =
    `<div class="row"><span>tick</span><span>${s.tick} (${(s.tick/20).toFixed(1)}s)</span></div>` +
    `<div class="row"><span>status</span><span>${s.status}</span></div>` +
    `<div class="row"><span>you are</span><span class="mine">side ${s.local_side}</span></div>` +
    `<div class="row"><span>objects</span><span>${s.entity_count}</span></div>` +
    `<div class="row"><span>read</span><span>${s.read_us} µs</span></div>` +
    `<div class="row"><span>frame age</span><span>${(s.age*1000).toFixed(0)} ms</span></div>` +
    `<div class="row"><span>client lag</span><span>${s.lag_ticks ?? '—'} ticks (${s.lag_ticks!=null?(s.lag_ticks*50):'—'} ms)</span></div>` +
    `<div class="row"><span>your pending</span><span>${(s.pending||[]).map(p=>p.name).join(', ') || 'none'}</span></div>` +
    `<div class="row"><span>recording</span><span>${s.session || ''}</span></div>`;
  const me = s.players.find(p => p.side === s.local_side);
  const them = s.players.find(p => p.side !== s.local_side);
  if (me) {
    document.getElementById('meElixir').innerHTML =
      `<div class="row"><span>elixir</span><span>${me.elixir.toFixed(2)}</span></div>`;
    document.getElementById('meBar').style.width = (me.elixir * 10) + '%';
    document.getElementById('meHand').innerHTML = me.hand.map(card).join('');
    document.getElementById('meNext').textContent = me.next ? me.next.name : '—';
  }
  if (them) {
    document.getElementById('themElixir').innerHTML =
      `<div class="row"><span>elixir</span><span>${them.elixir.toFixed(2)}</span></div>`;
    document.getElementById('themBar').style.width = (them.elixir * 10) + '%';
  }
}
setInterval(tick, 150); tick();
</script></body></html>
"""


def main() -> int:
    threading.Thread(target=pump, daemon=True).start()
    threading.Thread(target=pump_queue, daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    print(f'viewer on http://127.0.0.1:{PORT}  (read-only; ctrl-c to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
