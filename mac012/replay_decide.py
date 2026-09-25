"""Replay a recorded match through the live decision path (build -> decide), turn by turn.

Checks what the offline tests cannot: real frames, including the moments the live loop finds
awkward (a card being drawn, towers falling, the match ending). Reports decide errors, turns
where our hand had an empty slot, and what the policy chose on those turns.

    python3 mac012/replay_decide.py [SESSION_DIR] [MODEL]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac012 import firstlight_obs as FLO  # noqa: E402
from mac012 import firstlight_bot as FLB  # noqa: E402
import viewer as V  # noqa: E402

CLAPHA = Path(__file__).resolve().parents[1]
args = [a for a in sys.argv[1:]]
directory = Path(args[0]) if args else max(
    (d for d in (CLAPHA / 'artifacts/viewer-sessions').iterdir()
     if (d / 'frames.jsonl').is_file() and (d / 'frames.jsonl').stat().st_size), key=lambda d: d.name)
model = args[1] if len(args) > 1 else 'fl:hog2'
rows = [json.loads(line) for line in (directory / 'frames.jsonl').open()]
# A session can open on the previous match's result screen: take the battle with most frames.
counts = {}
for r in rows:
    if r['frame'].get('battle_active'):
        b = r['frame'].get('chain', {}).get('battle')
        counts[b] = counts.get(b, 0) + 1
battle_id = max(counts, key=counts.get)
first = next(r['frame'] for r in rows if r['frame'].get('chain', {}).get('battle') == battle_id)
side = next(r['health']['local_side'] for r in rows
            if r['frame'].get('chain', {}).get('battle') == battle_id
            and r['health'].get('local_side') in (0, 1))
frames = [(r['frame'], r['health']) for r in rows if r['frame'].get('chain', {}).get('battle') == battle_id]
queue = [json.loads(l) for l in (directory / 'queue.jsonl').open() if l.strip()]
queue = [q for q in queue if q.get('battle') == battle_id] or queue
plays, pending = [], {}
for row in queue:
    plays += V.executed_plays(row, row.get('queue', {}).get('entries', []), pending)

runner = FLB.FirstLightRunner(model)
me0 = next(p for p in first['players'] if p['side'] == side)
deck = me0['deck_card_ids']
battle = None
started = False
last = -10 ** 9
turns = errors = gap_turns = gap_plays = 0
messages = []
for frame, health in frames:
    if not frame.get('battle_active') or not health.get('can_control'):
        continue
    me = next(p for p in frame['players'] if p['side'] == side)
    if not any(i >= 0 for i in me['hand_deck_indices']):
        continue
    turn = frame['game_tick'] - frame['game_tick'] % 5
    if turn <= last:
        continue
    last = turn
    gap = any(i < 0 for i in me['hand_deck_indices'])
    try:
        runner.register_plays([p for p in plays if p['tick'] <= frame['game_tick']], {})
        obs, battle = FLO.build(frame, {**health, 'local_side': side}, '1', battle=battle,
                                plays=plays, decks=runner.tracked_decks(),
                                hand_forms=runner.hand_forms(deck, me.get('deck_form_flags'),
                                                             me.get('evo_progress')))
        if not started:
            runner.start_battle(deck, deck, side, obs,
                                {p['side']: p['elixir_raw'] / 10000.0 for p in frame['players']},
                                our_forms=me.get('deck_form_flags'))
            started = True
        if turn < runner.first_decision_tick:
            runner.observe(obs)
            continue
        moves = runner.decide(obs)
        turns += 1
        if gap:
            gap_turns += 1
            for move in moves:
                if str(getattr(move[0], 'value', move[0])) == 'play_card':
                    gap_plays += 1
                    slot = move[1]
                    if me['hand_deck_indices'][slot] < 0:
                        messages.append(f'tick {turn}: chose the EMPTY slot {slot}')
    except Exception as error:  # noqa: BLE001
        errors += 1
        messages.append(f'tick {turn}{" (draw gap)" if gap else ""}: '
                        f'{type(error).__name__}: {str(error)[:140]}')
print(f'{directory.name} {model} side {side}: {turns} decisions, {errors} errors, '
      f'{gap_turns} turns during a card draw ({gap_plays} plays chosen in them)')
for line in messages[:15]:
    print('  ', line)
raise SystemExit(1 if errors or any('EMPTY' in m for m in messages) else 0)
