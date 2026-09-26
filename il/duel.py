"""Model against model in Null's engine, both sides fed by the live code, with a real command delay.

Each side is a FirstLightRunner (a FirstLight checkpoint or one of ours) given exactly what the
console gives it: the engine snapshot as the reader's frame (il/samples.reader_frame: its own hand
and elixir as the screen shows them, its sent commands taken out), firstlight_obs.build, the
tracker with the opponent's deck adopted as from the API. What differs between the sides is only
when their plays execute:
  delay 'live'  the console's timing: the tap goes out `overhead` ticks after the decision (drawn
                per play from the measured live distribution, il/params.OWN_OVERHEAD_TICKS) plus
                the play's offset in its window, waits for elixir at the tap as the console does,
                and executes 21 ticks after the tap (the game's command age)
  delay 'none'  FirstLight's sandbox: the play executes on the next tick after the decision
Decks come from converted Hog 2.6 replays (il/frames.py): the Hog side's deck for both sides
(a mirror), sides swapped every match.

    ./py -m il.duel --a fl:hog2 --a-delay live --b fl:hog2 --b-delay none --matches 20
Needs the engine free (CR_4k with the run-lean probe); the conversion uses the same engine.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
if str(CLAPHA) not in sys.path:
    sys.path.insert(0, str(CLAPHA))

import il.samples as S                       # noqa: E402  (live code on sys.path first)
from il.params import COMMAND_AGE_TICKS, DECISION_TICKS, OWN_OVERHEAD_TICKS  # noqa: E402

DEFER_TICKS = 30        # a tap that cannot be afforded is dropped after this long (console: 1.5 s)


@dataclass
class Command:
    """One decided play on its way: tapped at `tap`, executes at `execute` (the engine step after)."""
    side: int
    card_id: int
    grid: tuple[int, int]
    decided: int
    tap: int
    execute: int | None = None
    kind: str = 'card'
    seq: int = 0
    injected: bool = False


def _overhead(rng: random.Random) -> int:
    values, weights = zip(*sorted(OWN_OVERHEAD_TICKS.items()))
    return rng.choices(values, weights)[0]


def play_match(native, runners, delays, leads, config, deck_forms, rng, match_id: str) -> dict:
    """One battle to the end; returns the result and per-side counters.

    The engine is stepped to the next thing that happens: a decision turn (every 5 ticks), a tap
    coming due (checked for elixir at that tick; one that cannot be afforded is re-checked every
    tick for up to DEFER_TICKS, as the console defers it), or a command to inject the tick before
    it executes (as il/engine_convert queues recorded plays)."""
    import firstlight_obs as FLO
    from il.engine_convert import run_lean
    from native_runner.arena import cell_to_world
    from native_runner.cr_native_env import HandAction

    native.create_match(config)
    battles = {0: None, 1: None}
    commands: list[Command] = []
    executed_rows: list[dict] = []
    counters = {side: {'decided': 0, 'tapped': 0, 'executed': 0, 'dropped_elixir': 0, 'dropped_hand': 0,
                       'abilities_ignored': 0} for side in (0, 1)}
    tick, ended, seq = 0, False, 0

    def screen_elixir(side: int, logic_raw: int) -> float:
        in_flight = [c for c in commands if c.side == side and c.execute is not None]
        return logic_raw / 10000.0 - sum(S._card_cost(c.card_id) or 0 for c in in_flight)

    def process_taps(state_players) -> None:
        for command in sorted((c for c in commands if c.execute is None and c.tap <= tick), key=lambda c: c.seq):
            logic = {p['owner']: p['elixirRaw'] for p in state_players}[command.side]
            cost = S._card_cost(command.card_id) or 0
            if screen_elixir(command.side, logic) >= cost - 1e-6:
                command.execute = tick + COMMAND_AGE_TICKS
                counters[command.side]['tapped'] += 1
            elif tick - command.tap > DEFER_TICKS:
                counters[command.side]['dropped_elixir'] += 1
                commands.remove(command)

    def inject_due() -> None:
        for command in [c for c in commands if c.execute is not None and not c.injected and c.execute - 1 == tick]:
            state = next(p for p in native.observe()['players'] if p['owner'] == command.side)
            slot = next((h['handIndex'] for h in state['hand'] if h['cardId'] == command.card_id), None)
            if slot is None:
                counters[command.side]['dropped_hand'] += 1
                commands.remove(command)
                continue
            x, y = cell_to_world(command.grid)
            native.queue_hand_action_at(HandAction(command.side, slot, x, y), execute_tick=command.execute)
            command.injected = True

    while True:
        waiting = any(c.execute is None and c.tap <= tick for c in commands)
        candidates = [tick + DECISION_TICKS - tick % DECISION_TICKS]
        candidates += [c.tap for c in commands if c.execute is None and c.tap > tick]
        candidates += [c.execute - 1 for c in commands if c.execute is not None and not c.injected
                       and c.execute - 1 > tick]
        if waiting:
            candidates.append(tick + 1)
        frames, tick, ended = run_lean(native, min(candidates))
        if ended:
            break
        decision = tick % DECISION_TICKS == 0 and frames
        players = frames[-1]['state']['players'] if decision else native.observe()['players']
        process_taps(players if decision else [{'owner': p['owner'], 'elixirRaw': p['elixirRaw']} for p in players])
        inject_due()
        if not decision:
            continue
        frame = frames[-1]
        # executed: out of the hand in the snapshot at their execute tick; dated like the viewer's
        for command in [c for c in commands if c.execute is not None and c.execute <= tick]:
            x, y = cell_to_world(command.grid)
            executed_rows.append({'tick': command.execute, 'side': command.side, 'card_id': command.card_id,
                                  'form_code': 0, 'kind': 'card', 'x': x, 'y': y, 'seq': command.seq,
                                  'issue_tick': command.execute - COMMAND_AGE_TICKS})
            counters[command.side]['executed'] += 1
            commands.remove(command)
        revealed = {0: [], 1: []}
        for row in executed_rows:
            if row['card_id'] not in revealed[row['side']]:
                revealed[row['side']].append(row['card_id'])
        for side in (0, 1):
            runner = runners[side]
            # every play this side has decided and that has not executed is gone from its screen
            # hand, tapped or not: the console taps within the turn, and training counts a play as
            # sent from the turn after its decision (il/samples in_flight)
            sent = [_Sent(c.card_id) for c in sorted(commands, key=lambda c: c.seq) if c.side == side]
            try:
                raw = S.reader_frame(frame, side, deck_forms, sent)
            except ValueError:
                state = {p['owner']: p for p in frame['state']['players']}[side]
                print('DEBUG tick', tick, 'side', side, 'hand', [(h['handIndex'], h['deckSlot'], h['cardId']) for h in state['hand']],
                      'cycle', [c['cardId'] for c in state['cycle']],
                      'commands', [(c.side, c.card_id, c.decided, c.tap, c.execute, c.injected) for c in commands],
                      'executed', [(r['side'], r['card_id'], r['tick']) for r in executed_rows[-6:]], flush=True)
                raise
            health = {'local_side': side, 'visible_sides': [side]}
            if battles[side] is None:
                observation, battles[side] = FLO.build(raw, health, episode_id=f'{match_id}:{side}')
                me = raw['players'][side]
                runner.start_battle(me['deck_card_ids'], None, side, observation,
                                    {p['side']: p['elixir_raw'] / 10000.0 for p in raw['players']},
                                    our_forms=me['deck_form_flags'])
                opponent = {p['owner']: p for p in frame['state']['players']}[1 - side]
                runner.adopt_api_deck([d['cardId'] for d in sorted(opponent['deck'], key=lambda d: d['deckSlot'])])
            runner.register_plays(executed_rows, revealed)
            seen = {side: revealed[side], 1 - side: [c for c in revealed[1 - side] if c in runner.opponent_seen]}
            me = raw['players'][side]
            observation, battles[side] = FLO.build(
                raw, health, episode_id=f'{match_id}:{side}', battle=battles[side], revealed=seen, reserved=0.0,
                plays=executed_rows, decks=runner.tracked_decks(),
                hand_forms=runner.hand_forms(me['deck_card_ids'], me['deck_form_flags'], me['evo_progress']),
                elixir_lead_ticks=leads[side])
            if tick < runner.first_decision_tick:
                runner.observe(observation)
                continue
            for kind, _slot, card_id, target_grid, offset in runner.decide(observation):
                if str(getattr(kind, 'value', kind)) != 'play_card' or target_grid is None:
                    counters[side]['abilities_ignored'] += int(str(getattr(kind, 'value', kind)) == 'activate_ability')
                    continue
                seq += 1
                counters[side]['decided'] += 1
                command = Command(side=side, card_id=int(card_id), grid=(int(target_grid[0]), int(target_grid[1])),
                                  decided=tick, tap=tick + int(offset) + (_overhead(rng) if delays[side] == 'live' else 0),
                                  seq=seq)
                if delays[side] != 'live':
                    # FirstLight's sandbox: execute_in_ticks = max(1, offset), no tap and no wait
                    command.execute = tick + max(1, int(offset))
                    counters[side]['tapped'] += 1
                commands.append(command)
        # a tap due right now (no-delay side, offset 0) goes out this tick
        process_taps(frame['state']['players'])
        inject_due()
    final = native.observe()
    for runner in runners.values():
        runner.end_battle()
    return {'winner': final.get('winner'), 'crowns': final.get('crownsRaw'), 'tick': final.get('tick'),
            'counters': counters}


@dataclass
class _Sent:
    card_id: int
    kind: str = 'card'


def hog26_matchups(frames_dir: Path, count: int, seed: int):
    """(match config, deck forms) for Hog 2.6 mirrors, from converted replays' Hog decks."""
    from il.frames import load_replay
    from native_runner.royaleapi_replay import _episode_match_config
    from dataclasses import replace
    rng = random.Random(seed)
    paths = sorted((frames_dir / 'frames').glob('*/*.jsonl.zst'))
    rng.shuffle(paths)
    out = []
    for path in paths:
        header, _frames = load_replay(path)
        timeline = header['timeline']
        sides = [i for i, deck in enumerate(timeline['decks']) if S.HOG26_CARDS <= set(deck)]
        if not sides:
            continue
        hog = sides[0]
        episode = header['calibrated'].replay.episode_config
        tags = dict(episode.tags)
        deck = tuple(timeline['decks'][hog])
        forms = tuple(timeline['form_availability'][hog])
        tags['deck0_form_availability'] = tags['deck1_form_availability'] = forms
        mirrored = replace(episode, deck0=deck, deck1=deck, tags=tags)
        out.append((_episode_match_config(mirrored), {0: list(forms), 1: list(forms)}, header['replay_tag']))
        if len(out) >= count:
            break
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', default='fl:hog2')
    parser.add_argument('--a-delay', default='live', choices=('live', 'none'))
    parser.add_argument('--b', default='fl:hog2')
    parser.add_argument('--b-delay', default='none', choices=('live', 'none'))
    parser.add_argument('--a-lead', type=int, default=0, help="elixir_lead_ticks for a's mask (0 = live today)")
    parser.add_argument('--b-lead', type=int, default=0)
    parser.add_argument('--matches', type=int, default=20)
    parser.add_argument('--frames', type=Path, default=CLAPHA / 'runs/conv-hog26')
    parser.add_argument('--out', type=Path, default=CLAPHA / 'runs/duels.jsonl')
    parser.add_argument('--port', type=int, default=26789)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args(argv)
    import firstlight_bot as FLB
    from il.engine_convert import connect
    native = connect(args.port)
    native.wait_ready(timeout=60.0)
    models = {'a': FLB.FirstLightRunner(args.a), 'b': FLB.FirstLightRunner(args.b) if args.b != args.a
              else FLB.FirstLightRunner(args.b)}
    rng = random.Random(args.seed)
    wins = {'a': 0, 'b': 0, 'draw': 0}
    for index, (config, forms, tag) in enumerate(hog26_matchups(args.frames, args.matches, args.seed)):
        a_side = index % 2          # sides swapped every match
        runners = {a_side: models['a'], 1 - a_side: models['b']}
        delays = {a_side: args.a_delay, 1 - a_side: args.b_delay}
        leads = {a_side: args.a_lead, 1 - a_side: args.b_lead}
        started = time.time()
        try:
            result = play_match(native, runners, delays, leads, config, forms, rng, f'{tag[:8]}-{index}')
        except Exception as error:  # noqa: BLE001  (one broken match must not end the set)
            print(f'match {index + 1} failed: {type(error).__name__}: {error}', flush=True)
            for runner in runners.values():
                try:
                    runner.end_battle()
                except Exception:  # noqa: BLE001
                    pass
            continue
        winner = result['winner']
        label = 'draw' if winner not in (0, 1) else ('a' if winner == a_side else 'b')
        wins[label] += 1
        row = {'a': args.a, 'a_delay': args.a_delay, 'b': args.b, 'b_delay': args.b_delay, 'deck_from': tag,
               'a_side': a_side, 'result': label, 'crowns': result['crowns'], 'end_tick': result['tick'],
               'counters': result['counters'], 'seconds': round(time.time() - started)}
        with args.out.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(f"match {index + 1}: {label} (crowns {result['crowns']}, a on side {a_side}); "
              f"running score a {wins['a']} - b {wins['b']} ({wins['draw']} draws); {row['seconds']} s", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
