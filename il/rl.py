"""Reinforcement learning under the real game's conditions: self-play collection (this file's
first half) and PPO (il/rl_learn.py).

Collection is il.duel's live path, so a training game is played exactly as a benchmark or the
console plays: every play lands TARGET_DELAY ticks after its moment (console.TARGET_DELAY), the
model sees the screen's hand and elixir, both sides' pending cards (the opponent's only once
visible, as the queue shows them), each pending card's arrival, exact opponent elixir, both
players' hero and champion controllers. FirstLight's own PPO collector plays its sandbox instead
(cards land 1-4 ticks after the decision), which is not the game.

Both sides are policies: the learner's current weights, or a league snapshot. Every decision of a
recorded side is kept -- the model input it saw, the action it sampled, that action's log-prob
and the value under the weights that chose it -- one lane per side per game, with our reward (the
user's, 2026-09-26, for 2.6 mirrors):
  terminal  win +1, loss -1, draw 0
  towers    + princess-tower damage dealt, - princess-tower damage taken, PRINCESS_WEIGHT per
            tower's worth of health (a whole princess tower = PRINCESS_WEIGHT)
  king      nothing for King Tower damage. KING_ACTIVATION_PENALTY when the opponent's King Tower
            is first damaged while both their princess towers stand and one of our spells is on it
            (activating it is a gift to them). An activation the opponent engineers -- pulling our
            troops onto it -- is not charged to the model; it shows in the result.

    ./py -m il.rl collect --policy CKPT [--opponent CKPT] --games N --out runs/rl/<run>/games
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))

PRINCESS_WEIGHT = 0.3
KING_ACTIVATION_PENALTY = 0.2
KING_SPELL_RADIUS = 4500            # native units: a spell of ours this close to their King Tower
TARGET_DELAY = 26                   # as mac012/console.TARGET_DELAY
LANE_FORMAT = 'clapha-rl-lane.v1'

# engine tower objects (il.samples.TOWER_IDS): 5000000 + firstlight_obs.TOWERS index
KING = {0: 5000000, 1: 5000003}
PRINCESS = {0: (5000001, 5000002), 1: (5000004, 5000005)}


def _is_spell(card_id: int) -> bool:
    return 28000000 <= int(card_id) < 29000000


def tower_health(frame: dict) -> dict[int, int]:
    """nativeObjectId -> health of the towers standing in one snapshot (a destroyed one is gone)."""
    return {int(o['nativeObjectId']): int(o['hp']) for o in frame.get('objects') or ()
            if 5000000 <= int(o['nativeObjectId']) <= 5000005 and o.get('hp') is not None}


def _spell_impacts(plays, side: int, enemy_king: tuple[int, int]) -> list[int]:
    """Impact ticks of `side`'s spells that land on the enemy King Tower (il.duel's timeline plays:
    execute tick + the measured flight, il/flight.py; the landing cell within KING_SPELL_RADIUS)."""
    from il.flight import flight_ticks
    from native_runner.arena import cell_to_world
    impacts = []
    for play in plays or ():
        if play.get('owner') != side or play.get('kind') != 'card' or play.get('grid') is None:
            continue
        if not _is_spell(int(play.get('card_id') or 0)):
            continue
        x, y = cell_to_world(tuple(play['grid']))
        if math.hypot(x - enemy_king[0], y - enemy_king[1]) <= KING_SPELL_RADIUS:
            impacts.append(int(play['lands']) + 1 + flight_ticks(int(play['card_id']), side, tuple(play['grid'])))
    return impacts


def side_rewards(frames: list[dict], side: int, decision_ticks: list[int], winner, plays=None) -> tuple[list[float], dict]:
    """One reward per decision of `side`: what happened from that decision to the next (the last
    to the end), plus the result on the last. frames: the match's decision-tick snapshots; plays: the
    match timeline's plays (il.duel recording), which date our spells' impacts on their King Tower."""
    by_tick = {int(f['tick']): f for f in frames}
    ticks = sorted(by_tick)
    first = tower_health(by_tick[ticks[0]])
    full = {tower: max(1, health) for tower, health in first.items()}
    enemy = 1 - side

    def princess(health: dict, owner: int) -> float:
        # a tower's share of its full health, summed (0..2); a missing tower is destroyed
        return sum(health.get(tower, 0) / full.get(tower, 1) for tower in PRINCESS[owner])

    standing = [princess(tower_health(by_tick[t]), side) for t in ticks]
    theirs = [princess(tower_health(by_tick[t]), enemy) for t in ticks]
    index_at = {t: i for i, t in enumerate(ticks)}
    rewards, stats = [], {'dealt': 0.0, 'taken': 0.0, 'king_activation': None}
    # the King Tower activation this side caused with a spell, if any, dated by its snapshot
    activation_tick = None
    king_full = full.get(KING[enemy])
    for i in range(1, len(ticks)):
        health = tower_health(by_tick[ticks[i]])
        if king_full is None or health.get(KING[enemy], 0) >= king_full:
            continue
        both_up = all(tower_health(by_tick[ticks[i - 1]]).get(t, 0) > 0 for t in PRINCESS[enemy])
        if both_up:
            kx, ky = _king_xy(by_tick[ticks[i - 1]], enemy)
            # one of our spells landing on it between the last clean snapshot and this one (a
            # Fireball in flight moves ~3 tiles per snapshot, so its landing point, not where a
            # snapshot caught it), or a spell object of ours on it (a rolling Log)
            landed = any(ticks[i - 1] - 5 <= impact <= ticks[i] + 5
                         for impact in _spell_impacts(plays, side, (kx, ky)))
            spells = [o for f in (by_tick[ticks[i - 1]], by_tick[ticks[i]]) for o in f.get('objects') or ()
                      if int(o['owner']) == side and _is_spell(int(o.get('cardId', 0)))
                      and math.hypot(int(o['x']) - kx, int(o['y']) - ky) <= KING_SPELL_RADIUS]
            if landed or spells:
                activation_tick = ticks[i]
        break          # the first damage is the activation, whoever caused it
    for k, tick in enumerate(decision_ticks):
        start = index_at.get(tick)
        end = index_at.get(decision_ticks[k + 1]) if k + 1 < len(decision_ticks) else len(ticks) - 1
        reward = 0.0
        if start is not None and end is not None and end > start:
            dealt = theirs[start] - theirs[end]
            taken = standing[start] - standing[end]
            reward += PRINCESS_WEIGHT * (dealt - taken)
            stats['dealt'] += dealt
            stats['taken'] += taken
            if activation_tick is not None and ticks[start] < activation_tick <= ticks[end]:
                reward -= KING_ACTIVATION_PENALTY
                stats['king_activation'] = activation_tick
        rewards.append(reward)
    if rewards:
        rewards[-1] += 0.0 if winner not in (0, 1) else (1.0 if winner == side else -1.0)
    return rewards, stats


def _king_xy(frame: dict, owner: int) -> tuple[int, int]:
    for o in frame.get('objects') or ():
        if int(o['nativeObjectId']) == KING[owner]:
            return int(o['x']), int(o['y'])
    return (9000, 3000) if owner == 0 else (9000, 29000)


class Recorder:
    """Keeps every decision a runner's model makes: the model input, the sampled action, its
    log-prob and value (FirstLight's sample_for_ppo_rollout output), in decision order."""

    def __init__(self, runner) -> None:
        self.runner = runner
        self.steps: list[dict] = []
        model = runner.model
        original = model.sample_for_ppo_rollout
        recorder = self

        def sample_for_ppo_rollout(batch, state, *, episode_start=None, validate=True):
            output = original(batch, state, episode_start=episode_start, validate=validate)
            import torch
            recorder.steps.append({
                'observation': batch.to_storage('cpu', float_dtype=torch.float16),
                'action': output.actions.to('cpu'),
                'log_prob': float(output.log_prob.reshape(-1)[0]),
                'value': float(output.value.reshape(-1)[0])})
            return output

        model.sample_for_ppo_rollout = sample_for_ppo_rollout
        self.temperatures = (float(model.ppo_gate_temperature), float(model.ppo_action_temperature),
                             float(model.ppo_continue_temperature))
        start = runner.start_battle

        def start_battle(*args, **kwargs):
            result = start(*args, **kwargs)
            # every decision through the sampling call above: a CUDA graph would bypass it
            if getattr(runner.session, '_cuda_graph', None) is not None:
                runner.session._cuda_graph = None
            return result

        runner.start_battle = start_battle

    def take(self) -> list[dict]:
        steps, self.steps = self.steps, []
        return steps


def save_lane(path: Path, lane: dict) -> int:
    """One side of one game, packed (il/pack.py) and zstd-compressed."""
    import zstandard
    from il.pack import pack
    structure, layout, data = pack(lane)
    body = pickle.dumps({'format': LANE_FORMAT, 'structure': structure, 'layout': layout,
                         'data': zstandard.ZstdCompressor(level=3).compress(bytes(data))},
                        protocol=pickle.HIGHEST_PROTOCOL)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    temporary.write_bytes(body)
    temporary.replace(path)
    return len(body)


def load_lane(path: Path) -> dict:
    import zstandard
    from il.pack import unpack
    saved = pickle.loads(Path(path).read_bytes())
    if saved.get('format') != LANE_FORMAT:
        raise ValueError(f'{path}: not a {LANE_FORMAT} lane')
    data = bytearray(zstandard.ZstdDecompressor().decompress(saved['data']))
    return unpack((saved['structure'], saved['layout'], data))


def collect(args) -> int:
    import il.duel as D
    import firstlight_bot as FLB
    from il.engine_convert import connect
    from il.speed import install
    install()           # identical results, less Python per decision (il/speed.py)
    native = connect(args.port)
    native.wait_ready(timeout=60.0)
    def load(policy):
        from il.extras import checkpoint_extra
        runner = FLB.FirstLightRunner(policy, device=args.device)
        version = int(checkpoint_extra(policy).get('update', 0)) if Path(str(policy)).is_file() else 0
        return runner, version

    def runners_now():
        learner, version = load(args.policy)
        opponent, _v = load(args.opponent or args.policy)
        recorders = {'learner': Recorder(learner)}
        if not args.opponent:
            recorders['opponent'] = Recorder(opponent)      # self-play: both sides are experience
        return learner, opponent, recorders, version

    learner, opponent, recorders, version = runners_now()
    watched = Path(args.policy) if args.follow else None
    watched_mtime = watched.stat().st_mtime if watched and watched.exists() else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    overheads = {'learner': [], 'opponent': []}
    if args.real_decks:
        # real games' decks and deals: the learner takes the Hog 2.6 side, the opponent model the other
        # player's real deck (FirstLight's General plays any deck; mirrors never show air units)
        deals = [(config, forms, tag, hog) for config, forms, tag, hog, _plays
                 in D.replay_matchups(Path(args.frames), args.deals, args.seed)]
    else:
        deals = [(config, forms, tag, None) for config, forms, tag in D.hog26_matchups(Path(args.frames), args.deals, args.seed)]
    print(f'{len(deals)} deals ({"real decks" if args.real_decks else "Hog 2.6 mirrors"}), cycled', flush=True)

    def schedule():
        while True:
            order = list(deals)
            rng.shuffle(order)
            yield from order

    opponent_delay = args.opponent_delay if args.opponent else 'live'
    for index, (config, forms, tag, hog) in zip(range(args.games), schedule()):
        if watched is not None and watched.exists() and watched.stat().st_mtime != watched_mtime:
            # the learner wrote new weights: play the next game with them
            try:
                mtime = watched.stat().st_mtime
                learner, opponent, recorders, version = runners_now()
                watched_mtime = mtime
                print(f'reloaded {watched} (update {version})', flush=True)
            except Exception as error:  # noqa: BLE001  (being replaced right now: next game)
                print(f'reload failed, keeping update {version}: {type(error).__name__}: {error}', flush=True)
        learner_side = index % 2 if hog is None else hog
        runners = {learner_side: learner, 1 - learner_side: opponent}
        delays = {learner_side: 'live', 1 - learner_side: opponent_delay}
        leads = {learner_side: 'auto', 1 - learner_side: 'auto' if opponent_delay == 'live' else 0}
        started = time.time()
        try:
            result = D.play_match(native, runners, delays, leads, config, forms,
                                  rng, f'rl-{tag[:8]}-{index}', record=True,
                                  measured={learner_side: overheads['learner'], 1 - learner_side: overheads['opponent']},
                                  target_delay=TARGET_DELAY)
        except Exception as error:  # noqa: BLE001  (one broken game must not end the collection)
            print(f'game {index + 1} failed: {type(error).__name__}: {error}', flush=True)
            for recorder in recorders.values():
                recorder.take()
            for runner in (learner, opponent):
                try:
                    runner.end_battle()
                except Exception:  # noqa: BLE001
                    pass
            continue
        frames = result['recording']['frames']
        winner = result['winner']
        row = {'game': index, 'tag': tag, 'learner_side': learner_side, 'winner': winner, 'crowns': result['crowns'],
               'end_tick': result['tick'], 'tower_hp': result.get('tower_hp'), 'seconds': round(time.time() - started),
               'policy': args.policy, 'policy_version': version, 'opponent': args.opponent or 'self',
               'opponent_delay': opponent_delay, 'real_decks': bool(args.real_decks), 'lanes': []}
        for role, recorder in recorders.items():
            steps = recorder.take()
            if not steps:
                continue
            side = learner_side if role == 'learner' else 1 - learner_side
            runner = learner if role == 'learner' else opponent
            decision_ticks = [runner.first_decision_tick + 5 * k for k in range(len(steps))]
            rewards, stats = side_rewards(frames, side, decision_ticks, winner,
                                          plays=result['recording']['header']['timeline']['plays'])
            lane = {'observations': [s['observation'] for s in steps], 'actions': [s['action'] for s in steps],
                    'log_prob': [s['log_prob'] for s in steps], 'value': [s['value'] for s in steps],
                    'reward': rewards, 'side': side, 'role': role, 'winner': winner,
                    'temperatures': recorder.temperatures, 'policy': args.policy, 'tag': tag,
                    'target_delay': TARGET_DELAY, 'policy_version': version}
            name = f'{int(time.time() * 1000)}-{index:05d}-{role}-v{version}.lane'
            size = save_lane(out / name, lane)
            row['lanes'].append({'file': name, 'side': side, 'steps': len(steps), 'bytes': size,
                                 'return': round(sum(rewards), 4), **{k: (round(v, 4) if isinstance(v, float) else v)
                                                                      for k, v in stats.items()}})
        with (out / 'games.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps({k: row[k] for k in ('game', 'winner', 'crowns', 'end_tick', 'seconds')}
                         | {'lanes': [(l['side'], l['steps'], l['return']) for l in row['lanes']]}), flush=True)
    return 0


LEAGUE = 'self=4,snap=2,anchor=1,hog2=1,general=2'


def _league(text: str) -> list[tuple[str, float]]:
    import re
    # 'self=4,snap=2' (or '/' between: PowerShell treats a bare comma list as an array)
    pairs = [item.split('=') for item in re.split(r'[,/ ]+', text) if item]
    league = [(kind.strip(), float(weight)) for kind, weight in pairs]
    unknown = {kind for kind, _ in league} - {'self', 'snap', 'anchor', 'hog2', 'general'}
    if unknown:
        raise SystemExit(f'unknown league members {sorted(unknown)}')
    return league


def _snapshots(run: Path, every: int, keep: int) -> list[Path]:
    """League snapshots: the run's policy-NNNN.pt at every `every` updates, the latest `keep`."""
    found = []
    for path in run.glob('policy-*.pt'):
        try:
            number = int(path.stem.split('-')[1])
        except (IndexError, ValueError):
            continue
        if number % every == 0:
            found.append((number, path))
    return [path for _number, path in sorted(found)[-keep:]]


def collect_remote(args) -> int:
    """Collection with the models on the inference server (il/rl_serve.py) and the opponent drawn
    per game from the league: self (both sides trained), snap (a past snapshot), anchor (the fixed
    starting model), hog2 (FirstLight's hog2 with no delay), general (FirstLight's General on a real
    game's opponent deck, no delay). The trained side is always `latest`."""
    import torch
    import il.duel as D
    from il.engine_convert import connect
    from il.rl_serve import WAITED, Client, _address, decide_many, remote_runner
    from il.speed import install
    install()
    torch.set_num_threads(1)            # this process builds observations; the server does the model
    native = connect(args.port)
    native.wait_ready(timeout=60.0)
    client = Client(_address(args.server))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = out.parent
    league = _league(args.league)
    kinds, weights = zip(*league)
    rng = random.Random(args.seed)
    frames_dir = Path(args.frames)
    pool_file = Path(args.deal_pool) if args.deal_pool else None
    if pool_file is not None and pool_file.is_file():
        # the shared pool (il.rl deals): each collector plays its own shuffled share of it
        pool = pickle.loads(pool_file.read_bytes())
        mirrors, real = list(pool['mirror']), list(pool['real']) if 'general' in kinds else []
        rng.shuffle(mirrors)
        rng.shuffle(real)
    else:
        mirrors = [(config, forms, tag, None) for config, forms, tag in D.hog26_matchups(frames_dir, args.deals, args.seed)]
        real = ([(config, forms, tag, hog) for config, forms, tag, hog, _plays
                 in D.replay_matchups(frames_dir, args.deals, args.seed)] if 'general' in kinds else [])
    print(f'{len(mirrors)} mirror deals, {len(real)} real-deck deals; league {dict(league)}', flush=True)

    def cycle(deals):
        while True:
            order = list(deals)
            rng.shuffle(order)
            yield from order
    pools = {'mirror': cycle(mirrors), 'real': cycle(real) if real else None}
    overheads = {'learner': [], 'opponent': []}
    failures = 0
    for index in range(args.games):
        # the learner may be waiting for the shared GPU: do not run far ahead of it (its lanes would go stale)
        while len(list(out.glob('*.lane'))) > args.max_backlog:
            time.sleep(30)
        kind = rng.choices(kinds, weights)[0]
        snapshot = None
        if kind == 'snap':
            snapshots = _snapshots(run, args.snapshot_every, args.snapshots)
            snapshot = rng.choice(snapshots) if snapshots else None
            if snapshot is None:
                kind = 'self'
        opponent_key, opponent_checkpoint, opponent_delay, pool = {
            'self': ('latest', args.policy, 'live', 'mirror'),
            'snap': (str(snapshot), str(snapshot), 'live', 'mirror'),
            'anchor': (args.anchor, args.anchor, 'live', 'mirror'),
            'hog2': ('fl:hog2', None, 'none', 'mirror'),
            'general': ('fl:general', None, 'none', 'real'),
        }[kind]
        config, forms, tag, hog = next(pools[pool])
        learner = remote_runner(client, 'latest', args.policy)
        opponent = remote_runner(client, opponent_key, opponent_checkpoint)
        learner_side = index % 2 if hog is None else hog
        runners = {learner_side: learner, 1 - learner_side: opponent}
        delays = {learner_side: 'live', 1 - learner_side: opponent_delay}
        leads = {learner_side: 'auto', 1 - learner_side: 'auto' if opponent_delay == 'live' else 0}
        started = time.time()
        WAITED[0] = 0.0
        try:
            result = D.play_match(native, runners, delays, leads, config, forms, rng, f'rl-{tag[:8]}-{index}',
                                  record=True, measured={learner_side: overheads['learner'],
                                                         1 - learner_side: overheads['opponent']},
                                  target_delay=TARGET_DELAY, decide_many=decide_many)
        except Exception as error:  # noqa: BLE001  (one broken game must not end the collection)
            print(f'game {index + 1} failed: {type(error).__name__}: {error}', flush=True)
            for runner in (learner, opponent):
                try:
                    runner.end_battle()
                except Exception:  # noqa: BLE001
                    pass
            failures += 1
            if failures >= 5:
                # the engine or the server is gone, not one game: end, so the keeper (il/rl_turns.py)
                # sees it and brings the part back
                print('5 games in a row failed: stopping this collector', flush=True)
                return 3
            continue
        if result.get('timing'):
            # the decide time split: waiting for the server's answer vs our own tensorize / decode
            result['timing']['wait'] = round(WAITED[0], 2)
        try:
            row = _finish_game(args, out, run, index, kind, tag, opponent_key, opponent_delay, learner_side, learner,
                               opponent, result, started)
        except MemoryError:
            # the PC's memory is shared with the other user's job: lose this game, not the collector
            print(f'game {index + 1}: out of memory while saving it', flush=True)
            failures += 1
            if failures >= 5:
                print('5 games in a row failed: stopping this collector', flush=True)
                return 3
            continue
        failures = 0
        print(json.dumps({k: row[k] for k in ('game', 'league', 'learner_won', 'crowns', 'seconds')}
                         | {'return': row['sides']['learner']['return']}), flush=True)
    return 0


def _finish_game(args, out: Path, run: Path, index: int, kind: str, tag: str, opponent_key: str, opponent_delay: str,
                 learner_side: int, learner, opponent, result: dict, started: float) -> dict:
    """Rewards, lanes, the games.jsonl row and (sometimes) the whole recording of one played game."""
    from il.frames import save_recorded
    from il.habits import behaviour
    recording = result['recording']
    frames, plays = recording['frames'], recording['header']['timeline']['plays']
    winner = result['winner']
    row = {'game': index, 'time': round(time.time()), 'league': kind, 'opponent': opponent_key,
           'opponent_delay': opponent_delay, 'tag': tag, 'learner_side': learner_side,
           'learner_version': learner.version, 'opponent_version': opponent.version, 'winner': winner,
           'learner_won': None if winner not in (0, 1) else winner == learner_side,
           'crowns': result['crowns'], 'end_tick': result['tick'], 'tower_hp': result.get('tower_hp'),
           'seconds': round(time.time() - started, 1), 'timing': result.get('timing'), 'lanes': [], 'sides': {}}
    trained = [('learner', learner_side, learner)] + ([('opponent', 1 - learner_side, opponent)] if kind == 'self' else [])
    for role, side, runner in [('learner', learner_side, learner), ('opponent', 1 - learner_side, opponent)]:
        decision_ticks = [runner.first_decision_tick + 5 * k for k in range(len(runner.last_steps))]
        rewards, stats = side_rewards(frames, side, decision_ticks, winner, plays=plays)
        row['sides'][role] = {'side': side, 'return': round(sum(rewards), 4),
                              **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()},
                              **behaviour(frames, plays, side)}
        if (role, side, runner) not in trained or not runner.last_steps:
            continue
        steps = runner.last_steps
        lane = {'observations': [s['observation'] for s in steps], 'actions': [s['action'] for s in steps],
                'log_prob': [s['log_prob'] for s in steps], 'value': [s['value'] for s in steps],
                'reward': rewards, 'side': side, 'role': role, 'winner': winner,
                'temperatures': runner.temperatures, 'policy': 'latest', 'tag': tag,
                'target_delay': TARGET_DELAY, 'policy_version': runner.version, 'league': kind}
        name = f'{int(time.time() * 1000)}-{index:05d}-p{args.port}-{role}-v{runner.version}.lane'
        size = save_lane(out / name, lane)
        row['lanes'].append({'file': name, 'side': side, 'steps': len(steps), 'bytes': size})
    if args.save_every and index % args.save_every == 0:
        # an occasional whole game for watching (il.duel's recording format; the viewer reads it)
        header = dict(recording['header'])
        header['played'] = {'a': f'latest v{learner.version}', 'b': opponent_key, 'a_side': learner_side,
                            'league': kind, 'result': 'a' if row['learner_won'] else ('b' if winner in (0, 1) else 'draw')}
        save_recorded(run / 'recordings', header, frames)
    with (out / 'games.jsonl').open('a') as handle:
        handle.write(json.dumps(row) + '\n')
    return row


def deals(args) -> int:
    """Prepare the deal pool once for every collector: Hog 2.6 mirrors and real games' decks."""
    import il.duel as D
    import firstlight_bot  # noqa: F401
    frames_dir = Path(args.frames)
    started = time.time()
    pool = {'mirror': [(config, forms, tag, None) for config, forms, tag
                       in D.hog26_matchups(frames_dir, args.count, args.seed)],
            'real': [(config, forms, tag, hog) for config, forms, tag, hog, _plays
                     in D.replay_matchups(frames_dir, args.count, args.seed)]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix('.part')
    temporary.write_bytes(pickle.dumps(pool, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(out)
    print(f"{len(pool['mirror'])} mirror deals, {len(pool['real'])} real-deck deals -> {out} "
          f"({time.time() - started:.0f} s)", flush=True)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    d = commands.add_parser('deals', help='prepare the shared deal pool for collectors')
    d.add_argument('--frames', default=str(CLAPHA / 'runs/conv-hog26'))
    d.add_argument('--count', type=int, default=600)
    d.add_argument('--seed', type=int, default=0)
    d.add_argument('--out', default=str(CLAPHA / 'runs/rl/deals.pkl'))
    c = commands.add_parser('collect', help='play self-play games through the live path and save lanes')
    c.add_argument('--policy', required=True, help='the learner: a checkpoint (or a name in firstlight_bot)')
    c.add_argument('--opponent', help='a fixed opponent (league snapshot); default: self-play, both sides recorded')
    c.add_argument('--games', type=int, default=10)
    c.add_argument('--deals', type=int, default=400, help='distinct deals to prepare; games cycle through them')
    c.add_argument('--real-decks', action='store_true',
                   help="real games' decks: the learner on the Hog 2.6 side, --opponent on the other real deck")
    c.add_argument('--opponent-delay', default='live', choices=('live', 'none'),
                   help='a fixed opponent plays with the live delay, or as in FirstLight\'s sandbox (no delay)')
    c.add_argument('--out', required=True)
    c.add_argument('--frames', default=str(CLAPHA / 'runs/conv-hog26'), help='where the Hog 2.6 decks come from')
    c.add_argument('--port', type=int, default=26789)
    c.add_argument('--device', default='cpu')
    c.add_argument('--seed', type=int, default=7)
    c.add_argument('--follow', action='store_true', help='reload --policy whenever it changes (the learner\'s latest.pt)')
    c.add_argument('--server', help='host:port of il.rl_serve: the models run there, the league picks opponents')
    c.add_argument('--league', default=LEAGUE, help='with --server: opponent weights, e.g. ' + LEAGUE)
    c.add_argument('--anchor', help='with --server: the fixed starting checkpoint (the league\'s anchor)')
    c.add_argument('--snapshot-every', type=int, default=10, help='league snapshots: every N updates')
    c.add_argument('--snapshots', type=int, default=5, help='league snapshots kept (the latest N)')
    c.add_argument('--save-every', type=int, default=50, help='keep one whole game in N for watching (0: none)')
    c.add_argument('--max-backlog', type=int, default=64,
                   help='with --server: pause while this many lanes wait (more would go stale before the learner '
                        'gets to them)')
    c.add_argument('--deal-pool', default=str(CLAPHA / 'runs/rl/deals.pkl'),
                   help='with --server: the shared pool from il.rl deals (built on the spot when missing)')
    args = parser.parse_args(argv)
    if args.command == 'deals':
        return deals(args)
    if args.command == 'collect':
        return collect_remote(args) if args.server else collect(args)
    return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
