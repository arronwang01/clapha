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


def side_rewards(frames: list[dict], side: int, decision_ticks: list[int], winner) -> tuple[list[float], dict]:
    """One reward per decision of `side`: what happened from that decision to the next (the last
    to the end), plus the result on the last. frames: the match's decision-tick snapshots."""
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
            spells = [o for f in (by_tick[ticks[i - 1]], by_tick[ticks[i]]) for o in f.get('objects') or ()
                      if int(o['owner']) == side and _is_spell(int(o.get('cardId', 0)))
                      and math.hypot(int(o['x']) - kx, int(o['y']) - ky) <= KING_SPELL_RADIUS]
            if spells:
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
    games = D.hog26_matchups(Path(args.frames), args.games, args.seed)
    for index, (config, forms, tag) in enumerate(games):
        if watched is not None and watched.exists() and watched.stat().st_mtime != watched_mtime:
            # the learner wrote new weights: play the next game with them
            watched_mtime = watched.stat().st_mtime
            learner, opponent, recorders, version = runners_now()
            print(f'reloaded {watched} (update {version})', flush=True)
        learner_side = index % 2
        runners = {learner_side: learner, 1 - learner_side: opponent}
        started = time.time()
        try:
            result = D.play_match(native, runners, {0: 'live', 1: 'live'}, {0: 'auto', 1: 'auto'}, config, forms,
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
               'policy': args.policy, 'opponent': args.opponent or 'self', 'lanes': []}
        for role, recorder in recorders.items():
            steps = recorder.take()
            if not steps:
                continue
            side = learner_side if role == 'learner' else 1 - learner_side
            runner = learner if role == 'learner' else opponent
            decision_ticks = [runner.first_decision_tick + 5 * k for k in range(len(steps))]
            rewards, stats = side_rewards(frames, side, decision_ticks, winner)
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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    c = commands.add_parser('collect', help='play self-play games through the live path and save lanes')
    c.add_argument('--policy', required=True, help='the learner: a checkpoint (or a name in firstlight_bot)')
    c.add_argument('--opponent', help='a fixed opponent (league snapshot); default: self-play, both sides recorded')
    c.add_argument('--games', type=int, default=10)
    c.add_argument('--out', required=True)
    c.add_argument('--frames', default=str(CLAPHA / 'runs/conv-hog26'), help='where the Hog 2.6 decks come from')
    c.add_argument('--port', type=int, default=26789)
    c.add_argument('--device', default='cpu')
    c.add_argument('--seed', type=int, default=7)
    c.add_argument('--follow', action='store_true', help='reload --policy whenever it changes (the learner\'s latest.pt)')
    args = parser.parse_args(argv)
    if args.command == 'collect':
        return collect(args)
    return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
