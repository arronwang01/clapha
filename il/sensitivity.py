"""What our model's plays depend on: its memory of the match, the opponent's queued plays, one card of either
deck. Engine games only (test and training recordings), never the user's matches.

Every card our side played in the recorded games is decided again by the model, from that game's own inputs
turn by turn (il/samples.actor_samples: the live pipeline, the recorded plays fed back as the model's own),
once as it was and once with one thing changed:

  memory 10s / 30s / 60s  the recurrent memory rebuilt from only the last 10 / 30 / 60 s of the match:
                          what it remembered from before is gone; the board, both hands as tracked, the
                          opponent's card cycle and elixir, every input of the turn itself stay
  no queue                the opponent's queued plays left out (the model sees them ~0.7 s before they
                          land, with card and tile; a person sees them when they land)
  own Log->Arrows         our deck has Arrows where it has The Log, on turns where The Log is neither in the
                          hand nor next (so the hand and the choices are the same; only the deck differs)
  hand Log->Arrows        the other turns: The Log in the hand (an Arrows choice in its place, cost 3: not
                          playable under 3 elixir; spells may go on any tile, both) or next. A play of The Log
                          itself is compared with the Arrows choice in its place
  their Log->Arrows       the opponent's revealed The Log shown as Arrows

Per play, against the real decision: tile = how much of the probability of where the card goes moves
elsewhere (total variation, 0-1); top = whether its most likely tile changes; act = the change in the
probability of playing at all that turn (live gate temperature 0.2); card = the shift in which card.

    ./py -m il.sensitivity [--games runs/rl/check-full] [--model runs/pc/pilot3.pt] [--max 20] [--mirror-only]

Waits while either console is in a battle (il.strength_check.live_match) so it never loads a live match.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import replace
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))

LOG, ARROWS = 28000011, 28000001
NAMES = {26000021: 'Hog Rider', 26000014: 'Musketeer', 27000000: 'Cannon', 28000000: 'Fireball',
         28000011: 'The Log', 26000010: 'Skeletons', 26000038: 'Ice Golem', 26000030: 'Ice Spirit'}
GATE_TEMPERATURE = 0.2                 # the trained gate sampling temperature (every checkpoint)
TURNS_PER_SECOND = 4                   # a decision every five ticks
MEMORIES = (10, 30, 60)                # seconds of the match the rebuilt memory keeps
DELAY = 26                             # TARGET_DELAY: decision -> replay tick of our plays (live and tests)


def say(text: str) -> None:
    print(f'{time.strftime("%H:%M:%S")} {text}', flush=True)


def wait_for_quiet() -> None:
    from il.strength_check import live_match
    if live_match():
        say('a battle is running on a device: paused until it is over')
        while live_match():
            time.sleep(5)
        say('going on')


def tv(p, q) -> float:
    return float(0.5 * (p - q).abs().sum())


class Probe:
    def __init__(self, model):
        import torch
        from native_runner.training.v4.decoding import ShadowCandidateLegality
        self.torch, self.model, self.Shadow = torch, model, ShadowCandidateLegality
        catalog = model.catalog
        self.card_of = {catalog.vocab_id(card): card for card in (*NAMES, ARROWS)}
        self.log, self.arrows = catalog.vocab_id(LOG), catalog.vocab_id(ARROWS)

    def decide(self, batch, encoded, hidden, cell, sequence):
        """P(act) at the live gate temperature, the card distribution, and where the recorded card goes."""
        torch, model = self.torch, self.model
        context = model._policy_context(encoded, hidden, cell)
        shadow = self.Shadow(batch.candidates, model.config)
        legal = shadow.candidate_mask()
        act = torch.softmax(model.gate_head(context.policy_context)[0] / GATE_TEMPERATURE, -1)[1]
        decoder = model._decoder_context(model.decoder_initial(context.policy_context), context.encoded)
        cards = torch.softmax(model.candidate_logits(decoder, context.encoded.candidate_memory)[0]
                              .masked_fill(~legal[0], float('-inf')), -1)
        chosen = model._resolve_candidate_indices(batch, sequence)[:, 0]
        memory = model._selected_candidate_memory(context.encoded.candidate_memory, chosen)
        logits, _keys = model._location_logits(decoder, memory, context.encoded.spatial_memory)
        placement = shadow.selected_placement(chosen).flatten(1)[0]
        tiles = torch.softmax(logits.flatten(1)[0].masked_fill(~placement, float('-inf')), -1)
        return float(act), cards, tiles

    def step(self, batch, state):
        """Encode a (changed) turn and advance the memory from `state` with it."""
        encoded = self.model.encode_observation(batch, validate=False)
        hidden, cell = self.model.lstm_core(self.model._core_input(encoded), state)
        return encoded, hidden, cell

    def own_swapped(self, batch):
        own = batch.own_cards
        rows = (own.card_vocab_id[0] == self.log).nonzero().flatten().tolist()
        if not rows or own.runtime_features[0, rows[0], 0] or own.runtime_features[0, rows[0], 1]:
            return None                       # no Log, or it is in the hand or next
        ids = own.card_vocab_id.clone()
        ids[0, rows[0]] = self.arrows
        return replace(batch, own_cards=replace(own, card_vocab_id=ids))

    def hand_swapped(self, batch):
        """Arrows in place of The Log while it is in the hand or next: the deck row, and its choice."""
        own = batch.own_cards
        rows = (own.card_vocab_id[0] == self.log).nonzero().flatten().tolist()
        if not rows or not (own.runtime_features[0, rows[0], 0] or own.runtime_features[0, rows[0], 1]):
            return None
        ids = own.card_vocab_id.clone()
        ids[0, rows[0]] = self.arrows
        changed = replace(batch, own_cards=replace(own, card_vocab_id=ids))
        choices = batch.candidates
        hit = choices.mask[0] & (choices.visible_card_vocab_id[0] == self.log)
        if bool(hit.any()):
            visible, effective = choices.visible_card_vocab_id.clone(), choices.effective_card_vocab_id.clone()
            cost, runtime, native = choices.cost.clone(), choices.runtime_features.clone(), choices.native_visible_card_id.clone()
            visible[0, hit], effective[0, hit], native[0, hit] = self.arrows, self.arrows, ARROWS
            cost[0, hit], runtime[0, hit, 0] = 3.0, 0.3        # Arrows costs 3 (runtime[0] is cost / 10)
            changed = replace(changed, candidates=replace(
                choices, visible_card_vocab_id=visible, effective_card_vocab_id=effective, cost=cost,
                runtime_features=runtime, native_visible_card_id=native))
        return changed

    def their_swapped(self, batch):
        theirs = batch.opponent_cards
        rows = ((theirs.card_vocab_id[0] == self.log) & theirs.mask[0]).nonzero().flatten().tolist()
        if not rows:
            return None
        ids = theirs.card_vocab_id.clone()
        ids[0, rows[0]] = self.arrows
        return replace(batch, opponent_cards=replace(theirs, card_vocab_id=ids))

    @staticmethod
    def without_queue(batch):
        extras = batch.extras
        theirs = extras.pending_mask & (extras.pending_owner == 1)
        if not bool(theirs.any()):
            return None
        return replace(batch, extras=replace(extras, pending_mask=extras.pending_mask & ~theirs))


def game(path: Path, probe: Probe, stats: Counter, out, mirror_only: bool = False) -> int:
    import torch
    from il.frames import load_replay
    from il.samples import actor_samples
    from native_runner.training.v4.tensors import GATE_ACT, TARGET_GRID
    header, frames = load_replay(path)
    played = header.get('played') or {}
    actor = played.get('a_side')
    if actor not in (0, 1) or 'expert_actions' not in header:
        stats['games skipped (no side or plays)'] += 1
        return 0
    from il.samples import HOG26_CARDS
    if mirror_only and not all(HOG26_CARDS <= set(deck) for deck in header['timeline']['decks']):
        stats['games skipped (not a Hog 2.6 mirror)'] += 1
        return 0
    samples = actor_samples(header, frames, actor, stats, keep=True, delay=DELAY, extras=True)
    model = probe.model
    width = model.config.lstm_hidden_dim
    zeros = (torch.zeros(1, width), torch.zeros(1, width))
    state = zeros
    recent = deque(maxlen=max(MEMORIES) * TURNS_PER_SECOND)
    plays = 0
    for index, (tick, storage, sequence, usable) in enumerate(samples):
        if index % 200 == 199:
            wait_for_quiet()
        batch = storage.to_model_input('cpu')
        before = state
        encoded = model.encode_observation(batch, validate=False)
        core = model._core_input(encoded)
        recent.append(core)
        state = model.lstm_core(core, state)
        if not usable or int(sequence.gate[0]) != GATE_ACT:
            continue
        chosen = int(model._resolve_candidate_indices(batch, sequence)[0, 0])
        if int(batch.candidates.target_mode[0, chosen]) != TARGET_GRID:
            continue
        card = probe.card_of.get(int(batch.candidates.effective_card_vocab_id[0, chosen]))
        act, cards, tiles = probe.decide(batch, encoded, *state, sequence)
        row = {'game': path.stem[:12], 'tick': tick, 'card': NAMES.get(card, 'other'), 'act': round(act, 4),
               'top_p': round(float(tiles.max()), 4)}

        def compare(name, result):
            other_act, other_cards, other_tiles = result
            row[name] = {'tile': round(tv(tiles, other_tiles), 4),
                         'top': int(tiles.argmax()) != int(other_tiles.argmax()),
                         'act': round(other_act - act, 4), 'card': round(tv(cards, other_cards), 4),
                         'pick': round(float(other_cards[chosen] - cards[chosen]), 4)}

        for seconds in MEMORIES:
            kept = list(recent)[-seconds * TURNS_PER_SECOND:]
            if len(kept) < seconds * TURNS_PER_SECOND:
                continue                      # the match is not that old yet: nothing to forget
            short = zeros
            for item in kept:
                short = model.lstm_core(item, short)
            compare(f'memory {seconds}s', probe.decide(batch, encoded, *short, sequence))
        for name, changed in (('no queue', probe.without_queue(batch)), ('own Log->Arrows', probe.own_swapped(batch)),
                              ('hand Log->Arrows', probe.hand_swapped(batch)),
                              ('their Log->Arrows', probe.their_swapped(batch))):
            if changed is not None:
                encoded_2, hidden_2, cell_2 = probe.step(changed, before)
                compare(name, probe.decide(changed, encoded_2, hidden_2, cell_2, sequence))
        out.write(json.dumps(row) + '\n')
        plays += 1
    return plays


def summary(rows: list[dict]) -> None:
    names = ['memory 10s', 'memory 30s', 'memory 60s', 'no queue', 'own Log->Arrows', 'hand Log->Arrows',
             'their Log->Arrows']
    groups = defaultdict(list)
    for row in rows:
        groups[row['card']].append(row)
        groups['all cards'].append(row)
    print(f"\n{'':18}" + ''.join(f'{name:>22}' for name in names))
    print(f"{'card (plays)':18}" + ''.join(f"{'tile  top  act  card':>22}" for _ in names))
    for card in ['all cards', *NAMES.values()]:
        items = groups.get(card, [])
        if not items:
            continue
        line = f'{card} ({len(items)})'
        line = f'{line:18}'
        for name in names:
            values = [row[name] for row in items if name in row]
            if not values:
                line += f"{'-':>22}"
                continue
            n = len(values)
            line += (f"{sum(v['tile'] for v in values) / n:>9.2f}{100 * sum(v['top'] for v in values) / n:>4.0f}%"
                     f"{sum(abs(v['act']) for v in values) / n:>5.2f}{sum(v['card'] for v in values) / n:>5.2f}")
        print(line)
    print('\ntile: probability of the tile moved elsewhere (0-1, mean); top: most likely tile changed (% of plays);'
          '\nact: |change| in P(play this turn), gate at 0.2; card: shift in which card (0-1). A blank: not applicable'
          '\n(memory: match younger than that; no queue: nothing queued; own swap: The Log in hand or next; hand swap:'
          '\nit is not; their swap: The Log not revealed yet).')


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', action='append', help='a folder of recordings (repeatable)')
    parser.add_argument('--model', default=str(CLAPHA / 'runs/pc/pilot3.pt'))
    parser.add_argument('--max', type=int, default=20, help='games at most')
    parser.add_argument('--out', default=str(CLAPHA / 'runs/sensitivity.jsonl'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--mirror-only', action='store_true', help='Hog 2.6 mirrors only, like the user plays')
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)
    import firstlight_bot  # noqa: F401  (FirstLight on sys.path)
    from il.extras import cache_static_encodings, load_policy
    model = load_policy(args.model, 'cpu')
    model.eval()
    cache_static_encodings(model)
    probe = Probe(model)
    folders = [Path(f) for f in (args.games or [str(CLAPHA / 'runs/rl/check-full')])]
    paths = sorted({p for folder in folders for p in folder.rglob('*.jsonl.zst')}, key=lambda p: p.stat().st_mtime)
    paths = paths[-args.max:]
    stats: Counter = Counter()
    say(f'{len(paths)} games, model {args.model}')
    with torch.no_grad(), open(args.out, 'w') as out:
        for number, path in enumerate(paths, 1):
            wait_for_quiet()
            started = time.time()
            plays = game(path, probe, stats, out, args.mirror_only)
            out.flush()
            say(f'game {number}/{len(paths)} {path.name[:14]}: {plays} plays ({time.time() - started:.0f} s)')
    rows = [json.loads(line) for line in open(args.out)]
    summary(rows)
    print('\n' + ', '.join(f'{k}: {v}' for k, v in stats.most_common() if 'not aligned' in k or 'skipped' in k
                           or k in ('decision turns', 'labels aligned')))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
