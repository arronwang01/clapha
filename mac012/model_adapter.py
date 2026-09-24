"""Bridge from our live frames to the user's sim_engine policies.

Their models were trained with the opponent's hand visible (sim_engine.tensors reads
them['hand_deck_indices'] and looks each index up in deck[1-side]). On build 160402012 the
opponent's hand is not readable, so we feed the vocabulary's unknown token (0) for all five
opponent slots. The models therefore run slightly out of distribution here; that is a property
of how they were trained, not a bug in the bridge.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPLAY = Path.home() / 'Documents/GitHub/cr-sim-upstream/replay'
EXTRACT = Path.home() / 'Documents/GitHub/cr-engine-extraction'
MODELS = {'14k': REPLAY / 'collected' / 'policy.pt',
          '40k': REPLAY / 'collected' / 'policy_50k.pt',
          '100k': REPLAY / 'collected' / 'policy_100k.pt'}
TOWERS = ((9000, 3000, 0, 12), (3500, 6500, 0, 13), (14500, 6500, 0, 13),
          (9000, 29000, 1, 12), (3500, 25500, 1, 13), (14500, 25500, 1, 13))


def load_engine():
    for path in (str(REPLAY), str(EXTRACT / 'macos-port')):
        if path not in sys.path:
            sys.path.insert(0, path)
    import sim_engine  # type: ignore
    return sim_engine


def to_state(frame: dict, local_side: int, deduced: dict | None = None) -> dict:
    """Our frame in the shape sim_engine.tensors() reads.

    Towers go in as entities (kind 12 king, 13 princess) with the live hp from the frame;
    units are kind 15. Coordinates and elixir stay in native units, as the engine expects.
    """
    entities = []
    live = {(e['x'], e['y']): e for e in frame['entities'] if e['card_id'] == -1}
    for x, y, side, kind in TOWERS:
        tower = live.get((x, y))
        entities.append({'kind': kind, 'side': side, 'card_id': -1, 'x': x, 'y': y,
                         'hp': (tower or {}).get('hp', 0),
                         'max_hp': (tower or {}).get('max_hp', 1)})
    for e in frame['entities']:
        if e['card_id'] == -1:
            continue
        # tensors() treats kind 12/13 as crown towers, and buildings (Cannon, Tombstone)
        # come through with those kinds - passing them straight through would corrupt the
        # tower tensor. Only kind 14 is preserved, because the frozen feature reads it;
        # everything else is a plain unit.
        kind = 14 if e['kind'] == 14 else 15
        entities.append({'kind': kind, 'side': e['side'], 'card_id': e['card_id'],
                         'x': e['x'], 'y': e['y'], 'hp': e['hp'], 'max_hp': e['max_hp'],
                         # column 9 (deploy_frozen) reads this; our reader has it, so unlike
                         # the user's own live console we do not leave it blank.
                         'behavior_state': e.get('behavior_state_raw')})
    players = {}
    for p in frame['players']:
        readable = p['hand_deck_indices'][0] != -1
        # The opponent's hand is never in memory during play, but it is DEDUCIBLE from their
        # deck plus the cards we have watched them play (cycle_tracker). Feeding the deduction
        # puts the model nearer its training distribution than four unknown tokens.
        if not readable and deduced and p['side'] in deduced:
            hand, nxt = deduced[p['side']]
            players[p['side']] = {'side': p['side'], 'elixir': p['elixir_raw'] / 10000,
                                  'hand_deck_indices': hand, 'next_deck_index': nxt,
                                  'readable': False, 'deduced': True}
            continue
        players[p['side']] = {
            'side': p['side'], 'elixir': p['elixir_raw'] / 10000,
            # Unreadable opponent hand -> four zero indices, which map to the unknown token
            # because we also give that side an all-unknown deck (see decks()).
            'hand_deck_indices': p['hand_deck_indices'] if readable else [0, 0, 0, 0],
            'next_deck_index': p['next_deck_index'] if readable else 0,
            'readable': readable}
    crowns = [sum(1 for x, y, s, k in TOWERS
                  if s != side and (live.get((x, y)) or {}).get('hp', 0) <= 0)
              for side in (0, 1)]
    return {'tick': frame['game_tick'], 'players': [players[0], players[1]],
            'entities': entities, 'local_side': local_side,
            'episode': {'terminated': False, 'crowns': crowns}}


def decks(adapter, engine, state: dict, frame: dict, local_side: int,
          opponent_deck: list[int] | None = None) -> None:
    """deck[side] is (card_id, vocab_id) by deck slot. Ours is read directly; the opponent's
    is unavailable except in friendlies, where deck_vector_scan can supply it."""
    for p in frame['players']:
        ids = p.get('deck_card_ids') or []
        if not ids and opponent_deck and p['side'] != local_side:
            ids = opponent_deck
        if len(ids) == 8:
            adapter.deck[p['side']] = [
                (cid, adapter.vocab.get(engine.to_sim(
                    engine.MV.cards.get(cid) or adapter.form_names.get(cid)), 0))
                for cid in ids]
        else:
            adapter.deck[p['side']] = [(0, 0)] * 8


def build(model: str, threshold: float | None = None):
    engine = load_engine()
    path = MODELS[model]
    adapter = engine.Adapter(str(path), {0: [], 1: []})
    calibrated = path.with_suffix('.threshold.json').exists()
    if calibrated:
        try:
            adapter.threshold = json.loads(
                path.with_suffix('.threshold.json').read_text())['threshold']
        except Exception:  # noqa: BLE001
            calibrated = False
    if threshold is not None:
        adapter.threshold = threshold
        calibrated = True
    return engine, adapter, calibrated
