"""How often a model's Log and Ice Golem plays make sense, from recorded engine matches.

Recordings are il.duel --record output (engine games only, never the user's battles). For every
play, the enemy units at the moment it executed (the snapshot at or just after, within 4 ticks):

  Log         what its path held: 1.95 tiles either side of its line (plus a unit's width), 10.1
              tiles on from where it lands, towards the enemy. 'ground' = an enemy ground unit or
              building (the Log's targets), 'air only' = enemy units there but all flying (it cannot
              touch them), 'tower only', or 'nothing'.
  Ice Golem   the enemies within 6 tiles: 'pullable' = a troop that targets troops (the golem can
              kite it), 'building-targeters only' (Hog Rider, Giant, ... walk past it; pros sometimes
              still use it for collision), or 'none near'.

    ./py -m il.habits DIR [DIR ...]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))

LOG, ICE_GOLEM = 28000011, 26000038
LOG_HALF_WIDTH, LOG_LENGTH, UNIT_MARGIN = 1950, 10100, 500
GOLEM_RADIUS = 6000


def _unit_facts() -> tuple[set[int], set[int]]:
    """(cards whose units fly, cards whose units target only buildings), from FirstLight's specs."""
    from native_runner.training.v4.factory import production_semantic_bundle
    flying, building_only = set(), set()
    for card, spec in production_semantic_bundle().card_specs.items():
        attributes = spec.attributes or {}
        forms = [attributes.get('resolved_summoned_form') or {}]
        forms += [item.get('definition') or {} for item in attributes.get('resolved_summoned_forms') or ()]
        if any(float(form.get('FlyingHeight') or 0) > 0 for form in forms):
            flying.add(int(card))
        if (spec.categorical_features or {}).get('target_only_buildings'):
            building_only.add(int(card))
    return flying, building_only


def _snapshot_at(frames: list[dict], tick: int) -> dict | None:
    for frame in frames:
        if int(frame['tick']) >= tick:
            return frame if int(frame['tick']) - tick <= 4 else None
    return None


def classify(path: Path, flying: set[int], building_only: set[int]) -> list[tuple[str, str, str]]:
    """(model label, card, verdict) per Log / Ice Golem play in one recording."""
    from il.frames import load_replay
    from il.samples import TOWER_IDS
    from il.timeline import timeline_from_json
    header, frames = load_replay(path, with_replay=False)
    played = header.get('played') or {}
    timeline = timeline_from_json(header['timeline'])
    a_side = played.get('a_side')
    label = {a_side: f"a={played.get('a')}", 1 - a_side: f"b={played.get('b')}"} if a_side in (0, 1) else {}
    rows = []
    for play in timeline.plays:
        if play.kind != 'card' or play.card_id not in (LOG, ICE_GOLEM) or play.grid is None:
            continue
        snapshot = _snapshot_at(frames, play.lands + 1)
        if snapshot is None:
            continue
        x0, y0 = play.grid[0] * 1000 + 500, play.grid[1] * 1000 + 500
        enemies = [o for o in snapshot.get('objects') or () if int(o['owner']) != play.owner and o.get('hp') is not None]
        towers = [o for o in enemies if int(o['nativeObjectId']) in TOWER_IDS]
        units = [o for o in enemies if int(o['nativeObjectId']) not in TOWER_IDS]
        who = label.get(play.owner, f'side {play.owner}')
        if play.card_id == LOG:
            direction = 1 if play.owner == 0 else -1

            def in_path(o) -> bool:
                along = (int(o['y']) - y0) * direction
                return (abs(int(o['x']) - x0) <= LOG_HALF_WIDTH + UNIT_MARGIN
                        and -UNIT_MARGIN <= along <= LOG_LENGTH + UNIT_MARGIN)
            hit = [o for o in units if in_path(o)]
            if any(int(o['cardId']) not in flying for o in hit):
                verdict = 'ground'
            elif hit:
                verdict = 'air only'
            elif any(in_path(o) for o in towers):
                verdict = 'tower only'
            else:
                verdict = 'nothing'
            rows.append((who, 'Log', verdict))
        else:
            near = [o for o in units if ((int(o['x']) - x0) ** 2 + (int(o['y']) - y0) ** 2) ** 0.5 <= GOLEM_RADIUS]
            if not near:
                verdict = 'none near'
            elif all(int(o['cardId']) in building_only for o in near):
                verdict = 'building-targeters only'
            else:
                verdict = 'pullable'
            rows.append((who, 'Ice Golem', verdict))
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', nargs='+', type=Path)
    args = parser.parse_args(argv)
    flying, building_only = _unit_facts()
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for directory in args.dirs:
        for path in sorted(directory.glob('frames/*/*.jsonl.zst')):
            for who, card, verdict in classify(path, flying, building_only):
                counts[(who, card)][verdict] += 1
    for (who, card), counter in sorted(counts.items()):
        total = sum(counter.values())
        shares = ', '.join(f'{verdict} {n} ({100 * n / total:.0f}%)' for verdict, n in counter.most_common())
        print(f'{who:<60} {card:<10} {total:4} plays: {shares}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
