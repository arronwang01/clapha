"""Opponent landings that outlive their command, for the MuMu overlay.

A command marker (console.pending_commands) disappears when the command leaves the queue, which
is when the game executes it. For most troops that is when the unit appears. Three kinds of
play are still on their way after that:

  * spells that fly (Fireball, Arrows, Rocket, Log, Snowball, ...): their range stays on the
    target until the projectile that appeared after the play is gone. A spell with no projectile
    within 0.6 s of executing was instant (Zap, Poison, Freeze, ...) and is done.
  * Miner, Goblin Drill, Goblin Barrel: the destination marker stays until an opponent unit that
    appeared after the play stands within 1.5 tiles of the target (the Miner or Drill surfacing,
    the Barrel's goblins), or 7 s have passed.

Only the destination is marked, never the path (the owner's choice).
"""
from __future__ import annotations

import json
from pathlib import Path

TRAVELLERS = {26000032: 'Miner', 27000013: 'Goblin Drill', 28000004: 'Goblin Barrel'}
ARRIVAL_RADIUS = 1500          # native units: 1.5 tiles
TRAVELLER_TIMEOUT = 140        # ticks (7 s)
SPELL_TIMEOUT = 100            # ticks (5 s)
INSTANT_AFTER = 12             # ticks (0.6 s) with no projectile -> the spell was instant

_CARDS = {c['card_id']: c for c in json.loads(
    (Path(__file__).resolve().parents[1] / 'live_card_catalog.json').read_text())['cards']}
# Spell radius in tiles (FirstLight's card specs, 15.535), and which spells fly to the target.
SPELL_RADIUS = {28000000: 2.5, 28000001: 3.5, 28000002: 3.0, 28000003: 2.0, 28000004: 1.5,
                28000005: 3.0, 28000007: 3.5, 28000008: 2.5, 28000009: 3.5, 28000010: 4.0,
                28000011: 1.95, 28000012: 5.5, 28000013: 3.0, 28000014: 3.5, 28000015: 1.3,
                28000016: 1.5, 28000017: 2.5, 28000018: 3.0, 28000023: 2.5, 28000024: 3.0,
                28000026: 2.5}
FLYING_SPELLS = {28000000, 28000001, 28000003, 28000004, 28000011, 28000015, 28000017}


def is_spell(card_id: int) -> bool:
    return (_CARDS.get(int(card_id)) or {}).get('type') == 'spell'


class LandingTracker:
    """Fed every /state poll; returns the landings still on their way."""

    def __init__(self) -> None:
        self.battle = None
        self.first_seen: dict[str, int] = {}     # object address -> first tick seen
        self.active: dict[tuple, dict] = {}
        self.done: set[tuple] = set()

    def update(self, battle, tick: int, opponent: int, plays, entities) -> list[dict]:
        if battle != self.battle:
            self.__init__()
            self.battle = battle
        live = {}
        for e in entities or ():
            address = str(e.get('address') or '')
            if not address or e.get('side') != opponent:
                continue
            self.first_seen.setdefault(address, tick)
            live[address] = e
        for play in plays or ():
            if play.get('kind', 'card') != 'card' or play.get('side') != opponent:
                continue
            card = int(play['card_id'])
            key = (play.get('issue_tick'), play.get('seq'), card)
            if key in self.done or key in self.active:
                continue
            if card in TRAVELLERS or card in FLYING_SPELLS:
                self.active[key] = {'card_id': card, 'form': int(play.get('form_code') or 0),
                                    'x': play.get('x'), 'y': play.get('y'),
                                    'executed': int(play['tick']), 'projectile_seen': False,
                                    'kind': 'traveller' if card in TRAVELLERS else 'spell'}
            else:
                self.done.add(key)
        out = []
        for key, landing in list(self.active.items()):
            if self._finished(landing, tick, live):
                del self.active[key]
                self.done.add(key)
                continue
            out.append(dict(landing))
        return out

    def _new_since(self, address: str, executed: int) -> bool:
        return self.first_seen.get(address, -10 ** 9) >= executed - 2

    def _finished(self, landing: dict, tick: int, live: dict) -> bool:
        age = tick - landing['executed']
        if landing['x'] is None or landing['y'] is None:
            return True
        if landing['kind'] == 'spell':
            flying = [a for a, e in live.items()
                      if e.get('kind') == 0 and self._new_since(a, landing['executed'])]
            if flying:
                landing['projectile_seen'] = True
                return age > SPELL_TIMEOUT
            return landing['projectile_seen'] or age > INSTANT_AFTER
        if age > TRAVELLER_TIMEOUT:
            return True
        for address, e in live.items():
            if e.get('kind') == 0 or not self._new_since(address, landing['executed']):
                continue
            dx, dy = e['x'] - landing['x'], e['y'] - landing['y']
            if dx * dx + dy * dy <= ARRIVAL_RADIUS ** 2:
                return True
        return False
