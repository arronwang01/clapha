"""Deduce the opponent's hand and next card from their deck plus the cards we watch them play.

No hidden data: the deck comes from deck_vector_scan, the plays come from the command queue,
which carries the card id of every command either side sends. The cycle is deterministic --
a played card goes to the back of the queue and the front of the queue enters the hand -- so
brute-forcing all 8! = 40320 initial orderings and keeping the ones consistent with the
observed play sequence collapses the state quickly.

State convention: a permutation of the 8 deck slots, first 4 = starting hand, last 4 = queue
in order. Playing a slot removes it from the hand, appends it to the queue, and the queue's
front moves into the hand.

    python3 mac012/cycle_tracker.py            # live, follows the current battle
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_profile import ADB, CLAPHA, MANAGER_RVA, ROOT_CONTEXT_OFFSET, SERIAL, apply  # noqa: E402

apply()
CARDS = {c['card_id']: c['display_name']
         for c in json.loads((CLAPHA / 'live_card_catalog.json').read_text())['cards']}


def sh(command: str) -> str:
    return subprocess.run([str(ADB), '-s', SERIAL, 'shell', command],
                          capture_output=True, text=True, timeout=60).stdout


def pid() -> str:
    return sh('pidof com.supercell.clashroyale').strip()


def parse(text: str):
    start = text.find('{')
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return None


class Tracker:
    """All initial orderings still consistent with the plays seen so far."""

    def __init__(self, deck: list[int]):
        self.deck = deck                       # card id by slot
        self.states = list(itertools.permutations(range(8)))

    @staticmethod
    def step(order: tuple[int, ...], slot: int):
        hand, queue = list(order[:4]), list(order[4:])
        if slot not in hand:
            return None
        hand[hand.index(slot)] = queue.pop(0)
        queue.append(slot)
        return tuple(hand + queue)

    def observe(self, card_id: int) -> int:
        """Filter states by a played card. Returns how many remain."""
        slots = [i for i, c in enumerate(self.deck) if c == card_id]
        if not slots:
            return len(self.states)
        survivors, advanced = [], []
        for order in self.states:
            for slot in slots:
                if slot in order[:4]:
                    nxt = self.step(order, slot)
                    if nxt:
                        survivors.append(order)
                        advanced.append(nxt)
                    break
        self.states = advanced
        self._initial = survivors
        return len(self.states)

    def hand(self) -> tuple[list[str], bool]:
        sets = {tuple(sorted(o[:4])) for o in self.states}
        certain = len(sets) == 1
        if certain:
            return [CARDS.get(self.deck[s], str(self.deck[s])) for s in sorted(next(iter(sets)))], True
        counts = {}
        for o in self.states:
            for s in o[:4]:
                counts[s] = counts.get(s, 0) + 1
        total = len(self.states) or 1
        likely = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        return [f'{CARDS.get(self.deck[s], s)} {100*n//total}%' for s, n in likely], False

    def next_card(self) -> tuple[str, bool]:
        fronts = {o[4] for o in self.states}
        if len(fronts) == 1:
            slot = next(iter(fronts))
            return CARDS.get(self.deck[slot], str(self.deck[slot])), True
        counts = {}
        for o in self.states:
            counts[o[4]] = counts.get(o[4], 0) + 1
        total = len(self.states) or 1
        slot, n = max(counts.items(), key=lambda kv: kv[1])
        return f'{CARDS.get(self.deck[slot], slot)} ({100*n//total}%)', False


def opponent_deck(process: str, revealed: list[int]) -> list[int] | None:
    """The opponent's deck is the scanned 8-card group containing every card they have
    revealed. With nothing revealed yet this is ambiguous, so wait for a play or two."""
    scan = parse(sh(f'/data/local/tmp/deck_vector_scan {process}'))
    if not scan or not revealed:
        return None
    need = set(revealed)
    groups = {tuple(h['cards']) for h in scan['hits']}
    matches = [g for g in groups if need <= set(g)]
    return list(matches[0]) if len(matches) == 1 else None


def main() -> int:
    process = pid()
    if not process:
        print('game not running')
        return 2
    tracker = None
    seen_seq = -1
    plays: list[int] = []
    print('watching... (ctrl-c to stop)')
    while True:
        probe = parse(sh(f'/data/local/tmp/queue_probe {process} {hex(MANAGER_RVA)} '
                         f'{hex(ROOT_CONTEXT_OFFSET)} 1 100'))
        if not probe or not probe.get('accounts') or probe['accounts'][0] is None:
            time.sleep(1)
            continue
        local = next((a for a in probe['accounts'] if a and a['lo'] == 79227807), None)
        side = local['side'] if local else 1
        revealed = probe['revealed'][1 - side]

        if tracker is None:
            deck = opponent_deck(process, revealed)
            if deck:
                tracker = Tracker(deck)
                print('opponent deck:', [CARDS.get(c, c) for c in deck])
                for card in revealed:            # replay what we already saw
                    tracker.observe(card)
                    plays.append(card)
                print(f'  replayed {len(plays)} known plays -> {len(tracker.states)} orderings left')
            else:
                time.sleep(1)
                continue

        for card in revealed[len(plays):]:
            left = tracker.observe(card)
            plays.append(card)
            hand, sure = tracker.hand()
            nxt, nsure = tracker.next_card()
            print(f'play {len(plays)}: {CARDS.get(card, card):<16} -> {left:>6} orderings left')
            print(f'   hand {"(certain)" if sure else "(likely)"}: {hand}')
            print(f'   next {"(certain)" if nsure else "(likely)"}: {nxt}')
        time.sleep(0.5)


if __name__ == '__main__':
    raise SystemExit(main())
