"""Read tap_probe's records: every touch against the game's own events, on the device's clock.

    ./py mac012/tap_probe_report.py build/tap_probe/<time>.jsonl [more files]

Whether the game took a gesture is decided here from the whole queue record, not from what the probe saw at
the time: a command first shows in the queue 1-12 ticks after its issue tick, so a probe that looks for 0.7 s
calls some taken plays refused. A gesture for a card was taken when the queue ever held a command of that card
whose issue tick is the tick the gesture's last touch went up in, or up to six ticks later.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TICK_US = 50_000
_IDS: dict[str, int] = {}


def card_id(name: str) -> int | None:
    if not _IDS:
        import viewer as V
        for card, row in V.CARDS.items():
            _IDS.setdefault(str(row.get('name')), int(card))
    return _IDS.get(name)


class Battle:
    """One probe file: its records, and the queue's commands still unclaimed by a gesture."""

    def __init__(self, path: str) -> None:
        self.rows = []
        for line in open(path, encoding='utf-8'):
            try:
                self.rows.append(json.loads(line))
            except ValueError:
                pass
        self.start = next((r for r in self.rows if r['k'] == 'start'), {})
        self.commands = sorted((r for r in self.rows if r['k'] == 'cmd'), key=lambda r: r['seen_us'])
        self.used: set[int] = set()
        self._clock = None
        self.hands = [r for r in self.rows if r['k'] == 'hand']
        self.trials = [r for r in self.rows if r['k'] == 'trial']
        self.flip = self.start.get('side') == 0 and not self.start.get('layout_fix')

    def tick_at(self, us: int) -> float:
        """The game's tick at a device time: from the nearest record that carries both (a tick's first sample, a
        hand change, a command's first sight), 50 ms a tick from there."""
        if self._clock is None:
            import bisect
            marks = sorted({(r['us'], r['tick']) for r in self.rows if r['k'] in ('tick', 'hand') and r.get('tick', 0) > 0}
                           | {(r['seen_us'], r['seen_tick']) for r in self.commands if r.get('seen_tick')})
            self._clock = (bisect, [u for u, _t in marks], marks)
        bisect, times, marks = self._clock
        index = bisect.bisect_right(times, us)
        near = [marks[i] for i in (index - 1, index) if 0 <= i < len(marks)]
        mark_us, mark_tick = min(near, key=lambda m: abs(m[0] - us))
        return mark_tick + (us - mark_us) / TICK_US

    def command(self, name: str, ack: dict | None) -> dict | None:
        """The command the gesture `ack` produced for the card `name`, if the game took it: one of that card
        whose issue tick is the tick its last touch went up in, or up to six later (measured: +0 to +4)."""
        if not ack or not ack.get('t'):
            return None
        tick = self.tick_at(ack['t'][-1])
        wanted = card_id(name)
        best = None
        for index, c in enumerate(self.commands):
            if index in self.used or c['card'] != wanted or not -2 <= c['issue'] - tick <= 6.5:
                continue
            if best is None or abs(c['issue'] - tick) < abs(self.commands[best]['issue'] - tick):
                best = index
        if best is None:
            return None
        self.used.add(best)
        return self.commands[best]

    def tile(self, tile):
        """The tile a gesture asked for, in the game's own coordinates (records made before the probe took
        mac012/layout_fix had side 0's columns turned: 17 - column)."""
        column, row = tile
        return (17 - column, row) if self.flip else (column, row)


def deal_events(hands: list[dict]):
    """Every change of one hand slot: when it was first seen, the sample before, the tick, the slot, old, new."""
    events = []
    for before, after in zip(hands, hands[1:]):
        if len(before['hand']) != 4 or len(after['hand']) != 4 or after['tick'] < before['tick']:
            continue
        for position in range(4):
            if before['hand'][position] != after['hand'][position]:
                events.append({'us': after['us'], 'prev_us': after['prev_us'], 'tick': after['tick'], 'position': position,
                               'old': before['hand'][position], 'new': after['hand'][position]})
    return events


def ms(us) -> float:
    return round(us / 1000.0, 1)


def report_deal(battles) -> None:
    out = []
    after_issue = Counter()
    for b in battles:
        events = deal_events(b.hands)
        for t in (r for r in b.trials if r.get('trial') == 'deal' and r.get('ok') and r.get('ack2')):
            touches = t['ack2'].get('t') or []
            if len(touches) < 4:
                continue
            deals = [e for e in events if e['position'] == t['position'] and e['new'] >= 0
                     and e['tick'] >= t['first_issue'] + 15 and e['us'] > t['ack1']['t0']]
            if not deals:
                continue
            deal = deals[0]
            after_issue[deal['tick'] - t['first_issue']] += 1
            taken = b.command(t['second'], t['ack2'])
            out.append({'down': ms(touches[0] - deal['us']), 'up': ms(touches[1] - deal['us']), 'taken': taken is not None,
                        'hold': t.get('hold', 16),
                        'card': t['second'], 'first': t['first'],
                        'issue': taken and taken['issue'] - deal['tick']})
    if not out:
        return
    print(f'\nDEAL: play a slot\'s card, then tap that slot for the card the game deals into it ({len(out)} trials)')
    print(f'  the game dealt the next card N ticks after the first card\'s issue tick: {dict(sorted(after_issue.items()))}')
    print('  the second tap\'s card touch went down this long after the first sample that had the new card in the hand')
    print('  (samples every ~20 ms: the deal itself is up to 20 ms earlier):')
    bins = [(-10 ** 6, 0, 'before the deal'), (0, 50, '0-49 ms'), (50, 60, '50-59'), (60, 70, '60-69'), (70, 80, '70-79'),
            (80, 90, '80-89'), (90, 100, '90-99'), (100, 120, '100-119'), (120, 10 ** 6, '120 ms and later')]
    for low, high, label in bins:
        sel = [r for r in out if low <= r['down'] < high]
        if sel:
            print(f'    {label:16s} taken {sum(r["taken"] for r in sel):3d}   refused {sum(not r["taken"] for r in sel):3d}')
    holds = sorted({r['hold'] for r in out})
    if len(holds) > 1:
        print('  by the length of that card touch (it is the touch going down that counts, not its end):')
        for hold in holds:
            sel = sorted((r for r in out if r['hold'] == hold and 0 <= r['down'] < 120), key=lambda r: r['down'])
            if hold != 16 and sel:
                print(f'    held {hold:2d} ms: ' + ', '.join(f"{r['down']:.0f}{'+' if r['taken'] else '-'}" for r in sel)
                      + '   (ms after the deal the touch went down; + taken, - refused)')
    issues = Counter(r['issue'] for r in out if r['taken'])
    print(f'  taken ones were issued N ticks after the deal\'s tick: {dict(sorted(issues.items()))}')


def report_pairs(battles) -> None:
    out = []
    for b in battles:
        for t in (r for r in b.trials if r.get('trial') == 'pairs' and r.get('ack1') and r.get('ack2')):
            if t.get('tick', 1000) < 110:
                continue                              # the first seconds of a battle refuse everything (START below)
            first, second = b.command(t['cards'][0], t['ack1']), b.command(t['cards'][1], t['ack2'])
            out.append((ms(t['ack2']['t'][0] - t['ack1']['t'][3]), first is not None, second is not None, t['cards']))
    if not out:
        return
    print(f'\nPAIRS: two cards long in the hand; the second gesture begins N ms after the first one\'s last touch went up ({len(out)} trials)')
    for low, high in ((0, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, 40), (40, 1000)):
        sel = [r for r in out if low <= r[0] < high]
        if sel:
            print(f'    {low:3d}-{high:<4d} ms: {len(sel):2d} pairs   both taken {sum(r[1] and r[2] for r in sel):2d}   '
                  f'second lost {sum(r[1] and not r[2] for r in sel)}   first lost {sum(r[2] and not r[1] for r in sel)}   '
                  f'both lost {sum(not r[1] and not r[2] for r in sel)}')


def report_shape(battles) -> None:
    table = defaultdict(lambda: [0, 0])
    for b in battles:
        for t in (r for r in b.trials if r.get('trial') == 'shape' and r.get('ack')):
            if t.get('tick', 1000) < 110 or t['card'] == 'Cannon':
                continue
            taken = b.command(t['card'], t['ack'])
            table[(t['hold'], t['gap'])][0] += 1
            table[(t['hold'], t['gap'])][1] += taken is not None
    if not table:
        return
    print('\nSHAPE: one troop or spell long in the hand; each touch held `hold` ms, `gap` ms between the card touch and the tile touch')
    for (hold, gap), (n, ok) in sorted(table.items()):
        print(f'    hold {hold:3d} ms  gap {gap:3d} ms   taken {ok}/{n}')


def report_select(battles) -> None:
    lines = Counter()
    meaning = {'second': 'card A touched, then card B, then a tile', 'tile': 'a tile touched, no card first',
               'twice': 'card A touched twice, then a tile'}
    for b in battles:
        for t in (r for r in b.trials if r.get('trial') == 'select'):
            last = (t.get('acks') or [None])[-1]
            a, c = b.command(t['cards'][0], last), b.command(t['cards'][1], last)
            what = meaning.get(t['case'], 'card A touched, the tile ' + t['case'].split()[-1] + ' ms later')
            lines[(t['case'], what, 'A put down' if a else 'B put down' if c else 'nothing put down')] += 1
    if not lines:
        return
    print('\nSELECT: the touches of a play sent apart')
    for (case, what, result), n in sorted(lines.items()):
        print(f'    {what:48s} {result:18s} x{n}')


def report_early(battles) -> None:
    table = defaultdict(lambda: [0, 0])
    for b in battles:
        for t in (r for r in b.trials if r.get('trial') == 'early'):
            taken = b.command(t['second'], (t.get('acks') or [None])[-1])
            table[t['before_ms']][0] += 1
            table[t['before_ms']][1] += taken is not None
    if not table:
        return
    print('\nEARLY: the slot touched before its next card is dealt (no tile), a tile touched 300 ms after the deal')
    for before, (n, ok) in sorted(table.items()):
        print(f'    slot touched {before:3d} ms before the deal: the dealt card was put down {ok}/{n}')


def report_places(battles) -> None:
    for name, title in (('illegal', 'ILLEGAL: a card asked onto a tile it may not take'),
                        ('build', 'BUILD: a building asked next to a tower'),
                        ('occupied', 'OCCUPIED: a building asked on a tile (8, 9), free or with our last one still standing on it')):
        lines = Counter()
        for b in battles:
            for t in (r for r in b.trials if r.get('trial') == name and r.get('ack')):
                want = b.tile(t['tile'])
                taken = b.command(t['card'], t['ack'])
                got = taken and (round(taken['x'] / 1000 - 0.5), round(taken['y'] / 1000 - 0.5))
                label = t.get('case') or ('standing' if t.get('on_a_standing_building') else 'free' if name == 'occupied' else '')
                result = 'REFUSED' if not taken else ('put there' if got == tuple(want) else f'moved to {got}')
                lines[(label, t['card'], tuple(want), result)] += 1
        if lines:
            print(f'\n{title}; tiles (column, row) in the game\'s own coordinates')
            for (label, card, want, result), n in sorted(lines.items()):
                print(f'    {label:9s} {card:12s} asked {str(want):9s} -> {result}' + (f'  x{n}' if n > 1 else ''))
    after = Counter()
    for b in battles:
        for t in (r for r in b.trials if r.get('trial') == 'occupied'):
            refused = b.command(t['card'], t['ack']) is None      # (claimed above already when taken: None here too)
            for index, play in enumerate(t.get('after') or ()):
                after[(index + 1, b.command(play['card'], play['ack']) is not None)] += 1
    if after:
        print('  plain plays right after a building the probe saw no command for (it was in fact taken, a little later):')
        for (index, ok), n in sorted(after.items()):
            print(f'    play {index} after it: {"taken" if ok else "not taken"} x{n}')


def report_burst(battles) -> None:
    lines = []
    for b in battles:
        for t in (r for r in b.trials if r.get('trial') == 'burst'):
            taken = [b.command(card, ack) is not None for card, ack in zip(t['cards'], t['acks'])]
            lines.append((t['cards'], t['costs'], taken, t['apart_ms']))
    if not lines:
        return
    print('\nBURST: the whole hand played from a full bar (10), cheapest first, the gestures a moment apart')
    for cards, costs, taken, apart in lines:
        text = ', '.join(f'{c} ({int(k)}) {"taken" if ok else "REFUSED"}' for c, k, ok in zip(cards, costs, taken))
        print(f'    {apart} ms apart: {text}')


def report_start(battles) -> None:
    lines = []
    for b in battles:
        for t in (r for r in b.trials if r.get('trial') == 'start'):
            for tick, card, ack in zip(t['ticks'], t['cards'], t['acks']):
                taken = b.command(card, ack)
                lines.append((tick, card, taken and taken['issue']))
        # the first plain trials of a battle the probe met early
        for t in b.trials[:4]:
            if t.get('trial') == 'deal' and not t.get('ok') and t.get('ack1'):
                pass
    if not lines:
        return
    print('\nSTART: one play at each of a few ticks right after a battle begins')
    for tick, card, issue in sorted(lines):
        print(f'    touched at tick {tick:3d} ({card}): ' + (f'taken, issued at tick {issue}' if issue else 'refused'))


def report_queue(battles) -> None:
    lag = Counter()
    for b in battles:
        for c in b.commands:
            if c.get('seen_tick') is not None and 0 <= c['seen_tick'] - c['issue'] <= 30:
                lag[c['seen_tick'] - c['issue']] += 1
    if lag:
        total = sum(lag.values())
        print(f'\nQUEUE: ticks from a command\'s issue tick to the tick it first showed in the queue ({total} of our commands)')
        print('    ' + ', '.join(f'{k}: {v}' for k, v in sorted(lag.items())))


def report_issue(battles) -> None:
    """A taken play's issue tick against the tick its tile touch went up in (files that carry the tick clock)."""
    lags = Counter()
    for b in battles:
        if not any(r['k'] == 'tick' for r in b.rows):
            continue
        fresh = Battle.__new__(Battle)
        fresh.rows, fresh.commands, fresh.used, fresh._clock = b.rows, b.commands, set(), None
        for t in b.trials:
            pairs = []
            if t.get('trial') in ('cycle', 'shape') and t.get('ack'):
                pairs.append((t['card'], t['ack']))
            if t.get('trial') == 'deal' and t.get('ok'):
                pairs.append((t['first'], t['ack1']))
            for card, ack in pairs:
                command = fresh.command(card, ack)
                if command is not None and len(ack.get('t') or []) >= 4:
                    lags[command['issue'] - int(fresh.tick_at(ack['t'][3]))] += 1
    if lags:
        print('\nISSUE TICK: a taken play\'s issue tick, less the tick its tile touch went up in')
        print('    ' + ', '.join(f'{k:+d}: {v}' for k, v in sorted(lags.items())))


def report_game_deal(battles) -> None:
    empty = Counter()
    for b in battles:
        last: dict[int, dict] = {}
        for event in deal_events(b.hands):
            before = last.get(event['position'])
            if event['new'] >= 0:
                # dealt: how long the slot had been seen empty (0: card to card within one sample)
                empty[event['tick'] - before['tick'] if before and before['new'] < 0 else 0] += 1
            last[event['position']] = event
    if empty:
        print('\nTHE SLOT IN THE GAME\'S MEMORY: ticks it was seen empty before its deal (0: card to card in one sample)')
        print('    ' + ', '.join(f'{k}: {v}' for k, v in sorted(empty.items())))


def main(argv: list[str]) -> int:
    battles = [Battle(path) for path in argv]
    for b in battles:
        if b.start:
            print(f"{Path(argv[battles.index(b)]).name}: side {b.start.get('side')}, from tick {b.start.get('tick')}, "
                  f"{len(b.trials)} trials, experiments {b.start.get('experiments')}")
    report_start(battles)
    report_deal(battles)
    report_early(battles)
    report_pairs(battles)
    report_shape(battles)
    report_select(battles)
    report_burst(battles)
    report_places(battles)
    report_queue(battles)
    report_issue(battles)
    report_game_deal(battles)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
