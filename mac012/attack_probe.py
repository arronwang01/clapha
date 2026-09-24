"""Find the attack component on this build, read-only.

Waits for a battle, takes the troops on the board, samples their components at 20 Hz with
comp_probe, and reports per component type (vtable offset in libg):
  * how many troops carry it, and whether it points back at its owner (+0x08)
  * whether +0x10 points at another live object (a target)
  * how +0x20 / +0x24 / +0x28 move over time (a timer counts down and resets per hit)
  * +0x1e0 on the movement component (charge progress)

FirstLight's layout for Null's 15.535.13: attack component +0x10 target, +0x20 sequence stage,
+0x24 timeline, +0x28 load remaining; movement component +0x1e0 charge progress. The object
fields we share with them (+0x18 components, +0x7c/+0x80 position, +0x11c state) already match
this build, so the component fields are expected to as well -- this verifies it.

    CR_MUMU_SERIAL=127.0.0.1:26624 python3 mac012/attack_probe.py [ROUNDS]
"""
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_profile import ADB, SERIAL, apply  # type: ignore  # noqa: E402

apply()
from native_core.mumu_live_protocol import install_reader, start_reader, verify_runtime  # noqa: E402
from mac_profile import READER  # type: ignore  # noqa: E402

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = Path(__file__).resolve().parents[1] / 'build' / 'attack_probe.jsonl'


def one_frame(pid):
    process = start_reader(ADB, SERIAL, pid, interval_ms=50, max_frames=3)
    frame = None
    for line in process.stdout:
        if '"mumu_live_frame"' in line:
            frame = json.loads(line)
    return frame


def main():
    runtime = verify_runtime(ADB, SERIAL)
    install_reader(ADB, SERIAL, READER)
    pid = runtime['pid']
    print(f'waiting for a battle on {SERIAL} ...', flush=True)
    rows = []
    rounds = 0
    while rounds < ROUNDS:
        frame = one_frame(pid)
        if not frame or not frame.get('battle_active'):
            time.sleep(1.0)
            continue
        troops = [e for e in frame['entities'] if e['card_id'] > 0 and e.get('hp', 0) > 0]
        if len(troops) < 2:
            time.sleep(0.5)
            continue
        live = {e['address'].lower(): e for e in frame['entities']}
        addresses = [e['address'] for e in troops][:12]
        command = f'/data/local/tmp/comp_probe {pid} 60 50 ' + ' '.join(addresses)
        output = subprocess.run([str(ADB), '-s', SERIAL, 'shell', command],
                                capture_output=True, text=True, timeout=30).stdout
        cards = {e['address'].lower(): e['card_id'] for e in troops}
        for line in output.splitlines():
            if not line.startswith('{'):
                continue
            row = json.loads(line)
            row['card'] = cards.get(row['obj'].lower())
            row['round'] = rounds
            row['live'] = sorted(live)
            rows.append(row)
        rounds += 1
        print(f'round {rounds}/{ROUNDS}: {len(addresses)} troops sampled', flush=True)
    OUT.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    report(rows)


def report(rows):
    by_vt = collections.defaultdict(lambda: {'objs': set(), 'owner': 0, 'n': 0, 'target': 0,
                                             'series': collections.defaultdict(list)})
    for row in rows:
        live = set(row.get('live') or [])
        for comp in row['comps']:
            slot = by_vt[comp['vt']]
            slot['objs'].add((row['round'], row['obj']))
            slot['n'] += 1
            slot['owner'] += comp['owner']
            if comp['p10'].lower() in live and comp['p10'].lower() != row['obj'].lower():
                slot['target'] += 1
            key = (row['round'], row['obj'])
            slot['series'][key].append((comp['w20'], comp['w24'], comp['w28'], comp['w1e0']))
    print('\nper component type:')
    for vt, slot in sorted(by_vt.items(), key=lambda kv: -len(kv[1]['objs'])):
        changes = [0, 0, 0, 0]
        resets = [0, 0, 0, 0]
        for series in slot['series'].values():
            for a, b in zip(series, series[1:]):
                for k in range(4):
                    if a[k] != b[k]:
                        changes[k] += 1
                        if b[k] > a[k]:
                            resets[k] += 1
        sample = next(iter(slot['series'].values()))[:3]
        print(f"  vt {vt}: troops {len(slot['objs']):3}  owner-back {slot['owner']}/{slot['n']}  "
              f"points-at-live-object {slot['target']}/{slot['n']}")
        print(f"      changes  +0x20 {changes[0]:4}  +0x24 {changes[1]:4}  +0x28 {changes[2]:4}  "
              f"+0x1e0 {changes[3]:4}   (increases {resets})  e.g. {sample}")


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == '--report':
        report([json.loads(l) for l in OUT.read_text().splitlines() if l.strip()])
    else:
        main()
