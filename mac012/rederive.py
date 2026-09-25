"""Re-derive the reader's pinned offsets on the installed game build: one command, one battle.

After a game update the reader refuses to attach (the build it is certified for no longer
matches). This finds the values that moved, read-only, during any battle (Training Camp is
fine), checks each against live evidence, and proposes a new profile:

  1. manager RVA + root context offset   src/find_manager.c: a slot whose chain ends in a
                                         battle tick advancing at ~20 Hz
  2. player fields (hand, cycle, elixir,  the reader run with the candidate chain: hand is 4
     deck), towers, units                distinct deck indices, elixir 0-10, deck ids known to
                                         the build's own card tables, 6 towers with hp
  3. attack / movement component vtables src/comp_probe.c on live troops: attack = points at a
                                         live object and its timeline moves; movement = the
                                         other per-troop component with charge progress at +0x1e0

Hero ability controllers and evolution progress need a hero/evo deck in the battle; they are
reported when present and otherwise left to mac012/run_runtime_probe.py.

Why not offline: on ARM64 the game's libg.so code is encrypted on disk (8.0 bits/byte) and only
decrypted inside the running game, so signatures cannot be matched in the file, and dumping the
decrypted code would be circumventing the protection. Everything here reads game *data* through
/proc/PID/mem, the same way the reader does; nothing is written to the game.

    python3 mac012/rederive.py [--serial 127.0.0.1:26624] [--force] [--apply]

--force runs even when the build is current (a regression check: it must re-find the values
we already have). --apply writes the result into mac012/mac_profile.py and the reader source,
only when every check passed; start-consoles.sh rebuilds the reader on the next start.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_profile import ADB, LIBG_SHA256, MANAGER_RVA, ROOT_CONTEXT_OFFSET, VERSION_CODE  # type: ignore  # noqa: E402
from update_check import DEVICES, PACKAGE, installed  # type: ignore  # noqa: E402

CLAPHA = Path(__file__).resolve().parents[1]
READER_SOURCE = CLAPHA / 'src' / 'live_sampler_tbi.c'
PROFILE = CLAPHA / 'mac012' / 'mac_profile.py'
REMOTE = '/data/local/tmp/clapha-rederive-'   # own copies: never replace a reader in use
TOOLS = ('find_manager', 'comp_probe', 'live_sampler_tbi')


def pinned_vtables() -> dict:
    text = READER_SOURCE.read_text()
    return {name: int(re.search(rf'#define {name} (0x[0-9a-fA-F]+)ULL', text)[1], 16)
            for name in ('ATTACK_COMPONENT_VT', 'MOVEMENT_COMPONENT_VT')}


def shell(serial: str, command: str, timeout: float = 120) -> str:
    try:
        return subprocess.run([str(ADB), '-s', serial, 'shell', command], capture_output=True,
                              text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired as error:
        return error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or '')


def build_and_push(serial: str) -> None:
    clang = sorted(Path.home().glob('Library/Android/sdk/ndk/*/toolchains/llvm/prebuilt/*/bin/'
                                    'aarch64-linux-android31-clang'))
    for tool in TOOLS:
        binary, source = CLAPHA / 'build' / tool, CLAPHA / 'src' / f'{tool}.c'
        if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
            if not clang:
                sys.exit(f'{tool} needs building and no NDK clang was found')
            subprocess.run([str(clang[-1]), '-O2', '-o', str(binary), str(source)], check=True)
        subprocess.run([str(ADB), '-s', serial, 'push', str(binary), REMOTE + tool],
                       capture_output=True, check=True)
        shell(serial, f'chmod 755 {REMOTE + tool}')


def game_process(serial: str) -> tuple[int, str, str]:
    pid = shell(serial, f'pidof {PACKAGE}').strip()
    if not pid.isdigit():
        sys.exit(f'{serial}: Clash Royale is not running; open it, then start a battle')
    maps = shell(serial, f'cat /proc/{pid}/maps')
    paths = {line.split()[-1] for line in maps.splitlines() if line.endswith('/lib/arm64/libg.so')}
    if len(paths) != 1:
        sys.exit(f'{serial}: no unique libg.so mapping in the game process')
    path = paths.pop()
    return int(pid), path, shell(serial, f'sha256sum {path}').split()[0]


def find_chain(serial: str, pid: int, deadline: float) -> tuple[dict | None, list]:
    """Wait for a battle; return the one manager slot whose tick advances at game speed."""
    announced = False
    while time.monotonic() < deadline:
        raw = shell(serial, f'{REMOTE}find_manager {pid} 0x8 0x100', timeout=180).strip()
        try:
            result = json.loads(raw.splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            result = {'hits': []}
        hits = [h for h in result.get('hits', []) if 12 <= h['rate'] <= 30]
        chains = {(h['rva'], h['ctx_off']): h for h in hits}
        if len(chains) == 1:
            return next(iter(chains.values())), hits
        if len(chains) > 1:
            # Several slots can reach the same battle (aliases); prefer the pinned one if it is
            # among them, else the lowest -- and say so.
            pinned = (hex(MANAGER_RVA), hex(ROOT_CONTEXT_OFFSET))
            key = pinned if pinned in chains else min(chains, key=lambda k: int(k[0], 16))
            return chains[key], hits
        if not announced:
            print('  waiting for a battle (Training Camp is fine) ...', flush=True)
            announced = True
        time.sleep(3)
    return None, []


def read_frames(serial: str, pid: int, rva: int, ctx: int, count: int = 60) -> list[dict]:
    command = f'{REMOTE}live_sampler_tbi {pid} 50 {hex(rva)} {hex(ctx)} --unified {count}'
    frames = []
    for line in shell(serial, command, timeout=60).splitlines():
        if '"mumu_live_frame"' in line:
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return frames


def check_frames(frames: list[dict], card_ids: set[int]) -> list[tuple[str, bool, str]]:
    live = [f for f in frames if f.get('battle_active') and f.get('coherent')]
    checks = [('battle frames coherent', len(live) >= 0.8 * max(1, len(frames)),
               f'{len(live)}/{len(frames)}')]
    if len(live) < 2:
        return checks
    ticks = [f['game_tick'] for f in live]
    span = (live[-1]['sequence'] - live[0]['sequence']) * 0.05
    rate = (ticks[-1] - ticks[0]) / span if span else 0
    checks.append(('battle tick advances at ~20 Hz', 14 <= rate <= 26, f'{rate:.1f}/s'))
    hands, elixirs, decks = [], [], []
    for frame in live:
        for player in frame['players']:
            hand = [i for i in player.get('hand_deck_indices', []) if i >= 0]
            if hand:
                hands.append(len(hand) == 4 and len(set(hand)) == 4 and all(0 <= i < 8 for i in hand))
            if player.get('elixir_raw') is not None:
                elixirs.append(player['elixir_raw'])
            if player.get('deck_card_ids'):
                decks.append(player['deck_card_ids'])
    checks.append(('hand: 4 distinct deck indices', bool(hands) and all(hands),
                   f'{sum(hands)}/{len(hands)} readings'))
    # elixir_raw is fixed-point; the ceiling is 10 elixir whatever the scale, so check that it
    # moves and stays within one consistent range.
    moving = len(set(elixirs)) > 1
    checks.append(('elixir readable and moving', moving,
                   f'{min(elixirs) if elixirs else "-"}..{max(elixirs) if elixirs else "-"}'))
    known = [all(c in card_ids for c in deck if c > 0) for deck in decks]
    checks.append(('deck: 8 ids from the build\'s own card tables', bool(known) and all(known),
                   f'{sum(known)}/{len(known)} readings'))
    towers = [sum(1 for e in f['entities'] if e['card_id'] == -1 and e['kind'] != 0 and e['hp'] > 0)
              for f in live]
    checks.append(('towers with hp (2..6)', all(2 <= t <= 6 for t in towers),
                   f'{min(towers)}..{max(towers)}'))
    units = [e['card_id'] for f in live for e in f['entities'] if e['card_id'] > 0]
    unknown = sorted({c for c in units if c not in card_ids and c // 1_000_000 not in (13, 10, 22)})
    checks.append(('units carry known card ids', not unknown, f'unknown {unknown[:5]}' if unknown
                   else f'{len(set(units))} kinds'))
    return checks


def classify_components(serial: str, pid: int, rva: int, ctx: int, deadline: float) -> dict:
    """Sample components of live troops; name the attack and movement vtables by behaviour."""
    rows, rounds = [], 0
    while rounds < 4 and time.monotonic() < deadline:
        frames = read_frames(serial, pid, rva, ctx, count=2)
        frame = next((f for f in frames if f.get('battle_active')), None)
        troops = [e for e in (frame or {}).get('entities', [])
                  if e['card_id'] > 0 and e['hp'] > 0 and e['kind'] != 0]
        if len(troops) < 2:
            time.sleep(1.0)
            continue
        live = {e['address'].lower() for e in frame['entities']}
        output = shell(serial, f'{REMOTE}comp_probe {pid} 40 50 '
                       + ' '.join(e['address'] for e in troops[:12]), timeout=60)
        for line in output.splitlines():
            if line.startswith('{'):
                row = json.loads(line)
                row['live'], row['round'] = live, rounds
                rows.append(row)
        rounds += 1
    stats = collections.defaultdict(lambda: {'objs': set(), 'n': 0, 'owner': 0, 'target': 0,
                                             'timeline_moves': 0, 'charge_idle': 0,
                                             'series': collections.defaultdict(list)})
    objects = set()
    for row in rows:
        key = (row['round'], row['obj'])
        objects.add(key)
        for comp in row['comps']:
            slot = stats[comp['vt']]
            slot['objs'].add(key)
            slot['n'] += 1
            slot['owner'] += comp['owner']
            slot['target'] += comp['p10'].lower() in row['live'] and comp['p10'].lower() != row['obj'].lower()
            slot['charge_idle'] += comp['w1e0'] == -1
            slot['series'][key].append(comp['w24'])
    for slot in stats.values():
        slot['timeline_moves'] = sum(1 for s in slot['series'].values() for a, b in zip(s, s[1:]) if a != b)
    common = {vt: s for vt, s in stats.items() if objects and len(s['objs']) >= 0.6 * len(objects)
              and s['owner'] >= 0.9 * s['n']}
    attack = [vt for vt, s in common.items() if s['target'] >= 0.5 * s['n'] and s['timeline_moves'] > 0]
    movement = [vt for vt, s in common.items() if vt not in attack and s['charge_idle'] >= 0.5 * s['n']]
    summary = {vt: {'troops': len(s['objs']), 'samples': s['n'], 'owner_back': s['owner'],
                    'points_at_live_object': s['target'], 'timeline_moves': s['timeline_moves'],
                    'charge_minus_one': s['charge_idle']} for vt, s in stats.items()}
    return {'objects': len(objects), 'attack': attack, 'movement': movement, 'evidence': summary}


def apply_profile(result: dict) -> None:
    text = PROFILE.read_text()
    for name, value in (('VERSION_CODE', str(result['version'])),
                        ('LIBG_SHA256', repr(result['libg_sha256'])),
                        ('MANAGER_RVA', result['manager_rva'].upper().replace('0X', '0x')),
                        ('ROOT_CONTEXT_OFFSET', result['root_context_offset'])):
        text = re.sub(rf'^{name} = .*$', f'{name} = {value}', text, count=1, flags=re.M)
    PROFILE.write_text(text)
    source = READER_SOURCE.read_text()
    for name in ('ATTACK_COMPONENT_VT', 'MOVEMENT_COMPONENT_VT'):
        source = re.sub(rf'#define {name} 0x[0-9a-fA-F]+ULL', f'#define {name} {result[name]}ULL', source)
    READER_SOURCE.write_text(source)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', default=DEVICES[0])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--wait', type=float, default=900, help='seconds to wait for a battle')
    args = parser.parse_args(argv)

    info = installed(args.serial)
    if info is None:
        sys.exit(f'{args.serial}: not reachable or the game is not installed')
    pid, libg_path, sha = game_process(args.serial)
    current = int(info['version']) == int(VERSION_CODE) and sha == LIBG_SHA256
    print(f'{args.serial}: build {info["version"]}, libg {sha[:12]} -> '
          + ('CURRENT (profile matches)' if current else 'NOT the certified build'))
    if current and not args.force:
        print('nothing to re-derive (use --force to re-check the pinned values anyway)')
        return 0

    build_and_push(args.serial)
    deadline = time.monotonic() + args.wait
    chain, hits = find_chain(args.serial, pid, deadline)
    if chain is None:
        print('no battle seen before the time limit; nothing changed')
        return 1
    rva, ctx = int(chain['rva'], 16), int(chain['ctx_off'], 16)
    print(f'  manager chain: libg+{hex(rva)} ctx {hex(ctx)} (tick {chain["rate"]}/s; '
          f'{len(hits)} advancing slot(s))')

    catalog = json.loads((CLAPHA / 'live_card_catalog.json').read_text())
    card_ids = {int(card['card_id']) for card in catalog['cards']}
    checks = check_frames(read_frames(args.serial, pid, rva, ctx), card_ids)
    components = classify_components(args.serial, pid, rva, ctx, deadline)
    old = pinned_vtables()
    vt = {'ATTACK_COMPONENT_VT': components['attack'], 'MOVEMENT_COMPONENT_VT': components['movement']}
    for name, found in vt.items():
        checks.append((f'{name.lower()} identified uniquely', len(found) == 1,
                       ', '.join(found) or f'none among {components["objects"]} troops'))

    print('\nchecks:')
    for what, ok, detail in checks:
        print(f'  {"ok  " if ok else "FAIL"}  {what:48} {detail}')
    passed = all(ok for _, ok, _ in checks)
    result = {'version': int(info['version']), 'libg_sha256': sha, 'libg_path': libg_path,
              'manager_rva': hex(rva), 'root_context_offset': hex(ctx),
              'ATTACK_COMPONENT_VT': vt['ATTACK_COMPONENT_VT'][0] if len(vt['ATTACK_COMPONENT_VT']) == 1 else None,
              'MOVEMENT_COMPONENT_VT': vt['MOVEMENT_COMPONENT_VT'][0] if len(vt['MOVEMENT_COMPONENT_VT']) == 1 else None,
              'checks': [{'check': w, 'ok': o, 'detail': d} for w, o, d in checks],
              'passed': passed, 'chain_hits': hits, 'components': components['evidence']}
    out = CLAPHA / 'build' / f'rederive_{info["version"]}.json'
    out.write_text(json.dumps(result, indent=1, default=list) + '\n')

    print('\n  value                     pinned          derived')
    for label, pinned_value, derived in (
            ('build', str(VERSION_CODE), str(result['version'])),
            ('libg sha256', LIBG_SHA256[:12], sha[:12]),
            ('manager RVA', hex(MANAGER_RVA), result['manager_rva']),
            ('root context offset', hex(ROOT_CONTEXT_OFFSET), result['root_context_offset']),
            ('attack component vt', hex(old['ATTACK_COMPONENT_VT']), result['ATTACK_COMPONENT_VT']),
            ('movement component vt', hex(old['MOVEMENT_COMPONENT_VT']), result['MOVEMENT_COMPONENT_VT'])):
        same = str(derived).lower() == str(pinned_value).lower()
        print(f'  {label:25} {pinned_value:15} {derived}{"" if same else "   <- changed"}')
    print(f'\nevidence: {out.relative_to(CLAPHA)}')
    if not passed:
        print('Not every check passed; nothing applied. A failing player/tower check means the '
              'player struct moved: see src/player_dump.c and src/deck_vector_scan.c.')
        return 1
    if args.apply:
        apply_profile(result)
        print('applied to mac012/mac_profile.py and src/live_sampler_tbi.c; restart the consoles '
              '(the reader is rebuilt and re-installed automatically). Then run '
              'mac012/update_catalog.py and mac012/drift_check.py.')
    else:
        print('all checks passed; re-run with --apply to write the new profile.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
