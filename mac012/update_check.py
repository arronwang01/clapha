"""Is everything still current? One report for the game build, the reader, the card catalog,
the models and the reference repos.

Per device: the installed game build against the build the reader is certified for
(mac_profile: versionCode + libg SHA-256). When the game has updated, the new build is pulled
from that device (the user's own install; nothing is downloaded) into runtime/<version>/, its
tables are decoded, the card list is diffed, and the report says which pinned values must be
re-derived before the bot can run again, and with which probe.

    python3 mac012/update_check.py            (report only)
    python3 mac012/update_check.py --pull     (also pull + decode a new build if there is one)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_profile import ADB, LIBG_SHA256, VERSION_CODE  # type: ignore  # noqa: E402

CLAPHA = Path(__file__).resolve().parents[1]
DEVICES = ('127.0.0.1:26624', '127.0.0.1:26656')
PACKAGE = 'com.supercell.clashroyale'

# Everything pinned to one build, and how it is re-derived (all read-only, need a live battle
# unless noted).
PINNED = [
    ('manager RVA / root context (the reader chain)', 'src/find_manager.c (battle)'),
    ('player struct fields: hand, cycle, elixir, deck', 'src/player_dump.c, src/deck_vector_scan.c (battle)'),
    ('command queue layout', 'src/queue_probe.c (battle, both sides playing)'),
    ('troop component vtables: attack, movement', 'src/comp_probe.c + mac012/attack_probe.py (battle)'),
    ('hero ability controllers, evolution progress', 'src/runtime_probe.c + mac012/run_runtime_probe.py (battle, hero deck)'),
    ('card catalog', 'mac012/update_catalog.py (offline, from the pulled build)'),
    ('FirstLight drift (cards the models never saw)', 'mac012/drift_check.py (offline)'),
]


def adb(serial: str, *args: str) -> str:
    try:
        return subprocess.run([str(ADB), '-s', serial, *args], capture_output=True, text=True,
                              timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ''


def installed(serial: str) -> dict | None:
    text = adb(serial, 'shell', 'dumpsys', 'package', PACKAGE)
    code = next((line.split('versionCode=')[1].split()[0] for line in text.splitlines()
                 if 'versionCode=' in line), None)
    if code is None:
        return None
    paths = [line.replace('package:', '').strip()
             for line in adb(serial, 'shell', 'pm', 'path', PACKAGE).splitlines() if line.strip()]
    return {'version': code, 'paths': paths}


def libg_sha(apk: Path) -> str | None:
    try:
        with zipfile.ZipFile(apk) as archive:
            return hashlib.sha256(archive.read('lib/arm64-v8a/libg.so')).hexdigest()
    except (KeyError, OSError, zipfile.BadZipFile):
        return None


def pull(serial: str, version: str, paths: list[str]) -> Path:
    out = CLAPHA / 'runtime' / version / 'apks'
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = out / Path(path).name
        if not target.exists():
            adb(serial, 'pull', path, str(target))
    for apk in out.glob('*.apk'):
        with zipfile.ZipFile(apk) as archive:
            names = [n for n in archive.namelist() if n.startswith('assets/csv_logic/')]
            if names:
                archive.extractall(CLAPHA / 'runtime' / version / 'assets', names)
    return out


def repo_status(path: Path, github: str | None = None) -> str:
    if not (path / '.git').exists():
        if github is None:
            return 'not a git checkout'
        # A downloaded copy: say what upstream is at, so a newer release is not missed.
        latest = subprocess.run(['gh', 'api', f'repos/{github}/commits?per_page=1', '--jq',
                                 '.[0].sha[0:7] + " " + .[0].commit.committer.date[0:10] + " "'
                                 ' + (.[0].commit.message | split("\\n")[0])'],
                                capture_output=True, text=True, timeout=60).stdout.strip()
        return (f'downloaded copy (no git); upstream latest: {latest}' if latest
                else 'downloaded copy (no git); upstream not reachable')
    subprocess.run(['git', '-C', str(path), 'fetch', '-q'], capture_output=True, timeout=60)
    behind = subprocess.run(['git', '-C', str(path), 'rev-list', '--count', 'HEAD..@{u}'],
                            capture_output=True, text=True).stdout.strip()
    head = subprocess.run(['git', '-C', str(path), 'log', '-1', '--format=%h %cs'],
                          capture_output=True, text=True).stdout.strip()
    return f'at {head}; ' + (f'{behind} new upstream commit(s) -- review before pulling'
                             if behind not in ('', '0') else 'up to date with upstream')


def main(argv: list[str]) -> int:
    do_pull = '--pull' in argv
    print(f'reader certified for build {VERSION_CODE} (libg {LIBG_SHA256[:12]})')
    stale = False
    for serial in DEVICES:
        info = installed(serial)
        if info is None:
            print(f'  {serial}: not reachable or the game is not installed')
            continue
        same = int(info['version']) == int(VERSION_CODE)
        sha = None
        local = CLAPHA / 'runtime' / info['version'] / 'apks' / 'split_config.arm64_v8a.apk'
        if local.exists():
            sha = libg_sha(local)
        verdict = 'CURRENT' if same and (sha is None or sha == LIBG_SHA256) else 'GAME UPDATED'
        print(f'  {serial}: installed {info["version"]}  ->  {verdict}'
              + (f' (libg {sha[:12]})' if sha else ''))
        if verdict != 'CURRENT':
            stale = True
            if do_pull:
                folder = pull(serial, info['version'], info['paths'])
                print(f'    pulled into {folder.relative_to(CLAPHA)}')
                subprocess.run([sys.executable, str(CLAPHA / 'mac012' / 'update_catalog.py'),
                                info['version']])
    catalog = json.loads((CLAPHA / 'live_card_catalog.json').read_text())
    checked = catalog.get('game_version_checked')
    print(f'card catalog: checked against {checked or "an older build (15.535.29)"}'
          + ('' if str(checked) == str(VERSION_CODE) else '  -> run mac012/update_catalog.py'))
    if stale:
        print('\nThe bot must not run on the new build until these are re-derived:')
        for what, how in PINNED:
            print(f'  - {what:48} {how}')
        print('The reader refuses an unverified build by itself ("reader not attached").')
    print('\nreference repos:')
    for name, path, github in (
            ('FirstLight_CR', CLAPHA / 'ref-firstlight', 'Jaasssoooonnnnn/FirstLight_CR'),
            ('cr-native-sandbox', CLAPHA / 'ref-cr-native-sandbox', 'IMAX9D/cr-native-sandbox')):
        print(f'  {name:18} {repo_status(path, github)}')
    return 1 if stale else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
