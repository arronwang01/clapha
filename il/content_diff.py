"""How far the training engine's game content is from the live game's: every named entry (units,
buildings, projectiles, spells, area effects, buffs...) in both versions' logic tables, field by field.

Engine: Null's APK csv_logic with its update folder on top (content 15.535.86). Live: the 160402012
APK's csv_logic (runtime/160402012, pulled from the user's install). Both are game data, not code:
LZMA-packed CSV (header row, types row, named rows) and TOML ([Name] sections). Values are level-1
base stats; levels scale both versions alike.

    ./py -m il.content_diff [--focus HogRider,Musketeer,...] [--all]
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))
NULLS_APK = Path.home() / 'crtrain-stage/engine/nulls-offline.apk'
NULLS_UPDATE = Path.home() / 'crtrain-stage/engine/nulls-update.tar'
LIVE = CLAPHA / 'runtime/160402012/assets/assets/csv_logic'
# the Hog 2.6 cards and what they are made of (units, projectiles, spells, evolutions, hero forms)
FOCUS = ('HogRider', 'Musketeer', 'Cannon', 'Fireball', 'Log', 'Skeleton', 'IceGolem', 'IceSpirit', 'Princess',
         'KingTower', 'Tower')


def _decompress(raw: bytes) -> str:
    from game_data import decompress
    return decompress(raw).decode('utf-8-sig', 'replace')


def _csv_entries(text: str) -> dict[str, dict]:
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        return {}
    header = rows[0]
    entries = {}
    for row in rows[2:]:
        if row and row[0].strip():
            entries[row[0].strip()] = {k: v for k, v in zip(header, row) if k and v != ''}
    return entries


_SECTION = re.compile(r'^\[+\s*([^\]]+?)\s*\]+\s*$')
_PAIR = re.compile(r'^([A-Za-z0-9_.]+)\s*=\s*(.+?)\s*$')


def _toml_entries(text: str) -> dict[str, dict]:
    """[Name] sections with key = value lines (a tolerant reader: the files are simple and some are
    not strict TOML)."""
    entries, current = {}, None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        section = _SECTION.match(line)
        if section:
            current = entries.setdefault(section.group(1).strip('"'), {})
            continue
        pair = _PAIR.match(line)
        if pair and current is not None:
            current[pair.group(1)] = pair.group(2).strip('"')
    return entries


def _entries(files: dict[str, bytes]) -> dict[str, tuple[str, dict]]:
    """name -> (file, fields) over every table; a later file never replaces an earlier name."""
    out = {}
    for name in sorted(files):
        text = _decompress(files[name])
        found = _csv_entries(text) if name.endswith('.csv') else _toml_entries(text) if name.endswith('.toml') else {}
        for entry, fields in found.items():
            out.setdefault(entry, (name, fields))
    return out


def engine_files() -> dict[str, bytes]:
    import tarfile
    files = {}
    with zipfile.ZipFile(NULLS_APK) as apk:
        for info in apk.infolist():
            if info.filename.startswith('assets/csv_logic/') and not info.is_dir():
                files[info.filename[len('assets/csv_logic/'):]] = apk.read(info)
    with tarfile.open(NULLS_UPDATE) as tar:          # the update folder replaces the APK's copy
        for member in tar.getmembers():
            if member.isfile() and member.name.startswith('update/csv_logic/'):
                files[member.name[len('update/csv_logic/'):]] = tar.extractfile(member).read()
    return files


def live_files() -> dict[str, bytes]:
    return {str(path.relative_to(LIVE)): path.read_bytes() for path in LIVE.rglob('*') if path.is_file()}


def _number(value: str):
    try:
        return float(value)
    except ValueError:
        return None


# the simulation's numbers; a field renamed or folded into a Filter in the new schema is not a
# balance change (HitsAir/OnlyEnemies -> Filter, Damage -> { BaseDamage = ... })
STATS = ('Hitpoints', 'ShieldHitpoints', 'Damage', 'BaseDamage', 'TowerDamage', 'CrownTowerDamagePercent',
         'HitSpeed', 'LoadTime', 'LoadFirstHit', 'Speed', 'Range', 'MinimumRange', 'SightRange', 'DeployTime',
         'DeployDelay', 'DeathDamage', 'DeathDamageRadius', 'DeathSpawnCount', 'DeathSpawnCharacter', 'Radius',
         'AreaDamageRadius', 'LifeTime', 'LifeDuration', 'BuffTime', 'SpawnNumber', 'SpawnPauseTime',
         'SpawnInterval', 'SpawnCharacter', 'Mass', 'CollisionRadius', 'ChargeRange', 'DamageSpecial',
         'ChargeSpeedMultiplier', 'JumpHeight', 'JumpSpeed', 'ProjectileRange', 'Pushback', 'PushbackAll',
         'SpeedMultiplier', 'HitSpeedMultiplier', 'MultipleProjectiles', 'Mirror', 'RollingRadius',
         'RollingRange', 'RollingSpeed', 'ElixirCost', 'ManaCost', 'Cooldown', 'AbilityCooldown')


def _stat(value: str | None):
    """A stat as a number: plain, or the BaseDamage of a { BaseDamage = x, ... } table."""
    if value is None:
        return None
    match = re.search(r'BaseDamage\s*=\s*(-?[0-9.]+)', value)
    if match:
        return float(match.group(1))
    return _number(value)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--focus', default=','.join(FOCUS), help='entry-name substrings to detail')
    parser.add_argument('--all', action='store_true', help='detail every changed entry')
    args = parser.parse_args(argv)
    engine, live = _entries(engine_files()), _entries(live_files())
    shared = sorted(set(engine) & set(live))
    changed = {}
    for name in shared:
        before, after = engine[name][1], live[name][1]
        diffs = {}
        for key in STATS:
            old, new = _stat(before.get(key)), _stat(after.get(key))
            if key == 'Damage' and new is None and before.get(key) is not None:
                new = _stat(after.get('BaseDamage'))        # moved into the damage table
            if old is None or new is None or old == new:
                continue            # absent in one schema, or unchanged
            diffs[key] = (before.get(key), after.get(key))
        if diffs:
            changed[name] = diffs
    only_live = sorted(set(live) - set(engine))
    print(f'engine entries {len(engine)}, live {len(live)}, shared {len(shared)}; '
          f'{len(changed)} shared entries have a different gameplay number; {len(only_live)} only in the live game')
    focus = [f.lower() for f in args.focus.split(',') if f]
    for name, diffs in sorted(changed.items()):
        if not args.all and not any(f in name.lower() for f in focus):
            continue
        print(f'{name} ({engine[name][0]} -> {live[name][0]}):')
        for key, (old, new) in diffs.items():
            print(f'    {key}: {old} -> {new}')
    new_focus = [n for n in only_live if any(f in n.lower() for f in focus)]
    if new_focus:
        print('only in the live game (focus):', ', '.join(new_focus))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
