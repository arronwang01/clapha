"""Faster drop-ins for FirstLight hot spots that give identical results.

native_runner.contracts._freeze makes every contract recursively immutable. Each observation's
action mask carries a placement grid per hand slot (32 rows of 18 flags) and is frozen three times
per decision; the recursion, with three ABC isinstance checks per element, was 20 s of a 108 s
self-play game (il/rl.py profile, 2026-09-27). The drop-in below takes the common shapes by exact
type -- a primitive, a plain dict, a plain list or tuple -- and hands everything else to the
original, so the frozen value is the same object structure either way (test_freeze below).

    install()   patch native_runner.contracts (idempotent)
"""
from __future__ import annotations

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import native_runner.contracts as contracts
    original = contracts._freeze
    frozen = contracts.FrozenMapping
    primitive = frozenset((int, float, str, bool, type(None)))

    def _freeze(value):
        kind = type(value)
        if kind in primitive or kind is frozen:
            return value
        if kind is dict:
            return frozen(value)
        if kind is tuple or kind is list:
            return tuple([item if type(item) in primitive else _freeze(item) for item in value])
        return original(value)      # sets, other mappings, subclasses: exactly as before

    contracts._freeze = _freeze     # FrozenMapping.__init__ looks it up at call time
    _INSTALLED = True


def test_freeze(rounds: int = 3000, seed: int = 5) -> None:
    """The drop-in and the original agree on random nested values (structure, types, equality)."""
    import random
    from collections import OrderedDict, namedtuple
    from enum import Enum, IntEnum
    import native_runner.contracts as contracts
    install()
    fast = contracts._freeze

    class Color(Enum):
        RED = 'red'

    class Level(IntEnum):
        ONE = 1

    Pair = namedtuple('Pair', 'a b')
    rng = random.Random(seed)

    def make(depth: int):
        roll = rng.random()
        if depth > 3 or roll < 0.35:
            return rng.choice([0, 1, -7, 2.5, True, False, None, 'x', '', Color.RED, Level.ONE])
        if roll < 0.5:
            return {str(rng.randint(0, 9)): make(depth + 1) for _ in range(rng.randint(0, 4))}
        if roll < 0.6:
            return OrderedDict((str(i), make(depth + 1)) for i in range(rng.randint(0, 3)))
        if roll < 0.75:
            return [make(depth + 1) for _ in range(rng.randint(0, 6))]
        if roll < 0.85:
            return tuple(make(depth + 1) for _ in range(rng.randint(0, 6)))
        if roll < 0.9:
            return Pair(make(depth + 1), make(depth + 1))
        if roll < 0.95:
            return {rng.randint(0, 5) for _ in range(rng.randint(0, 4))}
        return frozenset(rng.choice('abc') for _ in range(rng.randint(0, 3)))

    def shape(value):
        if isinstance(value, contracts.FrozenMapping):
            return ('FM', tuple((key, shape(item)) for key, item in value.items()))
        if isinstance(value, tuple):
            return (type(value).__name__, tuple(shape(item) for item in value))
        return (type(value).__name__, repr(value))

    # the original algorithm, verbatim, recursing into itself rather than the patched global
    def frozen_reference(value):
        if isinstance(value, dict):
            mapping = contracts.FrozenMapping.__new__(contracts.FrozenMapping)
            from types import MappingProxyType
            mapping._data = MappingProxyType({str(k): frozen_ref_item(v) for k, v in value.items()})
            mapping._hash = None
            return mapping
        return frozen_ref_item(value)

    def frozen_ref_item(value):
        from collections.abc import Mapping
        if isinstance(value, contracts.FrozenMapping):
            return value
        if isinstance(value, Mapping):
            return frozen_reference(dict(value))
        if isinstance(value, (list, tuple)):
            return tuple(frozen_ref_item(item) for item in value)
        if isinstance(value, (set, frozenset)):
            return tuple(sorted((frozen_ref_item(item) for item in value), key=repr))
        return value

    for _ in range(rounds):
        value = make(0)
        got, want = fast(value), frozen_ref_item(value)
        if shape(got) != shape(want) or got != want:
            raise AssertionError(f'freeze differs for {value!r}: {shape(got)} vs {shape(want)}')
        if isinstance(value, dict):
            mapping = contracts.frozen_mapping(value)
            if shape(mapping) != shape(frozen_reference(value)):
                raise AssertionError(f'frozen_mapping differs for {value!r}')
    print(f'freeze: {rounds} random values agree')


if __name__ == '__main__':
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'mac012'))
    import firstlight_bot  # noqa: F401  (puts FirstLight's native_runner on the path)
    test_freeze()
