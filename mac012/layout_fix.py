"""Fix x-mirroring for side 0 in the upstream ScreenLayout.

Upstream mirrors the canonical x fraction only when side == 1, because the author only ever
verified side 1. Measured here on 160402012 (friendly battle, local side 0): requesting
canonical (3500, 9500) landed the Cannon at native (14500, 9500) — y exact, x mirrored.

Working the two cases through in canonical space:
    side 1: canonical_x = 18000 - native_x, and the camera keeps native x
            -> screen fraction = native_x/18000 = 1 - canonical_x/18000
    side 0: canonical_x = native_x, and the camera mirrors it
            -> screen fraction = 1 - native_x/18000 = 1 - canonical_x/18000
So the mirror applies to **both** sides. This matches the user's prior project, which used
x' = 18000 - x for owner 0 and x' = x for owner 1 in native terms.

Import this module before sending taps; it patches the method in place.
"""
from __future__ import annotations

from native_core.mumu_live_actions import ScreenLayout

_original = ScreenLayout.deployment_point


def deployment_point(self, canonical_position: int, side: int = 0) -> tuple[int, int]:
    if type(canonical_position) is not int or not 0 <= canonical_position < 576:
        raise ValueError('placement must be a cell in the 18x32 grid')
    if type(side) is not int or side not in (0, 1):
        raise ValueError('side must be 0 or 1')
    row, column = divmod(canonical_position, 18)
    x_fraction = 1 - (column + .5) / 18  # mirrored for both sides; see module docstring
    return (round(self.arena_left + x_fraction * (self.arena_right - self.arena_left)),
            round(self.arena_bottom - (row + .5) / 32 * (self.arena_bottom - self.arena_top)))


ScreenLayout.deployment_point = deployment_point
