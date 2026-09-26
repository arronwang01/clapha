"""Timing constants for imitation samples, each measured on the live game (2026-09-25).

One command, live (queue probe) and in Null's engine alike: it is issued at I (the tap registers)
and executes at X = I + COMMAND_AGE_TICKS: hand, elixir and the viewer's executed plays change at
X, and its unit is on the board from X + 1 (the measured "queued -> unit 22 ticks"). A replay tick
L (RoyaleAPI data-t, FirstLight's source_command_tick) is X - 1: FirstLight queues it for native
execution at L + 1 (its native_observable_tick), and the engine's hand still holds the card in the
snapshot at L. So L = I + 20 (REPLAY_TICK_AFTER_ISSUE). The human tapped at about L - 20; our bot
decides on a frame d and its tap registers OWN_OVERHEAD ticks later, so a play at replay tick L is
decided at d = L - (20 + overhead) (command_delay).
"""
from __future__ import annotations

import hashlib

COMMAND_AGE_TICKS = 21          # issue -> execute, both players (queue_lead, NOTES "Delay")
REPLAY_TICK_AFTER_ISSUE = COMMAND_AGE_TICKS - 1   # issue -> replay tick L (= execute - 1)
DECISION_TICKS = 5              # the policy's decision grid (FirstLight POLICY_DECISION_TICKS)
FIRST_DECISION_TICK = 90        # FirstLight FIRST_POLICY_DECISION_TICK

# Decision frame -> issue tick for the bot's own plays: 908 plays in the 2026-09-25 friendlies
# (bot_8777.log decision tick vs queue issue tick). The tail beyond 7 is deferred plays waiting
# for elixir, not pipeline latency, so it is left out.
OWN_OVERHEAD_TICKS = {3: 65, 4: 153, 5: 267, 6: 51, 7: 128}

# How many ticks before execution an opponent command is in our queue: 1131 commands from real
# opponents over 34 recorded sessions. Leads above 21 (about 10%, up to 48) would mean the
# command was visible before it was issued; they are held out until explained.
OPPONENT_LEAD_TICKS = {2: 1, 4: 1, 5: 2, 7: 1, 8: 8, 9: 42, 10: 104, 11: 182, 12: 92, 13: 109,
                       14: 85, 15: 113, 16: 78, 17: 88, 18: 57, 19: 26, 20: 4, 21: 1}


def _draw(distribution: dict[int, int], *key: object) -> int:
    """A deterministic draw from a count table, seeded by the key (reproducible samples)."""
    digest = hashlib.sha256('|'.join(map(str, key)).encode()).digest()
    point = int.from_bytes(digest[:8], 'big') % sum(distribution.values())
    for value, count in sorted(distribution.items()):
        if point < count:
            return value
        point -= count
    raise AssertionError('unreachable')


def command_delay(replay_tag: str, owner: int) -> int:
    """Ticks from the bot's decision frame to a play's replay tick L, for one actor in one replay."""
    return REPLAY_TICK_AFTER_ISSUE + _draw(OWN_OVERHEAD_TICKS, 'overhead', replay_tag, owner)


def opponent_lead(replay_tag: str, source_index: int) -> int:
    """Ticks before it executes (X = L + 1) that one opponent command becomes visible to the actor."""
    return _draw(OPPONENT_LEAD_TICKS, 'lead', replay_tag, source_index)


def opponent_seen_tick(replay_tag: str, source_index: int, lands: int) -> int:
    """The tick an opponent command with replay tick `lands` shows in the actor's queue."""
    return lands + 1 - opponent_lead(replay_tag, source_index)
