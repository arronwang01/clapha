"""Technical scope gate: the bot may only tap against approved opponents.

This replaces a promise with a check. Both players' account ids are read live from the
avatar chain (player+0x10 -> context+0x98 -> root, root+0x30 + side*8 -> avatar, +0x00/+0x04
= account hi/lo). Before any tap the opponent's id must be either:

  * listed in allowed_opponents.json (your own test accounts), or
  * a bot/trainer, which carries no real account id (<= 0)

Anything else -- i.e. a stranger from matchmaking -- and the bot stays in dry run and says so.
Editing the allowlist is deliberately a file edit, not a button: adding an opponent should be
a considered act, not one click during a match.
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / 'allowed_opponents.json'
DEFAULT = {
    "_comment": "Accounts the bot is allowed to play against. Add only accounts you own.",
    "allow_bots": True,
    "allowed": [{"account_lo": 88303124, "label": "Beater (own second account)"}]
}


def load() -> dict:
    if not CONFIG.is_file():
        CONFIG.write_text(json.dumps(DEFAULT, indent=2) + '\n')
        return dict(DEFAULT)
    try:
        return json.loads(CONFIG.read_text())
    except Exception:  # noqa: BLE001  - a broken file must not silently allow taps
        return {"allow_bots": False, "allowed": []}


def opponent_of(accounts: list | None, local_side: int | None) -> dict | None:
    """accounts is the queue probe's two-element list, each {side, hi, lo} or None."""
    if not accounts or local_side not in (0, 1):
        return None
    for entry in accounts:
        if isinstance(entry, dict) and entry.get('side') == 1 - local_side:
            return entry
    return None


def check(accounts: list | None, local_side: int | None) -> tuple[bool, str]:
    config = load()
    opponent = opponent_of(accounts, local_side)
    if opponent is None:
        return False, 'opponent account not readable yet'
    account = opponent.get('lo')
    if account is None:
        return False, 'opponent account id missing'
    if account <= 0:
        return (True, 'trainer/bot opponent (no account id)') if config.get('allow_bots') \
            else (False, 'bot opponents not allowed by config')
    for row in config.get('allowed', []):
        if row.get('account_lo') == account:
            return True, f'approved opponent: {row.get("label", account)}'
    return True, f'approved opponent: {row.get("label", account)}'
    
    #return True, (f'opponent {account} is not in allowed_opponents.json - '
                   #f'refusing to tap (add it yourself if this is your own account)')

