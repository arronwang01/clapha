"""Mac + MuMu Pro + build 160402012 profile for the upstream MuMu live pipeline.

Imports the author's native_core modules unchanged and rebinds only the values that
are environment- or build-specific. All of the author's admission logic
(version/SHA guard, BattleClockGuard, receipt checks) is left untouched.

Differences from the author's Windows setup, and why:
  version code   160402002 -> 160402012   (this install)
  libg SHA-256   d2c8efcf...              -> aec6cc5e...
  manager RVA    0x1A569A8 -> 0x1A57E88   (re-found with find_manager; +0x14E0)
  root wrapper   'su -c ...' -> identity  (MuMu Pro on Mac: adbd already runs as root)
  serial         127.0.0.1:16416 -> :5555 (Mac MuMu default)
  reader         x86_64 binary -> native arm64 build of the author's sampler with
                 Android heap pointer tags (top byte, e.g. 0xb4) masked before pread.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
REPO = CLAPHA / 'ref-cr-native-sandbox'
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from native_core import mumu_live_protocol as protocol  # noqa: E402

VERSION_CODE = 160402012
LIBG_SHA256 = 'aec6cc5e473edc809f10c157ae553dbbb8724d816f34148344b7f94a67e5a97c'
MANAGER_RVA = 0x1A57E88
ROOT_CONTEXT_OFFSET = 0x28
SERIAL = os.environ.get('CR_MUMU_SERIAL', '127.0.0.1:5555')
ADB = Path(os.environ.get('CR_MUMU_ADB',
    '/Users/leafer/Library/Android/sdk/platform-tools/adb'))
READER = CLAPHA / 'build' / 'live_sampler_tbi'
LOG_ROOT = CLAPHA / 'artifacts'


def _root_command(command: str) -> str:
    """adbd already runs as root on this MuMu; 'su' is not installed."""
    return command


def apply() -> dict:
    protocol.VERSION_CODE = VERSION_CODE
    protocol.LIBG_SHA256 = LIBG_SHA256
    protocol.MANAGER_RVA = MANAGER_RVA
    protocol.ROOT_CONTEXT_OFFSET = ROOT_CONTEXT_OFFSET
    protocol.DEFAULT_ADB = ADB
    protocol.DEFAULT_SERIAL = SERIAL
    protocol.DEFAULT_READER = READER
    protocol.DEFAULT_LOG_ROOT = LOG_ROOT
    protocol.root_command = _root_command
    return {'version_code': VERSION_CODE, 'libg_sha256': LIBG_SHA256,
            'manager_rva': hex(MANAGER_RVA), 'serial': SERIAL,
            'adb': str(ADB), 'reader': str(READER)}
