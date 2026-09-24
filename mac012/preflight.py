"""Say, per device, whether the live reader can attach -- and if not, exactly why.

Without this a device whose game is closed, unrooted or on a different build looks the same
from the console as a device that is simply between battles.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, '/Users/leafer/clapha/mac012')
from mac_profile import ADB, apply  # type: ignore  # noqa: E402

apply()  # rebind VERSION_CODE / LIBG_SHA256 / offsets for this build, as the viewer does
from native_core.mumu_live_protocol import root_run, verify_runtime  # noqa: E402

# mumu-live-reader-v2 is the name start_reader actually executes; the console installs it
# itself now, but a device missing it emits no frames at all, so it is worth naming here.
TOOLS = ('mumu-live-reader-v2', 'queue_probe')

for serial in sys.argv[1:]:
    label = f'  {serial}'
    try:
        runtime = verify_runtime(ADB, serial)
    except Exception as error:  # noqa: BLE001
        print(f'{label}  READER WILL NOT ATTACH: {type(error).__name__}: {error}')
        continue
    try:
        listing = root_run(ADB, serial, 'ls /data/local/tmp/')
    except Exception as error:  # noqa: BLE001
        print(f'{label}  no root shell: {error}')
        continue
    absent = [tool for tool in TOOLS if tool not in listing.split()]
    state = f'pid {runtime["pid"]}, build {runtime["version_code"]}'
    print(f'{label}  ready ({state})' if not absent
          else f'{label}  tools missing: {" ".join(absent)} ({state})')
