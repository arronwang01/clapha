#!/bin/bash
# Play against FirstLight in its own sandbox (Human vs Model) on CR_4k.
#
#   tools/play_firstlight.sh              realistic: the AI's cards land ~21-25 ticks after it decides
#   tools/play_firstlight.sh --no-delay   like FirstLight's current release and video: 1 tick
#   tools/play_firstlight.sh --install-only   just boot CR_4k and install FirstLight's probe
#
# FirstLight's sandbox refuses to run unless the probe inside Null's is exactly the one it was built
# with (sha256 2257d2c7...). CR_4k normally carries the HUD + lean-snapshot probe, so this installs
# FirstLight's original probe for the session. The emulator boots from its saved snapshot and never
# saves over it, so the next boot is back to the HUD probe automatically.
#
# CR_4k is found by its AVD name as the device itself reports it (ro.boot.qemu.avd_name), never as
# "emulator-5554": MuMu Pro's adbd listens on port 5555 and adb lists anything there as emulator-5554
# -- the live MuMu device -- and an emulator booted without -port then takes console 5554 next to it.
# So CR_4k always boots on its own pair, 5580/5581 (emulator-5580).
set -euo pipefail
SDK="$HOME/Library/Android/sdk"
ADB="$SDK/platform-tools/adb"
PKG=nullsroyale.rel.free
PROBE="$HOME/Documents/GitHub/FirstLight_CR/native_runner/probe/out/libcrprobe.so"
EXPECTED=2257d2c7051c5b4dc33d381bc0be2d40ecff3d71dbda517b18bbfba3bb88429d
PORT_DIR="$HOME/Documents/GitHub/cr-engine-extraction/macos-port"
PY="$(cd "$(dirname "$0")/.." && pwd)/py"

cr4k() {
  local serial
  for serial in $("$ADB" devices | awk '$2 == "device" {print $1}'); do
    [ "$("$ADB" -s "$serial" shell getprop ro.boot.qemu.avd_name 2>/dev/null | tr -d '\r')" = CR_4k ] \
      && { echo "$serial"; return 0; }
  done
  return 1
}

no_delay=0; install_only=0
[ "${1:-}" = "--no-delay" ] && no_delay=1
[ "${1:-}" = "--install-only" ] && install_only=1

[ "$(shasum -a 256 "$PROBE" | cut -d' ' -f1)" = "$EXPECTED" ] || { echo "FirstLight's probe at $PROBE is not the attested build" >&2; exit 1; }

SERIAL=$(cr4k || true)
if [ -z "$SERIAL" ]; then
  echo "booting CR_4k..."
  # the same flags as the Null's Console: never write over the saved snapshot, public DNS
  nohup "$SDK/emulator/emulator" -avd CR_4k -port 5580 -no-snapshot-save -no-boot-anim \
    -dns-server 1.1.1.1,8.8.8.8 -grpc 8554 -grpc-use-token >/dev/null 2>&1 &
  for _ in $(seq 1 120); do
    SERIAL=$(cr4k || true)
    [ -n "$SERIAL" ] && [ "$("$ADB" -s "$SERIAL" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = 1 ] && break
    sleep 2
  done
  [ -n "$SERIAL" ] || { echo "CR_4k did not come up" >&2; exit 1; }
fi
a() { "$ADB" -s "$SERIAL" "$@"; }
export CR_ADB_SERIAL="$SERIAL" CR_VM_NAME="$SERIAL"
echo "CR_4k is $SERIAL"
a root >/dev/null 2>&1 || true
a wait-for-device

LIB=$(a shell "ls -d /data/app/*/$PKG*/lib/arm64" | tr -d '\r')
a push "$PROBE" /data/local/tmp/libcrprobe.firstlight.so >/dev/null
a shell "am force-stop $PKG; cp /data/local/tmp/libcrprobe.firstlight.so $LIB/libcrprobe.so;
  chown system:system $LIB/libcrprobe.so; chmod 755 $LIB/libcrprobe.so; restorecon $LIB/libcrprobe.so 2>/dev/null; true"
installed=$(a shell "sha256sum $LIB/libcrprobe.so" | cut -d' ' -f1)
[ "$installed" = "$EXPECTED" ] || { echo "probe install failed ($installed)" >&2; exit 1; }
echo "FirstLight's original probe installed."
[ "$install_only" = 1 ] && exit 0

# its console preselects the newest checkpoint: make that the Hog 2.6 specialist 2 (fl:hog2, the stronger
# one in model-vs-model play); both decks already default to the Hog 2.6 mirror (interface_mac presets)
touch "$HOME/Documents/GitHub/FirstLight_CR/checkpoints/2_6hog_expert/hog26-specialist2.pt"
cd "$PORT_DIR"
if [ "$no_delay" = 1 ]; then
  echo "opening FirstLight's interface: NO-DELAY AI (cards land 1 tick after it decides)"
  CR_AI_NO_DELAY=1 exec "$PY" interface_mac.py
else
  echo "opening FirstLight's interface: realistic AI (cards land ~21-25 ticks after it decides)"
  exec "$PY" interface_mac.py
fi
