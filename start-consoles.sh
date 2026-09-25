#!/bin/bash
# Start both bot consoles (one per MuMu instance) and open them in the browser.
# Idempotent: safe to run again; it replaces anything already running.
set -u

CLAPHA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADB="$HOME/Library/Android/sdk/platform-tools/adb"
MUMU="/Applications/MuMuPlayer Pro.app/Contents/MacOS/mumu-cli"
DEV0_PORT=26624     # main account
DEV1_PORT=26656     # second account
TOOLS="live_sampler_tbi queue_probe deck_vector_scan player_dump peek fast_tap runtime_probe"

say() { printf '%s\n' "$*"; }

say "== clapha consoles =="

# 1. emulators
for index in 0 1; do
  state=$("$MUMU" info "$index" 2>/dev/null | sed -n 's/.*"state" : "\([a-z]*\)".*/\1/p' | head -1)
  if [ "$state" != "running" ]; then
    say "starting emulator $index (this takes ~40s)..."
    "$MUMU" open "$index" >/dev/null 2>&1
  else
    say "emulator $index already running"
  fi
done

# 2. wait for adb, then connect
for port in $DEV0_PORT $DEV1_PORT; do
  for _ in $(seq 1 45); do
    "$ADB" connect "127.0.0.1:$port" 2>/dev/null | grep -q connected && break
    sleep 2
  done
done
"$ADB" devices | sed -n '2,$p' | sed '/^$/d' | sed 's/^/  /'

# 3. rebuild the native tools whose source is newer than their binary, then make sure each
#    device has them
NDK_CC="$(ls -d "$HOME"/Library/Android/sdk/ndk/*/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android31-clang 2>/dev/null | tail -1)"
for tool in live_sampler_tbi queue_probe fast_tap runtime_probe comp_probe; do
  if [ "$CLAPHA/src/$tool.c" -nt "$CLAPHA/build/$tool" ]; then
    if [ -n "$NDK_CC" ]; then
      mkdir -p "$CLAPHA/build"
      "$NDK_CC" -O2 -o "$CLAPHA/build/$tool" "$CLAPHA/src/$tool.c" &&
        say "rebuilt $tool" || say "$tool build FAILED"
    else
      say "$tool source changed but no NDK clang found"
    fi
  fi
done
for port in $DEV0_PORT $DEV1_PORT; do
  serial="127.0.0.1:$port"
  "$ADB" -s "$serial" shell 'id' >/dev/null 2>&1 || continue
  # Push unconditionally. The old "skip if it already exists" test relied on adb shell
  # propagating the test's exit status, which it does not do reliably here -- so the push was
  # skipped even on a device that had never received the tools, and its reader could never
  # start. They are a few hundred KB; copying them every time is cheaper than that failure.
  missing=""
  for tool in $TOOLS; do
    if [ -f "$CLAPHA/build/$tool" ]; then
      "$ADB" -s "$serial" push "$CLAPHA/build/$tool" /data/local/tmp/ >/dev/null 2>&1 ||
        missing="$missing $tool"
    else
      missing="$missing $tool(not built)"
    fi
  done
  "$ADB" -s "$serial" shell 'chmod 755 /data/local/tmp/* 2>/dev/null' >/dev/null 2>&1
  present=$("$ADB" -s "$serial" shell 'ls /data/local/tmp/ 2>/dev/null' | tr -d '\r' | tr '\n' ' ')
  for tool in $TOOLS; do
    case " $present " in *" $tool "*) ;; *) missing="$missing $tool(absent)" ;; esac
  done
  [ -n "$missing" ] && say "  $serial MISSING TOOLS:$missing"
done

# 4. say, per device, whether the reader can actually attach
say "checking readers..."
CR_MUMU_SERIAL="127.0.0.1:$DEV0_PORT" python3 "$CLAPHA/mac012/preflight.py" \
  "127.0.0.1:$DEV0_PORT" "127.0.0.1:$DEV1_PORT" 2>&1 | sed 's/^/  /'

# 5. replace any running consoles
pkill -9 -f "mac012/console.py" 2>/dev/null
for p in 8777 8778; do lsof -ti:$p 2>/dev/null | xargs kill -9 2>/dev/null; done
sleep 1

cd "$CLAPHA" || exit 1
CR_MUMU_SERIAL="127.0.0.1:$DEV0_PORT" CR_CONSOLE_PORT=8777 \
  nohup python3 mac012/console.py > build/console_dev0.log 2>&1 &
CR_MUMU_SERIAL="127.0.0.1:$DEV1_PORT" CR_CONSOLE_PORT=8778 \
  nohup python3 mac012/console.py > build/console_dev1.log 2>&1 &

# 6. wait until they answer, then open them
for p in 8777 8778; do
  for _ in $(seq 1 30); do
    curl -sf "http://127.0.0.1:$p/state" >/dev/null 2>&1 && break
    sleep 1
  done
done

for p in 8777 8778; do
  if curl -sf "http://127.0.0.1:$p/state" >/dev/null 2>&1; then
    say "  console on http://127.0.0.1:$p  (ready)"
    open "http://127.0.0.1:$p"
  else
    say "  console on http://127.0.0.1:$p  FAILED - see build/console_dev*.log"
  fi
done

say ""
say "device 0 (main account)   -> http://127.0.0.1:8777"
say "device 1 (second account) -> http://127.0.0.1:8778"
say "stop them with:  pkill -f mac012/console.py"
