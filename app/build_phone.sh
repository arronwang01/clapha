#!/bin/bash
# Build the phone link helper (build/ClaphaPhone.app) -- only when its source is newer than the build: macOS
# gives the permission to record the screen to this exact build, and a rebuild has to be allowed again.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
APP="$ROOT/build/ClaphaPhone.app"
BIN="$APP/Contents/MacOS/ClaphaPhone"
if [ "${1:-}" != "--force" ] && [ -x "$BIN" ] && [ "$BIN" -nt "$HERE/Phone/PhoneLink.swift" ]; then
  echo "up to date: $APP"; exit 0
fi
mkdir -p "$APP/Contents/MacOS"
swiftc -O -parse-as-library -target arm64-apple-macosx14.0 "$HERE/Phone/PhoneLink.swift" -o "$BIN"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Clapha Phone</string>
  <key>CFBundleDisplayName</key><string>Clapha Phone</string>
  <key>CFBundleIdentifier</key><string>local.clapha.phone</string>
  <key>CFBundleExecutable</key><string>ClaphaPhone</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force --sign - "$APP" >/dev/null 2>&1 || true
echo "built $APP"
