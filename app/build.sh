#!/bin/bash
# Build Clapha.app (native macOS, SwiftUI) into the repo root. No Xcode project needed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
APP="$ROOT/Clapha.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -O -parse-as-library -target arm64-apple-macosx14.0 \
  "$HERE"/Sources/*.swift -o "$APP/Contents/MacOS/Clapha"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Clapha</string>
  <key>CFBundleDisplayName</key><string>Clapha</string>
  <key>CFBundleIdentifier</key><string>local.clapha.console</string>
  <key>CFBundleExecutable</key><string>Clapha</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>$(date +%Y%m%d%H%M)</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict></plist>
PLIST
[ -f "$HERE/AppIcon.icns" ] && cp "$HERE/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
# Landing-warning art (Supercell's own assets, project use only): unpacked outside git.
if [ -f "$ROOT/landing-hud-kit.zip" ] && [ ! -d "$ROOT/build/hud/landing-hud-kit" ]; then
  mkdir -p "$ROOT/build/hud" && unzip -q -o "$ROOT/landing-hud-kit.zip" -d "$ROOT/build/hud"
fi
codesign --force --sign - "$APP" >/dev/null 2>&1 || true
echo "built $APP"
