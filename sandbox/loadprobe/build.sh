#!/bin/bash
# Build the load-probe APK with the game's ARM64 libraries from runtime/<version>/ (local only,
# never distributed). Needs build-tools 36 + platform 37 + a JDK.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VERSION="${1:-160402012}"
SDK="$HOME/Library/Android/sdk"
BT="$SDK/build-tools/36.0.0"
JAR="$SDK/platforms/android-37.0/android.jar"
OUT="$HERE/out"; rm -rf "$OUT"; mkdir -p "$OUT/classes" "$OUT/apk/lib/arm64-v8a"
javac --release 17 -classpath "$JAR" -d "$OUT/classes" $(find "$HERE/src" -name '*.java')
"$BT/d8" --min-api 31 --lib "$JAR" --output "$OUT/apk" $(find "$OUT/classes" -name '*.class')
"$BT/aapt2" link -o "$OUT/base.apk" --manifest "$HERE/AndroidManifest.xml" -I "$JAR"
( cd "$OUT/apk/lib/arm64-v8a" && unzip -q -o "$ROOT/runtime/$VERSION/apks/split_config.arm64_v8a.apk" 'lib/arm64-v8a/*' && mv lib/arm64-v8a/*.so . && rm -rf lib )
cp "$OUT/base.apk" "$OUT/unsigned.apk"
( cd "$OUT/apk" && zip -q -r -0 "$OUT/unsigned.apk" lib && zip -q "$OUT/unsigned.apk" classes.dex )
"$BT/zipalign" -f -p 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"
KEY="$HERE/debug.keystore"
[ -f "$KEY" ] || keytool -genkeypair -keystore "$KEY" -storepass android -keypass android \
  -alias probe -keyalg RSA -keysize 2048 -validity 3650 -dname "CN=clapha probe" >/dev/null 2>&1
"$BT/apksigner" sign --ks "$KEY" --ks-pass pass:android --key-pass pass:android \
  --out "$OUT/loadprobe.apk" "$OUT/aligned.apk"
echo "built $OUT/loadprobe.apk ($(du -h "$OUT/loadprobe.apk" | cut -f1))"
