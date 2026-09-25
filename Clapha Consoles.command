#!/bin/bash
# Opens the Clapha app (builds it first if needed). Everything else -- emulators, readers,
# both devices, the bot, checks -- is done from inside the app.
cd "$(dirname "$0")" || exit 1
[ -d Clapha.app ] || ./app/build.sh
open Clapha.app
