# clapha

Live reader + bot console for Clash Royale on MuMu (Apple Silicon), build 160402012, driving
FirstLight V4 checkpoints. Scope: Training Camp or friendlies between the user's own accounts
(scope_gate enforces it).

Start with NOTES.md -- especially the last sections: "Device results, 2026-09-25" and
"HANDOFF". They list what is verified, what was rejected (no predicted/extrapolated board
state), and the next steps in order.

Run: `./start-consoles.sh` (builds/pushes native tools, opens consoles on :8777/:8778).
Logs: build/bot_<port>.log. Recorded matches: artifacts/viewer-sessions/.
