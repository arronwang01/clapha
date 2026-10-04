# The game on a phone

`nulls-viewer.apk` is the viewer app that goes on the phone ("Null's Viewer", 16 KB): it shows the pictures the
Mac sends and sends back every touch. Built for the earlier Null's setup; its source and build script are in
`~/Documents/GitHub/cr-engine-extraction/macos-port/phone/android` (plain Java, built without Gradle).

Here the Mac side is `app/Phone/PhoneLink.swift` (-> `build/ClaphaPhone.app`, started by the Phone switch in
Clapha): the game picture of the MuMu window, with Clapha's overlay on it, taken from the screen, sent as JPEG
over USB (`adb reverse tcp:8766`), and the phone's touches written to the MuMu touchscreen (`src/fast_tap.c`,
`d` / `m` / `u`).

To use it: phone on USB with USB debugging on; MuMu's window in view; Overlay on MuMu on if the overlay is
wanted on the phone; press **Phone**. The first time, macOS asks whether "Clapha Phone" may record the screen:
allow it in System Settings, then switch Phone off and on.

Protocol (the app's own): to the phone `type, u32 length, bytes` -- 1 config JSON `{"w","h","mode":"jpeg"}`,
3 one JPEG; from the phone `N` `j` once, then 1 touch `[action, finger, x u16, y u16]` (0 down, 1 up, 2 move,
3 cancel; picture pixels), 3 "frame shown" (two frames on their way at most), 2 statistics `[u16 length, JSON]`.
