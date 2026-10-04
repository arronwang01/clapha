import AppKit
import SwiftUI

/// The Phone switch: the game picture on a phone over USB, the phone's touches back to the game. The work is done
/// by a helper of its own (app/Phone/PhoneLink.swift -> build/ClaphaPhone.app), started here through Launch
/// Services so that macOS asks for -- and remembers -- the permission to record the screen for the helper, not for
/// this app (which is rebuilt often, and each rebuild would have to be allowed again).
@MainActor
final class PhoneLinkControl: ObservableObject {
    @Published var on = false
    /// What the helper last said: nil until it has, then its words for the state it is in.
    @Published var note = ""
    @Published var good = false
    private var helper: NSRunningApplication?
    private var timer: Timer?
    private let statusURL = claphaRoot().appendingPathComponent("build/phone_link.json")

    func set(_ wanted: Bool, serial: String, width: Int) {
        wanted ? start(serial: serial, width: width) : stop()
    }

    private func start(serial: String, width: Int) {
        stop()
        let app = claphaRoot().appendingPathComponent("build/ClaphaPhone.app")
        guard FileManager.default.fileExists(atPath: app.path) else {
            note = "The phone helper is not built yet: run app/build_phone.sh."
            return
        }
        try? FileManager.default.removeItem(at: statusURL)
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = false
        configuration.createsNewApplicationInstance = true
        configuration.arguments = [
            "--serial", serial,
            "--adb", NSString(string: "~/Library/Android/sdk/platform-tools/adb").expandingTildeInPath,
            "--width", String(width),
            "--apk", claphaRoot().appendingPathComponent("phone/nulls-viewer.apk").path,
            "--status", statusURL.path,
        ]
        on = true
        note = "Starting…"
        good = false
        NSWorkspace.shared.openApplication(at: app, configuration: configuration) { [weak self] running, error in
            Task { @MainActor in
                guard let self else { return }
                if let running {
                    self.helper = running
                } else {
                    self.on = false
                    self.note = "The phone helper did not start: \(error?.localizedDescription ?? "unknown reason")"
                }
            }
        }
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.read() }
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
        helper?.terminate()
        helper = nil
        // one left over from an earlier run of this app
        for other in NSRunningApplication.runningApplications(withBundleIdentifier: "local.clapha.phone") {
            other.terminate()
        }
        on = false
        note = ""
        good = false
    }

    private struct Status: Decodable {
        let capture: String
        let phone: String
        let connected: Bool
        let framesPerSecond: Int
        let sentPerSecond: Int
        let phoneStats: String
    }

    private func read() {
        guard on else { return }
        if let helper, helper.isTerminated {
            self.helper = nil
            on = false
            note = "The phone helper stopped."
            return
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let data = try? Data(contentsOf: statusURL), let status = try? decoder.decode(Status.self, from: data) else { return }
        if status.capture != "capturing" && status.capture != "test pattern" {
            good = false
            note = "Phone: " + status.capture + "."
        } else if status.phone.isEmpty {
            good = false
            note = "Phone: no phone on USB (USB debugging on, and the computer allowed on the phone?)."
        } else if !status.connected {
            good = false
            note = "Phone: \(status.phone) is on the cable; waiting for the viewer app on it to connect."
        } else {
            good = true
            var shown = ""
            if let stats = try? JSONSerialization.jsonObject(with: Data(status.phoneStats.utf8)) as? [String: Any],
               let fps = stats["fps"] as? Double, let ms = stats["ms"] as? Double {
                shown = String(format: ", the phone shows %.0f a second, %.0f ms to decode each", fps, ms)
            }
            note = "Phone: \(status.sentPerSecond) pictures a second sent\(shown)."
        }
    }
}
