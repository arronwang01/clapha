import AppKit
import SwiftUI

// MARK: - Watch in Null's: a recorded training game played again in the real game
//
// il/watch_nulls.py does the work (CR_4k, FirstLight's direct scheduling, snapshots for seeking).
// This runs it, sends it the controls on stdin (pause, resume, speed X, seek TICK, stop) and reads
// its progress, one JSON object per line, from stdout. The game shows in the CR_4k emulator window;
// while it plays, the game's own play button, progress bar and speed drive it (GamePlayer).

@MainActor
final class NullsPlayer: ObservableObject {
    @Published var gameTag: String?
    @Published var state = "idle"        // preparing, starting, playing, paused, done, stopped, failed, closed
    @Published var message = ""
    @Published var tick: Int?
    @Published var plays = 0
    @Published var of = 0
    @Published var skipped = 0
    @Published var speed: Double = 1
    @Published var end: Int?
    @Published var explored = 0        // how far it has played: seeks up to here are near-instant
    @Published var seeking: Int?       // a far seek, running there at 4x
    @Published private(set) var running = false
    private var process: Process?
    private var input: FileHandle?
    private var buffer = Data()
    private var lastLog = ""
    let root: URL

    init(root: URL) { self.root = root }

    var active: Bool { running && ["preparing", "starting", "playing", "paused", "done"].contains(state) }

    /// This game is up in Null's: its controls drive Null's.
    func linked(_ game: TrainingGame) -> Bool {
        running && gameTag == game.id && ["playing", "paused", "done"].contains(state)
    }

    func watch(_ game: TrainingGame) {
        guard let path = game.path, !running else { return }
        gameTag = game.id
        state = "preparing"; message = "Starting…"; tick = nil; plays = 0; of = 0; skipped = 0; lastLog = ""
        end = nil; explored = 0; seeking = nil
        buffer = Data()
        let speedText = speed == speed.rounded() ? String(Int(speed)) : String(speed)
        launch("exec ./py -m il.watch_nulls \(path.shellQuoted) --speed \(speedText)")
    }

    /// Closes the CR_4k emulator (it idles at ~85% CPU): before a live test on MuMu.
    func shutDown() {
        if running { send("stop") }
        gameTag = nil
        state = "preparing"; message = "Shutting down CR_4k…"
        let launchShutdown = { [weak self] in self?.launch("exec ./py -m il.watch_nulls --shutdown") }
        if running {
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) { launchShutdown() }
        } else {
            launchShutdown()
        }
    }

    func send(_ command: String) {
        try? input?.write(contentsOf: Data((command + "\n").utf8))
    }

    func pauseOrResume() {
        if state == "done" { seek(0); send("resume") }       // from the start again
        else { send(state == "paused" ? "resume" : "pause") }
    }

    func seek(_ tick: Int) { send("seek \(tick)") }

    func setSpeed(_ value: Double) {
        speed = value
        if running { send("speed \(value)") }
    }

    func stop() {
        send("stop")
        let process = self.process
        DispatchQueue.main.asyncAfter(deadline: .now() + 10) { if process?.isRunning == true { process?.terminate() } }
    }

    private func launch(_ command: String) {
        guard !running else { return }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-lc", "cd \(root.path.shellQuoted) && \(command)"]
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env
        let output = Pipe(), input = Pipe()
        process.standardOutput = output
        process.standardError = output
        process.standardInput = input
        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            Task { @MainActor in self?.receive(data) }
        }
        process.terminationHandler = { [weak self] finished in
            output.fileHandleForReading.readabilityHandler = nil
            Task { @MainActor in
                guard let self else { return }
                self.running = false
                self.process = nil
                self.input = nil
                if ["preparing", "starting", "playing", "paused"].contains(self.state) {
                    self.state = finished.terminationStatus == 0 ? "stopped" : "failed"
                    if finished.terminationStatus != 0 {
                        self.message = self.lastLog.isEmpty ? "il.watch_nulls exited (\(finished.terminationStatus))" : self.lastLog
                    }
                }
            }
        }
        do {
            try process.run()
            self.process = process
            self.input = input.fileHandleForWriting
            running = true
        } catch {
            state = "failed"
            message = error.localizedDescription
        }
    }

    private func receive(_ data: Data) {
        buffer.append(data)
        while let newline = buffer.firstIndex(of: 0x0A) {
            let line = buffer.subdata(in: buffer.startIndex..<newline)
            buffer.removeSubrange(buffer.startIndex...newline)
            guard let object = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any] else {
                if let text = String(data: line, encoding: .utf8)?.trimmingCharacters(in: .whitespaces), !text.isEmpty {
                    lastLog = String(text.suffix(300))
                }
                continue
            }
            if let value = object["state"] as? String { state = value }
            if let value = object["message"] as? String { message = value }
            if let value = object["tick"] as? Int { tick = value }
            if let value = object["plays"] as? Int { plays = value }
            if let value = object["of"] as? Int { of = value }
            if let value = object["skipped"] as? Int { skipped = value }
            if let value = object["speed"] as? Double { speed = value }
            if let value = object["end"] as? Int { end = value }
            if let value = object["explored"] as? Int { explored = value }
            if object.keys.contains("seeking") { seeking = object["seeking"] as? Int }
        }
    }
}

/// Under a game: the button that plays it in Null's; while it plays, where it is and Stop (the
/// game's own play button, progress bar and speed drive it).
struct NullsBar: View {
    @ObservedObject var nulls: NullsPlayer
    let game: TrainingGame

    private var mine: Bool { nulls.gameTag == game.id }

    var body: some View {
        HStack(spacing: 10) {
            if mine && nulls.active {
                if nulls.state == "preparing" || nulls.state == "starting" || nulls.seeking != nil {
                    ProgressView().controlSize(.small)
                }
                Image(systemName: "play.tv").foregroundStyle(.secondary)
                Text(status).font(.callout).foregroundStyle(.secondary).lineLimit(2)
                Button(role: .destructive) { nulls.stop() } label: { Label("Stop", systemImage: "stop.fill") }
                    .help("Ends the game in Null's (CR_4k stays open for the next one).")
            } else {
                Button { nulls.watch(game) } label: { Label("Watch in Null's", systemImage: "play.tv") }
                    .disabled(nulls.running || game.path == nil)
                    .help("Plays this game again in Null's Royale itself, on the CR_4k emulator (its window shows the game). Play, the progress bar and speed here then drive it.")
                if mine || nulls.state == "closed" || (nulls.state == "failed" && nulls.gameTag == nil) {
                    Text(nulls.message).font(.callout)
                        .foregroundStyle(nulls.state == "failed" ? Color.red : .secondary).lineLimit(3)
                        .textSelection(.enabled)
                } else if nulls.active {
                    Text("Null's is playing another game.").font(.callout).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 0)
            Button { nulls.shutDown() } label: { Image(systemName: "power") }
                .disabled(nulls.running && !nulls.active)
                .help("Shut down the CR_4k emulator (it uses ~85% CPU while open): before you play live on MuMu.")
        }
    }

    private var status: String {
        if let target = nulls.seeking {
            return "Running ahead at 4× to \(gameClock(Double(target)))…"
        }
        if nulls.state == "playing" || nulls.state == "paused", let tick = nulls.tick {
            let cards = nulls.of > 0 ? " · card \(nulls.plays) of \(nulls.of)" : ""
            let skipped = nulls.skipped > 0 ? " · \(nulls.skipped) could not be played" : ""
            return (nulls.state == "paused" ? "In Null's, paused at " : "In Null's: ") + gameClock(Double(tick)) + cards + skipped
        }
        return nulls.message
    }
}

// MARK: - Import: games brought over from the PC

/// Copies what the user picked or dropped (zips, folders, recordings; the Desktop is fine) into
/// runs/inbox, where il.recordings files it under runs/rl/<run>. The copy is ours, so the Python
/// side never needs access to the user's folders.
func stageForImport(_ urls: [URL], root: URL) -> URL? {
    let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
    let inbox = root.appendingPathComponent("runs/inbox/\(stamp)")
    do {
        try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
        for url in urls {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            try FileManager.default.copyItem(at: url, to: inbox.appendingPathComponent(url.lastPathComponent))
        }
        return inbox
    } catch {
        return nil
    }
}

@MainActor
func pickRecordings() -> [URL] {
    let panel = NSOpenPanel()
    panel.title = "Import training games"
    panel.message = "Pick clapha-recordings.zip from the PC (pack-recordings.cmd), or folders / .jsonl.zst recordings."
    panel.canChooseFiles = true
    panel.canChooseDirectories = true
    panel.allowsMultipleSelection = true
    let staged = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("crtrain-stage/engine")
    if FileManager.default.fileExists(atPath: staged.path) { panel.directoryURL = staged }
    return panel.runModal() == .OK ? panel.urls : []
}
