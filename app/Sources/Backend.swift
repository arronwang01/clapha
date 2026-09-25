import Foundation
import SwiftUI

// MARK: - What the Python engine reports (mac012/console.py, GET /state)

struct CardInfo: Decodable, Hashable {
    let cardId: Int
    let name: String
    let elixir: Double?
    let type: String?
}

struct PlayerInfo: Decodable {
    let side: Int
    let elixir: Double
    let hand: [CardInfo?]
    let next: CardInfo?
}

struct EntityInfo: Decodable, Identifiable {
    var id: String { "\(x):\(y):\(cardId):\(side):\(hp)" }
    let x: Int
    let y: Int
    let side: Int
    let hp: Int
    let maxHp: Int
    let cardId: Int
    let name: String
    let tower: Bool
    let effect: Bool?
}

/// A command in the game's queue: played, not yet on the board (it lands `remainingTicks`
/// later -- 21 ticks after it was issued). Drawn apart from real units.
struct PendingInfo: Decodable {
    let x: Int?
    let y: Int?
    let side: Int
    let cardId: Int
    let form: Int?
    let kind: String
    let name: String
    let remainingTicks: Int
    let screenX: Int?
    let screenY: Int?
    let screenW: Int?
    let screenH: Int?
}

struct BotInfo: Decodable {
    let running: Bool
    let armed: Bool
    let model: String?
    let status: String
    let plays: Int
    let lastPlay: String?
    let log: [String]
    let models: [String]
    let mode: String?
    let serial: String?
    let readerError: String?
    let gate: String?
    let gateOk: Bool?
    let decoding: String?
    let decodingUsed: String?
}

struct DeviceState: Decodable {
    let ok: Bool
    let age: Double?
    let bot: BotInfo
    let status: String?
    let tick: Int?
    let players: [PlayerInfo]?
    let entities: [EntityInfo]?
    let pendingCommands: [PendingInfo]?
    let localSide: Int?
}

// MARK: - One device: polls its engine and sends commands

/// The bot log on its own, so a board update does not re-lay out 120 log lines.
@MainActor
final class LogModel: ObservableObject {
    @Published var lines: [String] = []
}

/// What the board and hand draw: changes several times a second in a battle, cheap to draw.
@MainActor
final class BoardModel: ObservableObject {
    @Published var state: DeviceState? { didSet { receivedAt = Date() } }
    /// When `state` arrived, so countdowns can run smoothly between polls.
    var receivedAt = Date()
}

/// Everything the controls and status show. Only republished when it actually changes, so the
/// larger part of the panel is not re-laid out 4 times a second.
struct DeviceSummary: Equatable {
    var reachable = false
    var inBattle = false
    var clock = ""
    var status = ""
    var mode = "off"
    var model: String?
    var running = false
    var models: [String] = []
    var serial: String?
    var readerError: String?
    var gate: String?
    var gateOk: Bool?
    var botStatus = "engine not running"
    var decoding = "auto"
    var decodingUsed: String?
}

@MainActor
final class DeviceModel: ObservableObject, Identifiable {
    let id: Int
    let title: String
    let port: Int
    var state: DeviceState?
    @Published var reachable = false
    @Published var lastError: String?
    @Published var pendingMode: String?
    @Published var summary = DeviceSummary()
    let logs = LogModel()
    let board = BoardModel()
    private var timer: Timer?
    private var slowUntil = Date.distantPast
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    init(id: Int, title: String, port: Int) {
        self.id = id
        self.title = title
        self.port = port
    }

    private var polls = 0
    private var lastPayload: Data?

    func startPolling() {
        appLog("device \(id): polling port \(port)")
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                // 4 Hz in a battle; once a second otherwise.
                if self.state?.ok != true && Date() < self.slowUntil { return }
                self.slowUntil = Date().addingTimeInterval(1.0)
                await self.poll()
            }
        }
    }

    func poll() async {
        guard let url = URL(string: "http://127.0.0.1:\(port)/state?log=120") else { return }
        var request = URLRequest(url: url)
        request.timeoutInterval = 1.0
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            if data == lastPayload && reachable { return }   // nothing changed: no redraw
            lastPayload = data
            let decoded = try decoder.decode(DeviceState.self, from: data)
            if decoded.bot.log != logs.lines { logs.lines = decoded.bot.log }
            state = decoded
            board.state = decoded
            // @Published fires on every assignment, even an unchanged one: assigning these each
            // poll re-rendered the whole window 8 times a second.
            if !reachable { reachable = true }
            if lastError != nil { lastError = nil }
            publishSummary()
            polls += 1
            if polls == 1 || polls == 40 { appLog("device \(id): poll \(polls) ok, battle=\(decoded.ok) mode=\(decoded.bot.mode ?? "?")") }
        } catch {
            if reachable || polls == 0 { appLog("device \(id): poll failed: \(error)") }
            if reachable { reachable = false }
            if lastError != error.localizedDescription { lastError = error.localizedDescription }
            publishSummary()
        }
    }

    func refreshSummary() { publishSummary() }

    private func publishSummary() {
        var next = DeviceSummary()
        next.reachable = reachable
        if let s = state, reachable {
            next.inBattle = s.ok
            if s.ok, let tick = s.tick {
                next.clock = String(format: "%d:%02d", tick / 20 / 60, (tick / 20) % 60)
            }
            next.status = s.status ?? ""
            next.mode = s.bot.mode ?? (s.bot.running ? (s.bot.armed ? "play" : "watch") : "off")
            next.model = s.bot.model
            next.running = s.bot.running
            next.models = s.bot.models
            next.serial = s.bot.serial
            next.readerError = s.bot.readerError
            next.gate = s.bot.gate
            next.gateOk = s.bot.gateOk
            next.botStatus = s.bot.status
            next.decoding = s.bot.decoding ?? "auto"
            next.decodingUsed = s.bot.decodingUsed
        }
        if next != summary { summary = next }
    }

    /// off / watch / play, as one decision (console.py BOT.set_mode).
    func setMode(_ mode: String, model: String?) async {
        pendingMode = mode
        var parts = URLComponents(string: "http://127.0.0.1:\(port)/api/mode")!
        parts.queryItems = [URLQueryItem(name: "mode", value: mode)]
        if let model { parts.queryItems?.append(URLQueryItem(name: "model", value: model)) }
        var request = URLRequest(url: parts.url!)
        request.timeoutInterval = 8.0
        _ = try? await URLSession.shared.data(for: request)
        await poll()
        pendingMode = nil
    }

    var mode: String { summary.mode }

    func setDecoding(_ value: String) async {
        guard let url = URL(string: "http://127.0.0.1:\(port)/api/decoding?value=\(value)") else { return }
        var request = URLRequest(url: url)
        request.timeoutInterval = 3.0
        _ = try? await URLSession.shared.data(for: request)
        await poll()
    }
}

// MARK: - Shell tasks (backends, checks, reports), run from the repo root

@MainActor
final class TaskRunner: ObservableObject {
    @Published var title = ""
    @Published var output = ""
    @Published var running = false
    @Published var lastExit: Int32?
    private var process: Process?
    let root: URL

    init(root: URL) { self.root = root }

    /// Runs through a login zsh so the user's PATH (python, adb) is what the scripts see.
    func run(_ title: String, _ command: String) {
        guard !running else { return }
        self.title = title
        output = "$ \(command)\n"
        running = true
        lastExit = nil
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-lc", "cd \(root.path.shellQuoted) && \(command)"]
        var env = ProcessInfo.processInfo.environment
        env["CLAPHA_NO_BROWSER"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in
                self.output += text
                if self.output.count > 400_000 { self.output = String(self.output.suffix(300_000)) }
            }
        }
        process.terminationHandler = { finished in
            pipe.fileHandleForReading.readabilityHandler = nil
            Task { @MainActor in
                self.running = false
                self.lastExit = finished.terminationStatus
                self.output += "\n[exit \(finished.terminationStatus)]\n"
            }
        }
        do {
            try process.run()
            self.process = process
        } catch {
            running = false
            output += "failed to start: \(error.localizedDescription)\n"
        }
    }

    func cancel() {
        process?.terminate()
    }
}

extension String {
    var shellQuoted: String { "'" + replacingOccurrences(of: "'", with: "'\\''") + "'" }
}

/// A small diagnostic log at build/app.log: startup, first polls, errors. The app is often run
/// with nobody watching the screen, so this is how its behaviour can be checked.
func appLog(_ message: String) {
    let url = claphaRoot().appendingPathComponent("build/app.log")
    let line = "\(Date()) \(message)\n"
    if let handle = try? FileHandle(forWritingTo: url) {
        handle.seekToEndOfFile(); handle.write(line.data(using: .utf8)!); try? handle.close()
    } else {
        try? line.data(using: .utf8)!.write(to: url)
    }
}

/// The repository this app belongs to: the folder the app sits in, else ~/clapha.
func claphaRoot() -> URL {
    var url = Bundle.main.bundleURL.deletingLastPathComponent()
    for _ in 0..<3 {
        if FileManager.default.fileExists(atPath: url.appendingPathComponent("mac012/console.py").path) {
            return url
        }
        url = url.deletingLastPathComponent()
    }
    return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("clapha")
}
