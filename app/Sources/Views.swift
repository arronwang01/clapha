import SwiftUI

let modelNames: [String: String] = [
    "fl:hog1": "FirstLight · 2.6 Hog specialist 1",
    "fl:hog2": "FirstLight · 2.6 Hog specialist 2",
    "fl:il": "FirstLight · Imitation (29k steps)",
    "fl:active-il": "FirstLight · Active imitation",
    "fl:general": "FirstLight · General",
    "14k": "Own sim model · 14k",
    "40k": "Own sim model · 40k",
    "100k": "Own sim model · 100k",
]

func modelLabel(_ id: String) -> String { modelNames[id] ?? id }

struct ContentView: View {
    @StateObject private var tasks = TaskRunner(root: claphaRoot())
    @StateObject private var device1: DeviceModel
    @StateObject private var device2: DeviceModel
    @State private var showTasks = true
    private let live: Bool

    @MainActor
    init(device1: DeviceModel? = nil, device2: DeviceModel? = nil, live: Bool = true) {
        _device1 = StateObject(wrappedValue: device1 ?? DeviceModel(id: 1, title: "Device 1 · main account", port: 8777))
        _device2 = StateObject(wrappedValue: device2 ?? DeviceModel(id: 2, title: "Device 2 · second account", port: 8778))
        self.live = live
    }

    var body: some View {
        VStack(spacing: 0) {
            TopBar(tasks: tasks, devices: [device1, device2], showTasks: $showTasks)
            Divider()
            if !device1.reachable && !device2.reachable {
                HStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    Text(tasks.running ? "Starting… (emulators take ~40 s the first time)"
                                       : "The engines are not running. Start everything brings up the emulators, readers and both devices.")
                    Spacer()
                    if !tasks.running {
                        Button("Start everything") {
                            tasks.run("Start emulators and engines", "./start-consoles.sh")
                        }.buttonStyle(.borderedProminent)
                    }
                }
                .padding(10)
                .background(Color.orange.opacity(0.12))
            }
            HStack(alignment: .top, spacing: 0) {
                DevicePanel(device: device1)
                Divider()
                DevicePanel(device: device2)
            }
            if showTasks {
                Divider()
                TasksPanel(tasks: tasks)
                    .frame(height: 230)
            }
        }
        .onAppear {
            appLog("window appeared (live=\(live))")
            guard live else { return }
            device1.startPolling()
            device2.startPolling()
        }
    }
}

// MARK: - Top bar: bring everything up / down, run checks

struct TopBar: View {
    @ObservedObject var tasks: TaskRunner
    let devices: [DeviceModel]
    @Binding var showTasks: Bool

    var body: some View {
        HStack(spacing: 10) {
            Text("Clapha").font(.title2.weight(.semibold))
            Text("live reader · FirstLight bot").foregroundStyle(.secondary)
            Spacer()
            Button {
                tasks.run("Start emulators and engines", "./start-consoles.sh")
            } label: { Label("Start everything", systemImage: "play.circle.fill") }
                .buttonStyle(.borderedProminent)
                .disabled(tasks.running)
            Button {
                tasks.run("Stop engines", "pkill -f mac012/console.py; sleep 1; echo engines stopped")
            } label: { Label("Stop engines", systemImage: "stop.circle") }
                .disabled(tasks.running)
            Button {
                tasks.run("Check devices", "CR_MUMU_SERIAL=127.0.0.1:26624 ./py mac012/preflight.py 127.0.0.1:26624 127.0.0.1:26656")
            } label: { Label("Check devices", systemImage: "checkmark.shield") }
                .disabled(tasks.running)
            Toggle(isOn: $showTasks) { Label("Tasks", systemImage: "terminal") }
                .toggleStyle(.button)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }
}

// MARK: - One device

struct DevicePanel: View {
    @ObservedObject var device: DeviceModel
    @State private var chosenModel: String = "fl:hog2"

    private var sum: DeviceSummary { device.summary }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            HStack(alignment: .top, spacing: 12) {
                BoardView(board: device.board)
                    .frame(width: 250, height: 444)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                VStack(alignment: .leading, spacing: 10) {
                    controls
                    HandView(board: device.board)
                    LogView(logs: device.logs)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .onAppear { if let m = sum.model { chosenModel = m } }
        .onChange(of: sum.model) { _, new in if let new, sum.running { chosenModel = new } }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text(device.title).font(.headline)
            Text(sum.serial ?? "port \(device.port)").font(.caption).foregroundStyle(.secondary)
            Spacer()
            StatusChip(text: readerText, color: readerColor)
            StatusChip(text: battleText, color: battleColor)
        }
    }

    private var readerText: String {
        if !sum.reachable { return "engine not running" }
        if let err = sum.readerError, !err.isEmpty { return "reader: \(err.prefix(40))" }
        return "reader live"
    }
    private var readerColor: Color {
        if !sum.reachable { return .gray }
        if let err = sum.readerError, !err.isEmpty { return .orange }
        return .green
    }
    private var battleText: String {
        guard sum.reachable else { return "—" }
        if sum.inBattle { return "battle \(sum.clock)" }
        switch sum.status {
        case "terminal_or_unverified_towers": return "battle over"
        case "paused_or_stalled": return "paused"
        default: return "no battle"
        }
    }
    private var battleColor: Color { sum.inBattle ? .blue : .gray }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            ModeSwitch(mode: device.pendingMode ?? device.mode,
                       enabled: device.reachable && device.pendingMode == nil) { mode in
                Task { await device.setMode(mode, model: chosenModel) }
            }
            Text(modeHelp).font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            DecodingSwitch(value: sum.decoding, used: sum.decodingUsed, enabled: sum.reachable) { value in
                Task { await device.setDecoding(value) }
            }
            ModelList(models: orderedModels, chosen: $chosenModel, running: sum.running ? sum.model : nil)
            if sum.running, let m = sum.model, m != chosenModel {
                Button("Switch the running model to \(modelLabel(chosenModel))") {
                    Task { await device.setMode(device.mode, model: chosenModel) }
                }.font(.caption)
            }
            Text(sum.botStatus)
                .font(.callout.monospaced())
                .lineLimit(2)
                .foregroundStyle(sum.botStatus.contains("FAIL") ? .red : .primary)
            if let gate = sum.gate, !gate.isEmpty {
                Label(gate, systemImage: sum.gateOk == true ? "lock.open" : "lock")
                    .font(.caption).foregroundStyle(sum.gateOk == true ? Color.secondary : Color.red)
            }
        }
    }

    private var orderedModels: [String] {
        let all = sum.models.isEmpty ? Array(modelNames.keys) : sum.models
        return all.sorted { a, b in
            let fa = a.hasPrefix("fl:"), fb = b.hasPrefix("fl:")
            return fa != fb ? fa : a < b
        }
    }

    private var modeHelp: String {
        switch device.pendingMode ?? device.mode {
        case "play": return "Play: the model decides and taps. Only against Training Camp bots or your own accounts (scope gate)."
        case "watch": return "Watch: the model decides every turn and logs what it would play. No taps."
        default: return "Off: no model running. Pick a model, then Watch or Play — before the battle starts."
        }
    }
}

/// Off · Watch · Play as one big switch: which one is lit is what the engine is doing.
struct ModeSwitch: View {
    let mode: String
    let enabled: Bool
    let choose: (String) -> Void

    private let options: [(String, String, String, Color)] = [
        ("off", "Off", "power", .gray),
        ("watch", "Watch", "eye", .blue),
        ("play", "Play", "hand.tap.fill", .green),
    ]

    var body: some View {
        HStack(spacing: 0) {
            ForEach(options, id: \.0) { option in
                let selected = mode == option.0
                Button { if !selected { choose(option.0) } } label: {
                    Label(option.1, systemImage: option.2)
                        .font(.callout.weight(selected ? .semibold : .regular))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 7)
                        .foregroundStyle(selected ? Color.white : Color.primary)
                        .background(selected ? option.3 : Color.secondary.opacity(0.10))
                }
                .buttonStyle(.plain)
                .disabled(!enabled)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.3)))
        .frame(maxWidth: 330)
        .opacity(enabled ? 1 : 0.6)
    }
}

/// How the policy picks among its options, matched to how FirstLight runs each setup.
struct DecodingSwitch: View {
    let value: String
    let used: String?
    let enabled: Bool
    let choose: (String) -> Void

    var body: some View {
        HStack(spacing: 8) {
            Text("Decoding").font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 0) {
                ForEach([("auto", "Auto"), ("sampled", "Sampled"), ("greedy", "Greedy")], id: \.0) { option in
                    let selected = value == option.0
                    Button { if !selected { choose(option.0) } } label: {
                        Text(option.1).font(.caption.weight(selected ? .semibold : .regular))
                            .padding(.horizontal, 10).padding(.vertical, 3)
                            .foregroundStyle(selected ? Color.white : Color.primary)
                            .background(selected ? Color.accentColor : Color.secondary.opacity(0.10))
                    }
                    .buttonStyle(.plain)
                    .disabled(!enabled)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            if let used {
                Text("this battle: \(used)").font(.caption2).foregroundStyle(.secondary)
            }
        }
        .help("Auto: sampled against you or a bot (FirstLight's human-vs-AI), greedy when both devices run models against each other (FirstLight's AI duel).")
    }
}

/// The models as a short list; the dot marks the one the engine is running.
struct ModelList: View {
    let models: [String]
    @Binding var chosen: String
    let running: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Model").font(.caption).foregroundStyle(.secondary)
            ForEach(models, id: \.self) { id in
                Button { chosen = id } label: {
                    HStack(spacing: 6) {
                        Image(systemName: chosen == id ? "largecircle.fill.circle" : "circle")
                            .foregroundStyle(chosen == id ? Color.accentColor : .secondary)
                        Text(modelLabel(id)).font(.caption)
                        if running == id {
                            Text("running").font(.caption2.weight(.semibold)).foregroundStyle(.green)
                        }
                        Spacer()
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .frame(maxWidth: 330)
    }
}

struct StatusChip: View {
    let text: String
    let color: Color
    var body: some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 7, height: 7)
            Text(text).font(.caption)
        }
        .padding(.horizontal, 8).padding(.vertical, 3)
        .background(Capsule().fill(color.opacity(0.12)))
    }
}

// MARK: - Board: drawn from our side, like the game

struct BoardView: View {
    @ObservedObject var board: BoardModel
    private var state: DeviceState? { board.state }

    var body: some View {
        Canvas { context, size in
            let w = size.width, h = size.height
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(red: 0.36, green: 0.55, blue: 0.30)))
            let local = state?.localSide ?? 0
            func point(_ x: Int, _ y: Int) -> CGPoint {
                let fx = Double(x) / 18000.0, fy = Double(y) / 32000.0
                return local == 1 ? CGPoint(x: (1 - fx) * w, y: fy * h)
                                  : CGPoint(x: fx * w, y: (1 - fy) * h)
            }
            let tile = w / 18.0
            // river and bridges
            let river = CGRect(x: 0, y: h * 15.0 / 32.0, width: w, height: h * 2.0 / 32.0)
            context.fill(Path(river), with: .color(Color(red: 0.25, green: 0.52, blue: 0.80)))
            for bx in [3.0, 14.0] {
                let bridge = CGRect(x: (local == 1 ? 18 - bx - 1 : bx - 1) * tile, y: river.minY, width: tile * 2, height: river.height)
                context.fill(Path(bridge), with: .color(Color(red: 0.55, green: 0.42, blue: 0.28)))
            }
            guard let state, state.ok, let entities = state.entities else {
                context.draw(Text(state == nil ? "engine not running" : "no battle").foregroundStyle(.white),
                             at: CGPoint(x: w / 2, y: h / 2))
                return
            }
            for e in entities where e.tower && e.hp > 0 {
                let p = point(e.x, e.y)
                let side: Double = e.name == "King" ? 4 : 3
                let rect = CGRect(x: p.x - side * tile / 2, y: p.y - side * tile / 2, width: side * tile, height: side * tile)
                context.fill(Path(roundedRect: rect, cornerRadius: 3), with: .color(e.side == local ? .blue : .red))
                let frac = e.maxHp > 0 ? Double(e.hp) / Double(e.maxHp) : 0
                context.fill(Path(CGRect(x: rect.minX, y: rect.minY - 5, width: rect.width, height: 3)), with: .color(.black.opacity(0.5)))
                context.fill(Path(CGRect(x: rect.minX, y: rect.minY - 5, width: rect.width * frac, height: 3)), with: .color(.green))
            }
            for e in entities where e.effect == true {
                // a shot or spell in flight
                let p = point(e.x, e.y)
                let r = tile * 0.22
                context.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)),
                             with: .color(.yellow))
            }
            for e in entities where !e.tower && e.effect != true && e.hp > 0 {
                let p = point(e.x, e.y)
                let r = tile * 0.45
                let circle = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
                context.fill(circle, with: .color(e.side == local ? Color(red: 0.35, green: 0.62, blue: 1) : Color(red: 1, green: 0.38, blue: 0.35)))
                context.stroke(circle, with: .color(.white.opacity(0.8)), lineWidth: 0.8)
                context.draw(Text(String(e.name.prefix(4))).font(.system(size: 7)).foregroundStyle(.white),
                             at: CGPoint(x: p.x, y: p.y - r - 5))
            }
        }
    }
}

// MARK: - Hand and elixir

struct HandView: View {
    @ObservedObject var board: BoardModel
    private var state: DeviceState? { board.state }

    var body: some View {
        let local = state?.localSide
        let me = state?.players?.first { $0.side == local }
        let them = state?.players?.first { $0.side != local }
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                ForEach(0..<4, id: \.self) { index in
                    let card = (me?.hand.count ?? 0) > index ? me?.hand[index] : nil
                    CardChip(card: card ?? nil)
                }
                if let next = me?.next {
                    Text("next: \(next.name)").font(.caption2).foregroundStyle(.secondary)
                }
            }
            ElixirBar(label: "you", value: me?.elixir ?? 0, color: .purple)
            ElixirBar(label: "them", value: them?.elixir ?? 0, color: .gray)
        }
        .opacity(state?.ok == true ? 1 : 0.35)
    }
}

struct CardChip: View {
    let card: CardInfo?
    var body: some View {
        VStack(spacing: 2) {
            Text(card?.name ?? "—").font(.caption2.weight(.medium)).lineLimit(1)
            Text(card?.elixir.map { String(Int($0)) } ?? "").font(.caption2).foregroundStyle(.purple)
        }
        .frame(width: 68, height: 34)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.12)))
    }
}

struct ElixirBar: View {
    let label: String
    let value: Double
    let color: Color
    var body: some View {
        HStack(spacing: 6) {
            Text(label).font(.caption2).foregroundStyle(.secondary).frame(width: 30, alignment: .leading)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3).fill(Color.secondary.opacity(0.15))
                    RoundedRectangle(cornerRadius: 3).fill(color).frame(width: geo.size.width * min(1, max(0, value / 10)))
                }
            }.frame(height: 8)
            Text(String(format: "%.1f", value)).font(.caption2.monospacedDigit()).frame(width: 28)
        }
    }
}

// MARK: - Log

/// Set by the off-screen snapshot tool: scroll views cannot be drawn there.
nonisolated(unsafe) var renderingSnapshot = false

struct LogView: View {
    @ObservedObject var logs: LogModel
    private var lines: [String] { logs.lines }
    var body: some View {
        if renderingSnapshot {
            VStack(alignment: .leading, spacing: 1) {
                ForEach(Array(lines.suffix(16).enumerated()), id: \.offset) { _, line in
                    Text(line).font(.system(size: 10.5, design: .monospaced))
                        .foregroundStyle(color(for: line)).lineLimit(1)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                Spacer(minLength: 0)
            }
            .padding(6)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
        } else {
            scrolling
        }
    }

    private var scrolling: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(lines.enumerated()), id: \.offset) { index, line in
                        Text(line)
                            .font(.system(size: 10.5, design: .monospaced))
                            .foregroundStyle(color(for: line))
                            .lineLimit(1)
                            .frame(height: 14)
                            .textSelection(.enabled)
                            .id(index)
                    }
                }
                .padding(6)
            }
            .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
            .onChange(of: lines.count) { _, count in
                if count > 0 { proxy.scrollTo(count - 1, anchor: .bottom) }
            }
        }
    }

    private func color(for line: String) -> Color {
        if line.contains("failed") || line.contains("MISPLACED") || line.contains("BLOCK") || line.contains("NOT ") { return .red }
        if line.contains("missed") || line.contains("MISMATCH") || line.contains("unknown") { return .orange }
        if line.contains("ABILITY") { return .purple }
        if line.contains("latency") || line.contains("placement") { return .secondary }
        return .primary
    }
}

// MARK: - Tasks: every check and report, one click each

struct TasksPanel: View {
    @ObservedObject var tasks: TaskRunner

    private let actions: [(String, String, String)] = [
        ("Input report (last match)", "doc.text.magnifyingglass",
         "./py mac012/input_coverage.py | sed -n '/=== match_scalars/,$p' | grep -E '^===|^!!!|^ +[0-9]+ '"),
        ("Replay last match through the model", "arrow.counterclockwise",
         "./py mac012/replay_decide.py"),
        ("Hero / evolution checks", "sparkles", "./py mac012/test_hero_evo.py fl:hog2"),
        ("All-cards sweep", "square.grid.3x3", "./py mac012/test_all_cards.py fl:il | tail -12"),
        ("Tap benchmark (Training Camp)", "hand.tap", "./py mac012/tap_bench.py"),
        ("Bot log (device 1)", "text.alignleft", "tail -80 build/bot_8777.log"),
    ]

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Tasks").font(.headline)
                ForEach(actions, id: \.0) { action in
                    Button {
                        tasks.run(action.0, action.2)
                    } label: {
                        Label(action.0, systemImage: action.1).frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .disabled(tasks.running)
                }
                if tasks.running {
                    Button("Cancel", role: .destructive) { tasks.cancel() }
                }
                Spacer()
            }
            .frame(width: 270)
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(tasks.title.isEmpty ? "Output" : tasks.title).font(.headline)
                    if tasks.running { ProgressView().controlSize(.small) }
                    if let code = tasks.lastExit {
                        Text(code == 0 ? "done" : "exit \(code)").font(.caption)
                            .foregroundStyle(code == 0 ? .green : .red)
                    }
                    Spacer()
                }
                ScrollViewReader { proxy in
                    ScrollView {
                        Text(tasks.output.isEmpty ? "Output of a task shows here." : tasks.output)
                            .font(.system(size: 11, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                            .id("end")
                    }
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
                    .onChange(of: tasks.output) { _, _ in proxy.scrollTo("end", anchor: .bottom) }
                }
            }
        }
        .padding(12)
    }
}
