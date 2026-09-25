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
    @State private var page = 1
    @State private var showTools = false
    @State private var overlayOn = false
    @State private var overlay = MuMuOverlay()
    private let live: Bool

    @MainActor
    init(device1: DeviceModel? = nil, device2: DeviceModel? = nil, live: Bool = true) {
        _device1 = StateObject(wrappedValue: device1 ?? DeviceModel(id: 1, title: "Main account", port: 8777))
        _device2 = StateObject(wrappedValue: device2 ?? DeviceModel(id: 2, title: "Second account", port: 8778))
        self.live = live
    }

    var body: some View {
        VStack(spacing: 0) {
            TopBar(tasks: tasks, page: $page, device1: device1, device2: device2, showTools: $showTools,
                   overlayOn: $overlayOn)
            Divider()
            if !device1.reachable && !device2.reachable {
                HStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    Text(tasks.running ? "Starting… (emulators take ~40 s the first time)"
                                       : "Nothing is running yet. Press Start to bring up the emulators and the bot.")
                    Spacer()
                }
                .padding(10)
                .background(Color.orange.opacity(0.12))
            }
            // One device per page: the tab in the top bar picks which.
            if page == 1 {
                DevicePanel(device: device1).id(1)
            } else {
                DevicePanel(device: device2).id(2)
            }
            if showTools {
                Divider()
                TasksPanel(tasks: tasks)
                    .frame(height: 340)
            }
        }
        .onChange(of: overlayOn) { _, on in updateOverlay(on) }
        .onChange(of: page) { _, _ in updateOverlay(overlayOn) }
        .onAppear {
            appLog("window appeared (live=\(live))")
            guard live else { return }
            device1.startPolling()
            device2.startPolling()
        }
    }
}

extension ContentView {
    /// The overlay follows the device on the current page.
    func updateOverlay(_ on: Bool) {
        if on {
            overlay.show(board: (page == 1 ? device1 : device2).board)
        } else {
            overlay.hide()
        }
    }
}

// MARK: - Top bar: which device, Start, and everything else under More

struct TopBar: View {
    @ObservedObject var tasks: TaskRunner
    @Binding var page: Int
    @ObservedObject var device1: DeviceModel
    @ObservedObject var device2: DeviceModel
    @Binding var showTools: Bool
    @Binding var overlayOn: Bool

    private func tab(_ device: DeviceModel) -> String {
        device.summary.inBattle ? "\(device.title) · in battle" : device.title
    }

    var body: some View {
        HStack(spacing: 12) {
            Text("Clapha").font(.title2.weight(.semibold))
            Picker("Device", selection: $page) {
                Text(tab(device1)).tag(1)
                Text(tab(device2)).tag(2)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: 380)
            Toggle(isOn: $overlayOn) { Label("Overlay on MuMu", systemImage: "circle.dashed") }
                .toggleStyle(.button)
                .help("Draws cards that are played but not landed yet on top of the MuMu window. Click-through.")
            Spacer()
            Button {
                tasks.run("Start emulators and bot", "./start-consoles.sh")
            } label: { Label("Start", systemImage: "play.circle.fill") }
                .buttonStyle(.borderedProminent)
                .disabled(tasks.running)
                .help("Starts both emulators and the bot engines. Use after a reboot or an update, between matches.")
            Menu {
                Button("Stop the bot engines") {
                    tasks.run("Stop engines", "pkill -f mac012/console.py; sleep 1; echo engines stopped")
                }
                Button("Check devices") {
                    tasks.run("Check devices", "CR_MUMU_SERIAL=127.0.0.1:26624 ./py mac012/preflight.py 127.0.0.1:26624 127.0.0.1:26656")
                }
                Divider()
                Toggle("Show tools and output", isOn: $showTools)
            } label: { Label("More", systemImage: "ellipsis.circle") }
                .fixedSize()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }
}

// MARK: - One device

struct DevicePanel: View {
    @ObservedObject var device: DeviceModel
    @State private var chosenModel: String = "fl:hog2"
    @State private var showAdvanced = false

    private var sum: DeviceSummary { device.summary }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            HStack(alignment: .top, spacing: 16) {
                BoardView(board: device.board)
                    .frame(width: 300, height: 533)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                VStack(alignment: .leading, spacing: 12) {
                    controls
                    HandView(board: device.board)
                    LogView(logs: device.logs)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .onAppear { if let m = sum.model { chosenModel = m } }
        .onChange(of: sum.model) { _, new in if let new, sum.running { chosenModel = new } }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text(device.title).font(.headline)
            Spacer()
            StatusChip(text: readerText, color: readerColor)
            StatusChip(text: battleText, color: battleColor)
        }
    }

    private var readerText: String {
        if !sum.reachable { return "not running" }
        if let err = sum.readerError, !err.isEmpty { return "can't read the game: \(err.prefix(40))" }
        return "reading the game"
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
            Picker("Model", selection: $chosenModel) {
                ForEach(orderedModels, id: \.self) { id in
                    Text(modelLabel(id) + (sum.running && sum.model == id ? "  — running" : "")).tag(id)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 360)
            if sum.running, let m = sum.model, m != chosenModel {
                Button("Switch the running model to \(modelLabel(chosenModel))") {
                    Task { await device.setMode(device.mode, model: chosenModel) }
                }.font(.caption)
            }
            Text(sum.botStatus)
                .font(.callout.monospaced())
                .lineLimit(2)
                .foregroundStyle(sum.botStatus.contains("FAIL") ? .red : .primary)
            // The scope gate only needs attention when it blocks.
            if let gate = sum.gate, !gate.isEmpty, sum.gateOk == false {
                Label(gate, systemImage: "lock").font(.caption).foregroundStyle(.red)
            }
            DisclosureGroup("Advanced", isExpanded: $showAdvanced) {
                DecodingSwitch(value: sum.decoding, used: sum.decodingUsed, enabled: sum.reachable) { value in
                    Task { await device.setDecoding(value) }
                }
                .padding(.top, 4)
            }
            .font(.caption)
            .frame(maxWidth: 360, alignment: .leading)
        }
    }

    /// FirstLight models first; the old simulator models only if nothing else is available.
    private var orderedModels: [String] {
        let all = sum.models.isEmpty ? Array(modelNames.keys) : sum.models
        let firstLight = all.filter { $0.hasPrefix("fl:") }.sorted()
        var list = firstLight.isEmpty ? all.sorted() : firstLight
        if !list.contains(chosenModel) { list.append(chosenModel) }
        return list
    }

    private var modeHelp: String {
        switch device.pendingMode ?? device.mode {
        case "play": return "Play: the bot plays this account. Training Camp or your own accounts only."
        case "watch": return "Watch: the bot thinks along and logs what it would play, without touching the game."
        default: return "Off. Pick a model, then Watch or Play before the battle starts."
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
            // Played but not landed yet: a dashed ghost ring with the card and the seconds
            // until it lands. Opponent ones show ~0.7 s ahead; a real unit is a filled circle.
            for pending in state.pendingCommands ?? [] where pending.kind == "card" {
                guard let x = pending.x, let y = pending.y else { continue }
                let p = point(x, y)
                let r = tile * 0.55
                let ring = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
                let color: Color = pending.side == local ? Color(red: 0.35, green: 0.62, blue: 1)
                                                         : Color(red: 1, green: 0.38, blue: 0.35)
                context.stroke(ring, with: .color(color), style: StrokeStyle(lineWidth: 1.6, dash: [3, 2]))
                let seconds = String(format: "%.1fs", Double(max(0, pending.remainingTicks)) / 20.0)
                context.draw(Text("\(String(pending.name.prefix(5))) \(seconds)")
                                .font(.system(size: 7, weight: .semibold)).foregroundStyle(color),
                             at: CGPoint(x: p.x, y: p.y + r + 5))
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

    /// (button, icon, what it does in plain words, command)
    private let actions: [(String, String, String, String)] = [
        ("Show the bot's log", "text.alignleft",
         "The last 80 lines of what the bot on the main account did and why.",
         "tail -80 build/bot_8777.log"),
        ("What did the bot see last match?", "doc.text.magnifyingglass",
         "Lists every kind of game information the model gets, and which ones stayed empty.",
         "./py mac012/input_coverage.py | sed -n '/=== match_scalars/,$p' | grep -E '^===|^!!!|^ +[0-9]+ '"),
        ("Re-run last match through the model", "arrow.counterclockwise",
         "Feeds the recorded match back in, offline, to check nothing errors. Nothing is tapped.",
         "./py mac012/replay_decide.py"),
        ("Check hero & evolution handling", "sparkles",
         "Offline test that heroes, abilities and evolved cards reach the model correctly.",
         "./py mac012/test_hero_evo.py fl:hog2"),
        ("Check every card can be placed", "square.grid.3x3",
         "Offline test: every card in the game, both sides, gets a legal placement.",
         "./py mac012/test_all_cards.py fl:il | tail -12"),
        ("Measure tap speed", "hand.tap",
         "Run inside a Training Camp battle: places cheap cards to time how fast taps register.",
         "./py mac012/tap_bench.py"),
    ]

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Tools").font(.headline)
                ForEach(actions, id: \.0) { action in
                    VStack(alignment: .leading, spacing: 1) {
                        Button {
                            tasks.run(action.0, action.3)
                        } label: {
                            Label(action.0, systemImage: action.1).frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .disabled(tasks.running)
                        Text(action.2).font(.caption2).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                if tasks.running {
                    Button("Cancel", role: .destructive) { tasks.cancel() }
                }
                Spacer()
            }
            .frame(width: 300)
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
