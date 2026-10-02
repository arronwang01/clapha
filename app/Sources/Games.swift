import SwiftUI

// MARK: - Training games: whole games an RL run recorded on the PC, replayed
//
// The data is il/game_viewer.py's (`--json runs/viewer/games.json`): per game every snapshot
// (half a second apart) with the board, both players' elixir and hands, and every play. The
// learner is always at the bottom. Reload rebuilds the file from the newest recordings; Import
// brings games over from the PC (il/recordings.py: pack-recordings.cmd's zip, folders, files);
// Watch in Null's plays one again in the real game on CR_4k (Nulls.swift, il/watch_nulls.py).

struct GameFile: Decodable {
    let labels: [String]
    let games: [TrainingGame]
    let about: String
}

struct GameFrame: Decodable {
    let tick: Int
    let elixir: [Int]        // tenths of elixir: learner, opponent
    let hands: [[Int]]       // four cards and the next one, as labels: learner, opponent
    let objects: [Int]       // flat: id, (label * 4 + kind) * 2 + side, x / 100, y / 100, health %

    init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        tick = try c.decode(Int.self)
        elixir = [try c.decode(Int.self), try c.decode(Int.self)]
        hands = [try c.decode([Int].self), try c.decode([Int].self)]
        objects = try c.decode([Int].self)
    }
}

struct TrainingGame: Decodable, Identifiable {
    let tag: String
    let league: String
    let run: String?         // runs/rl/<run>
    let path: String?        // the recording, for Watch in Null's
    let title: String
    let result: String
    let sub: String
    let learner: String
    let opponent: String
    let decks: [[String]]
    let frames: [GameFrame]
    let plays: [[Int]]       // tick, side (0 = learner), label, grid x, grid y
    var id: String { tag }
    var start: Int { frames.first?.tick ?? 0 }
    var end: Int { frames.last?.tick ?? 0 }

    /// The last snapshot at or before `tick`.
    func frameIndex(at tick: Double) -> Int {
        var lo = 0, hi = frames.count - 1
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if Double(frames[mid].tick) <= tick { lo = mid } else { hi = mid - 1 }
        }
        return lo
    }
}

struct GameObject {
    let id: Int, side: Int, label: Int, kind: Int     // kind: 0 troop, 1 building, 2 tower, 3 spell
    var x: Double, y: Double
    let health: Int

    static func all(in frame: GameFrame) -> [GameObject] {
        var out: [GameObject] = []
        let a = frame.objects
        var i = 0
        while i + 4 < a.count {
            let packed = a[i + 1]
            out.append(GameObject(id: a[i], side: packed & 1, label: packed >> 3, kind: (packed >> 1) & 3,
                                  x: Double(a[i + 2]), y: Double(a[i + 3]), health: a[i + 4]))
            i += 5
        }
        return out
    }
}

@MainActor
final class GamesModel: ObservableObject {
    @Published var file: GameFile?
    @Published var problem: String?
    let url = claphaRoot().appendingPathComponent("runs/viewer/games.json")

    func load() {
        let url = self.url
        Task {
            let result = await Task.detached(priority: .userInitiated) { () -> Result<GameFile, Error> in
                Result { try JSONDecoder().decode(GameFile.self, from: Data(contentsOf: url)) }
            }.value
            switch result {
            case .success(let file): self.file = file; self.problem = nil
            case .failure: self.problem = "No games loaded yet. Press Reload to read the newest recordings."
            }
        }
    }
}

let learnerColor = Color(red: 0.35, green: 0.62, blue: 1)
let opponentColor = Color(red: 1, green: 0.38, blue: 0.35)

func gameClock(_ tick: Double) -> String {
    let seconds = max(0, tick / 20)
    return String(format: "%d:%02d", Int(seconds) / 60, Int(seconds) % 60) + (seconds >= 180 ? " OT" : "")
}

// MARK: - The window

struct TrainingGamesView: View {
    @StateObject private var model: GamesModel
    @StateObject private var builder = TaskRunner(root: claphaRoot())
    @StateObject private var nulls = NullsPlayer(root: claphaRoot())
    @State private var selection: String?
    @State private var run = "all"
    @State private var opponent = "all"
    @State private var outcome = "all"
    @State private var tick: Double = 0
    @State private var playing = false
    @State private var speed: Double = 2
    @State private var arrived: URL?          // a zip from the PC not imported yet (newFromPC)
    @State private var importing: URL?
    private let timer = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()
    private let look = Timer.publish(every: 5, on: .main, in: .common).autoconnect()

    @MainActor
    init(model: GamesModel? = nil, selection: String? = nil, tick: Double = 0) {
        _model = StateObject(wrappedValue: model ?? GamesModel())
        _selection = State(initialValue: selection)
        _tick = State(initialValue: tick)
    }

    private static let opponents: [(String, String)] = [
        ("all", "Everyone"), ("anchor", "The anchor: v2, or ex1's frozen pilot3"), ("hog2", "hog2, no delay"),
        ("general", "General, real decks"), ("self", "Itself"), ("snap", "Its snapshots"),
    ]

    private static let reload = "./py -m il.game_viewer runs/rl --max 80 --json runs/viewer/games.json"

    private var runs: [String] { Array(Set((model.file?.games ?? []).map { $0.run ?? "other" })).sorted() }

    private var shown: [TrainingGame] {
        (model.file?.games ?? []).filter { game in
            (run == "all" || (game.run ?? "other") == run)
                && (opponent == "all" || game.league == opponent)
                && (outcome == "all" || game.result.hasPrefix(outcome))
        }
    }

    /// pack-recordings.cmd's zip where ToDesk puts files (its last folder, ~/crtrain-stage/engine) or
    /// in the home folder, newer than the last one imported and done arriving (5 s unchanged).
    private func newFromPC() -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let places = ["crtrain-stage/engine", "crtrain-stage", ""].map { home.appendingPathComponent($0) }
        var newest: (URL, Date)?
        for place in places {
            let names = (try? FileManager.default.contentsOfDirectory(atPath: place.path)) ?? []
            for name in names where name.hasPrefix("clapha-recordings") && name.hasSuffix(".zip") {
                let url = place.appendingPathComponent(name)
                guard let date = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date,
                      Date().timeIntervalSince(date) > 5 else { continue }
                if newest == nil || date > newest!.1 { newest = (url, date) }
            }
        }
        guard let (url, date) = newest else { return nil }
        let done = UserDefaults.standard.string(forKey: "importedFromPC") ?? ""
        return done == "\(url.path)|\(date.timeIntervalSince1970)" ? nil : url
    }

    private func markImported(_ url: URL) {
        guard let date = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date else { return }
        UserDefaults.standard.set("\(url.path)|\(date.timeIntervalSince1970)", forKey: "importedFromPC")
    }

    private func importRecordings(_ urls: [URL]) {
        guard !urls.isEmpty, !builder.running else { return }
        importing = urls.count == 1 && urls[0] == arrived ? arrived : nil
        guard let inbox = stageForImport(urls, root: claphaRoot()) else {
            builder.output = "Could not copy what was picked into runs/inbox."
            return
        }
        let relative = "runs/inbox/" + inbox.lastPathComponent
        builder.run("Import training games",
                    "./py -m il.recordings import \(relative.shellQuoted) --remove && \(Self.reload)")
    }
    private var current: TrainingGame? { shown.first { $0.id == selection } ?? shown.first }

    var body: some View {
        HStack(spacing: 0) {
            sidebar.frame(width: 290)
            Divider()
            if let game = current, let labels = model.file?.labels {
                VStack(alignment: .leading, spacing: 10) {
                    GamePlayer(game: game, labels: labels, tick: $tick, playing: $playing, speed: $speed, nulls: nulls)
                    Divider()
                    NullsBar(nulls: nulls, game: game, speed: speed)
                }
                .padding(14)
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "film").font(.largeTitle).foregroundStyle(.secondary)
                    Text(model.problem ?? "Loading…").foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onReceive(timer) { _ in
            guard playing, let game = current else { return }
            tick = min(Double(game.end), tick + speed * 20 / 30)
            if tick >= Double(game.end) { playing = false }
        }
        .onAppear {
            if model.file == nil { model.load() }
            arrived = newFromPC()
        }
        .onReceive(look) { _ in if !builder.running { arrived = newFromPC() } }
        .dropDestination(for: URL.self) { urls, _ in
            importRecordings(urls)
            return !urls.isEmpty
        }
        .onChange(of: current?.id) { _, _ in
            playing = false
            tick = Double(current?.start ?? 0)
        }
        .onChange(of: builder.running) { _, running in
            guard !running else { return }
            if builder.lastExit == 0 {
                model.load()
                if let url = importing { markImported(url) }
            }
            importing = nil
            arrived = newFromPC()
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Training games").font(.headline)
                Spacer()
                if builder.running { ProgressView().controlSize(.small) }
                Button {
                    builder.run("Load training games", Self.reload)
                } label: { Label("Reload", systemImage: "arrow.clockwise") }
                    .disabled(builder.running)
                    .help("Reads each run's newest recorded games (runs/rl/*/recordings).")
                Button { importRecordings(pickRecordings()) } label: { Label("Import", systemImage: "square.and.arrow.down") }
                    .disabled(builder.running)
                    .help("Adds games from the PC: pick clapha-recordings.zip (made by pack-recordings.cmd), or folders and recordings. You can also drop them on this window.")
            }
            if let zip = arrived, !builder.running {
                Button { importRecordings([zip]) } label: {
                    Label("Import \(zip.lastPathComponent) from the PC", systemImage: "tray.and.arrow.down.fill")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.borderedProminent)
                .help("A new zip from pack-recordings.cmd arrived in \(zip.deletingLastPathComponent().path).")
            }
            if runs.count > 1 {
                Picker("Run", selection: $run) {
                    Text("All runs").tag("all")
                    ForEach(runs, id: \.self) { Text($0).tag($0) }
                }
            }
            Picker("Opponent", selection: $opponent) {
                ForEach(Self.opponents, id: \.0) { Text($0.1).tag($0.0) }
            }
            Picker("Result", selection: $outcome) {
                Text("Wins and losses").tag("all")
                Text("Wins").tag("won")
                Text("Losses").tag("lost")
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            Text("\(shown.count) of \(model.file?.games.count ?? 0) games, newest first")
                .font(.caption).foregroundStyle(.secondary)
            if renderingSnapshot {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(shown.prefix(12)) { row($0) }
                }
                Spacer()
            } else {
                List(shown, selection: $selection) { game in
                    row(game).tag(game.id)
                }
                .listStyle(.sidebar)
            }
            if builder.lastExit.map({ $0 != 0 }) == true {
                Text("\(builder.title) failed:\n" + String(builder.output.suffix(300)))
                    .font(.caption2.monospaced()).foregroundStyle(.red)
            } else if builder.title == "Import training games", !builder.running,
                      let line = builder.output.split(separator: "\n").last(where: { $0.hasPrefix("imported") }) {
                Text(line).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(12)
    }

    private func row(_ game: TrainingGame) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(game.title).font(.callout.weight(game.id == current?.id ? .semibold : .regular)).lineLimit(1)
            Text("\(game.result) · \(game.sub)").font(.caption)
                .foregroundStyle(game.result.hasPrefix("won") ? Color.green : game.result.hasPrefix("lost") ? Color.red : .secondary)
        }
        .padding(.vertical, 2)
    }
}

// MARK: - One game

struct GamePlayer: View {
    let game: TrainingGame
    let labels: [String]
    @Binding var tick: Double
    @Binding var playing: Bool
    @Binding var speed: Double
    @ObservedObject var nulls: NullsPlayer
    @State private var dragging = false

    private func label(_ index: Int) -> String { labels.indices.contains(index) ? labels[index] : "?" }

    var body: some View {
        let frame = game.frames[game.frameIndex(at: tick)]
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(game.title).font(.title3.weight(.semibold))
                Text(game.result).font(.callout.weight(.medium))
                    .foregroundStyle(game.result.hasPrefix("won") ? Color.green : game.result.hasPrefix("lost") ? Color.red : .secondary)
                Text(game.sub).font(.callout).foregroundStyle(.secondary)
            }
            HStack(alignment: .top, spacing: 16) {
                GameBoard(game: game, labels: labels, tick: tick)
                    .frame(maxHeight: .infinity)
                VStack(alignment: .leading, spacing: 10) {
                    side(1, frame: frame)
                    side(0, frame: frame)
                    Text("Plays").font(.subheadline.weight(.semibold))
                    VStack(alignment: .leading, spacing: 3) {
                        ForEach(Array(game.plays.filter { Double($0[0]) <= tick }.suffix(14).reversed().enumerated()),
                                id: \.offset) { _, play in
                            HStack(spacing: 8) {
                                Text(gameClock(Double(play[0])).replacingOccurrences(of: " OT", with: ""))
                                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary).frame(width: 36, alignment: .leading)
                                Circle().fill(play[1] == 0 ? learnerColor : opponentColor).frame(width: 8, height: 8)
                                Text(label(play[2])).font(.caption)
                            }
                        }
                    }
                    Spacer(minLength: 0)
                }
                .frame(minWidth: 280, maxWidth: 360, alignment: .leading)
            }
            // while the game is up in Null's these drive it: play / pause, drag anywhere, its speed
            let linked = nulls.linked(game)
            HStack(spacing: 10) {
                Button {
                    if linked { nulls.pauseOrResume(); return }
                    if !playing && tick >= Double(game.end) { tick = Double(game.start) }
                    playing.toggle()
                } label: {
                    Image(systemName: (linked ? nulls.state == "playing" : playing) ? "pause.fill" : "play.fill").frame(width: 18)
                }
                    .keyboardShortcut(.space, modifiers: [])
                Slider(value: $tick, in: Double(game.start)...Double(max(game.end, game.start + 1))) { editing in
                    dragging = editing
                    if !editing && linked { nulls.seek(Int(tick)) }
                }
                // one click each (a pop-up menu loses its choice while the board redraws under it):
                // the board's own speeds, or Null's while the game is up there (its fastest is 4x)
                let speeds: [Double] = linked ? [0.25, 0.5, 1, 2, 4] : [0.25, 0.5, 1, 2, 4, 8]
                Picker("Speed", selection: Binding(get: { linked ? nulls.speed : speed },
                                                   set: { value in
                                                       if linked { nulls.setSpeed(value) }
                                                       speed = value
                                                   })) {
                    ForEach(speeds, id: \.self) { Text(speedText($0)).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize()
                .help(linked ? "Null's speed (its fastest is 4×)." : "The board's speed.")
                Text(gameClock(tick)).font(.callout.monospacedDigit()).frame(width: 64, alignment: .trailing)
            }
        }
        .onChange(of: nulls.tick) { _, value in
            // the board follows the game in Null's, except while the bar is being dragged
            guard let value, nulls.linked(game), !dragging else { return }
            playing = false
            tick = Double(value)
        }
    }

    private func side(_ side: Int, frame: GameFrame) -> some View {
        let hand = frame.hands[side]
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Circle().fill(side == 0 ? learnerColor : opponentColor).frame(width: 10, height: 10)
                Text(side == 0 ? game.learner : game.opponent).font(.callout.weight(.semibold))
                Text(side == 0 ? "bottom" : "top").font(.caption).foregroundStyle(.secondary)
            }
            Text(game.decks[side].joined(separator: " · ")).font(.caption2).foregroundStyle(.secondary).lineLimit(2)
            HStack(spacing: 4) {
                ForEach(Array(hand.prefix(4).enumerated()), id: \.offset) { _, card in
                    Text(label(card)).font(.caption2.weight(.medium)).lineLimit(1)
                        .padding(.horizontal, 6).padding(.vertical, 3)
                        .background(RoundedRectangle(cornerRadius: 5).fill(Color.secondary.opacity(0.14)))
                }
                if hand.count > 4 {
                    Text("next: \(label(hand[4]))").font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                }
            }
            ElixirBar(label: "elixir", value: Double(frame.elixir[side]) / 10, color: .purple)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.07)))
    }
}

// MARK: - The arena, the learner at the bottom

struct GameBoard: View {
    let game: TrainingGame
    let labels: [String]
    let tick: Double

    var body: some View {
        Canvas { context, size in
            let w = size.width, h = size.height, tile = w / 18
            func point(_ x: Double, _ y: Double) -> CGPoint { CGPoint(x: x / 180 * w, y: (1 - y / 320) * h) }
            func name(_ index: Int) -> String { labels.indices.contains(index) ? labels[index] : "?" }
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(red: 0.36, green: 0.55, blue: 0.30)))
            let river = CGRect(x: 0, y: h * 15 / 32, width: w, height: h * 2 / 32)
            context.fill(Path(river), with: .color(Color(red: 0.25, green: 0.52, blue: 0.80)))
            for bx in [3.0, 14.0] {
                context.fill(Path(CGRect(x: (bx - 1) * tile, y: river.minY, width: tile * 2, height: river.height)),
                             with: .color(Color(red: 0.55, green: 0.42, blue: 0.28)))
            }
            guard !game.frames.isEmpty else { return }
            let i = game.frameIndex(at: tick)
            let now = game.frames[i], next = game.frames[min(i + 1, game.frames.count - 1)]
            let a = next.tick > now.tick ? min(1, max(0, (tick - Double(now.tick)) / Double(next.tick - now.tick))) : 0
            var ahead: [Int: GameObject] = [:]
            for o in GameObject.all(in: next) { ahead[o.id] = o }
            let objects = GameObject.all(in: now).map { o -> GameObject in
                guard let m = ahead[o.id] else { return o }
                var moved = o
                moved.x += (m.x - o.x) * a
                moved.y += (m.y - o.y) * a
                return moved
            }
            // where cards landed: a ring for a second
            for play in game.plays {
                let age = tick - Double(play[0])
                guard age >= 0, age <= 20 else { continue }
                let p = point(Double(play[3] * 10 + 5), Double(play[4] * 10 + 5))
                let r = tile * 0.9
                context.stroke(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)),
                               with: .color((play[1] == 0 ? learnerColor : opponentColor).opacity(1 - age / 20)), lineWidth: 2)
            }
            for o in objects where o.kind == 2 || o.kind == 1 {
                let p = point(o.x, o.y)
                let side = o.kind == 2 ? (name(o.label) == "King" ? 4.0 : 3.0) : 1.6
                let rect = CGRect(x: p.x - side * tile / 2, y: p.y - side * tile / 2, width: side * tile, height: side * tile)
                let color = o.side == 0 ? (o.kind == 2 ? Color.blue : learnerColor) : (o.kind == 2 ? Color.red : opponentColor)
                context.fill(Path(roundedRect: rect, cornerRadius: 3), with: .color(color.opacity(o.kind == 2 ? 0.85 : 0.75)))
                if o.health >= 0 {
                    context.fill(Path(CGRect(x: rect.minX, y: rect.minY - 5, width: rect.width, height: 3)), with: .color(.black.opacity(0.5)))
                    context.fill(Path(CGRect(x: rect.minX, y: rect.minY - 5, width: rect.width * Double(o.health) / 100, height: 3)),
                                 with: .color(.green))
                }
                context.draw(Text(o.kind == 2 ? "\(o.health)%" : String(name(o.label).prefix(6)))
                                .font(.system(size: 9, weight: .semibold)).foregroundStyle(.white), at: p)
            }
            for o in objects where o.kind == 0 {
                let p = point(o.x, o.y)
                let r = tile * 0.45
                let circle = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
                context.fill(circle, with: .color(o.side == 0 ? learnerColor : opponentColor))
                context.stroke(circle, with: .color(.white.opacity(0.8)), lineWidth: 0.8)
                if o.health >= 0 && o.health < 100 {
                    context.fill(Path(CGRect(x: p.x - r, y: p.y - r - 4, width: 2 * r * Double(o.health) / 100, height: 2)),
                                 with: .color(.green))
                }
                context.draw(Text(String(name(o.label).prefix(7))).font(.system(size: 8)).foregroundStyle(.white),
                             at: CGPoint(x: p.x, y: p.y + r + 6))
            }
            for o in objects where o.kind == 3 {
                let p = point(o.x, o.y)
                let r = tile * 0.8
                context.stroke(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)),
                               with: .color(o.side == 0 ? learnerColor : opponentColor), style: StrokeStyle(lineWidth: 2, dash: [4, 3]))
                context.draw(Text(name(o.label)).font(.system(size: 9, weight: .bold)).foregroundStyle(.yellow),
                             at: CGPoint(x: p.x, y: p.y - r - 7))
            }
        }
        .aspectRatio(18.0 / 32.0, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
