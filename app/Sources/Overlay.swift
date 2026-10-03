import AppKit
import SwiftUI

/// Ghost markers drawn over the MuMu window itself: a transparent, click-through window that
/// follows MuMu's game view. Each queued command (played, not landed yet) is a dashed ring
/// with the card and a countdown, at the device pixel the engine computed with the same
/// geometry the bot's taps use. Nothing is sent to the game; clicks pass straight through.
/// Where things are inside the overlay window: the game picture, and a strip beside it for the opponent panel
/// when the screen has room (so the panel covers nothing of the game).
@MainActor
final class OverlayLayout: ObservableObject {
    @Published var game = CGRect.zero
    @Published var side: CGRect?
}

@MainActor
final class MuMuOverlay {
    private var window: NSWindow?
    private var timer: Timer?
    private let layout = OverlayLayout()

    var isShown: Bool { window != nil }

    func show(board: BoardModel) {
        hide()
        let window = NSWindow(contentRect: .zero, styleMask: [.borderless], backing: .buffered, defer: false)
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.ignoresMouseEvents = true
        window.level = .floating
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        window.contentView = NSHostingView(rootView: OverlayView(board: board, layout: layout))
        self.window = window
        follow()
        window.orderFrontRegardless()
        timer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.follow() }
        }
    }

    func hide() {
        timer?.invalidate()
        timer = nil
        window?.orderOut(nil)
        window = nil
    }

    /// Put the overlay exactly over the game picture of the frontmost MuMu window: the
    /// window minus its title bar, fitted to the game's 9:16 screen.
    private func follow() {
        guard let window else { return }
        guard let game = MuMuOverlay.gameRect() else {
            window.orderOut(nil)
            return
        }
        // A strip beside the game for the opponent panel: to the right if the screen has the room, else to the
        // left, else none (the panel is then drawn small inside the picture's top left).
        let screen = NSScreen.screens.first { $0.frame.intersects(game) }?.visibleFrame ?? game
        let strip = min(320, max(230, game.width * 0.6)), space = 8.0
        var frame = game
        var inside = CGRect(x: 0, y: 0, width: game.width, height: game.height)
        var side: CGRect?
        if screen.maxX - game.maxX >= strip + space {
            frame.size.width += strip + space
            side = CGRect(x: game.width + space, y: 0, width: strip, height: game.height)
        } else if game.minX - screen.minX >= strip + space {
            frame.origin.x -= strip + space
            frame.size.width += strip + space
            inside.origin.x = strip + space
            side = CGRect(x: 0, y: 0, width: strip, height: game.height)
        }
        if window.frame != frame { window.setFrame(frame, display: true) }
        if layout.game != inside { layout.game = inside }
        if layout.side != side { layout.side = side }
        if !window.isVisible { window.orderFrontRegardless() }
    }

    static func gameRect() -> NSRect? {
        let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else { return nil }
        // Front-to-back order: the first large MuMu window is the one on top.
        for info in list {
            guard let owner = info[kCGWindowOwnerName as String] as? String, owner.contains("MuMu"),
                  (info[kCGWindowLayer as String] as? Int) == 0,
                  let bounds = info[kCGWindowBounds as String] as? [String: CGFloat],
                  let x = bounds["X"], let y = bounds["Y"], let w = bounds["Width"], let h = bounds["Height"],
                  w > 200, h > 300 else { continue }
            // Title bar = what is left above a full-width 9:16 picture; if the window is wider
            // than 9:16, the picture is letterboxed at full height instead.
            var gameW = w, gameH = w * 16.0 / 9.0
            var left = x, top = y + max(0, h - gameH)
            if gameH > h {
                gameH = h
                gameW = h * 9.0 / 16.0
                left = x + (w - gameW) / 2
                top = y
            }
            // CoreGraphics measures from the top of the main display; AppKit from its bottom.
            let mainHeight = NSScreen.screens.first?.frame.height ?? 0
            return NSRect(x: left, y: mainHeight - (top + gameH), width: gameW, height: gameH)
        }
        return nil
    }
}

/// The landing-warning art (landing-hud-kit, the owner's style): unit models in `figures/`,
/// card portraits in `card_icons/`. Unpacked by app/build.sh into build/hud (git-ignored:
/// these are Supercell's own assets, for use inside this project only).
@MainActor
enum HudArt {
    private static var cache: [String: NSImage] = [:]
    private static var missing: Set<String> = []
    static let root = claphaRoot().appendingPathComponent("build/hud/landing-hud-kit")

    private static func load(_ relative: String) -> NSImage? {
        if let image = cache[relative] { return image }
        if missing.contains(relative) { return nil }
        guard let image = NSImage(contentsOf: root.appendingPathComponent(relative)) else {
            missing.insert(relative)
            return nil
        }
        cache[relative] = image
        return image
    }

    /// Kit lookup rule: the unit's figure (forms use the base card's); otherwise the card
    /// icon, the evolution/hero portrait when the card was played in that form.
    static func figure(_ card: Int) -> NSImage? { load("figures/\(card).png") }

    static func icon(_ card: Int, form: Int) -> NSImage? {
        let suffix = form == 1 ? "_evo" : form == 2 ? "_hero" : ""
        return (suffix.isEmpty ? nil : load("card_icons/\(card)\(suffix).png"))
            ?? load("card_icons/\(card).png")
    }
}

/// Landing warning for the opponent's plays, as LANDING-HUD-DESIGN.md specifies (sizes there are at 1080 px wide and are
/// scaled to the overlay's width here). Positions come from the engine's tap geometry.
struct OverlayView: View {
    @ObservedObject var board: BoardModel
    @ObservedObject var layout: OverlayLayout
    /// The opponent panel (deck guessed, cards seen, hand, elixir): a debugging aid, switched in the top bar.
    @AppStorage("overlayOpponent") private var showOpponent = true

    private static let red = Color(red: 235 / 255, green: 70 / 255, blue: 70 / 255)
    private static let gold = Color(red: 255 / 255, green: 205 / 255, blue: 60 / 255)
    private static let elixir = Color(red: 214 / 255, green: 92 / 255, blue: 255 / 255)

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas { context, size in
                draw(&context, size: size, now: timeline.date)
            }
        }
        .allowsHitTesting(false)
    }

    private func draw(_ whole: inout GraphicsContext, size window: CGSize, now: Date) {
        guard let state = board.state, state.ok else { return }
        // the game picture inside the window (the whole of it when there is no strip beside)
        let game = layout.game.width > 0 ? layout.game : CGRect(origin: .zero, size: window)
        let size = game.size
        if showOpponent, let opponent = state.opponent {
            if let side = layout.side {
                // beside the game: as wide as the strip
                drawOpponent(&whole, left: side.minX + 8, top: side.minY + 10, k: (side.width - 16) / 334, info: opponent)
            } else {
                // no room beside: small, inside the picture's top left, under the opponent's name
                drawOpponent(&whole, left: game.minX + 10 * size.width / 1080, top: 118 * size.width / 1080,
                             k: size.width / 1080, info: opponent)
            }
        }
        var context = whole
        context.translateBy(x: game.minX, y: game.minY)
        let local = state.localSide ?? 0
        let k = size.width / 1080.0                       // spec pixels -> overlay points
        let t = now.timeIntervalSinceReferenceDate
        let pulse = 0.5 + 0.5 * sin(t * 9)
        // Ticks elapsed since this state was read, so the countdown runs between polls.
        let elapsedTicks = max(0, now.timeIntervalSince(board.receivedAt)) * 20
        // Opponent only: your own plays are not marked on the game.
        for pending in state.pendingCommands ?? [] where pending.kind == "card" && pending.side != local {
            guard let sx = pending.screenX, let sy = pending.screenY,
                  let sw = pending.screenW, let sh = pending.screenH, sw > 0, sh > 0 else { continue }
            let p = CGPoint(x: Double(sx) / Double(sw) * size.width,
                            y: Double(sy) / Double(sh) * size.height)
            // remaining_ticks -1: executed but still on its way (a spell in flight, a Miner
            // underground, a Barrel in the air) -- no countdown, just "incoming".
            let incoming = pending.remainingTicks < 0
            let seconds = max(0, Double(pending.remainingTicks) - elapsedTicks) / 20
            let colour = Self.red

            // A spell's range, until it hits.
            if let rx = pending.radiusX, let ry = pending.radiusY, rx > 0, ry > 0 {
                let ex = Double(rx) / Double(sw) * size.width
                let ey = Double(ry) / Double(sh) * size.height
                let area = Path(ellipseIn: CGRect(x: p.x - ex, y: p.y - ey, width: 2 * ex, height: 2 * ey))
                context.fill(area, with: .color(colour.opacity(0.10 + 0.08 * pulse)))
                context.stroke(area, with: .color(colour.opacity(0.85)),
                               style: StrokeStyle(lineWidth: 4 * k, dash: [14 * k, 9 * k]))
            }

            // Ring: flat ellipse (the board is foreshortened), pulsing radius and opacity.
            let r = (95 + 12 * pulse) * k
            let ring = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - 0.78 * r, width: 2 * r,
                                              height: 2 * 0.78 * r))
            context.stroke(ring, with: .color(colour.opacity((150 + 80 * pulse) / 255)),
                           lineWidth: 7 * k)

            let label = incoming ? "incoming" : String(format: "%.1fs", seconds)
            let top: Double
            if let figure = HudArt.figure(pending.cardId) {
                // Unit: its model standing in the ring, feet 22 px below the centre.
                let rect = CGRect(x: p.x - 75 * k, y: p.y + 22 * k - 150 * k,
                                  width: 150 * k, height: 150 * k)
                context.draw(Image(nsImage: figure), in: rect)
                top = rect.minY
            } else if let icon = HudArt.icon(pending.cardId, form: pending.form ?? 0) {
                // Spell (or a unit without a figure): the card icon on the spot.
                let rect = CGRect(x: p.x - 36 * k, y: p.y - 36 * k, width: 72 * k, height: 72 * k)
                context.draw(Image(nsImage: icon), in: rect)
                top = rect.minY
            } else {
                context.draw(Text(pending.name).font(.system(size: 22 * k, weight: .bold))
                                .foregroundStyle(colour), at: p)
                top = p.y - 20 * k
            }
            // Gold countdown 44 px above the figure, never above the top 215 px.
            let y = max(215 * k, top - 44 * k)
            context.draw(Text(label).font(.system(size: (incoming ? 26 : 34) * k, weight: .heavy))
                            .foregroundStyle(Self.gold),
                         at: CGPoint(x: p.x, y: y))
        }
    }

    // MARK: The opponent panel

    /// Beside the game picture when the screen has room, else small in its top left: up to three rows of cards and
    /// their elixir. `k` scales the panel: its cards are 40 wide and it is 334 across at k = 1.
    ///   guess   the deck held for theirs before they play, drawn faint with a dashed edge and named for where
    ///           it comes from. Not drawn at all once a card they played is not in it.
    ///   seen    the cards they have actually played, in the order first seen; the ones in their hand now edged
    ///           in gold, the ones waiting in their cycle dimmed.
    ///   hand    their four hand cards as they follow from the order of their plays ("?" for a card not seen
    ///           yet), and the card that comes back next. Not known until they have played four.
    private func drawOpponent(_ context: inout GraphicsContext, left: Double, top: Double, k: Double, info: OpponentInfo) {
        let cardW = 40 * k, cardH = 59 * k, gap = 2 * k
        let captionH = 21 * k, rowH = captionH + cardH + 7 * k
        let width = 8 * cardW + 7 * gap
        let rows = (info.guess == nil ? 0.0 : 1.0) + 2.0
        let height = rows * rowH + 40 * k
        let back = Path(roundedRect: CGRect(x: left - 6 * k, y: top - 6 * k, width: width + 12 * k, height: height + 8 * k),
                        cornerRadius: 8 * k)
        context.fill(back, with: .color(.black.opacity(0.5)))
        var y = top

        func caption(_ context: inout GraphicsContext, _ text: String, _ colour: Color) {
            context.draw(Text(text).font(.system(size: 16 * k, weight: .semibold)).foregroundStyle(colour),
                         at: CGPoint(x: left, y: y), anchor: .topLeading)
        }
        func slot(_ index: Int) -> CGRect {
            CGRect(x: left + Double(index) * (cardW + gap), y: y + captionH, width: cardW, height: cardH)
        }

        if let guess = info.guess {
            caption(&context, info.guessSource ?? "guess", .white.opacity(0.75))
            for (index, card) in guess.prefix(8).enumerated() {
                drawCard(&context, card, in: slot(index), k: k, opacity: 0.55, edge: .white.opacity(0.7), dashed: true)
            }
            y += rowH
        }

        let hand = info.hand ?? []
        let held = Set(hand.compactMap { $0?.cardId })
        caption(&context, "seen \(info.revealed.count) of 8 · \(info.plays ?? 0) plays", .white)
        for (index, card) in info.revealed.prefix(8).enumerated() {
            let inHand = held.contains(card.cardId)
            drawCard(&context, card, in: slot(index), k: k, opacity: info.hand == nil || inHand ? 1 : 0.45,
                     edge: inHand ? Self.gold : .clear, dashed: false)
        }
        y += rowH

        if info.hand == nil {
            caption(&context, "hand · known after their fourth play", .white.opacity(0.75))
        } else {
            caption(&context, "hand", Self.gold)
            for index in 0..<4 {
                let card = index < hand.count ? hand[index] : nil
                drawCard(&context, card ?? nil, in: slot(index), k: k, opacity: 1, edge: Self.gold, dashed: false)
            }
            if let next = info.next {
                context.draw(Text("next").font(.system(size: 15 * k, weight: .semibold)).foregroundStyle(.white.opacity(0.8)),
                             at: CGPoint(x: slot(4).midX + 6 * k, y: slot(4).midY), anchor: .center)
                drawCard(&context, next, in: slot(5), k: k, opacity: 0.8, edge: .white.opacity(0.6), dashed: false)
            }
        }
        y += rowH

        // their elixir: ten segments and the number
        let value = min(10, max(0, info.elixir ?? 0))
        let barW = width - 86 * k, barH = 16 * k
        for segment in 0..<10 {
            let rect = CGRect(x: left + Double(segment) * barW / 10 + 1 * k, y: y + 6 * k, width: barW / 10 - 2 * k, height: barH)
            let fill = min(1, max(0, value - Double(segment)))
            context.fill(Path(roundedRect: rect, cornerRadius: 3 * k), with: .color(.white.opacity(0.16)))
            if fill > 0 {
                let part = CGRect(x: rect.minX, y: rect.minY, width: rect.width * fill, height: rect.height)
                context.fill(Path(roundedRect: part, cornerRadius: 3 * k), with: .color(Self.elixir))
            }
        }
        context.draw(Text(info.elixir == nil ? "elixir ?" : String(format: "elixir %.1f", value))
                        .font(.system(size: 18 * k, weight: .heavy)).foregroundStyle(Self.elixir),
                     at: CGPoint(x: left + width, y: y + 6 * k + barH / 2), anchor: .trailing)
    }

    /// One card: its portrait (the evolution's or hero's when it was played so), or its name where the art is
    /// missing; "?" for a card not seen yet.
    private func drawCard(_ context: inout GraphicsContext, _ card: OpponentCard?, in rect: CGRect, k: Double,
                          opacity: Double, edge: Color, dashed: Bool) {
        var layer = context
        layer.opacity = opacity
        let shape = Path(roundedRect: rect, cornerRadius: 4 * k)
        if let card, let icon = HudArt.icon(card.cardId, form: card.form ?? 0) {
            layer.draw(Image(nsImage: icon), in: rect)
        } else {
            layer.fill(shape, with: .color(Color(white: 0.16)))
            let text = card.map { String($0.name.prefix(6)) } ?? "?"
            layer.draw(Text(text).font(.system(size: (card == nil ? 30 : 11) * k, weight: .bold)).foregroundStyle(.white),
                       at: CGPoint(x: rect.midX, y: rect.midY))
        }
        if edge != .clear {
            context.stroke(shape, with: .color(edge),
                           style: StrokeStyle(lineWidth: 2.5 * k, dash: dashed ? [5 * k, 4 * k] : []))
        }
    }
}
