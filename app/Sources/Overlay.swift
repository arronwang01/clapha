import AppKit
import SwiftUI

/// Ghost markers drawn over the MuMu window itself: a transparent, click-through window that
/// follows MuMu's game view. Each queued command (played, not landed yet) is a dashed ring
/// with the card and a countdown, at the device pixel the engine computed with the same
/// geometry the bot's taps use. Nothing is sent to the game; clicks pass straight through.
@MainActor
final class MuMuOverlay {
    private var window: NSWindow?
    private var timer: Timer?

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
        window.contentView = NSHostingView(rootView: OverlayView(board: board))
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
        guard let frame = MuMuOverlay.gameRect() else {
            window.orderOut(nil)
            return
        }
        if window.frame != frame { window.setFrame(frame, display: true) }
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

    private func draw(_ context: inout GraphicsContext, size: CGSize, now: Date) {
        guard let state = board.state, state.ok else { return }
        if showOpponent, let opponent = state.opponent {
            drawOpponent(&context, size: size, info: opponent)
        }
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

    /// Inside the game picture, in the two margins beside the arena (the outer 5.5% on each side, where nothing
    /// of the game stands), so the picture a phone is sent carries it and nothing of the board is covered.
    ///   left    the cards they have actually played, in the order first seen, the ones in their hand now edged
    ///           in gold and the ones waiting in their cycle dimmed; under them their hand as it follows from
    ///           the order of their plays ("?" for a card not seen yet; not known before their fourth play),
    ///           and the card that comes back next.
    ///   right   the deck held for theirs before they play, faint with a dashed edge, for as long as nothing
    ///           they play contradicts it (then it is not drawn at all); under it their elixir.
    private func drawOpponent(_ context: inout GraphicsContext, size: CGSize, info: OpponentInfo) {
        let w = size.width
        let cardW = 0.052 * w, cardH = cardW * 420 / 285, step = cardH + 0.004 * w
        let caption = 0.026 * w, k = w / 1080
        let leftX = 0.002 * w, rightX = w - cardW - 0.002 * w
        let top = 0.102 * size.height, bottom = 0.775 * size.height

        func back(_ context: inout GraphicsContext, x: Double, from: Double, to: Double) {
            let rect = CGRect(x: x - 0.001 * w, y: from - 0.004 * w, width: cardW + 0.002 * w, height: to - from + 0.008 * w)
            context.fill(Path(roundedRect: rect, cornerRadius: 5 * k), with: .color(.black.opacity(0.45)))
        }
        func label(_ context: inout GraphicsContext, _ text: String, x: Double, y: Double, _ colour: Color) {
            context.draw(Text(text).font(.system(size: 0.019 * w, weight: .bold)).foregroundStyle(colour),
                         at: CGPoint(x: x + cardW / 2, y: y + caption / 2))
        }

        // left margin: seen, then hand and next
        let hand = info.hand ?? []
        let held = Set(hand.compactMap { $0?.cardId })
        let seen = Array(info.revealed.prefix(8))
        var y = top
        let leftEnd = top + caption + Double(max(seen.count, 1)) * step + caption + 4 * step + caption + step
        back(&context, x: leftX, from: top, to: min(bottom, leftEnd))
        label(&context, "seen", x: leftX, y: y, .white)
        y += caption
        for card in seen {
            let inHand = held.contains(card.cardId)
            drawCard(&context, card, in: CGRect(x: leftX, y: y, width: cardW, height: cardH), k: k,
                     opacity: info.hand == nil || inHand ? 1 : 0.45, edge: inHand ? Self.gold : .clear, dashed: false)
            y += step
        }
        if seen.isEmpty { y += step }
        label(&context, "hand", x: leftX, y: y, Self.gold)
        y += caption
        for index in 0..<4 {
            let rect = CGRect(x: leftX, y: y, width: cardW, height: cardH)
            if info.hand == nil {
                // not fixed before their fourth play: empty frames
                context.stroke(Path(roundedRect: rect, cornerRadius: 4 * k), with: .color(Self.gold.opacity(0.35)),
                               style: StrokeStyle(lineWidth: 2 * k, dash: [5 * k, 4 * k]))
            } else {
                drawCard(&context, index < hand.count ? hand[index] : nil, in: rect, k: k, opacity: 1, edge: Self.gold, dashed: false)
            }
            y += step
        }
        if let next = info.next {
            label(&context, "next", x: leftX, y: y, .white.opacity(0.85))
            y += caption
            drawCard(&context, next, in: CGRect(x: leftX, y: y, width: cardW, height: cardH), k: k, opacity: 0.85,
                     edge: .white.opacity(0.6), dashed: false)
        }

        // right margin: the deck held for theirs while nothing contradicts it, then their elixir
        var ry = 0.145 * size.height                    // below the clock and the elixir badge
        let barH = 10 * (0.021 * w)
        let guess = Array((info.guess ?? []).prefix(8))
        let rightEnd = ry + (guess.isEmpty ? 0 : caption + Double(guess.count) * step) + caption + 0.045 * w + barH
        back(&context, x: rightX, from: ry, to: rightEnd)
        if !guess.isEmpty {
            label(&context, info.guessKind == "published" ? "deck" : "guess", x: rightX, y: ry, .white.opacity(0.8))
            ry += caption
            for card in guess {
                drawCard(&context, card, in: CGRect(x: rightX, y: ry, width: cardW, height: cardH), k: k, opacity: 0.6,
                         edge: .white.opacity(0.7), dashed: true)
                ry += step
            }
        }
        label(&context, "elixir", x: rightX, y: ry, Self.elixir)
        ry += caption
        let value = min(10, max(0, info.elixir ?? 0))
        context.draw(Text(info.elixir == nil ? "?" : String(format: "%.1f", value))
                        .font(.system(size: 0.032 * w, weight: .heavy)).foregroundStyle(Self.elixir),
                     at: CGPoint(x: rightX + cardW / 2, y: ry + 0.02 * w))
        ry += 0.045 * w
        // ten segments, filling from the bottom
        let segment = 0.021 * w
        for index in 0..<10 {
            let rect = CGRect(x: rightX + 0.006 * w, y: ry + Double(9 - index) * segment + 1 * k,
                              width: cardW - 0.012 * w, height: segment - 2 * k)
            let fill = min(1, max(0, value - Double(index)))
            context.fill(Path(roundedRect: rect, cornerRadius: 2 * k), with: .color(.white.opacity(0.18)))
            if fill > 0 {
                let part = CGRect(x: rect.minX, y: rect.maxY - rect.height * fill, width: rect.width, height: rect.height * fill)
                context.fill(Path(roundedRect: part, cornerRadius: 2 * k), with: .color(Self.elixir))
            }
        }
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
