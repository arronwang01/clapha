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
        window.setFrame(frame, display: true)
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

    private static let red = Color(red: 235 / 255, green: 70 / 255, blue: 70 / 255)
    private static let gold = Color(red: 255 / 255, green: 205 / 255, blue: 60 / 255)

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
}
