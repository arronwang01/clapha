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

struct OverlayView: View {
    @ObservedObject var board: BoardModel

    var body: some View {
        Canvas { context, size in
            guard let state = board.state, state.ok else { return }
            let local = state.localSide ?? 0
            for pending in state.pendingCommands ?? [] where pending.kind == "card" {
                guard let sx = pending.screenX, let sy = pending.screenY,
                      let sw = pending.screenW, let sh = pending.screenH, sw > 0, sh > 0 else { continue }
                let p = CGPoint(x: Double(sx) / Double(sw) * size.width,
                                y: Double(sy) / Double(sh) * size.height)
                let r = size.width * 0.045
                let ring = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
                let color: Color = pending.side == local ? Color(red: 0.35, green: 0.62, blue: 1)
                                                         : Color(red: 1, green: 0.30, blue: 0.28)
                context.fill(ring, with: .color(color.opacity(0.18)))
                context.stroke(ring, with: .color(color), style: StrokeStyle(lineWidth: 2.5, dash: [6, 4]))
                let seconds = String(format: "%.1fs", Double(max(0, pending.remainingTicks)) / 20.0)
                context.draw(Text("\(pending.name) \(seconds)")
                                .font(.system(size: 12, weight: .bold)).foregroundStyle(color),
                             at: CGPoint(x: p.x, y: p.y + r + 10))
            }
        }
        .allowsHitTesting(false)
    }
}
