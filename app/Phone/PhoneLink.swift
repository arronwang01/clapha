import AppKit
import CoreImage
import Network
import ScreenCaptureKit

/// The game on your phone: the MuMu picture (with Clapha's overlay on it) shown on a phone over USB, the phone's
/// touches sent back to the game.
///
///     screen (ScreenCaptureKit, the game picture's rectangle) -> JPEG -> the viewer app on the phone
///     phone touches -> here -> the MuMu touchscreen (fast_tap's d / m / u, the ordinary Android input path)
///
/// The picture is taken from the Mac's screen, so what is on it is what the phone shows: the MuMu window has to
/// be in view, and the overlay drawn over it is in the picture. The phone side is the "Null's Viewer" app of the
/// earlier Null's setup (phone/nulls-viewer.apk, cr-engine-extraction/macos-port/phone/android), spoken to in its
/// own protocol through `adb reverse`:
///     to the phone:   type, u32 length, bytes     1 config JSON, 3 a JPEG frame
///     from the phone: 'N' and 'j' once; then 1 touch [action, finger, x u16, y u16] (action 0 down, 1 up, 2 move,
///                     3 cancel; x, y in picture pixels), 3 frame shown, 2 statistics [u16 length, JSON], 4 ignored
/// At most two frames are on their way at once and each is the newest, so the picture never queues up behind
/// the game.
///
/// A helper of its own, not part of Clapha.app, so that macOS's permission to record the screen is asked once
/// for it and survives the main app being rebuilt. Started and stopped by Clapha (the Phone switch).
///
///     ClaphaPhone --serial 127.0.0.1:26624 --adb <path> [--port 8766] [--width 720] [--quality 0.7]
///                 [--apk phone/nulls-viewer.apk] [--status build/phone_link.json] [--pattern] [--touch FILE]
@main
struct PhoneLinkMain {
    static func main() {
        let app = NSApplication.shared
        let link = PhoneLink()
        app.delegate = link
        app.setActivationPolicy(.accessory)
        app.run()
    }
}

final class PhoneLink: NSObject, NSApplicationDelegate, SCStreamOutput, SCStreamDelegate {
    // settings
    private var serial = "127.0.0.1:26624"
    private var adb = NSString(string: "~/Library/Android/sdk/platform-tools/adb").expandingTildeInPath
    private var port: UInt16 = 8766
    private var width = 720
    private var height = 1280
    private var quality = 0.7
    private var apk = ""
    private var statusPath = ""
    private var pattern = false               // made-up frames instead of the screen (to test without the permission)
    private var touchDevice = "auto"            // the device's touchscreen; a file path to test without touching anything

    // the picture
    private var stream: SCStream?
    private var streamRect = CGRect.zero
    private let captureQueue = DispatchQueue(label: "phone.capture")
    private let encoder = CIContext(options: [.cacheIntermediates: false])
    private let colour = CGColorSpace(name: CGColorSpace.sRGB)!
    private var captureNote = "starting"
    private var framesMade = 0, framesSent = 0

    // the phone
    private let net = DispatchQueue(label: "phone.net")
    private var listener: NWListener?
    private var client: NWConnection?
    private var latest: Data?
    private var latestNumber = 0, sentNumber = 0, inFlight = 0
    private var phoneStats = ""
    private var phoneSerial = ""
    private var touches = 0

    // the game's touchscreen
    private var touchProcess: Process?
    private var touchPipe: FileHandle?
    private var deviceSize = CGSize(width: 1440, height: 2560)

    func applicationDidFinishLaunching(_ notification: Notification) {
        var args = Array(CommandLine.arguments.dropFirst())
        while !args.isEmpty {
            let flag = args.removeFirst()
            func value() -> String { args.isEmpty ? "" : args.removeFirst() }
            switch flag {
            case "--serial": serial = value()
            case "--adb": adb = value()
            case "--port": port = UInt16(value()) ?? port
            case "--width": width = Int(value()) ?? width
            case "--quality": quality = Double(value()) ?? quality
            case "--apk": apk = value()
            case "--status": statusPath = value()
            case "--pattern": pattern = true
            case "--touch": touchDevice = value()
            default: break
            }
        }
        height = width * 16 / 9
        startTouch()
        startListener()
        if pattern {
            captureNote = "test pattern"
            Timer.scheduledTimer(withTimeInterval: 1.0 / 60, repeats: true) { [weak self] _ in self?.makePattern() }
        } else {
            Task { await self.startCapture() }
            // the MuMu window may be moved or resized, and the permission may be given while we run
            Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
                guard let self else { return }
                Task { await self.followWindow() }
            }
        }
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in self?.everySecond() }
        connectPhone()
    }

    // MARK: the game picture on the Mac's screen

    /// The game picture of the frontmost MuMu window, in the screen's own coordinates (points, origin top left):
    /// the window less its title bar, fitted to the game's 9:16 (as Clapha's overlay finds it).
    static func gameRect() -> CGRect? {
        let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else { return nil }
        for info in list {
            guard let owner = info[kCGWindowOwnerName as String] as? String, owner.contains("MuMu"),
                  (info[kCGWindowLayer as String] as? Int) == 0,
                  let bounds = info[kCGWindowBounds as String] as? [String: CGFloat],
                  let x = bounds["X"], let y = bounds["Y"], let w = bounds["Width"], let h = bounds["Height"],
                  w > 200, h > 300 else { continue }
            var gameW = w, gameH = w * 16.0 / 9.0
            var left = x, top = y + max(0, h - gameH)
            if gameH > h {
                gameH = h
                gameW = h * 9.0 / 16.0
                left = x + (w - gameW) / 2
                top = y
            }
            return CGRect(x: left, y: top, width: gameW, height: gameH)
        }
        return nil
    }

    private var asked = false
    private static let notAllowed = "macOS has not allowed Clapha Phone to record the screen yet: switch it on in System Settings > Privacy & Security > Screen & System Audio Recording, then switch Phone off and on"

    private func startCapture() async {
        // Without the permission nothing of ScreenCaptureKit is touched: asked for once (macOS shows its own
        // question and lists the helper in System Settings), then only said.
        guard CGPreflightScreenCaptureAccess() else {
            if !asked {
                asked = true
                CGRequestScreenCaptureAccess()
            }
            captureNote = PhoneLink.notAllowed
            return
        }
        guard let rect = PhoneLink.gameRect() else {
            captureNote = "no MuMu window on the screen"
            return
        }
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            guard let display = content.displays.first(where: { $0.frame.intersects(rect) }) ?? content.displays.first else {
                captureNote = "no display"
                return
            }
            let config = SCStreamConfiguration()
            config.sourceRect = CGRect(x: rect.minX - display.frame.minX, y: rect.minY - display.frame.minY,
                                       width: rect.width, height: rect.height)
            config.width = width
            config.height = height
            config.minimumFrameInterval = CMTime(value: 1, timescale: 60)
            config.pixelFormat = kCVPixelFormatType_32BGRA
            config.showsCursor = false
            config.queueDepth = 3
            let made = SCStream(filter: SCContentFilter(display: display, excludingWindows: []), configuration: config,
                                delegate: self)
            try made.addStreamOutput(self, type: .screen, sampleHandlerQueue: captureQueue)
            try await made.startCapture()
            stream = made
            streamRect = rect
            captureNote = "capturing"
        } catch {
            stream = nil
            captureNote = "screen capture failed: \(error.localizedDescription)"
        }
    }

    private func followWindow() async {
        guard CGPreflightScreenCaptureAccess() else { return }
        guard let rect = PhoneLink.gameRect() else {
            captureNote = "no MuMu window on the screen"
            return
        }
        if stream == nil {
            await startCapture()
        } else if rect != streamRect {
            // moved or resized: start again on the new rectangle
            try? await stream?.stopCapture()
            stream = nil
            await startCapture()
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        self.stream = nil
        captureNote = "screen capture stopped: \(error.localizedDescription)"
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sampleBuffer.isValid, let buffer = sampleBuffer.imageBuffer else { return }
        // only frames that carry a new picture (an idle screen sends status frames without one)
        if let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
           let raw = attachments.first?[.status] as? Int, let status = SCFrameStatus(rawValue: raw), status != .complete {
            return
        }
        var wanted = false
        net.sync { wanted = client != nil }
        guard wanted else { return }                       // nobody watching: nothing to encode
        let image = CIImage(cvPixelBuffer: buffer)
        guard let jpeg = encoder.jpegRepresentation(
            of: image, colorSpace: colour,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: quality]) else { return }
        offer(jpeg)
    }

    /// Frames without the screen: a moving bar and a counter, for testing everything but the capture.
    private func makePattern() {
        var wanted = false
        net.sync { wanted = client != nil }
        guard wanted else { return }
        let size = CGSize(width: width, height: height)
        let image = NSImage(size: size, flipped: true) { _ in
            NSColor(calibratedRed: 0.2, green: 0.35, blue: 0.2, alpha: 1).setFill()
            NSRect(origin: .zero, size: size).fill()
            NSColor.systemPurple.setFill()
            let y = CGFloat(self.framesMade % 120) / 120 * size.height
            NSRect(x: 0, y: y, width: size.width, height: 24).fill()
            let text = "frame \(self.framesMade)  touches \(self.touches)" as NSString
            text.draw(at: NSPoint(x: 20, y: 40), withAttributes: [.font: NSFont.boldSystemFont(ofSize: 36),
                                                                  .foregroundColor: NSColor.white])
            return true
        }
        guard let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
              let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: quality]) else { return }
        offer(jpeg)
    }

    private func offer(_ jpeg: Data) {
        net.async {
            self.framesMade += 1
            self.latest = jpeg
            self.latestNumber += 1
            self.pump()
        }
    }

    // MARK: the phone

    private func startListener() {
        let parameters = NWParameters.tcp
        if let tcp = parameters.defaultProtocolStack.transportProtocol as? NWProtocolTCP.Options { tcp.noDelay = true }
        parameters.requiredLocalEndpoint = NWEndpoint.hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!)
        parameters.allowLocalEndpointReuse = true
        guard let made = try? NWListener(using: parameters) else {
            captureNote = "port \(port) is taken (another phone link running?)"
            return
        }
        made.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
        made.start(queue: net)
        listener = made
    }

    private func accept(_ connection: NWConnection) {
        client?.cancel()
        client = connection
        sentNumber = 0
        inFlight = 0
        connection.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .failed, .cancelled:
                if self.client === connection { self.client = nil; self.liftFingers() }
            default: break
            }
        }
        connection.start(queue: net)
        read(connection, 2) { [weak self] hello in
            guard let self, hello.first == UInt8(ascii: "N") else { connection.cancel(); return }
            let config = "{\"w\":\(self.width),\"h\":\(self.height),\"mode\":\"jpeg\"}".data(using: .utf8)!
            self.send(connection, type: 1, config)
            self.pump()
            self.readMessage(connection)
        }
    }

    private func read(_ connection: NWConnection, _ count: Int, then: @escaping (Data) -> Void) {
        connection.receive(minimumIncompleteLength: count, maximumLength: count) { [weak self] data, _, _, error in
            guard let data, data.count == count, error == nil else {
                connection.cancel()
                if let self, self.client === connection { self.client = nil; self.liftFingers() }
                return
            }
            then(data)
        }
    }

    private func readMessage(_ connection: NWConnection) {
        read(connection, 1) { [weak self] kind in
            guard let self else { return }
            switch kind[0] {
            case 1:
                self.read(connection, 6) { body in
                    let x = Int(body[2]) << 8 | Int(body[3]), y = Int(body[4]) << 8 | Int(body[5])
                    self.touch(action: Int(body[0]), finger: Int(body[1]), x: x, y: y)
                    self.readMessage(connection)
                }
            case 2:
                self.read(connection, 2) { length in
                    self.read(connection, Int(length[0]) << 8 | Int(length[1])) { body in
                        self.phoneStats = String(data: body, encoding: .utf8) ?? ""
                        self.readMessage(connection)
                    }
                }
            case 3:
                self.inFlight = max(0, self.inFlight - 1)
                self.pump()
                self.readMessage(connection)
            default:
                self.readMessage(connection)
            }
        }
    }

    private func send(_ connection: NWConnection, type: UInt8, _ payload: Data) {
        var packet = Data([type, UInt8(payload.count >> 24 & 0xff), UInt8(payload.count >> 16 & 0xff),
                           UInt8(payload.count >> 8 & 0xff), UInt8(payload.count & 0xff)])
        packet.append(payload)
        connection.send(content: packet, completion: .contentProcessed { _ in })
    }

    /// Send the newest frame if the phone has room for it (two on their way at most).
    private func pump() {
        guard let client, let frame = latest, latestNumber != sentNumber, inFlight < 2 else { return }
        sentNumber = latestNumber
        inFlight += 1
        framesSent += 1
        send(client, type: 3, frame)
    }

    // MARK: touches to the game

    private func startTouch() {
        let size = run([adb, "-s", serial, "shell", "wm", "size"])
        if let match = size.range(of: #"(\d+)x(\d+)"#, options: .regularExpression) {
            let parts = size[match].split(separator: "x").compactMap { Double($0) }
            if parts.count == 2 { deviceSize = CGSize(width: parts[0], height: parts[1]) }
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: adb)
        process.arguments = ["-s", serial, "shell", "/data/local/tmp/fast_tap \(touchDevice)"]
        let input = Pipe()
        process.standardInput = input
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            touchProcess = process
            touchPipe = input.fileHandleForWriting
        } catch {
            captureNote = "could not start the touch relay: \(error.localizedDescription)"
        }
    }

    private var fingersDown = Set<Int>()

    private func touch(action: Int, finger: Int, x: Int, y: Int) {
        let slot = min(9, max(0, finger))
        let dx = Int((Double(x) * deviceSize.width / Double(width)).rounded())
        let dy = Int((Double(y) * deviceSize.height / Double(height)).rounded())
        var line: String
        switch action {
        case 0:
            line = "d \(slot) \(dx) \(dy)\n"
            fingersDown.insert(slot)
        case 1, 3:
            guard fingersDown.contains(slot) || action == 3 else { return }
            if action == 3 {
                line = fingersDown.map { "u \($0)\n" }.joined()
                fingersDown.removeAll()
            } else {
                line = "u \(slot)\n"
                fingersDown.remove(slot)
            }
        default:
            guard fingersDown.contains(slot) else { return }
            line = "m \(slot) \(dx) \(dy)\n"
        }
        touches += 1
        if let data = line.data(using: .utf8) { try? touchPipe?.write(contentsOf: data) }
    }

    /// The phone went away with a finger down: lift it, or the game keeps a touch held.
    private func liftFingers() {
        let line = fingersDown.map { "u \($0)\n" }.joined()
        fingersDown.removeAll()
        if !line.isEmpty, let data = line.data(using: .utf8) { try? touchPipe?.write(contentsOf: data) }
    }

    // MARK: the phone's cable

    @discardableResult
    private func run(_ command: [String], timeout: TimeInterval = 8) -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: command[0])
        process.arguments = Array(command.dropFirst())
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        guard (try? process.run()) != nil else { return "" }
        DispatchQueue.global().asyncAfter(deadline: .now() + timeout) { if process.isRunning { process.terminate() } }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return String(data: data, encoding: .utf8) ?? ""
    }

    /// A phone on USB (not one of the MuMu instances, which adb lists by address): forward our port to it and open
    /// the viewer, installing it first if it is not there.
    private func connectPhone() {
        DispatchQueue.global().async {
            let devices = self.run([self.adb, "devices"]).split(separator: "\n").dropFirst().compactMap { line -> String? in
                let parts = line.split(separator: "\t")
                guard parts.count == 2, parts[1] == "device" else { return nil }
                let name = String(parts[0])
                return name.hasPrefix("127.0.0.1:") || name.hasPrefix("emulator-") ? nil : name
            }
            guard let phone = devices.first else {
                self.net.async { self.phoneSerial = "" }
                return
            }
            var known = ""
            self.net.sync { known = self.phoneSerial }
            self.run([self.adb, "-s", phone, "reverse", "tcp:\(self.port)", "tcp:\(self.port)"])
            if phone != known {
                let installed = self.run([self.adb, "-s", phone, "shell", "pm", "path", "local.nulls.viewer"])
                if installed.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !self.apk.isEmpty {
                    self.run([self.adb, "-s", phone, "install", "-r", self.apk], timeout: 40)
                }
                self.run([self.adb, "-s", phone, "shell", "am", "start", "-n", "local.nulls.viewer/.ViewerActivity"])
                self.net.async { self.phoneSerial = phone }
            }
        }
    }

    private var lastMade = 0, lastSent = 0, seconds = 0

    private func everySecond() {
        seconds += 1
        var connected = false, made = 0, sent = 0, stats = "", phone = ""
        net.sync {
            connected = client != nil
            made = framesMade; sent = framesSent; stats = phoneStats; phone = phoneSerial
        }
        if !connected && seconds % 3 == 0 { connectPhone() }
        let state: [String: Any] = [
            "capture": captureNote, "phone": phone, "connected": connected, "frames_per_second": made - lastMade,
            "sent_per_second": sent - lastSent, "phone_stats": stats, "touches": touches, "width": width, "height": height,
            "serial": serial, "time": Date().timeIntervalSince1970,
        ]
        lastMade = made; lastSent = sent
        guard !statusPath.isEmpty, let data = try? JSONSerialization.data(withJSONObject: state) else { return }
        try? data.write(to: URL(fileURLWithPath: statusPath), options: .atomic)
    }

    func applicationWillTerminate(_ notification: Notification) {
        liftFingers()
        try? touchPipe?.write(contentsOf: "quit\n".data(using: .utf8)!)
        touchProcess?.terminate()
        if !statusPath.isEmpty { try? FileManager.default.removeItem(atPath: statusPath) }
    }
}
