import SwiftUI
import AppKit

/// Renders the app window off-screen from recorded engine states, to check the layout
/// without a display:  app/snapshot.sh  ->  build/app_snapshot.png
@main
struct Snapshot {
    @MainActor
    static func main() {
        let args = CommandLine.arguments
        renderingSnapshot = true
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        func load(_ path: String) -> DeviceState {
            try! decoder.decode(DeviceState.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        }
        let one = DeviceModel(id: 1, title: "Device 1 · main account", port: 8777)
        one.state = load(args[1]); one.reachable = true; one.logs.lines = one.state!.bot.log; one.board.state = one.state; one.refreshSummary()
        let two = DeviceModel(id: 2, title: "Device 2 · second account", port: 8778)
        two.state = load(args[2]); two.reachable = true; two.logs.lines = two.state!.bot.log; two.board.state = two.state; two.refreshSummary()
        let view = ContentView(device1: one, device2: two, live: false)
            .frame(width: 1000, height: 760)
            .background(Color(nsColor: .windowBackgroundColor))
        let renderer = ImageRenderer(content: view)
        renderer.scale = 1
        guard let image = renderer.nsImage, let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else {
            print("render failed"); exit(1)
        }
        try! png.write(to: URL(fileURLWithPath: args[3]))
        print("wrote \(args[3])")
    }
}
