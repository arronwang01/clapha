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
        if args.count > 3 && args[1] == "--games" {
            // the Training games window: snapshot --games runs/viewer/games.json out.png [game index]
            let model = GamesModel()
            model.file = try! JSONDecoder().decode(GameFile.self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
            let game = model.file!.games[args.count > 4 ? Int(args[4])! : 0]
            let view = TrainingGamesView(model: model, selection: game.id, tick: Double(game.start + (game.end - game.start) * 2 / 5))
                .frame(width: 1150, height: 780)
                .background(Color(nsColor: .windowBackgroundColor))
            write(view, to: args[3])
            return
        }
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
        write(view, to: args[3])
    }

    @MainActor
    static func write(_ view: some View, to path: String) {
        let renderer = ImageRenderer(content: view)
        renderer.scale = 1
        guard let image = renderer.nsImage, let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else {
            print("render failed"); exit(1)
        }
        try! png.write(to: URL(fileURLWithPath: path))
        print("wrote \(path)")
    }
}
