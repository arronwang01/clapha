import SwiftUI

// MARK: - App

@main
struct ClaphaApp: App {
    init() { appLog("app started, root \(claphaRoot().path)") }

    var body: some Scene {
        WindowGroup("Clapha") {
            ContentView()
                .frame(minWidth: 820, minHeight: 700)
        }
        .windowResizability(.contentMinSize)
        // recorded RL training games, replayed (app/Sources/Games.swift)
        Window("Training games", id: "games") {
            TrainingGamesView()
                .frame(minWidth: 1000, minHeight: 720)
        }
    }
}

