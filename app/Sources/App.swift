import SwiftUI

// MARK: - App

@main
struct ClaphaApp: App {
    init() { appLog("app started, root \(claphaRoot().path)") }

    var body: some Scene {
        WindowGroup("Clapha") {
            ContentView()
                .frame(minWidth: 1180, minHeight: 820)
        }
        .windowResizability(.contentMinSize)
    }
}

