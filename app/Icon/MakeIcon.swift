import SwiftUI
import AppKit

@main
struct MakeIcon {
    @MainActor
    static func main() {
        let out = CommandLine.arguments[1]
        let view = ZStack {
            RoundedRectangle(cornerRadius: 230, style: .continuous)
                .fill(LinearGradient(colors: [Color(red: 0.22, green: 0.33, blue: 0.85),
                                              Color(red: 0.55, green: 0.22, blue: 0.78)],
                                     startPoint: .topLeading, endPoint: .bottomTrailing))
                .frame(width: 900, height: 900)
            Image(systemName: "crown.fill")
                .font(.system(size: 430, weight: .bold))
                .foregroundStyle(.white)
                .offset(y: -40)
            Text("AI")
                .font(.system(size: 170, weight: .heavy, design: .rounded))
                .foregroundStyle(.white.opacity(0.9))
                .offset(y: 270)
        }
        .frame(width: 1024, height: 1024)
        let renderer = ImageRenderer(content: view)
        renderer.scale = 1
        renderer.isOpaque = false
        let rep = NSBitmapImageRep(data: renderer.nsImage!.tiffRepresentation!)!
        try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out))
    }
}
