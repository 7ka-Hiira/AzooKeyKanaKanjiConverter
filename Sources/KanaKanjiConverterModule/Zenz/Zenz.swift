import Foundation
import SwiftUtils

@MainActor final class Zenz {
    package var resourceURL: URL
    private var zenzContext: ZenzContext?
    init(resourceURL: URL, gpuLayers: Int32) throws {
        self.resourceURL = resourceURL
        do {
            #if canImport(Darwin)
            if #available(iOS 16, macOS 13, *) {
                self.zenzContext = try ZenzContext.createContext(path: resourceURL.path(percentEncoded: false), gpuLayers: gpuLayers)
            } else {
                // this is not percent-encoded
                self.zenzContext = try ZenzContext.createContext(path: resourceURL.path, gpuLayers: gpuLayers)
            }
            #elseif canImport(WinSDK)
            // remove first "/" from path (for windows)
            self.zenzContext = try ZenzContext.createContext(path: String(resourceURL.path.dropFirst()))
            #else
            // this is not percent-encoded
            self.zenzContext = try ZenzContext.createContext(path: resourceURL.path, gpuLayers: gpuLayers)
            #endif
            debug("Loaded model \(resourceURL.lastPathComponent)")
        } catch {
            throw error
        }
    }

    func startSession() {}

    func endSession() {
        try? self.zenzContext?.reset_context()
    }

    func candidateEvaluate(convertTarget: String, candidates: [Candidate], requestRichCandidates: Bool, versionDependentConfig: ConvertRequestOptions.ZenzaiVersionDependentMode) -> ZenzContext.CandidateEvaluationResult {
        guard let zenzContext else {
            return .error
        }
        for candidate in candidates {
            let result = zenzContext.evaluate_candidate(input: convertTarget.toKatakana(), candidate: candidate, requestRichCandidates: requestRichCandidates, versionDependentConfig: versionDependentConfig)
            return result
        }
        return .error
    }

    func predictNextCharacter(leftSideContext: String, count: Int) -> [(character: Character, value: Float)] {
        guard let zenzContext else {
            return []
        }
        let result = zenzContext.predict_next_character(leftSideContext: leftSideContext, count: count)
        return result
    }
}
