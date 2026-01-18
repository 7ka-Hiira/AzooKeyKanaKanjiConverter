import EfficientNGram
package import Foundation
import SwiftUtils

/// Enumerate available GGML backend devices
package func enumerateBackendDevices() -> [GGMLBackendDevice] {
    return enumerateGGMLBackendDevices()
}

package final class Zenz {
    package var resourceURL: URL
    private var zenzContext: ZenzContext?
    
    init(resourceURL: URL, deviceConfig: ZenzaiDeviceConfig = ZenzaiDeviceConfig()) throws {
        self.resourceURL = resourceURL
        do {
            #if canImport(Darwin)
            if #available(iOS 16, macOS 13, *) {
                self.zenzContext = try ZenzContext.createContext(path: resourceURL.path(percentEncoded: false), deviceConfig: deviceConfig)
            } else {
                // this is not percent-encoded
                self.zenzContext = try ZenzContext.createContext(path: resourceURL.path, deviceConfig: deviceConfig)
            }
            #else
            // this is not percent-encoded
            self.zenzContext = try ZenzContext.createContext(path: resourceURL.path, deviceConfig: deviceConfig)
            #endif
            debug("Loaded model \(resourceURL.lastPathComponent) with device config: \(deviceConfig)")
        } catch {
            throw error
        }
    }

    package func endSession() {
        try? self.zenzContext?.reset_context()
    }
    
    /// Update device configuration dynamically
    package func updateDeviceConfig(_ newConfig: ZenzaiDeviceConfig) throws {
        try self.zenzContext?.updateDeviceConfig(newConfig)
    }
    
    /// Get current device configuration
    package func getDeviceConfig() -> ZenzaiDeviceConfig? {
        return self.zenzContext?.getDeviceConfig()
    }

    func candidateEvaluate(
        convertTarget: String,
        candidates: [Candidate],
        requestRichCandidates: Bool,
        prefixConstraint: Kana2Kanji.PrefixConstraint,
        personalizationMode: (mode: ConvertRequestOptions.ZenzaiMode.PersonalizationMode, base: EfficientNGram, personal: EfficientNGram)?,
        versionDependentConfig: ConvertRequestOptions.ZenzaiVersionDependentMode
    ) -> ZenzContext.CandidateEvaluationResult {
        guard let zenzContext else {
            return .error
        }
        for candidate in candidates {
            return zenzContext.evaluate_candidate(
                input: convertTarget.toKatakana(),
                candidate: candidate,
                requestRichCandidates: requestRichCandidates,
                prefixConstraint: prefixConstraint,
                personalizationMode: personalizationMode,
                versionDependentConfig: versionDependentConfig
            )
        }
        return .error
    }

    func predictNextCharacter(leftSideContext: String, count: Int) -> [(character: Character, value: Float)] {
        guard let zenzContext else {
            return []
        }
        return zenzContext.predict_next_character(leftSideContext: leftSideContext, count: count)
    }

    package func pureGreedyDecoding(pureInput: String, maxCount: Int = .max) -> String {
        self.zenzContext?.pure_greedy_decoding(leftSideContext: pureInput, maxCount: maxCount) ?? ""
    }
}
