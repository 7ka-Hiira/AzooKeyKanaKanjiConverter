import EfficientNGram
package import Foundation
import SwiftUtils

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
        try? self.zenzContext?.resetContext()
    }

    /// Update device configuration dynamically
    public func updateDeviceConfig(_ newConfig: ZenzaiDeviceConfig) throws {
        try self.zenzContext?.updateDeviceConfig(newConfig)
    }

    /// Get current device configuration
    public func getDeviceConfig() -> ZenzaiDeviceConfig? {
        return self.zenzContext?.getDeviceConfig()
    }

    func candidateEvaluate(
        convertTarget: String,
        candidates: [Candidate],
        requestRichCandidates: Bool,
        prefixConstraint: Kana2Kanji.PrefixConstraint,
        personalizationMode: (mode: ConvertRequestOptions.ZenzaiMode.PersonalizationMode, base: EfficientNGram, personal: EfficientNGram)?,
        versionDependentConfig: ConvertRequestOptions.ZenzaiVersionDependentMode
    ) -> CandidateEvaluationResult {
        guard let zenzContext else {
            return .error
        }
        for candidate in candidates {
            return ZenzCandidateEvaluator.evaluate(
                context: zenzContext,
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

    func predictNextInputText(
        leftSideContext: String,
        composingText: String,
        count: Int,
        minLength: Int = 1,
        maxEntropy: Float?,
        versionDependentConfig: ConvertRequestOptions.ZenzaiVersionDependentMode,
        possibleNexts: [String] = []
    ) -> String {
        guard let zenzContext else {
            return ""
        }
        return ZenzInputTextGenerator.generate(
            context: zenzContext,
            leftSideContext: leftSideContext,
            composingText: composingText,
            count: count,
            minLength: minLength,
            maxEntropy: maxEntropy,
            versionDependentConfig: versionDependentConfig,
            possibleNexts: possibleNexts
        )
    }

    package func pureGreedyDecoding(pureInput: String, maxCount: Int = .max) -> String {
        guard let zenzContext else {
            return ""
        }
        return ZenzPureGreedyDecoder.decode(context: zenzContext, leftSideContext: pureInput, maxCount: maxCount)
    }
}
