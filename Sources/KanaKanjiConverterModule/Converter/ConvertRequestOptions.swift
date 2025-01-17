//
//  ConvertRequestOptions.swift
//  Keyboard
//
//  Created by ensan on 2022/12/20.
//  Copyright © 2022 ensan. All rights reserved.
//

import Foundation

public struct ConvertRequestOptions: Sendable {
    /// 変換リクエストに必要な設定データ
    ///
    /// - parameters:
    ///   - N_best: 変換候補の数。上位`N`件までの言語モデル上の妥当性を保証します。大きくすると計算量が増加します。
    ///   - requireJapanesePrediction: 日本語の予測変換候補の必要性。`false`にすると、日本語の予測変換候補を出力しなくなります。
    ///   - requireEnglishPrediction: 英語の予測変換候補の必要性。`false`にすると、英語の予測変換候補を出力しなくなります。ローマ字入力を用いた日本語入力では`false`にした方が良いでしょう。
    ///   - keyboardLanguage: キーボードの言語を指定します。
    ///   - typographyLetterCandidate: `true`の場合、「おしゃれなフォント」での英数字変換候補が出力に含まれるようになります。詳しくは`KanaKanjiConverter.typographicalCandidates(_:)`を参照してください。
    ///   - unicodeCandidate: `true`の場合、`U+xxxx`のような入力に対してUnicodeの変換候補が出力に含まれるようになります。詳しくは`KanaKanjiConverter.unicodeCandidates(_:)`を参照してください。`
    ///   - englishCandidateInRoman2KanaInput: `true`の場合、日本語ローマ字入力時に英語変換候補を出力します。`false`の場合、ローマ字入力時に英語変換候補を出力しません。
    ///   - fullWidthRomanCandidate: `true`の場合、全角英数字の変換候補が出力に含まれるようになります。
    ///   - halfWidthKanaCandidate: `true`の場合、半角カナの変換候補が出力に含まれるようになります。
    ///   - learningType: 学習モードを指定します。詳しくは`LearningType`を参照してください。
    ///   - maxMemoryCount: 学習が有効な場合に保持するデータの最大数を指定します。`0`の場合`learningType`を`nothing`に指定する方が適切です。
    ///   - shouldResetMemory: `true`の場合、変換を開始する前に学習データをリセットします。
    ///   - dictionaryResourceURL: 内蔵辞書データの読み出し先を指定します。
    ///   - memoryDirectoryURL: 学習データの保存先を指定します。書き込み可能なディレクトリを指定してください。
    ///   - sharedContainerURL: ユーザ辞書など、キーボード外で書き込んだ設定データの保存されているディレクトリを指定します。
    ///   - textReplacer: 予測変換のための置換機を指定します。
    ///   - metadata: メタデータを指定します。詳しくは`ConvertRequestOptions.Metadata`を参照してください。
    public init(N_best: Int = 10, requireJapanesePrediction: Bool, requireEnglishPrediction: Bool, keyboardLanguage: KeyboardLanguage, typographyLetterCandidate: Bool = false, unicodeCandidate: Bool = true, englishCandidateInRoman2KanaInput: Bool = false, fullWidthRomanCandidate: Bool = false, halfWidthKanaCandidate: Bool = false, learningType: LearningType, maxMemoryCount: Int = 65536, shouldResetMemory: Bool = false, dictionaryResourceURL: URL, memoryDirectoryURL: URL, sharedContainerURL: URL, textReplacer: TextReplacer = TextReplacer(), zenzaiMode: ZenzaiMode = .off, metadata: ConvertRequestOptions.Metadata?) {
        self.N_best = N_best
        self.requireJapanesePrediction = requireJapanesePrediction
        self.requireEnglishPrediction = requireEnglishPrediction
        self.keyboardLanguage = keyboardLanguage
        self.typographyLetterCandidate = typographyLetterCandidate
        self.unicodeCandidate = unicodeCandidate
        self.englishCandidateInRoman2KanaInput = englishCandidateInRoman2KanaInput
        self.fullWidthRomanCandidate = fullWidthRomanCandidate
        self.halfWidthKanaCandidate = halfWidthKanaCandidate
        self.learningType = learningType
        self.maxMemoryCount = maxMemoryCount
        self.shouldResetMemory = shouldResetMemory
        self.memoryDirectoryURL = memoryDirectoryURL
        self.sharedContainerURL = sharedContainerURL
        self.metadata = metadata
        self.textReplacer = textReplacer
        self.zenzaiMode = zenzaiMode
        self.dictionaryResourceURL = dictionaryResourceURL
    }

    package init(N_best: Int = 10, requireJapanesePrediction: Bool, requireEnglishPrediction: Bool, keyboardLanguage: KeyboardLanguage, typographyLetterCandidate: Bool = false, unicodeCandidate: Bool = true, englishCandidateInRoman2KanaInput: Bool = false, fullWidthRomanCandidate: Bool = false, halfWidthKanaCandidate: Bool = false, learningType: LearningType, maxMemoryCount: Int = 65536, shouldResetMemory: Bool = false, dictionaryResourceURL: URL, memoryDirectoryURL: URL, sharedContainerURL: URL, textReplacer: TextReplacer = TextReplacer(), zenzaiMode: ZenzaiMode = .off, metadata: ConvertRequestOptions.Metadata?, requestQuery: RequestQuery) {
        self.N_best = N_best
        self.requireJapanesePrediction = requireJapanesePrediction
        self.requireEnglishPrediction = requireEnglishPrediction
        self.keyboardLanguage = keyboardLanguage
        self.typographyLetterCandidate = typographyLetterCandidate
        self.unicodeCandidate = unicodeCandidate
        self.englishCandidateInRoman2KanaInput = englishCandidateInRoman2KanaInput
        self.fullWidthRomanCandidate = fullWidthRomanCandidate
        self.halfWidthKanaCandidate = halfWidthKanaCandidate
        self.learningType = learningType
        self.maxMemoryCount = maxMemoryCount
        self.shouldResetMemory = shouldResetMemory
        self.memoryDirectoryURL = memoryDirectoryURL
        self.sharedContainerURL = sharedContainerURL
        self.metadata = metadata
        self.textReplacer = textReplacer
        self.zenzaiMode = zenzaiMode
        self.dictionaryResourceURL = dictionaryResourceURL
    }

    public var N_best: Int
    public var requireJapanesePrediction: Bool
    public var requireEnglishPrediction: Bool
    public var keyboardLanguage: KeyboardLanguage
    // KeyboardSettingのinjection用途
    public var typographyLetterCandidate: Bool
    public var unicodeCandidate: Bool
    public var englishCandidateInRoman2KanaInput: Bool
    public var fullWidthRomanCandidate: Bool
    public var halfWidthKanaCandidate: Bool
    public var learningType: LearningType
    public var maxMemoryCount: Int
    public var shouldResetMemory: Bool
    /// 変換用
    public var textReplacer: TextReplacer
    // ディレクトリなど
    public var memoryDirectoryURL: URL
    public var sharedContainerURL: URL
    public var dictionaryResourceURL: URL
    public var zenzaiMode: ZenzaiMode
    // メタデータ
    public var metadata: Metadata?

    // MARK: プライベートAPI
    package var requestQuery: RequestQuery = .default

    static var `default`: Self {
        Self(
            N_best: 10,
            requireJapanesePrediction: true,
            requireEnglishPrediction: true,
            keyboardLanguage: .ja_JP,
            typographyLetterCandidate: false,
            unicodeCandidate: true,
            englishCandidateInRoman2KanaInput: true,
            fullWidthRomanCandidate: true,
            halfWidthKanaCandidate: false,
            learningType: .inputAndOutput,
            maxMemoryCount: 65536,
            shouldResetMemory: false,
            // dummy data, won't work
            dictionaryResourceURL: Bundle.main.bundleURL,
            // dummy data, won't work
            memoryDirectoryURL: (try? FileManager.default.url(for: .libraryDirectory, in: .userDomainMask, appropriateFor: nil, create: false)) ?? Bundle.main.bundleURL,
            // dummy data, won't work
            sharedContainerURL: Bundle.main.bundleURL,
            metadata: nil
        )
    }

    public struct Metadata: Sendable {
        /// - parameters:
        ///   - appVersionString: アプリのバージョンを指定します。このデータは`KanaKanjiCovnerter.toVersionCandidate(_:)`などで用いられます。
        @available(*, deprecated, renamed: "init(versionString:)", message: "it be removed in AzooKeyKanaKanjiConverter v1.0")
        public init(appVersionString: String) {
            self.versionString = "azooKey Version " + appVersionString
        }

        /// - parameters:
        ///   - versionString: アプリのバージョンを示す文字列全体を`"MyIME Version 0.7.1"`のように指定します。このデータは`KanaKanjiCovnerter.toVersionCandidate(_:)`などで用いられます。
        public init(versionString: String = "Powererd by AzooKeyKanaKanjiConverter") {
            self.versionString = versionString
        }
        var versionString: String
    }

    package enum RequestQuery: Sendable {
        case `default`
        case 完全一致
    }

    public struct ZenzaiV2DependentMode: Sendable, Equatable, Hashable {
        public init(profile: String? = nil, leftSideContext: String? = nil) {
            self.profile = profile
            self.leftSideContext = leftSideContext
        }

        /// プロフィールコンテクストを設定した場合、プロフィールを反映したプロンプトが自動的に付与されます。プロフィールは10〜20文字程度の長さにとどめることを推奨します。
        public var profile: String?
        /// 左側の文字列を文脈として与えます。
        public var leftSideContext: String?
    }

    public enum ZenzVersion: Sendable, Equatable, Hashable {
        case v1
        case v2
    }

    public enum ZenzaiVersionDependentMode: Sendable, Equatable, Hashable {
        case v1
        case v2(ZenzaiV2DependentMode)

        public var version: ZenzVersion {
            switch self {
            case .v1:
                return .v1
            case .v2:
                return .v2
            }
        }
    }

    public struct ZenzaiMode: Sendable, Equatable {
        public static let off = ZenzaiMode(
            enabled: false,
            weightURL: URL(fileURLWithPath: ""),
            inferenceLimit: 10,
            gpuLayers: 0,
            requestRichCandidates: false,
            versionDependentMode: .v2(.init())
        )

        /// activate *Zenzai* - Neural Kana-Kanji Conversiion Engine
        /// - Parameters:
        ///    - weight: path for model weight (gguf)
        ///    - inferenceLimit: applying inference count limitation. Smaller limit makes conversion faster but quality will be worse. (Default: 10)
        ///    - requestRichCandidates: when this flag is true, the converter spends more time but generate richer N-Best candidates for candidate list view. Usually this option is not recommended for live conversion.
        ///    - versionDependentMode: specify zenz model version and its configuration.
        public static func on(weight: URL, inferenceLimit: Int = 10, gpuLayers: Int32 = 0, requestRichCandidates: Bool = false, versionDependentMode: ZenzaiVersionDependentMode = .v2(.init())) -> Self {
            ZenzaiMode(
                enabled: true,
                weightURL: weight,
                inferenceLimit: inferenceLimit,
                gpuLayers: gpuLayers,
                requestRichCandidates: requestRichCandidates,
                versionDependentMode: versionDependentMode
            )
        }
        var enabled: Bool
        var weightURL: URL
        var inferenceLimit: Int
        var gpuLayers: Int32
        var requestRichCandidates: Bool
        var versionDependentMode: ZenzaiVersionDependentMode
    }
}
