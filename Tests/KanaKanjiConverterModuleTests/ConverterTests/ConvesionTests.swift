//
//  ConversionTests.swift
//
//
//  Created by miwa on 2023/08/16.
//

@testable import KanaKanjiConverterModule
import XCTest

final class ConverterTests: XCTestCase {
    func dictionaryURL() -> URL {
        Bundle(for: type(of: self)).bundleURL.appendingPathComponent("DictionaryMock", isDirectory: true)
    }
    func requestOptions() -> ConvertRequestOptions {
        ConvertRequestOptions(
            N_best: 5,
            requireJapanesePrediction: true,
            requireEnglishPrediction: false,
            keyboardLanguage: .ja_JP,
            englishCandidateInRoman2KanaInput: true,
            fullWidthRomanCandidate: false,
            halfWidthKanaCandidate: false,
            learningType: .nothing,
            maxMemoryCount: 0,
            shouldResetMemory: false,
            memoryDirectoryURL: URL(fileURLWithPath: ""),
            sharedContainerURL: URL(fileURLWithPath: ""),
            textReplacer: .empty,
            specialCandidateProviders: [],
            metadata: nil
        )
    }

    // 変換されてはいけないケースを示す
    func testMustNotCases() async throws {
        do {
            // 改行文字に対して本当に改行が入ってしまうケース
            let converter = KanaKanjiConverter(dictionaryURL: dictionaryURL())
            var c = ComposingText()
            c.insertAtCursorPosition("\\n", inputStyle: .direct)
            let results = converter.requestCandidates(c, options: requestOptions())
            XCTAssertFalse(results.mainResults.contains(where: {$0.text == "\n"}))
        }
    }
}
