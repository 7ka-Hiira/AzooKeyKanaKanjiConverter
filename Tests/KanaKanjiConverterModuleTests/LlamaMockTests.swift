@testable import KanaKanjiConverterModule
import XCTest

final class LlamaMockTests: XCTestCase {
    func testMockTokenizationAndDecode() {
        var modelParams = llama_model_default_params()
        guard let model = llama_model_load_from_file("mock", modelParams) else {
            XCTFail("failed to create mock model")
            return
        }
        defer { llama_model_free(model) }

        guard let vocab = llama_model_get_vocab(model) else {
            XCTFail("failed to get vocab")
            return
        }

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = 16
        guard let ctx = llama_init_from_model(model, ctxParams) else {
            XCTFail("failed to create mock context")
            return
        }
        defer { llama_free(ctx) }

        let tokensPointer = UnsafeMutablePointer<llama_token>.allocate(capacity: 3)
        defer { tokensPointer.deallocate() }
        let tokenCount = llama_tokenize(model, "ab", 2, tokensPointer, 3, true, false)
        XCTAssertEqual(tokenCount, 3)
        XCTAssertEqual(tokensPointer[0], llama_vocab_bos(vocab))

        var pieceBuffer = [Int8](repeating: 0, count: 4)
        let pieceCount = llama_token_to_piece(vocab, tokensPointer[1], &pieceBuffer, 4, 0, false)
        XCTAssertEqual(pieceCount, 1)
        XCTAssertEqual(UInt8(bitPattern: pieceBuffer[0]), UInt8(ascii: "a"))

        var batch = llama_batch_init(3, 0, 1)
        batch.token[0] = tokensPointer[0]
        batch.pos[0] = 0
        batch.n_seq_id[0] = 1
        batch.seq_id[0] = [0]
        batch.logits[0] = 1
        batch.n_tokens = 1

        XCTAssertEqual(llama_decode(ctx, batch), 0)
        XCTAssertNotNil(llama_get_logits(ctx))
        llama_batch_free(batch)
    }
}
