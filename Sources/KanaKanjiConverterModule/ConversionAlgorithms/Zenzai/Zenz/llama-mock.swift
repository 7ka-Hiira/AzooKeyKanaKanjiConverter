#if !Zenzai
import Foundation

// MARK: - Common typealiases
package typealias llama_token = Int32
package typealias llama_pos = Int32
package typealias llama_seq_id = Int32
package typealias llama_context = OpaquePointer
package typealias llama_model = OpaquePointer
package typealias llama_vocab = OpaquePointer
package typealias ggml_backend_dev_t = OpaquePointer

// MARK: - Helpers
private func retainOpaque<T: AnyObject>(_ object: T) -> OpaquePointer {
    OpaquePointer(Unmanaged.passRetained(object).toOpaque())
}

private func releaseOpaque<T: AnyObject>(_ pointer: OpaquePointer, as type: T.Type) {
    Unmanaged<T>.fromOpaque(UnsafeRawPointer(pointer)).release()
}

private func getObject<T: AnyObject>(_ pointer: OpaquePointer, as type: T.Type) -> T {
    Unmanaged<T>.fromOpaque(UnsafeRawPointer(pointer)).takeUnretainedValue()
}

// MARK: - Mock core objects
private final class MockLlamaVocab: NSObject {
    let bos: llama_token = 1
    let eos: llama_token = 2
    private let baseOffset: llama_token = 3
    let tokenCount: Int = 259 // 256 bytes + bos/eos/pad

    func token(for byte: UInt8) -> llama_token {
        self.baseOffset + llama_token(byte)
    }

    func piece(for token: llama_token) -> [Int8]? {
        if token == bos || token == eos || token == 0 { return [] }
        let raw = Int(token - baseOffset)
        guard (0...255).contains(raw) else { return nil }
        return [Int8(bitPattern: UInt8(raw))]
    }
}

private final class MockLlamaModel: NSObject {
    let path: String
    let params: llama_model_params
    let vocab: MockLlamaVocab

    init(path: String, params: llama_model_params) {
        self.path = path
        self.params = params
        self.vocab = MockLlamaVocab()
    }
}

private final class MockLlamaContext: NSObject {
    let model: MockLlamaModel
    let params: llama_context_params
    var cachedTokens: [llama_token] = []
    var logitsPointer: UnsafeMutablePointer<Float>?
    var logitsCount: Int = 0

    init(model: MockLlamaModel, params: llama_context_params) {
        self.model = model
        self.params = params
    }

    deinit {
        logitsPointer?.deallocate()
    }
}

private final class MockDevice: NSObject {
    let name: String
    let descriptionText: String
    let type: Int32

    init(name: String, description: String, type: Int32) {
        self.name = name
        self.descriptionText = description
        self.type = type
    }
}

// MARK: - Params
package struct llama_context_params {
    package var seed: Int32 = 0
    package var n_ctx: Int32 = 512
    package var n_threads: Int32 = 0
    package var n_threads_batch: Int32 = 0
    package var n_batch: Int32 = 512
    package var offload_kqv: Bool = false
}

package func llama_context_default_params() -> llama_context_params {
    llama_context_params()
}

package struct llama_model_params {
    package var use_mmap: Bool = false
    package var n_gpu_layers: Int32 = 0
    package var devices: UnsafeMutablePointer<ggml_backend_dev_t?>? = nil
}

package func llama_model_default_params() -> llama_model_params {
    llama_model_params()
}

// MARK: - Backend mock
package let GGML_BACKEND_DEVICE_TYPE_CPU: Int32 = 0
package let GGML_BACKEND_DEVICE_TYPE_GPU: Int32 = 1
package let GGML_BACKEND_DEVICE_TYPE_ACCEL: Int32 = 2

private let mockDeviceNameStorage: [CChar] = Array("MockCPU".utf8CString)
private let mockDeviceDescriptionStorage: [CChar] = Array("Mock GGML CPU device".utf8CString)
private let mockDevicePointer: ggml_backend_dev_t = {
    retainOpaque(MockDevice(name: "MockCPU", description: "Mock GGML CPU device", type: GGML_BACKEND_DEVICE_TYPE_CPU))
}()

package func ggml_backend_load_all() {}
package func ggml_backend_load_all_from_path(_: String) {}

package func ggml_backend_dev_count() -> Int { 1 }

package func ggml_backend_dev_get(_ index: Int) -> ggml_backend_dev_t? {
    index == 0 ? mockDevicePointer : nil
}

package func ggml_backend_dev_by_name(_ name: String) -> ggml_backend_dev_t? {
    name == "MockCPU" ? mockDevicePointer : nil
}

package func ggml_backend_dev_name(_: ggml_backend_dev_t) -> UnsafePointer<CChar>? {
    mockDeviceNameStorage.withUnsafeBufferPointer { $0.baseAddress }
}

package func ggml_backend_dev_description(_: ggml_backend_dev_t) -> UnsafePointer<CChar>? {
    mockDeviceDescriptionStorage.withUnsafeBufferPointer { $0.baseAddress }
}

package func ggml_backend_dev_type(_: ggml_backend_dev_t) -> Int32 {
    GGML_BACKEND_DEVICE_TYPE_CPU
}

package func llama_backend_init() {}
package func llama_backend_free() {}

// MARK: - Model / context lifecycle
package func llama_model_load_from_file(_ path: String, _ params: llama_model_params) -> llama_model? {
    retainOpaque(MockLlamaModel(path: path, params: params))
}

package func llama_init_from_model(_ model: llama_model, _ params: llama_context_params) -> llama_context? {
    let mockModel: MockLlamaModel = getObject(model, as: MockLlamaModel.self)
    let context = MockLlamaContext(model: mockModel, params: params)
    return retainOpaque(context)
}

package func llama_model_get_vocab(_ model: llama_model) -> llama_vocab? {
    let mockModel: MockLlamaModel = getObject(model, as: MockLlamaModel.self)
    return OpaquePointer(Unmanaged.passUnretained(mockModel.vocab).toOpaque())
}

package func llama_model_free(_ model: llama_model) {
    releaseOpaque(model, as: MockLlamaModel.self)
}

package func llama_free(_ context: llama_context) {
    releaseOpaque(context, as: MockLlamaContext.self)
}

// MARK: - KV cache
package func llama_kv_cache_seq_rm(_ ctx: llama_context, _: llama_seq_id, _ p0: llama_pos, _ p1: llama_pos) {
    let context: MockLlamaContext = getObject(ctx, as: MockLlamaContext.self)
    let start = max(0, Int(p0))
    let end = p1 < 0 ? context.cachedTokens.count : min(context.cachedTokens.count, Int(p1))
    if start < end {
        context.cachedTokens.removeSubrange(start..<end)
    }
}

package func llama_kv_cache_seq_pos_max(_ ctx: llama_context, _: llama_seq_id) -> Int {
    let context: MockLlamaContext = getObject(ctx, as: MockLlamaContext.self)
    return context.cachedTokens.count
}

// MARK: - Batch
package struct llama_batch {
    package var token: [llama_token]
    package var pos: [llama_pos]
    package var n_seq_id: [llama_seq_id]
    package var seq_id: [[llama_seq_id]?]
    package var logits: UnsafeMutablePointer<Float>
    package var n_tokens: Int32
}

package func llama_batch_init(_ n_tokens: Int, _: Int, _: Int) -> llama_batch {
    let logitsPointer = UnsafeMutablePointer<Float>.allocate(capacity: n_tokens)
    logitsPointer.initialize(repeating: 0, count: n_tokens)
    return llama_batch(
        token: Array(repeating: 0, count: n_tokens),
        pos: Array(repeating: 0, count: n_tokens),
        n_seq_id: Array(repeating: 0, count: n_tokens),
        seq_id: Array(repeating: nil, count: n_tokens),
        logits: logitsPointer,
        n_tokens: 0
    )
}

package func llama_batch_free(_ batch: llama_batch) {
    batch.logits.deallocate()
}

// MARK: - Context info
package func llama_n_ctx(_ ctx: llama_context) -> Int {
    let context: MockLlamaContext = getObject(ctx, as: MockLlamaContext.self)
    return Int(context.params.n_ctx == 0 ? 512 : context.params.n_ctx)
}

// MARK: - Tokenization
package func llama_vocab_n_tokens(_ vocab: llama_vocab) -> Int {
    let v: MockLlamaVocab = getObject(vocab, as: MockLlamaVocab.self)
    return v.tokenCount
}

package func llama_tokenize(
    _ model: llama_model,
    _ text: String,
    _ textLen: Int32,
    _ tokens: UnsafeMutablePointer<llama_token>,
    _ n_tokens: Int32,
    _ add_bos: Bool,
    _: Bool
) -> Int {
    _ = model // unused in mock
    let utf8 = Array(text.utf8.prefix(Int(textLen)))
    let vocab = MockLlamaVocab()
    var result: [llama_token] = add_bos ? [vocab.bos] : []
    result.append(contentsOf: utf8.map(vocab.token))
    guard result.count <= n_tokens else {
        return -Int(result.count)
    }
    for (i, token) in result.enumerated() {
        tokens[i] = token
    }
    return result.count
}

package func llama_vocab_eos(_ vocab: llama_vocab) -> llama_token {
    let v: MockLlamaVocab = getObject(vocab, as: MockLlamaVocab.self)
    return v.eos
}

package func llama_vocab_bos(_ vocab: llama_vocab) -> llama_token {
    let v: MockLlamaVocab = getObject(vocab, as: MockLlamaVocab.self)
    return v.bos
}

package func llama_token_to_piece(
    _ vocab: llama_vocab,
    _ token: llama_token,
    _ buffer: UnsafeMutablePointer<Int8>,
    _ length: Int32,
    _: Int32,
    _: Bool
) -> Int32 {
    let v: MockLlamaVocab = getObject(vocab, as: MockLlamaVocab.self)
    guard let piece = v.piece(for: token) else { return 0 }
    if piece.count > length {
        return -Int32(piece.count)
    }
    for (i, byte) in piece.enumerated() {
        buffer[i] = byte
    }
    return Int32(piece.count)
}

// MARK: - Decode / logits
package func llama_decode(_ ctx: llama_context, _ batch: llama_batch) -> Int {
    let context: MockLlamaContext = getObject(ctx, as: MockLlamaContext.self)
    context.logitsPointer?.deallocate()
    let vocab = context.model.vocab
    let vocabSize = vocab.tokenCount
    let flaggedIndices = (0..<Int(batch.n_tokens)).filter { Int(batch.logits[$0]) != 0 }
    guard !flaggedIndices.isEmpty else {
        context.logitsPointer = nil
        context.logitsCount = 0
        context.cachedTokens = Array(batch.token.prefix(Int(batch.n_tokens)))
        return 0
    }
    let total = flaggedIndices.count * vocabSize
    let buffer = UnsafeMutablePointer<Float>.allocate(capacity: total)
    for (outIndex, batchIndex) in flaggedIndices.enumerated() {
        let token = batch.token[batchIndex]
        for v in 0..<vocabSize {
            let base = abs(Int(token % llama_token(vocabSize)) - v)
            buffer[outIndex * vocabSize + v] = -Float(base)
        }
    }
    context.logitsPointer = buffer
    context.logitsCount = total
    context.cachedTokens = Array(batch.token.prefix(Int(batch.n_tokens)))
    return 0
}

package func llama_get_logits(_ ctx: llama_context) -> UnsafeMutablePointer<Float>? {
    let context: MockLlamaContext = getObject(ctx, as: MockLlamaContext.self)
    return context.logitsPointer
}
#endif
