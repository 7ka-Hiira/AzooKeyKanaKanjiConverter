#if Zenzai
// Zenzai が有効でない場合、llama-mock.swift の実装が利用される
import llama
#endif

import Algorithms
import Foundation
import SwiftUtils

public typealias ZenzaiDeviceConfig = ConvertRequestOptions.ZenzaiMode.DeviceConfig

public struct GGMLBackendDevice: Sendable {
    public let name: String
    public let description: String
    public let type: DeviceType

    public enum DeviceType: Sendable {
        case cpu
        case gpu
        case accel
        case unknown
    }

    #if Zenzai
    init(device: ggml_backend_dev_t) {
        if let namePtr = ggml_backend_dev_name(device) {
            self.name = String(cString: namePtr)
        } else {
            self.name = "Unknown"
        }

        if let descPtr = ggml_backend_dev_description(device) {
            self.description = String(cString: descPtr)
        } else {
            self.description = "Unknown"
        }

        let deviceType = ggml_backend_dev_type(device)
        switch deviceType {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            self.type = .cpu
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            self.type = .gpu
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            self.type = .accel
        default:
            self.type = .unknown
        }
    }
    #endif
}

/// Enumerate available GGML backend devices
/// Note: loadGGMLBackends() should be called once before using this function
public func enumerateGGMLBackendDevices() -> [GGMLBackendDevice] {
    #if Zenzai
    let deviceCount = ggml_backend_dev_count()
    var devices: [GGMLBackendDevice] = []
    for i in 0..<deviceCount {
        if let device = ggml_backend_dev_get(i) {
            devices.append(GGMLBackendDevice(device: device))
        }
    }
    return devices
    #else
    return []
    #endif
}

/// Load all available GGML backends
/// - Parameter path: Optional directory path to load backends from. If nil, uses default search paths.
/// This function should be called once at application startup before using any Zenzai features
public func loadGGMLBackends(from path: String? = nil) {
    #if Zenzai
    if let path = path {
        ggml_backend_load_all_from_path(path)
    } else {
        ggml_backend_load_all()
    }
    #endif
}

/// Create a device configuration based on device type detection
/// - Parameters:
///   - deviceName: Optional device name. If nil, uses the best available device.
///   - gpuLayers: Number of GPU layers when GPU is used. Default is 99.
/// - Returns: A configured ZenzaiDeviceConfig
public func createDeviceConfig(deviceName: String? = nil, gpuLayers: Int32 = 99) -> ZenzaiDeviceConfig {
    #if Zenzai
    let devices = enumerateGGMLBackendDevices()

    // If a specific device name is provided, try to find it
    if let targetName = deviceName {
        if let device = devices.first(where: { $0.name == targetName }) {
            switch device.type {
            case .gpu:
                return ZenzaiDeviceConfig(deviceName: targetName, gpuLayers: gpuLayers)
            case .cpu, .accel, .unknown:
                return ZenzaiDeviceConfig(deviceName: targetName, gpuLayers: 0)
            }
        }
    }

    // Fall back to CPU
    if let cpuDevice = devices.first(where: { $0.type == .cpu }) {
        return ZenzaiDeviceConfig(deviceName: cpuDevice.name, gpuLayers: 0)
    }

    // Default configuration
    return ZenzaiDeviceConfig(deviceName: nil, gpuLayers: 0)
    #else
    return ZenzaiDeviceConfig(deviceName: nil, gpuLayers: 0)
    #endif
}

enum ZenzError: LocalizedError {
    case couldNotLoadModel(path: String)
    case couldNotLoadContext
    case couldNotLoadVocab
    case noDevicesAvailable

    var errorDescription: String? {
        switch self {
        case .couldNotLoadContext: return "failed to load context"
        case .couldNotLoadModel(path: let path): return "could not load model weight at \(path)"
        case .couldNotLoadVocab: return "failed to load vocab"
        case .noDevicesAvailable: return "no llama.cpp compatible device available"
        }
    }
}

final class ZenzContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var vocab: OpaquePointer
    private var prevInputBySeq: [llama_seq_id: [llama_token]] = [:]
    private var prevPromptBySeq: [llama_seq_id: [llama_token]] = [:]
    private var currentDeviceConfig: ZenzaiDeviceConfig
    private var modelPath: String

    private let n_len: Int32 = 512
    private let evalSeqId: llama_seq_id = 0
    private let inputPredictionSeqId: llama_seq_id = 1

    init(model: OpaquePointer, context: OpaquePointer, vocab: OpaquePointer, deviceConfig: ZenzaiDeviceConfig, modelPath: String) {
        self.model = model
        self.context = context
        self.vocab = vocab
        self.currentDeviceConfig = deviceConfig
        self.modelPath = modelPath
    }

    deinit {
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
    }

    private static func ctx_params(deviceConfig: ZenzaiDeviceConfig) -> llama_context_params {
        let n_threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        debug("Using \(n_threads) threads")
        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = 512
        ctx_params.n_threads       = Int32(n_threads)
        ctx_params.n_threads_batch = Int32(n_threads)
        ctx_params.n_batch = 512

        // Configure offload_kqv based on device type
        #if Zenzai
        if deviceConfig.gpuLayers > 0 {
            ctx_params.offload_kqv = true
        } else {
            ctx_params.offload_kqv = false
        }
        #endif

        return ctx_params
    }

    static func createContext(path: String, deviceConfig: ZenzaiDeviceConfig = ZenzaiDeviceConfig()) throws -> ZenzContext {
        llama_backend_init()
        ggml_backend_load_all();

        let deviceCount = ggml_backend_dev_count()
        if deviceCount == 0 {
            debug("No backend devices available")
            throw ZenzError.noDevicesAvailable
        }

        var model_params = llama_model_default_params()
        model_params.use_mmap = true

        #if Zenzai
        // Configure GPU layers based on device config
        model_params.n_gpu_layers = deviceConfig.gpuLayers

        // Set device if specified
        if let deviceName = deviceConfig.deviceName {
            if let device = ggml_backend_dev_by_name(deviceName) {
                var deviceArray = [device, nil]
                let devicePtr = UnsafeMutablePointer<ggml_backend_dev_t?>.allocate(capacity: 2)
                devicePtr.initialize(from: &deviceArray, count: 2)
                model_params.devices = devicePtr
            }
        }
        #endif

        let model = llama_model_load_from_file(path, model_params)

        #if Zenzai
        // Free the allocated device pointer if it was set
        if model_params.devices != nil {
            model_params.devices?.deallocate()
        }
        #endif

        guard let model else {
            debug("Could not load model at \(path)")
            throw ZenzError.couldNotLoadModel(path: path)
        }

        let params = ctx_params(deviceConfig: deviceConfig)
        let context = llama_init_from_model(model, params)
        guard let context else {
            debug("Could not load context!")
            throw ZenzError.couldNotLoadContext
        }

        let vocab = llama_model_get_vocab(model)
        guard let vocab else {
            debug("Could not load vocab!")
            throw ZenzError.couldNotLoadVocab
        }

        return ZenzContext(model: model, context: context, vocab: vocab, deviceConfig: deviceConfig, modelPath: path)
    }

    /// Update device configuration dynamically
    /// This will unload and reload the model with new parameters
    func updateDeviceConfig(_ newConfig: ZenzaiDeviceConfig) throws {
        // Store the new config
        self.currentDeviceConfig = newConfig

        // Free existing resources
        llama_free(self.context)
        llama_model_free(self.model)

        // Reload model with new parameters
        var model_params = llama_model_default_params()
        model_params.use_mmap = true

        #if Zenzai
        // Configure GPU layers based on device config
        model_params.n_gpu_layers = newConfig.gpuLayers

        // Set device if specified
        if let deviceName = newConfig.deviceName {
            if let device = ggml_backend_dev_by_name(deviceName) {
                var deviceArray = [device, nil]
                let devicePtr = UnsafeMutablePointer<ggml_backend_dev_t?>.allocate(capacity: 2)
                devicePtr.initialize(from: &deviceArray, count: 2)
                model_params.devices = devicePtr
            }
        }
        #endif

        let model = llama_model_load_from_file(self.modelPath, model_params)

        #if Zenzai
        // Free the allocated device pointer if it was set
        if model_params.devices != nil {
            model_params.devices?.deallocate()
        }
        #endif

        guard let model else {
            debug("Could not reload model at \(self.modelPath)")
            throw ZenzError.couldNotLoadModel(path: self.modelPath)
        }
        self.model = model

        // Reload context with new parameters
        let params = Self.ctx_params(deviceConfig: newConfig)
        let context = llama_init_from_model(model, params)
        guard let context else {
            debug("Could not load context!")
            throw ZenzError.couldNotLoadContext
        }
        self.context = context

        // Reload vocab
        let vocab = llama_model_get_vocab(model)
        guard let vocab else {
            debug("Could not load vocab!")
            throw ZenzError.couldNotLoadVocab
        }
        self.vocab = vocab

        // Reset state
        self.prevInputBySeq = [:]
        self.prevPromptBySeq = [:]
    }

    /// Get current device configuration
    func getDeviceConfig() -> ZenzaiDeviceConfig {
        return self.currentDeviceConfig
    }

    func resetContext() throws {
        llama_free(self.context)
        let params = Self.ctx_params(deviceConfig: self.currentDeviceConfig)
        let context = llama_init_from_model(self.model, params)
        guard let context else {
            debug("Could not load context!")
            throw ZenzError.couldNotLoadContext
        }
        self.context = context
        self.prevInputBySeq = [:]
        self.prevPromptBySeq = [:]
    }

    private func getLogits(tokens: [llama_token], logits_start_index: Int = 0, seqId: llama_seq_id = 0) -> UnsafeMutablePointer<Float>? {
        let currentPrevInput = self.prevInputBySeq[seqId] ?? []
        var effectivePrevInput = currentPrevInput

        // Try to copy KV cache from the other sequence if it gives a longer prefix match.
        let otherSeqId: llama_seq_id? = if seqId == evalSeqId {
            inputPredictionSeqId
        } else if seqId == inputPredictionSeqId {
            evalSeqId
        } else {
            nil
        }
        if let otherSeqId, let otherPrevInput = self.prevInputBySeq[otherSeqId] {
            let currentPrefix = currentPrevInput.commonPrefix(with: tokens).count
            let otherPrefix = otherPrevInput.commonPrefix(with: tokens).count
            if otherPrefix > currentPrefix {
                let prefixCacheCount = min(otherPrefix, logits_start_index)
                if prefixCacheCount > 0 {
                    llama_kv_cache_seq_rm(context, seqId, 0, -1)
                    llama_kv_cache_seq_cp(context, otherSeqId, seqId, 0, llama_pos(prefixCacheCount))
                    effectivePrevInput = otherPrevInput
                }
            }
        }

        // Manage KV cache: remove entries that differ from previous input
        let prefixCacheCount: Int
        do {
            let pos_max = llama_kv_cache_seq_pos_max(self.context, seqId)
            debug("pos max:", pos_max, "prevInput count:", effectivePrevInput.count, "tokens count:", tokens.count)
            let commonTokens = effectivePrevInput.commonPrefix(with: tokens)
            // Remove KV cache from position commonTokens.count onwards to recompute divergent part
            // removed range: [llama_pos(commonTokens.count), inf)
            prefixCacheCount = min(commonTokens.count, logits_start_index)
            llama_kv_cache_seq_rm(context, seqId, llama_pos(prefixCacheCount), -1)
            debug("new pos max:", llama_kv_cache_seq_pos_max(self.context, seqId), "commonTokens:", commonTokens.count)
        }
        var batch = llama_batch_init(512, 0, 1)
        defer { llama_batch_free(batch) }
        let n_ctx = llama_n_ctx(context)
        let n_kv_req = tokens.count + (Int(n_len) - tokens.count)
        if n_kv_req > n_ctx {
            debug("error: n_kv_req > n_ctx, the required KV cache size is not big enough")
        }
        for i in tokens.indices.dropFirst(prefixCacheCount) {
            llama_batch_add(&batch, tokens[i], Int32(i), [seqId], logits: logits_start_index <= i)
        }
        // 評価
        if llama_decode(context, batch) != 0 {
            debug("llama_decode() failed")
            return nil
        }
        // update cached input for next call (for KV cache management)
        self.prevInputBySeq[seqId] = tokens
        return llama_get_logits(context)
    }

    func previousEvaluationPromptTokens() -> [llama_token] {
        self.prevPromptBySeq[evalSeqId] ?? []
    }

    func setPreviousEvaluationPromptTokens(_ tokens: [llama_token]) {
        self.prevPromptBySeq[evalSeqId] = tokens
    }

    func normalizeForModel(_ text: String) -> String {
        self.preprocessText(text: text)
    }

    func encode(_ text: String, addBOS: Bool, addEOS: Bool = false) -> [llama_token] {
        self.tokenize(text: self.preprocessText(text: text), add_bos: addBOS, add_eos: addEOS)
    }

    func encodeRaw(_ text: String, addBOS: Bool, addEOS: Bool = false) -> [llama_token] {
        self.tokenize(text: text, add_bos: addBOS, add_eos: addEOS)
    }

    func evaluationLogits(tokens: [llama_token], startOffset: Int) -> UnsafeMutablePointer<Float>? {
        self.getLogits(tokens: tokens, logits_start_index: startOffset, seqId: evalSeqId)
    }

    func inputPredictionLogits(tokens: [llama_token], startOffset: Int) -> UnsafeMutablePointer<Float>? {
        self.getLogits(tokens: tokens, logits_start_index: startOffset, seqId: inputPredictionSeqId)
    }

    var vocabSize: Int32 {
        llama_vocab_n_tokens(vocab)
    }

    var eosToken: llama_token {
        llama_vocab_eos(vocab)
    }

    func decodeTokens(_ tokens: [llama_token]) -> String {
        let cchars: [CChar] = tokens.flatMap(self.tokenToPiece)
        let data = Data(cchars.map { UInt8(bitPattern: $0) })
        return String(data: data, encoding: .utf8) ?? ""
    }

    private func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], logits: Bool) {
        batch.token   [Int(batch.n_tokens)] = id
        batch.pos     [Int(batch.n_tokens)] = pos
        batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
        for i in 0..<seq_ids.count {
            batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
        }
        batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0
        batch.n_tokens += 1
    }

    private func preprocessText(text: String) -> String {
        // replace space into ideographic space (\u3000) for zenz tokenizer
        // replace newline into null for zenz tokenizer
        text.replacingOccurrences(of: " ", with: "\u{3000}").replacingOccurrences(of: "\n", with: "")
    }
    private func tokenize(text: String, add_bos: Bool, add_eos: Bool = false) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0)
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(vocab, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, false)
        var swiftTokens: [llama_token] = if tokenCount < 0 {
            [llama_vocab_bos(vocab)]
        } else {
            (0..<tokenCount).map {tokens[Int($0)]}
        }
        tokens.deallocate()
        if add_eos {
            swiftTokens.append(llama_vocab_eos(vocab))
        }
        return swiftTokens
    }

    /// - note: The result does not contain null-terminator
    func tokenToPiece(token: llama_token) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
        let nTokens = llama_token_to_piece(vocab, token, result, 8, 0, false)

        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(vocab, token, newResult, Int32(-nTokens), 0, false)
            let bufferPointer: UnsafeBufferPointer<Int8> = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer: UnsafeBufferPointer<Int8> = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
}
