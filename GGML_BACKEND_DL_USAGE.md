# GGML Backend DL Support Usage Guide

This document describes how to use the GGML backend dynamic loading (DL) support added to AzooKeyKanaKanjiConverter.

## Overview

The GGML backend DL support allows:
1. Dynamic loading of GGML backends at runtime
2. Enumeration and selection of available backend devices (CPU, GPU, etc.)
3. Configuration of GPU layers and other device-specific parameters
4. Automatic device detection and configuration

## Features

### 1. Backend Device Enumeration

You can enumerate all available GGML backend devices:

```swift
#if Zenzai
import KanaKanjiConverterModule

// Enumerate all available backend devices
let devices = enumerateBackendDevices()
for device in devices {
    print("Device: \(device.name)")
    print("Description: \(device.description)")
    print("Type: \(device.type)")
}
#endif
```

### 2. Device Configuration

#### Manual Configuration

Configure a specific device with custom GPU layers:

```swift
let deviceConfig = ConvertRequestOptions.ZenzaiMode.DeviceConfig(
    deviceName: "CUDA0",  // Optional: specify device name
    gpuLayers: 13         // Number of layers to offload to GPU
)

let zenzaiMode = ConvertRequestOptions.ZenzaiMode.on(
    weight: modelURL,
    deviceConfig: deviceConfig
)
```

#### Auto-Detection

Use the helper function for automatic device detection:

```swift
#if Zenzai
// Auto-detect and prefer GPU if available
let autoConfig = createDeviceConfig(preferGPU: true, gpuLayers: 13)

let zenzaiMode = ConvertRequestOptions.ZenzaiMode.on(
    weight: modelURL,
    deviceConfig: autoConfig
)
#endif
```

### 3. Dynamic Device Configuration

You can update device configuration at runtime:

```swift
// Get the Zenz instance
let converter = KanaKanjiConverter()
// ... initialize converter with options ...

// Later, update device configuration
#if Zenzai
let newConfig = ConvertRequestOptions.ZenzaiMode.DeviceConfig(
    deviceName: nil,
    gpuLayers: 20  // Change number of GPU layers
)

// This will be applied on the next conversion
let updatedOptions = ConvertRequestOptions(
    // ... other options ...
    zenzaiMode: .on(weight: modelURL, deviceConfig: newConfig)
)
#endif
```

## Device Type Behavior

### CPU Device
When using a CPU device or `gpuLayers = 0`:
- `n_gpu_layers` = 0
- `split_mode` = LLAMA_SPLIT_MODE_NONE
- `offload_kqv` = false

### GPU Device
When using a GPU device with `gpuLayers > 0`:
- `n_gpu_layers` = specified value (default: 13)
- `split_mode` = LLAMA_SPLIT_MODE_LAYER
- `offload_kqv` = true

## Example: Complete Setup

```swift
import KanaKanjiConverterModule
import Foundation

// 1. Enumerate devices (optional)
#if Zenzai
let devices = enumerateBackendDevices()
print("Available devices:")
for device in devices {
    print("- \(device.name): \(device.description) (Type: \(device.type))")
}

// 2. Create device configuration
let deviceConfig: ConvertRequestOptions.ZenzaiMode.DeviceConfig

if let gpuDevice = devices.first(where: { $0.type == .gpu }) {
    // Use GPU with 13 layers
    deviceConfig = ConvertRequestOptions.ZenzaiMode.DeviceConfig(
        deviceName: gpuDevice.name,
        gpuLayers: 13
    )
    print("Using GPU device: \(gpuDevice.name)")
} else {
    // Fall back to CPU
    deviceConfig = ConvertRequestOptions.ZenzaiMode.DeviceConfig(
        deviceName: nil,
        gpuLayers: 0
    )
    print("Using CPU device")
}
#else
let deviceConfig = ConvertRequestOptions.ZenzaiMode.DeviceConfig()
#endif

// 3. Create converter options with device configuration
let options = ConvertRequestOptions(
    N_best: 10,
    requireJapanesePrediction: .autoMix,
    requireEnglishPrediction: .disabled,
    keyboardLanguage: .ja_JP,
    learningType: .inputAndOutput,
    memoryDirectoryURL: memoryURL,
    sharedContainerURL: sharedURL,
    textReplacer: .init(
        leftSideContext: "",
        dictionary: [:],
        learnedDictionary: [:],
        customKeys: []
    ),
    specialCandidateProviders: nil,
    zenzaiMode: .on(
        weight: modelURL,
        inferenceLimit: 10,
        requestRichCandidates: false,
        personalizationMode: nil,
        versionDependentMode: .v3(.init()),
        deviceConfig: deviceConfig
    ),
    metadata: nil
)

// 4. Use the converter
let converter = KanaKanjiConverter()
let result = converter.requestConversion(inputData, options: options)
```

## Build Requirements

### Without Zenzai Trait
The code compiles and runs normally without the Zenzai trait enabled. Backend enumeration functions return empty arrays.

```bash
swift build --target KanaKanjiConverterModule
```

### With Zenzai Trait
To use GGML backend DL support, build with the Zenzai trait:

```bash
swift build --target KanaKanjiConverterModule --enable-trait Zenzai
```

## Testing on Linux

When testing on Linux with llama.cpp built with BACKEND_DL support:

1. Ensure llama.cpp is built with `-DGGML_BACKEND_DL=ON`
2. Place backend shared libraries in the library search path
3. The `ggml_backend_load_all()` function will automatically discover and load available backends

## Notes

- The device configuration is applied when the model is first loaded
- If the device configuration changes, the context is recreated with new parameters
- GPU layers are only used when a GPU backend is available and configured
- The default configuration (no device specified, 0 GPU layers) uses CPU mode
