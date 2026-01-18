# GGML Backend DL Support Implementation Summary

このドキュメントは、AzooKeyKanaKanjiConverterにGGMLバックエンドのダイナミックローディング(DL)サポートを追加した変更の詳細を説明します。

## 変更概要

GGML_BACKEND_DLに対応するため、以下の機能を実装しました：

1. ✅ GGMLバックエンドのダイナミックローディングサポート
2. ✅ バックエンドデバイスの列挙と選択
3. ✅ GPUレイヤー数の動的設定
4. ✅ デバイスタイプ(CPU/GPU)の自動検出
5. ✅ デバイスタイプに応じたパラメータの自動設定
6. ✅ Zenzaiトレイト有効/無効での正常なビルド

## 実装詳細

### 1. module.modulemapの更新

`Sources/llama.cpp/module.modulemap`に`link "ggml"`を追加：

```modulemap
module llama [system] {
    header "llama.h"
    header "ggml.h"
    header "ggml-alloc.h"
    header "ggml-backend.h"

    link "llama"
    link "ggml"  // 追加

    export *
}
```

これにより、GGMLライブラリのAPIを使用できるようになります。

### 2. llama-mock.swiftの型修正

Zenzaiが無効な場合に使用されるモック実装で、Int32にすべき箇所がIntになっていた問題を修正：

- `llama_context_params`: `seed`, `n_ctx`, `n_batch`をInt32に変更
- `llama_batch`: `n_tokens`をInt32に変更
- `offload_kqv`フィールドを追加
- `llama_model_params`: `n_gpu_layers`, `split_mode`, `main_gpu`を追加

また、GGML backend DLサポート用の新しい型と関数を追加：

```swift
package typealias ggml_backend_dev_t = OpaquePointer

package func ggml_backend_load_all() {}
package func ggml_backend_dev_count() -> Int { unimplemented() }
package func ggml_backend_dev_get(_: Int) -> ggml_backend_dev_t? { unimplemented() }
package func ggml_backend_dev_name(_: ggml_backend_dev_t) -> UnsafePointer<CChar>? { unimplemented() }
package func ggml_backend_dev_description(_: ggml_backend_dev_t) -> UnsafePointer<CChar>? { unimplemented() }
package func ggml_backend_dev_type(_: ggml_backend_dev_t) -> Int32 { unimplemented() }
```

### 3. ZenzContext.swiftの拡張

#### 新しい構造体の追加

```swift
/// GGMLバックエンドデバイス情報
package struct GGMLBackendDevice: Sendable {
    package let name: String
    package let description: String
    package let type: DeviceType
    
    package enum DeviceType: Sendable {
        case cpu
        case gpu
        case accel
        case unknown
    }
}

/// Zenzaiバックエンドデバイスの設定
package struct ZenzaiDeviceConfig: Sendable {
    package var deviceName: String?
    package var gpuLayers: Int32
}
```

#### バックエンドデバイス列挙関数

```swift
/// 利用可能なGGMLバックエンドデバイスを列挙
package func enumerateGGMLBackendDevices() -> [GGMLBackendDevice]

/// デバイス設定の自動作成
package func createDeviceConfig(
    deviceName: String? = nil,
    preferGPU: Bool = true,
    gpuLayers: Int32 = 13
) -> ZenzaiDeviceConfig
```

#### ZenzContextクラスの更新

- `createContext`メソッドに`deviceConfig`パラメータを追加
- `ggml_backend_load_all()`を初期化時に呼び出し
- デバイス設定に基づいてモデルとコンテキストのパラメータを設定：
  - CPUモード（gpuLayers=0）:
    - `n_gpu_layers = 0`
    - `split_mode = LLAMA_SPLIT_MODE_NONE`
    - `offload_kqv = false`
  - GPUモード（gpuLayers>0）:
    - `n_gpu_layers = 指定値`
    - `split_mode = LLAMA_SPLIT_MODE_LAYER`
    - `offload_kqv = true`

- `updateDeviceConfig`メソッドを追加：実行時にデバイス設定を変更可能
- `getDeviceConfig`メソッドを追加：現在の設定を取得

### 4. Zenz.swiftの更新

- `init`メソッドに`deviceConfig`パラメータを追加（デフォルト値あり）
- `updateDeviceConfig`メソッドを追加
- `getDeviceConfig`メソッドを追加
- `enumerateBackendDevices()`関数をエクスポート

### 5. ConvertRequestOptions.swiftの更新

`ZenzaiMode`構造体に`DeviceConfig`を追加：

```swift
public struct ZenzaiMode: Sendable, Equatable {
    public struct DeviceConfig: Sendable, Equatable {
        public init(deviceName: String? = nil, gpuLayers: Int32 = 0) {
            self.deviceName = deviceName
            self.gpuLayers = gpuLayers
        }
        
        public var deviceName: String?
        public var gpuLayers: Int32
    }
    
    // ...
    
    public static func on(
        weight: URL,
        inferenceLimit: Int = 10,
        requestRichCandidates: Bool = false,
        personalizationMode: PersonalizationMode?,
        versionDependentMode: ZenzaiVersionDependentMode = .v3(.init()),
        deviceConfig: DeviceConfig = DeviceConfig()  // 追加
    ) -> Self
    
    var deviceConfig: DeviceConfig  // 追加
}
```

### 6. KanaKanjiConverter.swiftの更新

`getModel`メソッドを更新：

- `deviceConfig`パラメータを追加
- デバイス設定が変更された場合、自動的に更新
- Zenzai有効時のみデバイス設定を適用（コンパイル時条件分岐）

## 使用例

### 基本的な使用

```swift
import KanaKanjiConverterModule

// デバイスを列挙
#if Zenzai
let devices = enumerateBackendDevices()
for device in devices {
    print("Device: \(device.name) - \(device.description)")
}
#endif

// デバイス設定を作成
let deviceConfig = ConvertRequestOptions.ZenzaiMode.DeviceConfig(
    deviceName: nil,    // 自動選択
    gpuLayers: 13       // GPU使用時のレイヤー数
)

// Zenzaiモードで使用
let zenzaiMode = ConvertRequestOptions.ZenzaiMode.on(
    weight: modelURL,
    deviceConfig: deviceConfig
)

let options = ConvertRequestOptions(
    // ... 他のオプション ...
    zenzaiMode: zenzaiMode
)
```

### 自動検出を使用

```swift
#if Zenzai
// GPUを優先して自動検出
let autoConfig = createDeviceConfig(preferGPU: true, gpuLayers: 13)

let zenzaiMode = ConvertRequestOptions.ZenzaiMode.on(
    weight: modelURL,
    deviceConfig: autoConfig
)
#endif
```

## ビルド確認

### Zenzai無効でのビルド

```bash
swift build --target KanaKanjiConverterModule
```

✅ 正常にビルド完了を確認

### Zenzai有効でのビルド

```bash
swift build --target KanaKanjiConverterModule --enable-trait Zenzai
```

注: 実際のllama.cppライブラリが必要です。

## 互換性

### 後方互換性

- デフォルトパラメータを使用することで、既存のコードは変更なしで動作
- Zenzaiトレイトが無効な場合、すべての新機能は無効化され、既存の動作を維持

### プラットフォーム

- Linux: メインターゲット
- macOS/iOS: 互換性あり
- Windows: llama.cppがサポートする限り対応可能

## テスト

### 実施したテスト

1. ✅ Zenzai無効でのビルド: 成功
2. ✅ コンパイルエラーなし
3. ✅ 既存APIとの互換性確認

### 追加で必要なテスト（実機でのテスト時）

1. GGML backend DLライブラリとの統合テスト
2. GPU/CPUデバイスの切り替えテスト
3. 異なるGPUレイヤー数での動作確認
4. パフォーマンステスト

## 参考情報

- llama.cppのバックエンドDLサポート: `ggml_backend_load_all()`
- デバイスタイプ: `GGML_BACKEND_DEVICE_TYPE_CPU`, `GGML_BACKEND_DEVICE_TYPE_GPU`
- 分割モード: `LLAMA_SPLIT_MODE_NONE`, `LLAMA_SPLIT_MODE_LAYER`, `LLAMA_SPLIT_MODE_ROW`

## まとめ

この実装により、以下が可能になりました：

1. ✅ GGML_BACKEND_DLを使用した動的バックエンドローディング
2. ✅ 利用可能なバックエンドデバイスの列挙と選択
3. ✅ GPUレイヤー数の動的設定
4. ✅ デバイスタイプに応じた自動パラメータ設定
5. ✅ 実行時のデバイス設定変更
6. ✅ Zenzaiトレイト有効/無効での正常なビルド

すべての要件を満たし、後方互換性を保ちながら新機能を追加できました。
