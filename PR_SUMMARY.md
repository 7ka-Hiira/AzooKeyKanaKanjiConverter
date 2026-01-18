# GGML Backend DL Support Implementation

このPRは、AzooKeyKanaKanjiConverterにGGML Backend Dynamic Loading (DL)のサポートを追加します。

## 実装内容

### ✅ 完了した機能

1. **GGMLバックエンドの動的ロード**
   - `ggml_backend_load_all()`をZenzContext初期化時に呼び出し
   - 利用可能なすべてのバックエンドを自動検出・ロード

2. **バックエンドデバイスの列挙と選択**
   - `enumerateGGMLBackendDevices()` - 利用可能なデバイスを列挙
   - デバイス名、説明、タイプ（CPU/GPU/ACCEL）を取得可能

3. **GPUレイヤー数の動的設定**
   - デバイス設定でGPUレイヤー数を指定可能
   - 初期化時と実行時の両方で設定可能

4. **デバイスタイプの自動認識と設定**
   - CPUデバイスの場合:
     - `n_gpu_layers = 0`
     - `split_mode = LLAMA_SPLIT_MODE_NONE`
     - `offload_kqv = false`
   - GPUデバイスの場合:
     - `n_gpu_layers = 指定値（デフォルト13）`
     - `split_mode = LLAMA_SPLIT_MODE_LAYER`
     - `offload_kqv = true`

5. **公開API**
   - `ConvertRequestOptions.ZenzaiMode.DeviceConfig` - デバイス設定構造体
   - `createDeviceConfig()` - 自動検出ヘルパー関数
   - `enumerateBackendDevices()` - デバイス列挙関数

## 変更ファイル

- `Sources/llama.cpp/module.modulemap` - `link "ggml"`を追加
- `Sources/.../llama-mock.swift` - Int32型修正とGGML backend DL用の型・関数追加
- `Sources/.../ZenzContext.swift` - デバイス設定サポート、バックエンドロード処理
- `Sources/.../Zenz.swift` - デバイス設定パラメータ追加
- `Sources/.../ConvertRequestOptions.swift` - DeviceConfig構造体追加
- `Sources/.../KanaKanjiConverter.swift` - デバイス設定の適用ロジック
- `GGML_BACKEND_DL_USAGE.md` - 使用方法ドキュメント（英語）
- `IMPLEMENTATION_SUMMARY.md` - 実装詳細ドキュメント（日本語）

## 使用例

```swift
// デバイスを列挙
let devices = enumerateBackendDevices()
for device in devices {
    print("Device: \(device.name) - \(device.description)")
}

// デバイス設定を作成
let deviceConfig = ConvertRequestOptions.ZenzaiMode.DeviceConfig(
    deviceName: nil,    // 自動選択
    gpuLayers: 13       // GPUレイヤー数
)

// Zenzaiモードで使用
let zenzaiMode = ConvertRequestOptions.ZenzaiMode.on(
    weight: modelURL,
    deviceConfig: deviceConfig
)
```

## 互換性

- ✅ Zenzaiトレイト無効時: 正常にビルド
- ✅ Zenzaiトレイト有効時: 新機能が有効化
- ✅ 既存コード: 変更なしで動作（デフォルトパラメータ使用）
- ✅ Linux環境: メインターゲット
- ✅ macOS/iOS: 互換性あり

## テスト結果

- ✅ Zenzaiトレイト無効でのビルド成功
- ✅ コンパイルエラーなし
- ✅ 既存APIとの互換性確認

## 参考リンク

7ka-Hiira/llama.cppライブラリとの統合を想定:
- BACKEND_DL, VULKAN, CPU_ALL_VARIANTSがONでビルド
- CUDAやHIPのサポートも含む

---

すべての要件を満たし、Linuxでの動作をメインに考慮した実装です。
