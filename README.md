# Hello WebGPU Compute

demo https://novogrammer.github.io/hello-webgpu-compute/

## Runtime Toggles (TSL demos)

各 TSL デモページに「Force WebGL」「Show compute shader」のチェックボックスを追加しています。

- Force WebGL: three.js WebGPURenderer を WebGL フォールバックで実行（WebGPU未対応でも動作）
- Show compute shader: 実行に使われたコンピュートシェーダ文字列をメッセージ欄に出力

チェック状態は Run 実行ごとに反映されます（リロード不要）。
