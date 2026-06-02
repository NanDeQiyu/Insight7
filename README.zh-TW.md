[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17/20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-1140%2B%20passed-brightgreen.svg)](tests/)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Insight

**輕量級、工業級張量計算框架，專為訊號處理與 GPU 加速設計。**

Insight 是一個 C++ 張量庫，設計理念深受 **PaddlePaddle**（算子註冊機制、裝置 HAL）、**Torch7**（簡潔的 C API、TH/THC 架構精神）和 **NumPy/CuPy**（Python 端 API 習慣）啟發。提供 CPU/GPU 統一計算、零拷貝視圖、動態算子派發和完整的訊號處理支援。

## 特性

- **統一 API** -- `insight::Array` 無縫在 `CPUPlace` / `GPUPlace` 間切換
- **零拷貝視圖** -- 透過 `strides` + `offset` 實現 `reshape`、`transpose`、`slice`
- **動態算子註冊** -- `ops()["add"][CPU][F32](args)` 派發（Paddle 風格）
- **裝置 HAL** -- 透過 `Device` 基底類別 + `extern "C"` 工廠實現 ABI 穩定的外掛系統
- **訊號處理** -- 89 個函數，涵蓋 14 個子模組（窗函數、波形產生、B 樣條、濾波器設計、卷積、濾波、頻譜分析、小波、聲學、雷達、解調、峰值偵測、參數估計、I/O），全部配有 CPU 與 CUDA 後端 kernel
- **半精度支援** -- fp16/bf16 支援，透過 `half_utils.h` / `half_utils.cuh` 實現，116 個 kernel 檔案包含半精度覆蓋
- **語言繫結** -- Python（pybind11）、Lua（sol2）、Julia（ccall），按模組拆分的 wrapper，訊號子命名空間
- **現代 C++** -- C++17/20，OpenMP 平行，FFTW3，OpenBLAS，cuBLAS，cuFFT
- **無自動微分** -- 保持函式庫輕量、專注

## 架構

```
insight/
├── include/insight/
│   ├── core/           # Array, Shape, Strides, DType, Place
│   ├── ops/            # 前端 API（elementwise, fft, signal, linalg 等）
│   ├── io/             # I/O（csv, print, sndfile）
│   ├── c_api/          # C ABI 介面（array, kernel, dtype, place）
│   └── plugin/         # 算子註冊 + 裝置 HAL
├── src/
│   ├── core/           # Array 實作、記憶體管理
│   ├── ops/            # 前端算子邏輯
│   └── internal/       # 內部工具
├── backends/
│   ├── cpu/kernels/    # CPU kernel（OpenMP + FFTW + OpenBLAS）
│   │   ├── cast/       ├── elementwise/   ├── fft/
│   │   ├── creation/   ├── indexing/      ├── linalg/
│   │   ├── manipulation/ ├── random/     ├── reduction/
│   │   ├── unary/      └── signal/       （14 個子目錄）
│   └── cuda/kernels/   # CUDA kernel（cuBLAS + cuFFT + Thrust）
│       ├── cast/       ├── elementwise/   ├── fft/
│       ├── creation/   ├── indexing/      ├── linalg/
│       ├── manipulation/ ├── random/     ├── reduction/
│       ├── unary/      └── signal/       （14 個子目錄）
├── bindings/
│   ├── python/insight/ # pybind11 繫結（按模組拆分的 wrapper）
│   ├── lua/insight/    # sol2 繫結（雙呼叫約定）
│   └── julia/          # ccall 繫結（Insight.jl）
├── tests/
│   ├── cpu/            # CPU 測試（630+ 測試，27 個套件）
│   ├── cuda/           # CUDA 測試（510+ 測試，23 個套件）
│   └── python_align/   # NumPy 精度對齊測試（194 CPU + 190 CUDA）
└── demos/              # 範例程式（C++, Python, Lua, Julia）
```

## 快速開始

### 從原始碼編譯

```bash
git clone https://github.com/PlumBlossomMaid/insight.git
cd insight
mkdir build && cd build
cmake .. -DINSIGHT_WITH_CUDA=ON   # 啟用 CUDA 後端
cmake --build . -j$(nproc)
```

### C++ 範例

```cpp
#include "insight/insight.h"
using namespace insight;

int main() {
    // 在 GPU 上建立張量
    Array a = ones({1000, 1000}, F32, GPUPlace(0));
    Array b = randn({1000, 1000}, F32, GPUPlace(0));

    // 矩陣乘法（自動選擇 GPU kernel）
    Array c = matmul(a, b);

    // 移到 CPU 並存取資料
    Array cpu_c = c.to(CPUPlace());
    float value = cpu_c.at({0, 0}).item<float>();
}
```

### Python 範例

```python
import insight as ins

a = ins.zeros([2, 3], ins.float32)
b = ins.ones([2, 3], ins.float32)
c = a + b
s = ins.sum(c, axis=0)

# 訊號處理
w = ins.signal.hann(256)
f, Pxx = ins.signal.welch(x, fs=1000)
```

### Lua 範例

```lua
local ins = require("insight")
ins.init({"cpu"})
local a = ins.zeros({2, 3}, ins.float32)
local b = ins.ones({2, 3}, ins.float32)
local c = a + b

-- 雙呼叫約定
local w = ins.signal.hann(256)
local w2 = ins.signal.hann{n=256}
```

### Julia 範例

```julia
push!(LOAD_PATH, "/path/to/bindings/julia")
using Insight

a = Insight.zeros([2, 3], Insight.float32)
b = Insight.ones([2, 3], Insight.float32)
c = a + b
s = Insight.sum(c, axis=0)
```

## 依賴

| 依賴 | 版本 | 必需 | 說明 |
|------|------|------|------|
| CMake | 3.15+ | 是 | 建置系統 |
| C++17 編譯器 | -- | 是 | GCC 9+, Clang 12+, MSVC 2019+ |
| CUDA | 11.7+ | 否 | 可選 GPU 後端 |
| OpenBLAS | 任意 | 否 | CPU 線性代數 |
| FFTW3 | 3.3+ | 否 | CPU FFT |
| OpenMP | -- | 否 | CPU 多執行緒 |
| Thrust | 捆綁 | 否 | CUDA 排序/去重 |
| GoogleTest | 自動 | -- | 自動取得 |

## 測試狀態

**1140+ 測試全部通過** -- CPU（630+, 27 個套件）與 CUDA（510+, 23 個套件），另有 384 個精度對齊測試

| 套件 | CPU | CUDA | 備註 |
|------|-----|------|------|
| cast | 9 | 9 | |
| complex | 22 | 22 | |
| creation | 27 | 27 | |
| csv | 1 | 1 | |
| dtype | 9 | 9 | 共享 |
| elementwise | 28 | 28 | |
| fft | 19 | 19 | |
| indexing | 41 | 33 | |
| linalg | 43 | 43 | 15 原生 CUDA + 13 C_FALLBACK |
| manipulation | 42 | 42 | |
| operator | 50 | 50 | |
| print | 11 | 11 | |
| random | 31 | 31 | |
| reduction | 24 | 24 | |
| signal（核心） | 10 | 10 | 組合算子 |
| signal_windows | 30 | 30 | |
| signal_waveforms | 18 | 18 | |
| signal_bsplines | 13 | 13 | |
| signal_filter_design | 22 | 22 | |
| signal_convolution | 21 | 17 | |
| signal_filtering | 23 | 15 | |
| signal_spectral | 11 | -- | |
| signal_wavelets | 13 | -- | |
| signal_acoustics | 9 | -- | |
| signal_radar | 7 | -- | |
| signal_io | 11 | -- | |
| signal_peak_finding | 3 | -- | |
| signal_demod | 1 | -- | |
| signal_estimation | 1 | -- | |
| plot | 13 | -- | |
| unary | 27 | 27 | |
| audio | 9 | 9 | |
| **合計** | **630+** | **510+** | |

### 精度對齊（對比 NumPy）

| 套件 | CPU | CUDA |
|------|-----|------|
| cast | 14 | 14 |
| complex | 8 | 8 |
| creation | 14 | 14 |
| elementwise | 24 | 24 |
| fft | 18 | 18 |
| linalg | 22 | 22 |
| manipulation | 18 | 18 |
| reduction | 22 | 22 |
| signal | 20 | 16 |
| unary | 24 | 24 |
| numerical | 10 | 10 |
| **合計** | **194** | **190** |

### 語言繫結測試

| 語言 | 測試框架 | 測試數 |
|------|---------|--------|
| Python | pytest | 76 基礎 + 54 數值 + 384 對齊 |
| Lua | busted | 208 |
| Julia | Test stdlib | 74 |

## 範例程式

`demos/` 目錄下提供 4 種語言、6 個場景的範例程式：

| 範例 | C++ | Python | Lua | Julia |
|------|-----|--------|-----|-------|
| basic_ops | 有 | 有 | 有 | 有 |
| fft_demo | 有 | 有 | 有 | 有 |
| gpu_transfer | 有 | 有 | 有 | 有 |
| linalg_demo | 有 | 有 | 有 | 有 |
| radar_task1 | 有 | 有 | 有 | 有 |
| sndfile_demo | 有 | 有 | 有 | 有 |

## 免責聲明

> [!IMPORTANT]
> 本專案程式碼主要由 **DeepSeek**（深度求索）生成，是 **7 天快速原型開發** 的產物。作者完成了架構整合、測試驗證和系統整合。
>
> 儘管所有 1140+ 個測試在 CPU 與 CUDA 上均穩定通過且完全對齊，這仍然是一個**概念驗證**專案。效能最佳化仍在進行中，在特定硬體組態或邊界條件下可能存在未知 Bug。
>
> 本專案深度借鑑了 **PaddlePaddle** 的設計哲學（算子註冊、裝置 HAL），同時受 Torch7 和 NumPy/CuPy 啟發。

## 授權條款

[Apache License 2.0](LICENSE)

版權所有 2026 PlumBlossomMaid

## 貢獻指南

歡迎提交 Issue 和 Pull Request。請確保：
1. 程式碼遵循 `.clang-format` 風格
2. 所有現有測試通過
3. 新功能包含對應測試
