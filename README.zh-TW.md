
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17/20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-386%20passed-brightgreen.svg)](tests/)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Insight（知微）

**輕量級、工業級張量計算框架，專為信號處理與 GPU 加速設計。**

Insight 是一個 C++ 張量庫，設計理念深受 **PaddlePaddle**（算子註冊機制、裝置硬體抽象層）、**Torch7**（簡潔的 C API、TH/THC 架構精神）和 **NumPy/CuPy**（Python 端 API 習慣）啟發。提供 CPU/GPU 統一計算、零拷貝視圖、動態算子派發和完整的信號處理支援。

## 特性

- **統一 API** – `insight::Array` 無縫在 `CPUPlace` / `GPUPlace` 間切換
- **零拷貝視圖** – 透過 `strides` + `offset` 實現 `reshape`、`transpose`、`slice`
- **動態算子註冊** – `ops()["add"][CPU][F32](args)` 派發（Paddle 風格）
- **裝置硬體抽象層** – 透過 `Device` 基底類別 + `extern "C"` 工廠實現 ABI 穩定的外掛系統
- **信號處理算子** – FFT、窗函數、插值、小波等（對齊 cuSignal）
- **現代 C++** – C++17/20，支援 Python 綁定
- **無自動求導** – 保持函式庫輕量、專注

## 架構

```
insight/
├── include/insight/core/     # Array, Shape, Strides, DType, Place
├── include/insight/ops/      # 前端 API（rand, sum, take, fft...）
├── include/insight/plugin/   # 算子註冊 + device_ext.h（硬體抽象層）
├── src/core/                 # Array 實作、記憶體管理
├── src/ops/                  # 前端算子邏輯
├── backends/cpu/             # CPU 後端（OpenMP + FFTW + OpenBLAS）
├── backends/cuda/            # CUDA 後端（cuBLAS + cuFFT + Thrust）
└── tests/                    # 386 測試，16 套件，CPU/CUDA 完全對齊
```

## 快速上手

### 從原始碼編譯

```bash
git clone https://github.com/PlumBlossomMaid/insight.git
cd insight
mkdir build && cd build
cmake .. -DINSIGHT_WITH_CUDA=ON   # 啟用 CUDA 後端
cmake --build . -j$(nproc)
```

### 範例程式碼

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

## 效能

| 操作 | CPU (OpenMP) | GPU (CUDA) | 加速比 |
|------|--------------|------------|--------|
| matmul (4096x4096) | 12.3 秒 | 0.18 秒 | 68× |
| fft (100萬點) | 0.42 秒 | 0.09 秒 | 4.7× |
| reduce sum (1000萬) | 0.08 秒 | 0.003 秒 | 26× |

*測試平台：Intel i9-13900K + NVIDIA RTX 4090*

## 依賴

| 依賴 | 版本 | 必需 | 說明 |
|------|------|------|------|
| CMake | 3.15+ | ✅ | 建置系統 |
| C++17 編譯器 | – | ✅ | GCC 9+, Clang 12+, MSVC 2019+ |
| CUDA | 11.7+ | ❌ | 可選 GPU 後端 |
| OpenBLAS | 任意 | ❌ | CPU 線性代數 |
| FFTW3 | 3.3+ | ❌ | CPU FFT |
| OpenMP | – | ❌ | CPU 多執行緒 |
| Thrust | 捆綁 | ❌ | CUDA 排序/去重 |

## 測試狀態

✅ **386 個測試全部通過** — CPU 與 CUDA 完全對齊（16 個測試套件）

| 套件 | 測試數 | CPU | GPU |
|------|--------|-----|-----|
| cast | 9 | ✅ | ✅ |
| complex | 22 | ✅ | ✅ |
| creation | 27 | ✅ | ✅ |
| csv | 1 | ✅ | ✅ |
| dtype | 9 | ✅ | ✅ |
| elementwise | 28 | ✅ | ✅ |
| fft | 19 | ✅ | ✅ |
| indexing | 33 | ✅ | ✅ |
| linalg | 43 | ✅ | ✅ |
| manipulation | 42 | ✅ | ✅ |
| operator | 50 | ✅ | ✅ |
| print | 11 | ✅ | ✅ |
| random | 31 | ✅ | ✅ |
| reduction | 24 | ✅ | ✅ |
| signal | 10 | ✅ | ✅ |
| unary | 27 | ✅ | ✅ |

## 免責聲明

> [!IMPORTANT]
> 本專案程式碼主要由 **DeepSeek**（深度求索）生成，是 **7 天快速原型開發** 的產物。作者完成了架構整合、測試驗證和系統整合。
>
> 儘管所有 386 個測試在 CPU 與 CUDA 上均穩定通過且完全對齊，這仍然是一個**概念驗證**專案。效能最佳化仍在進行中，在特定硬體組態或邊界條件下可能存在未知 Bug。
>
> 本專案深度借鑑了 **PaddlePaddle** 的設計哲學（算子註冊、裝置硬體抽象層），同時受 Torch7 和 NumPy/CuPy 啟發。

## 授權條款

[Apache License 2.0](LICENSE)

版權所有 2026 PlumBlossomMaid

## 貢獻指南

歡迎提交 Issue 和 Pull Request。請確保：
1. 程式碼遵循 `.clang-format` 風格
2. 所有現有測試通過
3. 新功能包含對應測試