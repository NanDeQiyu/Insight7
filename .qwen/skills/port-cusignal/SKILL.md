---
name: port-cusignal
description: 将 cusignal 函数移植到 ins::signal 命名空间（composite 实现 + CUDA kernel 适配）
---

# port-cusignal — cuSignal → ins::signal 移植规范

## 目标

将 cusignal (Python/CUDA) 的所有信号处理函数移植为 C++ `ins::signal::` 命名空间下的函数，
通过 Insight7 HAL 实现 CPU/CUDA 自动切换。

## 命名空间映射

```
cusignal.convolve       → ins::signal::convolve
cusignal.hilbert        → ins::signal::hilbert
cusignal.welch          → ins::signal::welch
cusignal.windows.hann   → ins::signal::hann
...                     → ins::signal::...
```

## 文件结构

```
include/insight/ops/
├── signal.h                    # 总入口，#include 所有子头文件
└── signal/
    ├── convolution.h           # convolve, fftconvolve, correlate, ...
    ├── filtering.h             # wiener, lfilter, filtfilt, hilbert, ...
    ├── resample.h              # decimate, resample, resample_poly, upfirdn
    ├── spectral_analysis.h     # welch, periodogram, stft, istft, ...
    ├── wavelets.h              # morlet, ricker, cwt, ...
    ├── waveforms.h             # sawtooth, square, chirp, ...
    ├── windows.h               # 23 窗函数 + get_window
    ├── filter_design.h         # firwin, firwin2, kaiser_beta, ...
    ├── estimation.h            # KalmanFilter
    ├── radartools.h            # pulse_compression, ca_cfar, mvdr, ...
    ├── peak_finding.h          # argrelmin, argrelmax, ...
    ├── bsplines.h              # gauss_spline, cubic, quadratic
    ├── acoustics.h             # real_cepstrum, complex_cepstrum, ...
    ├── demod.h                 # fm_demod
    └── io.h                    # read_bin, write_bin, ...

src/ops/signal/
    ├── convolution.cpp
    ├── filtering.cpp
    ├── resample.cpp
    ├── spectral_analysis.cpp
    ├── wavelets.cpp
    ├── waveforms.cpp
    ├── windows.cpp
    ├── filter_design.cpp
    ├── estimation.cpp
    ├── radartools.cpp
    ├── peak_finding.cpp
    ├── bsplines.cpp
    ├── acoustics.cpp
    ├── demod.cpp
    └── io.cpp

tests/cpu/test_signal_convolution.cpp      # 每个子模块一个测试文件
tests/cpu/test_signal_filtering.cpp
tests/cpu/test_signal_spectral.cpp
...
tests/cuda/test_signal_convolution.cpp     # CUDA 测试镜像 CPU 测试
...
```

## 开发循环（每个函数，写一个测一个）

**核心原则：写一部分测一部分，不要批量写完再测。**

```
1. 读 cusignal Python 源码 → 理解算法、参数、边界条件
2. 写 C++ 头文件声明（include/insight/ops/signal/<module>.h）
3. 写 C++ 前端实现（src/ops/signal/<module>.cpp）
4. 写 CPU 测试 → 对比 scipy 输出 → 立即验证 ✅
5. 如果需要 CUDA kernel：
   a. 适配 cusignal .cu 参考实现 → 写 CUDA kernel
   b. 同时写对应的 CPU kernel（确保纯 CPU 机器兼容）
   c. 写 CUDA 测试 → 与 CPU 结果一致 → 立即验证 ✅
6. 更新 signal.h 总入口 #include
7. 构建验证 → 下一个函数
```

**绝不允许**：写完一堆函数再一起测试。每完成一个函数必须立即验证通过。

## 实现策略选择

### 策略 A: Composite（推荐，大多数函数）

用已有 Insight7 API 组合实现，不需要新 kernel。

**适用**: windows, waveforms, bsplines, filter_design, convolution, filtering (大部分),
spectral_analysis, resample, wavelets, acoustics, peak_finding, demod, radartools

**示例** — `hilbert`:
```cpp
namespace ins::signal {
Array hilbert(const Array &x, int N = -1, int axis = -1) {
    INS_CHECK(x.defined(), "hilbert: input is undefined");
    if (N < 0) N = x.shape().dim(axis);
    Array X = fft::fft(x, N, axis);
    Array h = zeros({N}, x.dtype(), x.place());
    // 构建 h[0]=1, h[1..N/2-1]=2, h[N/2]=1, 其余=0
    // ...
    X = X * h;  // 频域乘法
    return fft::ifft(X, N, axis);
}
}
```

**优势**: 零额外代码，自动继承 HAL CPU/CUDA 支持，FFT 用 FFTW/cuFFT。

### 策略 B: 独立 Kernel（性能关键路径）

当 composite 性能不够时，写独立 CPU/CUDA kernel。

**适用**: sosfilt（递推数据依赖）、upfirdn（stride 操作）、channelizer（cooperative groups）、
lombscargle（O(N*F) 非均匀采样）

**步骤**: 按 add-cpu-op.md 和 add-cuda-op.md 规范操作。

## Python 动态类型 → C++ 重载

### String enum → C++ enum class

```python
# Python: method='linear' | 'quadratic' | 'logarithmic' | 'hyperbolic'
chirp(t, f0, t1, f1, method='linear')
```

```cpp
enum class ChirpMethod { Linear, Quadratic, Logarithmic, Hyperbolic };
Array chirp(const Array &t, double f0, double t1, double f1,
            ChirpMethod method = ChirpMethod::Linear, ...);
```

### Union type → 函数重载

```python
# Python: window can be str or tuple
get_window('hann', 256)
get_window(('kaiser', 5.0), 256)
```

```cpp
Array get_window(const std::string &window, int64_t Nx, bool fftbins = true);
Array get_window(const std::string &window, double param, int64_t Nx, bool fftbins = true);
```

### Optional output → 返回值

```python
# Python: gausspulse can return (yi, yq, ye) or just yi
gausspulse(t, ..., retquad=False, retenv=False)
```

```cpp
// C++: 用 struct 返回多个值
struct GaussPulseResult { Array yi; Array yq; Array ye; };
GaussPulseResult gausspulse(const Array &t, double fc = 1000, ...);
// 单独获取包络的便捷重载
Array gausspulse(const Array &t, double fc = 1000, ...);
```

### bool/int defaults → 默认参数

```python
# Python
welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None, ...)
```

```cpp
// C++: nperseg=0 表示 None (自动计算)
Array welch(const Array &x, double fs = 1.0, const std::string &window = "hann",
            int64_t nperseg = 0, int64_t noverlap = 0, int64_t nfft = 0, ...);
```

## cusignal CUDA Kernel 适配（必须同时写 CPU kernel）

cusignal 有 8 个 .cu 文件（fatbin 格式），需要适配到 Insight7 的 kernel 注册系统。

**关键规则：写 CUDA kernel 时必须同时写对应的 CPU kernel，确保纯 CPU 机器也能运行。**
参考 cusignal 原始 .cu 实现理解算法，但 CPU 和 CUDA kernel 都要自己写。

### 适配步骤

1. 读取 cusignal 的 `.cu` 文件，理解 kernel 签名和算法
2. 将 kernel 复制到 `backends/cpu/kernels/signal/` 或 `backends/cuda/kernels/signal/`
3. 替换 CuPy 特定代码：
   - `cupy.h` → `insight/c_api/array.h`
   - CuPy metadata → Insight7 `InsightArray` 直接访问
   - `ElementwiseKernel` → 标准 CUDA kernel + 手动 launch
4. 添加 `REGISTER_GPU_KERNEL` 注册
5. 写 wrapper 函数 `C_Status xxx_kernel_gpu(void** inputs, void** outputs)`

### 已有 .cu 文件清单

| 文件 | 行数 | 函数 |
|------|------|------|
| `convolution/_convolution.cu` | 806 | convolve, correlate, convolve2d, correlate2d, convolve1d2o, convolve1d3o |
| `filtering/_channelizer.cu` | 936 | channelize_poly |
| `filtering/_sosfilt.cu` | 153 | sosfilt |
| `filtering/_upfirdn.cu` | 267 | upfirdn1d, upfirdn2d |
| `io/_reader.cu` | 262 | byte-swap, unpack |
| `io/_writer.cu` | 91 | pack |
| `peak_finding/_peak_finding.cu` | 261 | boolrelextrema |
| `spectral_analysis/_spectral.cu` | 159 | lombscargle |

### 另外还有 inline CUDA code（在 Python 字符串中）

- `estimation/_filters_cuda.py` — KalmanFilter predict/update kernels
- `radartools/radartools.py` — ca_cfar_1d, ca_cfar_2d kernels
- 大量 `cp.ElementwiseKernel` — windows, waveforms, wavelets, bsplines, acoustics 等

## 测试规范

### 数据来源

cusignal 的测试在 `~/plum/cusignal/python/cusignal/test/`。每个测试都是 CPU vs GPU 比较。
我们改为 **scipy 输出 vs ins::signal 输出** 比较。

### 测试模式

```cpp
// tests/cpu/test_signal_convolution.cpp
class SignalConvolutionTestCPU : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init({"cpu"});
        set_device(CPUPlace());
    }
};

TEST_F(SignalConvolutionTestCPU, FftConvolve) {
    if (!ins::is_compiled_with_fftw3()) {
        GTEST_SKIP() << "FFTW3 not available";
    }
    // 用 scipy.signal.fftconvolve 的预期结果验证
    std::vector<double> a = {1, 2, 3, 4, 5};
    std::vector<double> b = {1, 1, 1};
    Array a_arr = to_array(a);
    Array b_arr = to_array(b);
    Array result = ins::signal::fftconvolve(a_arr, b_arr);
    // scipy: [1, 3, 6, 9, 12, 9, 5]
    expect_float_equal<double>(result, {1, 3, 6, 9, 12, 9, 5});
}
```

### 精度容差

| 计算类型 | 容差 |
|----------|------|
| 直接数学 (windows, waveforms) | 1e-6 |
| FFT-based (convolve, welch, stft) | 1e-4 (FFTW vs cuFFT 差异) |
| 递推 (sosfilt, lfilter) | 1e-4 (累积误差) |
| 大动态范围 (cepstrum, lombscargle) | 相对误差 1e-3 |

## 移植进度（2026-06-01）— ALL PHASES COMPLETE

```
Phase 1a (无依赖): ✅ 完成
  1. windows         — 24 函数, 30+30 tests, 12 backend kernels (signal_ prefix)
  2. waveforms       — 5 函数, 18+18 tests, 5 backend kernels (signal_square, etc.)
  3. bsplines        — 3 函数, 13+13 tests, 3 backend kernels

Phase 1b (核心 DSP, 依赖 Phase 1a): ✅ 完成
  4. filter_design   — 5 FIR设计, 22+22 tests, 1 backend kernel (signal_firwin)
  5. convolution     — 9 卷积/相关, 21+17 tests, 3 backend kernels (convolve1d, correlate1d, convolve2d)
  6. filtering       — 17 滤波, 23+15 tests, 8 backend kernels (lfilter, signal_wiener, signal_hilbert, etc.)

Phase 1c (高级分析, 依赖 Phase 1b): ✅ 完成
  7. spectral_analysis — 9 频谱分析, 11 tests, 3 backend kernels (spectrogram_accum, signal_lombscargle, etc.)
  8. resample         — 4 重采样, (composite — uses filter_design + FFT)
  9. wavelets         — 5 小波, 13 tests, 3 backend kernels (signal_morlet, signal_ricker, signal_morlet2)
  10. acoustics       — 4 倒谱, 9 tests, 5 backend kernels (signal_mel2hz, signal_hz2mel, etc.)

Phase 2 (专用算法): ✅ 完成
  11. peak_finding    — 3 峰值检测, 3 tests, 2 backend kernels (boolrelextrema_1d, boolrelextrema_2d)
  12. demod           — 1 FM 解调, 1 test (pure composite — no kernel needed)
  13. estimation      — KalmanFilter, 1 test, 1 backend kernel (simple_inv)
  14. radartools      — 6 雷达工具, 7 tests, 2 backend kernels (ca_cfar, ambgfun)

Phase 3 (I/O): ✅ 完成
  15. io              — 6 二进制/SigMF 读写, 11 tests, 2 backend kernels (signal_pack_bin, signal_unpack_bin)

Total: ~89 functions across 14+1 submodules, 50 backend kernels with signal_ prefix convention.
All 14 signal submodules now have CPU+CUDA backend kernels. dtype support: CPU (F64, F32), CUDA (F64, F32, F16, BF16).
```

## cusignal 源码参考位置

```
~/plum/cusignal/python/cusignal/
├── convolution/         # 卷积相关
├── filtering/           # 滤波相关
├── spectral_analysis/   # 频谱分析
├── wavelets/            # 小波
├── waveforms/           # 波形生成
├── windows/             # 窗函数
├── filter_design/       # 滤波器设计
├── estimation/          # 估计 (KalmanFilter)
├── radartools/          # 雷达工具
├── peak_finding/        # 峰值检测
├── bsplines/            # B样条
├── acoustics/           # 声学 (倒谱)
├── demod/               # 解调
├── io/                  # I/O
└── test/                # 测试 (可参考预期值)
```

## 常见陷阱

1. **numpy vs insight 数组创建**: `np.array([1,2,3])` → `to_array({1.0, 2.0, 3.0})`
2. **axis 默认值**: numpy 默认 axis=-1，有些 cusignal 函数默认 axis=0，注意对齐
3. **None 参数**: Python 的 `None` → C++ 的 `0` 或 `-1`（根据语义选择）
4. **dtype 参数**: Python 的 `cp.float32` → C++ 的 `DType::F32`
5. **FFT 归一化**: scipy 默认 "backward"（正变换不归一化），确认 cusignal 是否一致
6. **复数处理**: cusignal 大量使用 `cp.complex64/128`，对应 `DType::C32/C64`
7. **边界模式**: `mode='full'/'same'/'valid'` 和 `boundary='fill'/'wrap'/'symm'` 用 string 参数
8. **连续性检查**: composite 实现中间步骤可能产生非连续数组，注意 stride 处理
9. **GPU composite 必须**: 所有 signal 函数必须用 composite 操作（arange/sin/cos/where/abs/add/mul），不能用 lambda + 直接写 `data<T>()` 指针，否则 GPU 上 segfault
10. **复杂算法用 CPU 计算 + to_array()**: chebwin/taylor/kaiser/triang 等镜像/递推逻辑难以用 composite 正确表达，用 `std::vector<double>` 在 CPU 计算后 `to_array()` 转移
11. **acosh 负数输入**: `std::acosh(负数)` = NaN，Chebyshev 多项式需要对负 x 取 `acosh(|x|)` 再处理符号
12. **unit_impulse 必须先在 CPU 创建**: `zeros(shape, dtype, CPUPlace())` → 设置 impulse → `.to(place)`
