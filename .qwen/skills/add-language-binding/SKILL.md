---
name: add-language-binding
description: Add Python/Lua/Julia language bindings for Insight7 using pybind11/sol2/ccall
source: auto-skill
extracted_at: '2026-05-29T22:30:00.000Z'
---

# add-language-binding.md — 语言绑定开发流程

Insight7 支持三种语言绑定：Python (pybind11), Lua (sol2 + LuaJIT), Julia (C ABI + ccall)。

## 文件结构

```
bindings/
├── CMakeLists.txt           # 顶层，按选项 include 子目录
├── python/
│   ├── CMakeLists.txt
│   └── insight_py.cpp       # pybind11 绑定
├── lua/
│   ├── CMakeLists.txt
│   ├── insight_lua.cpp      # sol2 绑定
│   ├── insight.lua          # Lua loader（luarocks 用）
│   └── insight-scm-1.rockspec
└── julia/
    ├── CMakeLists.txt
    ├── insight_julia_capi.cpp  # C ABI wrapper
    └── Insight.jl              # Julia 模块
```

## API 风格（PaddlePaddle 风格）

```python
import insight as ins
a = ins.zeros([2, 3], ins.float32)     # dtype 是模块级属性
b = ins.ones([2, 3], ins.float32)
c = ins.matmul(a, b.T)
p = ins.CPUPlace()                      # Place 用工厂函数
```

## CMake 集成

根 `CMakeLists.txt` 加：
```cmake
option(INSIGHT_BUILD_BINDINGS "Build language bindings" ON)
if(INSIGHT_BUILD_BINDINGS)
    add_subdirectory(bindings)
endif()
```

`bindings/CMakeLists.txt`：
```cmake
option(INSIGHT_BUILD_PYTHON_BINDING "Build Python bindings" ON)
option(INSIGHT_BUILD_LUA_BINDING "Build Lua bindings" ON)
option(INSIGHT_BUILD_JULIA_BINDING "Build Julia bindings" ON)
# 各自 add_subdirectory
```

## ⚠️ 关键陷阱

### 1. 函数指针歧义（最重要！）

`using namespace ins;` 后，`sin`, `cos`, `abs`, `div`, `pow`, `rand` 等函数名与 C/`std::` 冲突。
**必须用 lambda 包装，不能直接取函数指针：**

```cpp
// ❌ 错误 — ambiguous
m.def("sin", &sin, py::arg("x"));
m.def("div", &div, py::arg("a"), py::arg("b"));
m.def("rand", &rand, py::arg("shape"));

// ✅ 正确 — lambda 消歧义
m.def("sin", [](const Array &x) { return sin(x); }, py::arg("x"));
m.def("div", [](const Array &a, const Array &b) { return div(a, b); }, py::arg("a"), py::arg("b"));
m.def("rand", [](const Shape &s, DType d, const Place &p) { return rand(s, d, p); }, ...);
```

**规则：所有 `ins::` 命名空间下的函数，绑定时一律用 lambda 包装。**

### 2. FFT 子命名空间

FFT 函数在 `ins::fft::` 子命名空间，不是 `ins::`：
```cpp
// ❌ 错误
m.def("fft", &fft, ...);
// ✅ 正确
m.def("fft", [](const Array &x, int n, int axis, const std::string &norm) {
    return fft::fft(x, n, axis, norm);
}, ...);
```

### 3. Shape API

`Shape` 没有 `operator[]`，用 `shape.dim(i)`：
```cpp
// ❌ 错误
shape[i]
// ✅ 正确
shape.dim(i)
```

### 4. Place 工厂函数

`Place::cpu()` 不存在，用自由函数 `CPUPlace()` / `GPUPlace(id)`：
```cpp
// ❌ 错误
Place::cpu()
// ✅ 正确
CPUPlace()
GPUPlace(0)
```

### 5. 框架初始化

必须在模块加载时调用 `ins::init({"cpu"})`，否则内核注册表为空：
```cpp
// pybind11
PYBIND11_MODULE(insight, m) {
    if (!ins::is_initialized()) ins::init({"cpu"});
    // ...
}

// sol2
extern "C" int luaopen_insight(lua_State *L) {
    if (!ins::is_initialized()) ins::init({"cpu"});
    // ...
}

// Julia C ABI
void insight_jl_init_cpu() {
    if (!ins::is_initialized()) ins::init({"cpu"});
}
```

### 6. sol2 特殊问题

**sol::optional ≠ std::optional**，需要手动转换：
```cpp
m["sum"] = [](const Array &x, sol::optional<int> axis, sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return sum(x, ax, keepdims.value_or(false));
};
```

**属性 vs 方法**：
```cpp
// 属性用 sol::property（Lua 中用 . 访问）
array_type["numel"] = sol::property(&Array::numel);
// 方法直接绑定（Lua 中用 : 调用）
array_type["contiguous"] = &Array::contiguous;
```

### 7. 静态链接陷阱

如果 `insight_core` 是 STATIC 库，多个 binding .so 各自有一份内核注册表副本。
后端通过 `dlopen` 加载时注册的表与 binding 使用的不是同一份。

**症状**：`add` 能用但 `sum` 崩溃（`add` 是直接注册的 kernel，`sum` 是组合算子内部查注册表）。

**解决方案**：把 `insight_core` 改成 SHARED 库（影响所有 binding，需权衡）。

### 8. `round` 不存在

Insight 的四舍五入函数是 `rint()`，不是 `round()`：
```cpp
m.def("round", [](const Array &x) { return rint(x); }, py::arg("x"));
```

### 9. `clip` 不存在

Insight 没有 `clip`/`clamp` 函数，不要在绑定中暴露。

### 10. `split` 重载

`split` 有两个重载（`int sections` 和 `vector<int64_t> indices`），用 lambda 指定：
```cpp
m.def("split", [](const Array &x, int sections, int axis) {
    return split(x, sections, axis);
}, py::arg("x"), py::arg("sections"), py::arg("axis") = 0);
```

## NumPy 互操作（Python）

```cpp
// Insight → NumPy
static py::array to_numpy(const Array &arr) {
    Array cpu = arr.contiguous().to(CPUPlace());
    // 用 shape.dim(i) 获取维度，strides()[i] 获取步长
    return py::array(dtype_to_numpy(cpu.dtype()), shape_vec, strides_vec, cpu.data());
}

// NumPy → Insight
static Array from_numpy(py::array arr) {
    Array result(Shape(shape_vec), dt, CPUPlace());
    std::memcpy(result.data(), info.ptr, arr.nbytes());
    return result;
}
```

## 测试模式

```python
# tests/bindings/test_python_binding.py
import sys, os
sys.path.insert(0, os.path.join(..., "build", "bindings", "python"))
import pytest
import insight as ins

class TestArrayCreation:
    def test_zeros(self):
        a = ins.zeros(ins.Shape([2, 3]), ins.float32)
        assert a.defined()
        assert a.numel() == 6
```

运行需要设置 `LD_LIBRARY_PATH`：
```bash
PYTHONPATH=build/bindings/python LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda:$LD_LIBRARY_PATH \
    python3 -m pytest tests/bindings/ -v
```

## 运行时依赖路径

| 语言 | 环境变量 | 示例 |
|------|----------|------|
| Python | `PYTHONPATH` | `bindings/python:build/bindings/python` |
| Lua | `LUA_CPATH` | `build/bindings/lua/?.so;` |
| Julia | `LOAD_PATH` + `include()` | `include("build/bindings/julia/Insight.jl")` |
| 所有 | `LD_LIBRARY_PATH` | `build/backends/cpu:build/backends/cuda` |

## Python Wrapper 包结构（IDE 友好）

原生模块必须命名为 `_insight`（下划线前缀），wrapper 包命名为 `insight`：

```
bindings/python/
├── CMakeLists.txt
├── insight_py.cpp          # PYBIND11_MODULE(_insight, m) ← 注意下划线
└── insight/
    ├── __init__.py          # from ._insight import *  ← 包装层
    └── _insight.cpython-310-x86_64-linux-gnu.so  # 编译后复制到这里
```

**CMake 输出名必须和 PYBIND11_MODULE 名一致：**
```cmake
set_target_properties(insight_python PROPERTIES
    OUTPUT_NAME "_insight"  # ← 下划线前缀
)
```

**PYBIND11_MODULE 声明：**
```cpp
PYBIND11_MODULE(_insight, m) {  // ← 不是 insight
```

**`__init__.py` 导入：**
```python
from ._insight import *       # ← 不是 .insight
from ._insight import Array, DType, ...
```

**测试路径设置：**
```python
sys.path.insert(0, "bindings/python")          # wrapper 包
sys.path.insert(0, "build/bindings/python")    # 原生 .so（如果没复制）
```

## Lua Place Usertype

sol2 中 `Place` 必须注册为 usertype，否则 `cpu:is_cpu()` 会报错：
```cpp
sol::usertype<Place> place_type = m.new_usertype<Place>(
    "Place", sol::constructors<Place()>());
place_type["is_cpu"] = &Place::is_cpu;
place_type["is_gpu"] = &Place::is_gpu;
place_type["device_id"] = &Place::device_id;
```

Lua 中属性 vs 方法：
- **属性**（`sol::property`）：用 `.` 访问 → `a.numel`, `a.defined`, `a.ndim`
- **方法**：用 `:` 调用 → `a:contiguous()`, `a:reshape({2,3})`

## Julia DType 枚举对齐

**必须严格对齐 `include/insight/c_api/dtype.h`！** C 枚举有间隔（F8 类型、U16/U32/U64 排在后面）：

```julia
module DTypeValues
const UNKNOWN  = Int32(0)
const BOOL     = Int32(1)
const U8       = Int32(2)
const I8       = Int32(3)
const I16      = Int32(4)
const I32      = Int32(5)
const I64      = Int32(6)
const F16      = Int32(7)   # ← 不是 7
const BF16     = Int32(8)   # ← 不是 8
const F32      = Int32(9)   # ← 不是 12！
const F64      = Int32(10)  # ← 不是 13！
const C32      = Int32(11)
const C64      = Int32(12)
const F8_E4M3  = Int32(13)
const F8_E5M2  = Int32(14)
const U16      = Int32(15)  # ← 排在后面！
const U32      = Int32(16)
const U64      = Int32(17)
end
```

**症状**：`sum` 报 "kernel not found for 'sum', device=0, dtype=12"，因为 dtype=12 在 C 端是 C64 不是 F32。

## `ins::std` 命名空间冲突

`reduction.h` 中有 `ins::std()` 函数，与 `std::` 命名空间冲突。绑定时必须用 `ins::std`：
```cpp
m.def("std", [](const Array &x, std::optional<int> axis, bool keepdims, int ddof) {
    return ins::std(x, axis, keepdims, ddof);  // ← ins:: 前缀
}, ...);
```

## `where` 重载

`where` 有两个重载（3-arg 条件选择和 1-arg 返回索引），必须用 lambda 指定：
```cpp
m.def("where", [](const Array &cond, const Array &x, const Array &y) {
    return where(cond, x, y);
}, py::arg("cond"), py::arg("x"), py::arg("y"));
```

## 数值正确性测试（Python vs NumPy）

`tests/bindings/test_numerical_correctness.py` 验证 Insight 操作与 NumPy 数值一致：

```python
def assert_close(actual, expected, rtol=1e-5, atol=1e-6, label=""):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if np.iscomplexobj(actual) or np.iscomplexobj(expected):
        np.testing.assert_allclose(actual.real, expected.real, rtol=rtol, atol=atol)
        np.testing.assert_allclose(actual.imag, expected.imag, rtol=rtol, atol=atol)
    else:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
```

**注意**：FFT 返回复数，`assert_close` 需要分别比较实部和虚部，不能直接 `astype(float64)`。

**reduce axis 形状差异**：Insight 的 `mean(x, axis=0)` 返回 `(1, N)` 而 NumPy 返回 `(N,)`。测试中用 `.squeeze()` 对齐。

## 包管理配置

### pip (pyproject.toml)

```toml
[build-system]
requires = ["setuptools>=61.0", "pybind11>=2.10"]
build-backend = "setuptools.build_meta"

[project]
name = "insight"
version = "1.0.0"
dependencies = ["numpy>=1.20"]

[tool.setuptools.packages.find]
where = ["bindings/python"]
```

### luarocks (rockspec)

```lua
package = "insight"
version = "scm-1"
dependencies = { "lua >= 5.1", "penlight >= 1.5" }
build = {
    type = "cmake",
    variables = {
        INSIGHT_BUILD_PYTHON_BINDING = "OFF",
        INSIGHT_BUILD_JULIA_BINDING = "OFF",
        INSIGHT_BUILD_LUA_BINDING = "ON",
        INSIGHT_WITH_CUDA = "OFF",
    },
    install = {
        lua = { ["insight"] = "bindings/lua/insight/init.lua" },
        lib = { "insight.so" },
    },
}
```

## CI Workflow

`.github/workflows/language_bindings.yml` — 三个独立 job (python/lua/julia)：

```yaml
jobs:
  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { submodules: recursive }
      - name: Install dependencies
        run: sudo apt-get install -y python3-dev python3-numpy python3-pytest ...
      - name: Build
        run: cd build && cmake --build . -j$(nproc)
      - name: Test
        run: python3 -m pytest tests/bindings/ -v
```

每个 job 必须全量构建（`cmake --build . -j$(nproc)`），不能只挑 binding target：
```yaml
# ❌ 错误 — 缺少 vorbis/ogg/flac/opus 等传递依赖
run: cd build && make -j$(nproc) insight_python insight_core
# ✅ 正确 — 全量构建确保 libsndfile 的依赖链完整
run: cd build && cmake --build . -j$(nproc)
```

**原因**：`insight_core` 依赖 libsndfile，libsndfile 依赖 ogg → vorbis/flac/opus（FetchContent）。只构建 binding target 会跳过这些传递依赖。

## Doxygen 文档

`Doxyfile` 配置要点：
```
INPUT = include src bindings/python/insight bindings/lua/insight bindings/julia
RECURSIVE = YES
EXTRACT_ALL = YES
GENERATE_LATEX = NO
GENERATE_XML = YES
OUTPUT_DIRECTORY = docs
```

生成命令：`~/public/doxygen/bin/doxygen Doxyfile`

`docs/` 目录加入 `.gitignore`。

## Penlight 集成（Lua）

Lua wrapper 用 Penlight 增强可读性：
```lua
local has_penlight, utils = pcall(require, "pl.utils")
local has_stringx, stringx = pcall(require, "pl.stringx")
```

安装：`luarocks --local install penlight`

LuaJIT 使用 Penlight 需设置 `package.path`：
```
/home/aistudio/.luarocks/share/lua/5.1/?.lua
```
