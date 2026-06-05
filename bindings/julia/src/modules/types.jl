# modules/types.jl
# Core type documentation for Insight7 Julia bindings.

"""
    InsightArray

Core array type in Insight7. Wraps a C++ Array pointer with automatic
finalization via Julia's garbage collector.

# Properties
- `ndim(arr)` — Number of dimensions
- `numel(arr)` — Total number of elements
- `dtype(arr)` — Data type code (see DType constants)
- `shape(arr)` — Vector of dimension sizes
- `device_type(arr)` — 0 for CPU, 1 for GPU

# Operations
- Arithmetic: `a + b`, `a - b`, `a * b`, `a / b`
- Indexing: `a[1]` (1-based), `a["1:3, :]` (string slicing)
- Transfer: `to_array(arr, place)` — move to different device
- Copy: `copy(arr)` — deep copy
- Reshape: `reshape(arr, dims...)` — view with new shape

# Creation
- `zeros(dims...)`, `ones(dims...)`, `full(dims..., val)`
- `eye(n)`, `arange(start, stop)`, `linspace(start, stop, num)`
"""
InsightArray

"""
    CPU = 0

CPU device placement constant. Use with `to_array(arr, CPU)`.
"""
const CPU_doc = 0

"""
    GPU = 1

GPU device placement constant. Use with `to_array(arr, GPU)`.
"""
const GPU_doc = 1

# DType documentation
"""
    float32, float64, float16, bfloat16

Floating-point data type constants.
"""
const dtype_float_docs = nothing

"""
    int8, int16, int32, int64

Signed integer data type constants.
"""
const dtype_int_docs = nothing

"""
    uint8, uint16, uint32, uint64

Unsigned integer data type constants.
"""
const dtype_uint_docs = nothing

"""
    bool, complex64, complex128

Boolean and complex data type constants.
"""
const dtype_other_docs = nothing
