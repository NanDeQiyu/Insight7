---
name: add-fp16-bf16-kernel-support
description: Add half-precision (float16/bfloat16) support to CPU and CUDA kernels using software emulation and native types
source: auto-skill
extracted_at: '2026-06-01T00:24:01.968Z'
---

# Add fp16/bf16 Kernel Support

## Problem

DType enum already has `F16` and `BF16`, and kernels are REGISTERED for these
types, but the switch/case blocks have no handling — they fall through to
`default: return C_FAILED`. Need to add actual computation support.

## Architecture

- **CPU**: Software emulation — convert to float, compute, convert back
- **CUDA**: Native types — `__half` and `__nv_bfloat16` with hardware support

## Step 1: Create CPU half-precision utilities

Create `backends/cpu/kernels/common/half_utils.h`:

```cpp
#pragma once
#include <cstdint>
#include <cstring>

namespace insight {

// IEEE 754 float16: 1 sign, 5 exponent, 10 mantissa
inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mantissa == 0) { f = sign << 31; }
        else { // Denormalized
            exp = 1;
            while (!(mantissa & 0x400)) { mantissa <<= 1; exp--; }
            mantissa &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mantissa << 13); // Inf/NaN
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }
    float result; std::memcpy(&result, &f, 4); return result;
}

inline uint16_t f32_to_f16(float val) {
    uint32_t f; std::memcpy(&f, &val, 4);
    uint32_t sign = (f >> 31) & 1;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (f >> 13) & 0x3FF;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign << 15);
        mantissa = (mantissa | 0x400) >> (1 - exp);
        return static_cast<uint16_t>((sign << 15) | mantissa);
    }
    if (exp >= 31) return static_cast<uint16_t>((sign << 15) | 0x7C00);
    return static_cast<uint16_t>((sign << 15) | (exp << 10) | mantissa);
}

// bfloat16: truncated float32 (upper 16 bits)
inline float bf16_to_f32(uint16_t b) {
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float result; std::memcpy(&result, &f, 4); return result;
}

inline uint16_t f32_to_bf16(float val) {
    uint32_t f; std::memcpy(&f, &val, 4);
    uint32_t rounding_bias = 0x7FFF + ((f >> 16) & 1);
    return static_cast<uint16_t>((f + rounding_bias) >> 16);
}

} // namespace insight
```

## Step 2: Create CUDA half-precision utilities

Create `backends/cuda/kernels/common/half_utils.cuh`:

```cuda
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
// CUDA natively supports __half and __nv_bfloat16
// Arithmetic operators (+, -, *, /) work in device code for SM 5.3+
```

## Step 3: Add cases to elementwise kernels

Pattern for each kernel (add, sub, mul, div, pow, mod, maximum, minimum):

```cpp
#include "common/half_utils.h"

// Inside switch(out->dtype), BEFORE default:
case INSIGHT_DTYPE_F16: {
    const uint16_t* a_data = (const uint16_t*)a->data;
    const uint16_t* b_data = (const uint16_t*)b->data;
    uint16_t* out_data = (uint16_t*)out->data;
    for (int64_t i = 0; i < out->numel; ++i) {
        float va = insight::f16_to_f32(a_data[i]);
        float vb = insight::f16_to_f32(b_data[i]);
        out_data[i] = insight::f32_to_f16(va OP vb);
    }
    break;
}
case INSIGHT_DTYPE_BF16: {
    // Same but bf16_to_f32 / f32_to_bf16
    break;
}
```

### Comparison kernels (equal, not_equal, greater, less, etc.)

Output dtype is BOOL, not half. Convert inputs to float, compare, write bool:

```cpp
case INSIGHT_DTYPE_F16: {
    const uint16_t* a_data = (const uint16_t*)a->data;
    const uint16_t* b_data = (const uint16_t*)b->data;
    bool* out_data = (bool*)out->data;
    for (int64_t i = 0; i < out->numel; ++i) {
        float va = insight::f16_to_f32(a_data[i]);
        float vb = insight::f16_to_f32(b_data[i]);
        out_data[i] = (va == vb);  // or !=, <, >, <=, >=
    }
    break;
}
```

### Logical kernels (logical_and, logical_or, logical_xor)

For logical ops, half inputs are truthy if nonzero. Use `is_zero_f16()` helper or convert to float:

```cpp
case INSIGHT_DTYPE_F16: {
    const uint16_t* a_data = (const uint16_t*)a->data;
    const uint16_t* b_data = (const uint16_t*)b->data;
    bool* out_data = (bool*)out->data;
    for (int64_t i = 0; i < out->numel; ++i) {
        bool va = (a_data[i] != 0) && (a_data[i] != 0x8000);  // not +0 or -0
        bool vb = (b_data[i] != 0) && (b_data[i] != 0x8000);
        out_data[i] = va && vb;  // or ||, ^
    }
    break;
}
```

**Files to modify** (elementwise): add.cpp, sub.cpp, mul.cpp, div.cpp, pow.cpp, mod.cpp, maximum.cpp, minimum.cpp, equal.cpp, not_equal.cpp, greater.cpp, less.cpp, greater_equal.cpp, less_equal.cpp, logical_and.cpp, logical_or.cpp, logical_xor.cpp, logical_not.cpp

**Files to modify** (unary): abs.cpp, sqrt.cpp, exp.cpp, log.cpp, sin.cpp, cos.cpp, etc.

For unary, the pattern is:
```cpp
case INSIGHT_DTYPE_F16: {
    const uint16_t* in_data = (const uint16_t*)x->data;
    uint16_t* out_data = (uint16_t*)out->data;
    for (int64_t i = 0; i < out->numel; ++i) {
        float v = insight::f16_to_f32(in_data[i]);
        out_data[i] = insight::f32_to_f16(std::FUNC(v));
    }
    break;
}
```

## Step 3b: Reduction kernels (sum, mean, max, min, prod, etc.)

Reduction kernels accumulate in float, then convert result back to half:

```cpp
case INSIGHT_DTYPE_F16: {
    const uint16_t* in_data = (const uint16_t*)x->data;
    float acc = (reduction == SUM) ? 0.0f : insight::f16_to_f32(in_data[0]);
    for (int64_t i = (reduction == SUM ? 0 : 1); i < x->numel; ++i) {
        float v = insight::f16_to_f32(in_data[i]);
        switch (reduction) {
            case SUM:  acc += v; break;
            case PROD: acc *= v; break;
            case MAX:  acc = std::max(acc, v); break;
            case MIN:  acc = std::min(acc, v); break;
        }
    }
    uint16_t* out_data = (uint16_t*)out->data;
    out_data[0] = insight::f32_to_f16(acc);
    break;
}
```

For argmax/argmin, compare in float but return int64 index:

```cpp
case INSIGHT_DTYPE_F16: {
    const uint16_t* in_data = (const uint16_t*)x->data;
    float best = insight::f16_to_f32(in_data[0]);
    int64_t best_idx = 0;
    for (int64_t i = 1; i < x->numel; ++i) {
        float v = insight::f16_to_f32(in_data[i]);
        if (v > best) { best = v; best_idx = i; }  // or < for argmin
    }
    int64_t* out_data = (int64_t*)out->data;
    out_data[0] = best_idx;
    break;
}
```

**Files to modify** (reduction): sum.cpp, mean.cpp, prod.cpp, max.cpp, min.cpp, argmax.cpp, argmin.cpp, var.cpp, std.cpp, cumsum.cpp, cumprod.cpp, cummax.cpp, cummin.cpp

## Step 4: Create cast kernels

Create `backends/cpu/kernels/cast/cast_f16.cpp` and `cast_bf16.cpp`:

```cpp
#include "../../registry/cpu_registry.h"
#include "../common/half_utils.h"
#include "insight/c_api/array.h"

extern "C" {
C_Status cast_f16_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* in = (InsightArray*)inputs[0];
    InsightArray* out = (InsightArray*)outputs[0];
    const uint16_t* src = (const uint16_t*)in->data;
    switch (out->dtype) {
        case INSIGHT_DTYPE_F32: {
            float* dst = (float*)out->data;
            for (int64_t i = 0; i < in->numel; ++i)
                dst[i] = insight::f16_to_f32(src[i]);
            break;
        }
        case INSIGHT_DTYPE_F64: {
            double* dst = (double*)out->data;
            for (int64_t i = 0; i < in->numel; ++i)
                dst[i] = (double)insight::f16_to_f32(src[i]);
            break;
        }
        // ... other target types
        default: return C_FAILED;
    }
    return C_SUCCESS;
}
}
REGISTER_CPU_KERNEL(cast_f16, INSIGHT_DTYPE_F32, cast_f16_kernel_cpu);
```

Also update existing cast kernels (cast_f32.cpp, cast_f64.cpp, etc.) to ADD
target cases for INSIGHT_DTYPE_F16 and INSIGHT_DTYPE_BF16.

## Step 5: CUDA kernel support

For CUDA elementwise kernels, use native types:

```cuda
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// In kernel launch switch:
case INSIGHT_DTYPE_F16:
    add_kernel<__half><<<blocks, threads>>>(
        (__half*)a->data, (__half*)b->data, (__half*)out->data, meta);
    break;
case INSIGHT_DTYPE_BF16:
    add_kernel<__nv_bfloat16><<<blocks, threads>>>(
        (__nv_bfloat16*)a->data, (__nv_bfloat16*)b->data,
        (__nv_bfloat16*)out->data, meta);
    break;
```

**Note**: `__half` arithmetic works natively in device code for SM 5.3+.
For older GPUs, NVIDIA drivers provide software emulation automatically.

## Key Pitfalls

1. **Registration ≠ Implementation**: Kernels may be registered for F16/BF16
   but have no switch case. Always check the switch block, not just the
   REGISTER line.

2. **Switch-case brace mismatch (MOST COMMON BUG)**: When adding F16/BF16
   cases to an existing `switch(out->dtype)` block, verify the switch closing
   brace `}` is AFTER your new cases, not before. The typical mistake:

   ```cpp
   // BEFORE (original code):
   case INSIGHT_DTYPE_I64:
       // ... code ...
       break;
   }                    // ← switch closing brace
   default:             // ← default is INSIDE switch
       return C_FAILED;
   }

   // WRONG (agent adds F16 after switch closing brace):
   case INSIGHT_DTYPE_I64:
       break;
   }                    // ← switch closes here
   case INSIGHT_DTYPE_F16: {   // ← ERROR: "not within a switch statement"
       // ...
   }
   default:
       return C_FAILED;
   }

   // CORRECT:
   case INSIGHT_DTYPE_I64:
       break;
   case INSIGHT_DTYPE_F16: {
       // ... use local variables inside braces ...
       break;
   }
   default:
       return C_FAILED;
   }                    // ← switch closes AFTER all cases
   ```

   **Symptoms**: `error: case label 'INSIGHT_DTYPE_F16' not within a switch
   statement` or `error: jump to case label crosses initialization`.

   **Fix**: Remove the premature `}` after the last original case, add F16/BF16
   cases, then close the switch after `default:`.

3. **Variable scope in case blocks**: F16/BF16 cases must use `{ }` braces
   around variable declarations. Without them, `default:` jumps over
   initializations → `error: jump to case label crosses initialization of
   'int64_t total'`. Use unique variable names (e.g., `total_f16`) or put
   declarations inside the `{ }` block.

4. **Include path**: From `backends/cpu/kernels/<module>/`, the include is
   `#include "../common/half_utils.h"` (one level up). NOT
   `#include "common/half_utils.h"`.

5. **cummin/cummax use uint16_t comparison**: Bit-pattern comparison is wrong
   for negative numbers, denormals, NaN. Must convert to float for comparison.

6. **isnan/isinf/isfinite return trivial results**: Currently return false/true
   directly for half types. Should check actual bit patterns (exp=0x1F).

7. **Type promotion**: F16+BF16 promotes to BF16 (higher enum value), which
   is arguably wrong — should promote to F32.

8. **No cast to/from half**: Without cast kernels, you can't get half-precision
   data in or out. Cast kernels are prerequisite for any real usage.

9. **Agent conflicts**: When dispatching multiple agents to modify kernel files
   in the same directory, they may fight over the same file (one writes, another
   reverts). Either assign each agent to a unique set of files, or stop agents
   before making manual fixes.

## Verification

```bash
# Build
cmake --build build -j$(nproc)

# Run existing tests (should not regress)
cd build/tests && ./insight_tests_cpu --gtest_filter="CastTestCPU.*:ElementwiseTestCPU.*"
```
