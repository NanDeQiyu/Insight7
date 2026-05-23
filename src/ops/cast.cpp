// src/ops/cast.cpp
#include "insight/ops/cast.h"
#include "insight/core/op_registry.h"
#include "insight/core/exception.h"
#include <iostream>
namespace ins {

    Array cast(const Array& input, DType target_dtype, bool copy) {
        // Fast path: same dtype
        if (input.dtype() == target_dtype) {
            if (copy) {
                return input.copy();
            }
            else {
                return input;
            }
        }

        // Output stays on the same device as input
        Array output(input.shape(), target_dtype, input.place());

        // Convert DType enum to int32_t for C API
        int32_t target_dtype_int = static_cast<int32_t>(target_dtype);

        ops().launch("cast", input.place(), input.dtype(),
            { (void*)input.layout_ptr(), &target_dtype_int },
            { output.layout_ptr() });

        return output;
    }

    Array cast_like(const Array& input, const Array& other, bool copy) {
        return cast(input, other.dtype(), copy);
    }

} // namespace ins