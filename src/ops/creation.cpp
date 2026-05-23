// src/ops/creation.cpp
#include "insight/ops/creation.h"
#include "insight/core/op_registry.h"

namespace ins {

    // ========== Basic Creation ==========

    Array zeros(const Shape& shape, DType dtype, const Place& place) {
        return full(shape, 0.0, dtype, place);
    }

    Array ones(const Shape& shape, DType dtype, const Place& place) {
        return full(shape, 1.0, dtype, place);
    }

    Array full(const Shape& shape, double fill_value, DType dtype, const Place& place) {
        Array result(shape, dtype, place);
        ops().launch("full", place, dtype,
            { result.layout_ptr(), &fill_value },
            { result.layout_ptr() });
        return result;
    }

    Array eye(int64_t n, int64_t m, int64_t k, DType dtype, const Place& place) {
        if (m < 0) m = n;
        Shape shape({ n, m });
        Array result(shape, dtype, place);
        ops().launch("eye", place, dtype,
            { result.layout_ptr(), &k },
            { result.layout_ptr() });
        return result;
    }

    // ========== Range Creation ==========

    Array arange(double end, DType dtype, const Place& place) {
        return arange(0.0, end, 1.0, dtype, place);
    }

    Array arange(double start, double end, double step, DType dtype, const Place& place) {
        int64_t num = static_cast<int64_t>(std::ceil((end - start) / step));
        Array result(Shape({ num }), dtype, place);
        ops().launch("arange", place, dtype,
            { result.layout_ptr(), &start, &step },
            { result.layout_ptr() });
        return result;
    }

    Array linspace(double start, double stop, int64_t num, DType dtype, const Place& place) {
        Array result(Shape({ num }), dtype, place);
        ops().launch("linspace", place, dtype,
            { result.layout_ptr(), &start, &stop },
            { result.layout_ptr() });
        return result;
    }

    Array logspace(double start, double stop, int64_t num, double base, DType dtype, const Place& place) {
        Array result(Shape({ num }), dtype, place);
        ops().launch("logspace", place, dtype,
            { result.layout_ptr(), &start, &stop, &base },
            { result.layout_ptr() });
        return result;
    }

    // ========== Like Creation ==========

    Array zeros_like(const Array& arr) {
        return zeros(arr.shape(), arr.dtype(), arr.place());
    }

    Array ones_like(const Array& arr) {
        return ones(arr.shape(), arr.dtype(), arr.place());
    }

} // namespace ins