// src/ops/unary.cpp
#include "insight/ops/elementwise.h"
#include "insight/core/op_registry.h"
#include "insight/core/exception.h"

namespace ins {

    static DeviceKind get_device_kind(const Place& place) {
        return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
    }

    // ============================================================================
    // Helper macro for unary operations
    // ============================================================================

#define DEFINE_UNARY_OP(op_name) \
    Array op_name(const Array& x) { \
        INS_CHECK(x.defined(), #op_name ": input is undefined"); \
        \
        /* For logical_not, output is always bool */ \
        DType out_dtype = x.dtype(); \
        if (std::string(#op_name) == "logical_not") { \
            out_dtype = DType::BOOL; \
        } \
        \
        Array out(x.shape(), out_dtype, x.place()); \
        \
        ops().launch(#op_name, x.place(), x.dtype(), \
            { (void*)x.layout_ptr() }, \
            { out.layout_ptr() }); \
        \
        return out; \
    }

    // ============================================================================
    // Basic math operations
    // ============================================================================

    DEFINE_UNARY_OP(abs);
    DEFINE_UNARY_OP(negative);
    DEFINE_UNARY_OP(square);
    DEFINE_UNARY_OP(reciprocal);

    // ============================================================================
    // Exponential and logarithmic
    // ============================================================================

    DEFINE_UNARY_OP(exp);
    DEFINE_UNARY_OP(exp2);
    DEFINE_UNARY_OP(expm1);
    DEFINE_UNARY_OP(log);
    DEFINE_UNARY_OP(log2);
    DEFINE_UNARY_OP(log10);
    DEFINE_UNARY_OP(log1p);

    // ============================================================================
    // Power and root
    // ============================================================================

    DEFINE_UNARY_OP(sqrt);
    DEFINE_UNARY_OP(cbrt);

    // ============================================================================
    // Trigonometric
    // ============================================================================

    DEFINE_UNARY_OP(sin);
    DEFINE_UNARY_OP(cos);
    DEFINE_UNARY_OP(tan);
    DEFINE_UNARY_OP(asin);
    DEFINE_UNARY_OP(acos);
    DEFINE_UNARY_OP(atan);

    // ============================================================================
    // Hyperbolic
    // ============================================================================

    DEFINE_UNARY_OP(sinh);
    DEFINE_UNARY_OP(cosh);
    DEFINE_UNARY_OP(tanh);
    DEFINE_UNARY_OP(asinh);
    DEFINE_UNARY_OP(acosh);
    DEFINE_UNARY_OP(atanh);

    // ============================================================================
    // Rounding
    // ============================================================================

    DEFINE_UNARY_OP(floor);
    DEFINE_UNARY_OP(ceil);
    DEFINE_UNARY_OP(trunc);
    DEFINE_UNARY_OP(rint);

    // ============================================================================
    // Sign
    // ============================================================================

    DEFINE_UNARY_OP(sign);

    // ============================================================================
    // Logical and bitwise
    // ============================================================================

    DEFINE_UNARY_OP(logical_not);
    DEFINE_UNARY_OP(bitwise_not);

    // ============================================================================
    // Complex
    // ============================================================================

    Array conj(const Array& x) {
        INS_CHECK(x.defined(), "conj: input is undefined");

        // For real numbers, conjugate is identity
        if (!is_complex(x.dtype())) {
            return x.copy();
        }

        Array out(x.shape(), x.dtype(), x.place());

        ops().launch("conj", x.place(), x.dtype(),
            { (void*)x.layout_ptr() },
            { out.layout_ptr() });

        return out;
    }

    // ============================================================================
    // Degree/radian conversion
    // ============================================================================

    DEFINE_UNARY_OP(deg2rad);
    DEFINE_UNARY_OP(rad2deg);

	// ============================================================================
	// Is finite/inf/nan
	// ============================================================================

    DEFINE_UNARY_OP(isnan);
    DEFINE_UNARY_OP(isinf);
    DEFINE_UNARY_OP(isfinite);

} // namespace ins