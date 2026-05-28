// src/ops/elementwise.cpp
#include "insight/ops/elementwise.h"
#include "insight/core/op_registry.h"
#include "insight/ops/broadcast.h"
#include "insight/utils/promotion.h"

namespace ins {

static DeviceKind get_device_kind(const Place &place) {
  return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
}

// ============================================================================
// Generic binary operation scheduler (does not force serialization)
// ============================================================================
template <typename KernelName>
static Array binary_op(const Array &a, const Array &b,
                       KernelName &&kernel_name) {
  // 1. Type promotion
  DType out_dtype = promote_types(a.dtype(), b.dtype());

  // 2. Convert to unified type
  Array a1 = (a.dtype() == out_dtype) ? a : a.to(out_dtype);
  Array b1 = (b.dtype() == out_dtype) ? b : b.to(out_dtype);

  // 3. Unify equipment
  Place target_place = promote_places(a1.place(), b1.place());
  if (a1.place() != target_place)
    a1 = a1.to(target_place);
  if (b1.place() != target_place)
    b1 = b1.to(target_place);

  // 4. Broadcasting (shape alignment)
  if (a1.shape() != b1.shape()) {
    auto bc = broadcast_arrays({a1, b1});
    a1 = bc[0];
    b1 = bc[1];
  }

  // 5. Allocate output array
  Array out(a1.shape(), out_dtype, target_place);

  // 6. Call the backend kernel (do not force continuousization, let the backend
  // handle strides)
  ops().launch(kernel_name, target_place, out_dtype,
               {a1.layout_ptr(), b1.layout_ptr()}, {out.layout_ptr()});

  return out;
}

// ============================================================================
// Comparison operation scheduler (returns bool)
// ============================================================================
template <typename KernelName>
static Array cmp_op(const Array &a, const Array &b, KernelName &&kernel_name) {
  // The comparison operation needs to unify the type before comparison, but the
  // output is bool
  DType common_dtype = promote_types(a.dtype(), b.dtype());

  // Convert to unified type
  Array a1 = (a.dtype() == common_dtype) ? a : a.to(common_dtype);
  Array b1 = (b.dtype() == common_dtype) ? b : b.to(common_dtype);

  // unified device
  Place target_place = promote_places(a1.place(), b1.place());
  if (a1.place() != target_place)
    a1 = a1.to(target_place);
  if (b1.place() != target_place)
    b1 = b1.to(target_place);

  // broadcast
  if (a1.shape() != b1.shape()) {
    auto bc = broadcast_arrays({a1, b1});
    a1 = bc[0];
    b1 = bc[1];
  }

  // The output is bool
  Array out(a1.shape(), DType::BOOL, target_place);

  // Distribute the kernel with common_dtype (the backend selects the
  // implementation based on the input type)
  ops().launch(kernel_name, target_place, common_dtype,
               {a1.layout_ptr(), b1.layout_ptr()}, {out.layout_ptr()});

  return out;
}

// ============================================================================
// Arithmetic operations
// ============================================================================
Array add(const Array &a, const Array &b) { return binary_op(a, b, "add"); }

Array sub(const Array &a, const Array &b) { return binary_op(a, b, "sub"); }

Array mul(const Array &a, const Array &b) { return binary_op(a, b, "mul"); }

Array div(const Array &a, const Array &b) { return binary_op(a, b, "div"); }

// ============================================================================
// Power operation (special processing: converting integer exponent to floating
// point to avoid precision problems)
// ============================================================================
Array pow(const Array &a, const Array &b) {
  DType out_dtype = promote_types(a.dtype(), b.dtype());

  // If the exponent is a floating point, the result must be a floating point
  if (is_floating_point(b.dtype()) || is_complex(b.dtype())) {
    if (is_integer(out_dtype)) {
      out_dtype = DType::F64;
    }
  }

  // conversion type
  Array a1 = (a.dtype() == out_dtype) ? a : a.to(out_dtype);
  Array b1 = (b.dtype() == out_dtype) ? b : b.to(out_dtype);

  // unified device
  Place target_place = promote_places(a1.place(), b1.place());
  if (a1.place() != target_place)
    a1 = a1.to(target_place);
  if (b1.place() != target_place)
    b1 = b1.to(target_place);

  // broadcast
  if (a1.shape() != b1.shape()) {
    auto bc = broadcast_arrays({a1, b1});
    a1 = bc[0];
    b1 = bc[1];
  }

  // Assign output
  Array out(a1.shape(), out_dtype, target_place);

  // Call kernel
  ops().launch("pow", target_place, out_dtype,
               {a1.layout_ptr(), b1.layout_ptr()}, {out.layout_ptr()});

  return out;
}

// ============================================================================
// Modulo operation
// ============================================================================
Array mod(const Array &a, const Array &b) { return binary_op(a, b, "mod"); }

// ============================================================================
// comparison operation
// ============================================================================
Array equal(const Array &a, const Array &b) { return cmp_op(a, b, "equal"); }

Array not_equal(const Array &a, const Array &b) {
  return cmp_op(a, b, "not_equal");
}

Array greater(const Array &a, const Array &b) {
  return cmp_op(a, b, "greater");
}

Array less(const Array &a, const Array &b) { return cmp_op(a, b, "less"); }

Array greater_equal(const Array &a, const Array &b) {
  return cmp_op(a, b, "greater_equal");
}

Array less_equal(const Array &a, const Array &b) {
  return cmp_op(a, b, "less_equal");
}

// Alias
Array greater_than(const Array &a, const Array &b) { return greater(a, b); }

Array less_than(const Array &a, const Array &b) { return less(a, b); }

// ============================================================================
// Logical operation (convert to bool first)
// ============================================================================
static Array logical_op(const Array &a, const Array &b,
                        const char *kernel_name) {
  // convert to bool
  Array a1 = (a.dtype() == DType::BOOL) ? a : a.to(DType::BOOL);
  Array b1 = (b.dtype() == DType::BOOL) ? b : b.to(DType::BOOL);

  // unified device
  Place target_place = promote_places(a1.place(), b1.place());
  if (a1.place() != target_place)
    a1 = a1.to(target_place);
  if (b1.place() != target_place)
    b1 = b1.to(target_place);

  // broadcast
  if (a1.shape() != b1.shape()) {
    auto bc = broadcast_arrays({a1, b1});
    a1 = bc[0];
    b1 = bc[1];
  }

  // The output is bool
  Array out(a1.shape(), DType::BOOL, target_place);

  ops().launch(kernel_name, target_place, DType::BOOL,
               {a1.layout_ptr(), b1.layout_ptr()}, {out.layout_ptr()});

  return out;
}

Array logical_and(const Array &a, const Array &b) {
  return logical_op(a, b, "logical_and");
}

Array logical_or(const Array &a, const Array &b) {
  return logical_op(a, b, "logical_or");
}

Array logical_xor(const Array &a, const Array &b) {
  return logical_op(a, b, "logical_xor");
}

// ============================================================================
// Bit operations (integer types)
// ============================================================================
static Array bitwise_op(const Array &a, const Array &b,
                        const char *kernel_name) {
  // Bit operations are only defined on integers
  DType out_dtype = promote_types(a.dtype(), b.dtype());

  // Convert to unified type
  Array a1 = (a.dtype() == out_dtype) ? a : a.to(out_dtype);
  Array b1 = (b.dtype() == out_dtype) ? b : b.to(out_dtype);

  // unified device
  Place target_place = promote_places(a1.place(), b1.place());
  if (a1.place() != target_place)
    a1 = a1.to(target_place);
  if (b1.place() != target_place)
    b1 = b1.to(target_place);

  // broadcast
  if (a1.shape() != b1.shape()) {
    auto bc = broadcast_arrays({a1, b1});
    a1 = bc[0];
    b1 = bc[1];
  }

  // Assign output
  Array out(a1.shape(), out_dtype, target_place);

  ops().launch(kernel_name, target_place, out_dtype,
               {a1.layout_ptr(), b1.layout_ptr()}, {out.layout_ptr()});

  return out;
}

Array bitwise_and(const Array &a, const Array &b) {
  return bitwise_op(a, b, "bitwise_and");
}

Array bitwise_or(const Array &a, const Array &b) {
  return bitwise_op(a, b, "bitwise_or");
}

Array bitwise_xor(const Array &a, const Array &b) {
  return bitwise_op(a, b, "bitwise_xor");
}

Array bitwise_left_shift(const Array &a, const Array &b) {
  return bitwise_op(a, b, "bitwise_left_shift");
}

Array bitwise_right_shift(const Array &a, const Array &b) {
  return bitwise_op(a, b, "bitwise_right_shift");
}

// ============================================================================
// Max/Min
// ============================================================================
Array maximum(const Array &a, const Array &b) {
  return binary_op(a, b, "maximum");
}

Array minimum(const Array &a, const Array &b) {
  return binary_op(a, b, "minimum");
}

} // namespace ins