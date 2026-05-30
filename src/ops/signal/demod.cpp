// src/ops/signal/demod.cpp
#include "insight/ops/signal/demod.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include "insight/ops/signal.h"
#include "insight/ops/signal/filtering.h"

namespace ins {
namespace signal {

Array fm_demod(const Array &x, int axis) {
  INS_CHECK(x.defined(), "fm_demod: input is undefined");
  INS_CHECK(x.dtype() == DType::C32 || x.dtype() == DType::C64,
            "fm_demod: input must be complex-valued");

  // angle(x) — compute phase of complex signal
  // We need to implement this using available ops
  // angle(x) = atan2(imag(x), real(x))
  // For complex arrays, we can compute element-wise

  Place cpu = CPUPlace();
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);

  int ndim = x_cpu.shape().ndim();
  if (axis < 0)
    axis += ndim;

  int64_t n = x_cpu.shape().dim(axis);

  // Compute angle of complex input
  Array x_angle;
  {
    // Extract real and imaginary parts
    int64_t total = x_cpu.numel();
    if (x.dtype() == DType::C64) {
      const std::complex<double> *src =
          reinterpret_cast<const std::complex<double> *>(x_cpu.data<char>());
      std::vector<double> angle_data(total);
      for (int64_t i = 0; i < total; ++i) {
        angle_data[i] = std::arg(src[i]);
      }
      x_angle = to_array(angle_data, x_cpu.shape(), DType::F64, cpu);
    } else {
      const std::complex<float> *src =
          reinterpret_cast<const std::complex<float> *>(x_cpu.data<char>());
      std::vector<float> angle_data(total);
      for (int64_t i = 0; i < total; ++i) {
        angle_data[i] = std::arg(src[i]);
      }
      x_angle = to_array(angle_data, x_cpu.shape(), DType::F32, cpu);
    }
  }

  // unwrap along axis
  Array x_unwrapped = ins::unwrap(x_angle, axis);

  // diff along axis
  // diff reduces length by 1 along axis
  // We need to implement diff manually since it may not be available
  // Actually, let's check if ins::signal has diff... it's in manipulation
  // Let's compute it directly

  Shape out_shape = x_unwrapped.shape();
  int64_t out_n = n - 1;
  std::vector<int64_t> shape_vec;
  for (int d = 0; d < ndim; ++d) {
    if (d == axis)
      shape_vec.push_back(out_n);
    else
      shape_vec.push_back(x_unwrapped.shape().dim(d));
  }
  out_shape = Shape(shape_vec);

  DType out_dtype = (x.dtype() == DType::C64) ? DType::F64 : DType::F32;
  Array result = zeros(out_shape, out_dtype, cpu);

  // Compute diff along axis
  int64_t total_out = result.numel();
  Strides in_strides = x_unwrapped.strides();
  Strides out_strides = result.strides();

  if (out_dtype == DType::F64) {
    const double *in_data = x_unwrapped.data<double>();
    double *out_data = result.data<double>();
    for (int64_t i = 0; i < total_out; ++i) {
      // Compute multi-dimensional index from linear index using out_strides
      int64_t remaining = i;
      int64_t in_off = 0;
      for (int d = 0; d < ndim; ++d) {
        int64_t idx = remaining / out_strides[d];
        remaining %= out_strides[d];
        in_off += idx * in_strides[d];
      }
      out_data[i] = in_data[in_off + in_strides[axis]] - in_data[in_off];
    }
  } else {
    const float *in_data = x_unwrapped.data<float>();
    float *out_data = result.data<float>();
    for (int64_t i = 0; i < total_out; ++i) {
      int64_t remaining = i;
      int64_t in_off = 0;
      for (int d = 0; d < ndim; ++d) {
        int64_t idx = remaining / out_strides[d];
        remaining %= out_strides[d];
        in_off += idx * in_strides[d];
      }
      out_data[i] = in_data[in_off + in_strides[axis]] - in_data[in_off];
    }
  }

  if (x.place().kind() != DeviceKind::CPU) {
    result = result.to(x.place());
  }
  return result;
}

} // namespace signal
} // namespace ins
