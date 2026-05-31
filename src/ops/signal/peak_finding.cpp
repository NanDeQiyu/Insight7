// src/ops/signal/peak_finding.cpp
#include "insight/ops/signal/peak_finding.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/indexing.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/reduction.h"
#include "insight/ops/unary.h"
#include <algorithm>

namespace ins {
namespace signal {

namespace {

Array boolrelextrema(const Array &data, const std::string &comparator, int axis,
                     int order, const std::string &mode) {
  int ndim = data.shape().ndim();
  if (axis < 0)
    axis += ndim;

  int64_t datalen = data.shape().dim(axis);

  // Work on CPU
  Place cpu = CPUPlace();
  Array data_cpu =
      (data.place().kind() == DeviceKind::CPU) ? data : data.to(cpu);

  // Initialize results to true
  Array results = ones(data.shape(), DType::BOOL, cpu);

  // Get main data along axis
  for (int64_t shift = 1; shift <= order; ++shift) {
    // Compare data[..., i, ...] with data[..., i+shift, ...] and
    // data[..., i-shift, ...]
    // For "clip" mode, out-of-bounds indices are clamped

    int64_t n = datalen;
    for (int64_t i = 0; i < n; ++i) {
      int64_t p_idx = std::min(i + shift, n - 1);
      int64_t m_idx = std::max(i - shift, (int64_t)0);

      // For each element along other axes, compare
      // This is done element-by-element for correctness
      // We'll use a flat iteration with stride computation
    }
  }

  // More efficient approach: use array slicing
  // For each position i, check if data[i] > data[i+1:i+order+1] AND
  // data[i] > data[i-order:i] (for "greater" comparator)

  // Re-implement using flat CPU data for simplicity
  if (data.dtype() == DType::F64) {
    const double *src = data_cpu.data<double>();
    bool *dst = results.data<bool>();
    int64_t total = data_cpu.numel();

    // Compute strides for the axis
    int64_t axis_stride = 1;
    for (int d = axis + 1; d < ndim; ++d) {
      axis_stride *= data.shape().dim(d);
    }
    int64_t outer_count = total / datalen; // elements per slice along axis

    // For each "outer" position (all dims except axis)
    for (int64_t outer = 0; outer < outer_count; ++outer) {
      // Compute base offset for this outer index
      int64_t outer_rem = outer;
      int64_t base = 0;
      for (int d = 0; d < ndim; ++d) {
        if (d == axis)
          continue;
        int64_t dim_size = data.shape().dim(d);
        int64_t stride_d = 1;
        for (int dd = d + 1; dd < ndim; ++dd) {
          if (dd != axis)
            stride_d *= data.shape().dim(dd);
          else if (dd > axis)
            stride_d *= data.shape().dim(dd);
        }
        // Simpler: just iterate over flattened non-axis dims
      }

      // Actually, let's use a simpler approach: iterate over all elements
      // and for each, check its axis neighbors
    }

    // Simplest correct approach: iterate over all elements
    for (int64_t idx = 0; idx < total; ++idx) {
      if (!dst[idx])
        continue; // already marked as not-extremum

      // Compute position along axis
      int64_t axis_pos = (idx / axis_stride) % datalen;

      bool is_extremum = true;

      for (int64_t s = 1; s <= order; ++s) {
        // Compute neighbor indices
        int64_t p_pos = axis_pos + s;
        int64_t m_pos = axis_pos - s;

        if (mode == "clip") {
          p_pos = std::min(p_pos, datalen - 1);
          m_pos = std::max(m_pos, (int64_t)0);
        }

        int64_t p_idx = idx + (p_pos - axis_pos) * axis_stride;
        int64_t m_idx = idx + (m_pos - axis_pos) * axis_stride;

        double val = src[idx];
        double p_val = src[p_idx];
        double m_val = src[m_idx];

        if (comparator == "greater") {
          if (!(val > p_val) || !(val > m_val)) {
            is_extremum = false;
            break;
          }
        } else { // "less"
          if (!(val < p_val) || !(val < m_val)) {
            is_extremum = false;
            break;
          }
        }
      }

      dst[idx] = is_extremum;
    }
  } else if (data.dtype() == DType::F32) {
    const float *src = data_cpu.data<float>();
    bool *dst = results.data<bool>();
    int64_t total = data_cpu.numel();

    int64_t axis_stride = 1;
    for (int d = axis + 1; d < ndim; ++d) {
      axis_stride *= data.shape().dim(d);
    }

    for (int64_t idx = 0; idx < total; ++idx) {
      if (!dst[idx])
        continue;

      int64_t axis_pos = (idx / axis_stride) % datalen;
      bool is_extremum = true;

      for (int64_t s = 1; s <= order; ++s) {
        int64_t p_pos = axis_pos + s;
        int64_t m_pos = axis_pos - s;

        if (mode == "clip") {
          p_pos = std::min(p_pos, datalen - 1);
          m_pos = std::max(m_pos, (int64_t)0);
        }

        int64_t p_idx = idx + (p_pos - axis_pos) * axis_stride;
        int64_t m_idx = idx + (m_pos - axis_pos) * axis_stride;

        float val = src[idx];
        float p_val = src[p_idx];
        float m_val = src[m_idx];

        if (comparator == "greater") {
          if (!(val > p_val) || !(val > m_val)) {
            is_extremum = false;
            break;
          }
        } else {
          if (!(val < p_val) || !(val < m_val)) {
            is_extremum = false;
            break;
          }
        }
      }

      dst[idx] = is_extremum;
    }
  } else if (data.dtype() == DType::I32) {
    const int32_t *src = data_cpu.data<int32_t>();
    bool *dst = results.data<bool>();
    int64_t total = data_cpu.numel();

    int64_t axis_stride = 1;
    for (int d = axis + 1; d < ndim; ++d) {
      axis_stride *= data.shape().dim(d);
    }

    for (int64_t idx = 0; idx < total; ++idx) {
      if (!dst[idx])
        continue;

      int64_t axis_pos = (idx / axis_stride) % datalen;
      bool is_extremum = true;

      for (int64_t s = 1; s <= order; ++s) {
        int64_t p_pos = axis_pos + s;
        int64_t m_pos = axis_pos - s;

        if (mode == "clip") {
          p_pos = std::min(p_pos, datalen - 1);
          m_pos = std::max(m_pos, (int64_t)0);
        }

        int64_t p_idx = idx + (p_pos - axis_pos) * axis_stride;
        int64_t m_idx = idx + (m_pos - axis_pos) * axis_stride;

        int32_t val = src[idx];
        int32_t p_val = src[p_idx];
        int32_t m_val = src[m_idx];

        if (comparator == "greater") {
          if (!(val > p_val) || !(val > m_val)) {
            is_extremum = false;
            break;
          }
        } else {
          if (!(val < p_val) || !(val < m_val)) {
            is_extremum = false;
            break;
          }
        }
      }

      dst[idx] = is_extremum;
    }
  } else if (data.dtype() == DType::I64) {
    const int64_t *src = data_cpu.data<int64_t>();
    bool *dst = results.data<bool>();
    int64_t total = data_cpu.numel();

    int64_t axis_stride = 1;
    for (int d = axis + 1; d < ndim; ++d) {
      axis_stride *= data.shape().dim(d);
    }

    for (int64_t idx = 0; idx < total; ++idx) {
      if (!dst[idx])
        continue;

      int64_t axis_pos = (idx / axis_stride) % datalen;
      bool is_extremum = true;

      for (int64_t s = 1; s <= order; ++s) {
        int64_t p_pos = axis_pos + s;
        int64_t m_pos = axis_pos - s;

        if (mode == "clip") {
          p_pos = std::min(p_pos, datalen - 1);
          m_pos = std::max(m_pos, (int64_t)0);
        }

        int64_t p_idx = idx + (p_pos - axis_pos) * axis_stride;
        int64_t m_idx = idx + (m_pos - axis_pos) * axis_stride;

        int64_t val = src[idx];
        int64_t p_val = src[p_idx];
        int64_t m_val = src[m_idx];

        if (comparator == "greater") {
          if (!(val > p_val) || !(val > m_val)) {
            is_extremum = false;
            break;
          }
        } else {
          if (!(val < p_val) || !(val < m_val)) {
            is_extremum = false;
            break;
          }
        }
      }

      dst[idx] = is_extremum;
    }
  } else {
    INS_THROW("peak_finding: unsupported dtype");
  }

  // Transfer back if needed
  if (data.place().kind() != DeviceKind::CPU) {
    results = results.to(data.place());
  }

  return results;
}

} // anonymous namespace

std::vector<Array> argrelextrema(const Array &data,
                                 const std::string &comparator, int axis,
                                 int order, const std::string &mode) {
  INS_CHECK(data.defined(), "argrelextrema: input is undefined");
  INS_CHECK(order >= 1, "argrelextrema: order must be >= 1");
  INS_CHECK(comparator == "greater" || comparator == "less",
            "argrelextrema: comparator must be 'greater' or 'less'");

  Array mask = boolrelextrema(data, comparator, axis, order, mode);

  // Find nonzero indices (returns vector of index arrays, one per dim)
  Place cpu = CPUPlace();
  Array mask_cpu =
      (mask.place().kind() == DeviceKind::CPU) ? mask : mask.to(cpu);

  // Get indices where mask is true
  const bool *mask_data = mask_cpu.data<bool>();
  int64_t total = mask_cpu.numel();
  int ndim = data.shape().ndim();

  std::vector<std::vector<int64_t>> indices(ndim);
  for (int64_t i = 0; i < total; ++i) {
    if (mask_data[i]) {
      int64_t rem = i;
      for (int d = 0; d < ndim; ++d) {
        int64_t stride = 1;
        for (int dd = d + 1; dd < ndim; ++dd)
          stride *= data.shape().dim(dd);
        indices[d].push_back(rem / stride);
        rem %= stride;
      }
    }
  }

  std::vector<Array> result;
  Place original_place = data.place();
  for (int d = 0; d < ndim; ++d) {
    Array idx_arr = to_array(indices[d], DType::I64, cpu);
    if (original_place.kind() != DeviceKind::CPU) {
      idx_arr = idx_arr.to(original_place);
    }
    result.push_back(idx_arr);
  }

  return result;
}

std::vector<Array> argrelmax(const Array &data, int axis, int order,
                             const std::string &mode) {
  return argrelextrema(data, "greater", axis, order, mode);
}

std::vector<Array> argrelmin(const Array &data, int axis, int order,
                             const std::string &mode) {
  return argrelextrema(data, "less", axis, order, mode);
}

} // namespace signal
} // namespace ins
