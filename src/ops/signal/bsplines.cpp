// src/ops/signal/bsplines.cpp
#include "insight/ops/signal/bsplines.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/indexing.h"
#include "insight/ops/operator.h"
#include "insight/ops/unary.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ins {
namespace signal {

Array gauss_spline(const Array &x, int n) {
  INS_CHECK(x.defined(), "gauss_spline: input is undefined");
  INS_CHECK(n >= 0, "gauss_spline: order n must be >= 0");

  double sigma_sq = (n + 1) / 12.0;
  double r_sigma_sq = 0.5 / sigma_sq;
  double coeff = 1.0 / std::sqrt(2.0 * M_PI * sigma_sq);

  // output = coeff * exp(-x^2 * r_sigma_sq)
  Array x2 = mul(x, x);
  Array neg_r = full(x2.shape(), -r_sigma_sq, x.dtype(), x.place());
  Array exp_arg = mul(x2, neg_r);
  Array result =
      mul(exp(exp_arg), full(x2.shape(), coeff, x.dtype(), x.place()));
  return result;
}

Array cubic(const Array &x) {
  INS_CHECK(x.defined(), "cubic: input is undefined");

  DType work_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;
  Array x_work = (x.dtype() == work_dtype) ? x : x.to(work_dtype);

  Array ax = abs(x_work);
  Array one = full(ax.shape(), 1.0, work_dtype, ax.place());
  Array two = full(ax.shape(), 2.0, work_dtype, ax.place());
  Array zero = full(ax.shape(), 0.0, work_dtype, ax.place());
  Array two_thirds = full(ax.shape(), 2.0 / 3.0, work_dtype, ax.place());
  Array one_half = full(ax.shape(), 0.5, work_dtype, ax.place());
  Array one_sixth = full(ax.shape(), 1.0 / 6.0, work_dtype, ax.place());

  // Region 1: |x| < 1 → 2/3 - 0.5*|x|^2*(2-|x|)
  Array ax2 = mul(ax, ax);
  Array two_minus_ax = sub(two, ax);
  Array region1 = sub(two_thirds, mul(one_half, mul(ax2, two_minus_ax)));

  // Region 2: 1 <= |x| < 2 → (1/6)*(2-|x|)^3
  Array t2 = mul(two_minus_ax, two_minus_ax);
  Array t3 = mul(t2, two_minus_ax);
  Array region2 = mul(one_sixth, t3);

  // Region 3: |x| >= 2 → 0
  Array mask1 = less_than(ax, one);
  Array mask2 = less_than(ax, two);

  // result = mask1 ? region1 : (mask2 ? region2 : 0)
  Array result = where(mask1, region1, where(mask2, region2, zero));

  if (result.dtype() != x.dtype()) {
    result = result.to(x.dtype());
  }
  return result;
}

Array quadratic(const Array &x) {
  INS_CHECK(x.defined(), "quadratic: input is undefined");

  DType work_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;
  Array x_work = (x.dtype() == work_dtype) ? x : x.to(work_dtype);

  Array ax = abs(x_work);
  Array half = full(ax.shape(), 0.5, work_dtype, ax.place());
  Array three_halves = full(ax.shape(), 1.5, work_dtype, ax.place());
  Array zero = full(ax.shape(), 0.0, work_dtype, ax.place());
  Array p75 = full(ax.shape(), 0.75, work_dtype, ax.place());

  // Region 1: |x| < 0.5 → 0.75 - |x|^2
  Array ax2 = mul(ax, ax);
  Array region1 = sub(p75, ax2);

  // Region 2: 0.5 <= |x| < 1.5 → 0.5*(|x|-1.5)^2
  Array diff = sub(ax, three_halves);
  Array region2 = mul(half, mul(diff, diff));

  // Region 3: |x| >= 1.5 → 0
  Array mask1 = less_than(ax, half);
  Array mask2 = less_than(ax, three_halves);

  Array result = where(mask1, region1, where(mask2, region2, zero));

  if (result.dtype() != x.dtype()) {
    result = result.to(x.dtype());
  }
  return result;
}

} // namespace signal
} // namespace ins
