// src/ops/signal/waveforms.cpp
#include "insight/ops/signal/waveforms.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/indexing.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/operator.h"
#include "insight/ops/unary.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ins {
namespace signal {

// ============================================================================
// sawtooth
// ============================================================================

Array sawtooth(const Array &t, double width) {
  INS_CHECK(t.defined(), "sawtooth: input is undefined");
  INS_CHECK(width >= 0.0 && width <= 1.0,
            "sawtooth: width must be in [0, 1], got ", width);

  double period = 2.0 * M_PI;

  // tmod = t mod 2*pi (Python-style: result in [0, 2*pi))
  Array tmod = mod(t, full(t.shape(), period, t.dtype(), t.place()));

  // Two regions:
  // Rising:  tmod < width*2*pi  →  tmod/(pi*width) - 1
  // Falling: tmod >= width*2*pi →  (pi*(width+1) - tmod) / (pi*(1-width))
  Array threshold = full(t.shape(), width * period, t.dtype(), t.place());
  Array is_rising = less_than(tmod, threshold);

  Array pi_w = full(t.shape(), M_PI * width, t.dtype(), t.place());
  Array pi_1mw = full(t.shape(), M_PI * (1.0 - width), t.dtype(), t.place());
  Array one = full(t.shape(), 1.0, t.dtype(), t.place());
  Array pi_wp1 = full(t.shape(), M_PI * (width + 1.0), t.dtype(), t.place());

  Array rising = sub(div(tmod, pi_w), one);
  Array falling = div(sub(pi_wp1, tmod), pi_1mw);

  return where(is_rising, rising, falling);
}

// ============================================================================
// square
// ============================================================================

Array square(const Array &t, double duty) {
  INS_CHECK(t.defined(), "square: input is undefined");
  INS_CHECK(duty >= 0.0 && duty <= 1.0, "square: duty must be in [0, 1], got ",
            duty);

  double period = 2.0 * M_PI;

  // tmod = t mod 2*pi
  Array tmod = mod(t, full(t.shape(), period, t.dtype(), t.place()));

  Array threshold = full(t.shape(), duty * period, t.dtype(), t.place());
  Array is_high = less_than(tmod, threshold);

  Array pos = full(t.shape(), 1.0, t.dtype(), t.place());
  Array neg = full(t.shape(), -1.0, t.dtype(), t.place());

  return where(is_high, pos, neg);
}

// ============================================================================
// gausspulse
// ============================================================================

namespace {

// Compute gausspulse envelope parameter 'a'
// a = (pi*fc*bw)^2 / (4 * log10(10^(bwr/20)))
// Note: bwr is negative, so log10(10^(bwr/20)) = bwr/20 is negative → a < 0
double gausspulse_a(double fc, double bw, double bwr) {
  double log10_val = bwr / 20.0; // log10(10^(bwr/20)) = bwr/20
  double pi_fc_bw = M_PI * fc * bw;
  return (pi_fc_bw * pi_fc_bw) / (4.0 * log10_val);
}

} // anonymous namespace

Array gausspulse(const Array &t, double fc, double bw, double bwr, double tpr) {
  INS_CHECK(t.defined(), "gausspulse: input is undefined");

  DType work_dtype = (t.dtype() == DType::F32) ? DType::F32 : DType::F64;
  Array t_work = (t.dtype() == work_dtype) ? t : t.to(work_dtype);

  double a = gausspulse_a(fc, bw, bwr);
  Array t2 = mul(t_work, t_work);

  // Envelope: exp(a * t^2)  (note: a is negative)
  Array a_arr = full(t2.shape(), a, work_dtype, t_work.place());
  Array env = exp(mul(a_arr, t2));

  // In-phase: env * cos(2*pi*fc*t)
  Array phase = mul(
      t_work, full(t2.shape(), 2.0 * M_PI * fc, work_dtype, t_work.place()));
  Array yi = mul(env, cos(phase));

  if (yi.dtype() != t.dtype()) {
    yi = yi.to(t.dtype());
  }
  return yi;
}

GaussPulseResult gausspulse_full(const Array &t, double fc, double bw,
                                 double bwr, double tpr) {
  INS_CHECK(t.defined(), "gausspulse_full: input is undefined");

  DType work_dtype = (t.dtype() == DType::F32) ? DType::F32 : DType::F64;
  Array t_work = (t.dtype() == work_dtype) ? t : t.to(work_dtype);

  double a = gausspulse_a(fc, bw, bwr);
  Array t2 = mul(t_work, t_work);

  Array a_arr = full(t2.shape(), a, work_dtype, t_work.place());
  Array env = exp(mul(a_arr, t2));

  Array phase = mul(
      t_work, full(t2.shape(), 2.0 * M_PI * fc, work_dtype, t_work.place()));

  Array yi = mul(env, cos(phase));
  Array yq = mul(env, sin(phase));

  if (yi.dtype() != t.dtype()) {
    yi = yi.to(t.dtype());
    yq = yq.to(t.dtype());
    env = env.to(t.dtype());
  }
  return {yi, yq, env};
}

// ============================================================================
// chirp
// ============================================================================

Array chirp(const Array &t, double f0, double t1, double f1, ChirpMethod method,
            double phi, bool vertex_zero) {
  INS_CHECK(t.defined(), "chirp: input is undefined");
  INS_CHECK(t1 != 0, "chirp: t1 must not be zero");

  DType work_dtype = (t.dtype() == DType::F32) ? DType::F32 : DType::F64;
  Array t_work = (t.dtype() == work_dtype) ? t : t.to(work_dtype);

  Array t2 = mul(t_work, t_work);
  Array phase;

  switch (method) {
  case ChirpMethod::Linear: {
    // beta = (f1 - f0) / t1
    // phase = 2*pi*(f0*t + 0.5*beta*t^2) + phi
    double beta = (f1 - f0) / t1;
    Array f0_t = mul(t_work, full(t_work.shape(), f0, work_dtype));
    Array half_beta_t2 = mul(t2, full(t2.shape(), 0.5 * beta, work_dtype));
    Array inner = mul(add(f0_t, half_beta_t2),
                      full(t_work.shape(), 2.0 * M_PI, work_dtype));
    phase = add(inner, full(inner.shape(), phi, work_dtype));
    break;
  }
  case ChirpMethod::Quadratic: {
    // beta = (f1 - f0) / t1^2
    double beta = (f1 - f0) / (t1 * t1);
    if (vertex_zero) {
      // phase = 2*pi*(f0*t + beta*t^3/3) + phi
      Array t3 = mul(t2, t_work);
      Array f0_t = mul(t_work, full(t_work.shape(), f0, work_dtype));
      Array beta_t3_3 = mul(t3, full(t3.shape(), beta / 3.0, work_dtype));
      Array inner = mul(add(f0_t, beta_t3_3),
                        full(t_work.shape(), 2.0 * M_PI, work_dtype));
      phase = add(inner, full(inner.shape(), phi, work_dtype));
    } else {
      // phase = 2*pi*(f1*t + beta*((t1-t)^3 - t1^3)/3) + phi
      Array t1_arr = full(t_work.shape(), t1, work_dtype);
      Array diff_t = sub(t1_arr, t_work);
      Array diff_t2 = mul(diff_t, diff_t);
      Array diff_t3 = mul(diff_t2, diff_t);
      double t1_cubed = t1 * t1 * t1;
      Array f1_t = mul(t_work, full(t_work.shape(), f1, work_dtype));
      Array bracket = sub(diff_t3, full(diff_t3.shape(), t1_cubed, work_dtype));
      Array beta_bracket =
          mul(bracket, full(bracket.shape(), beta / 3.0, work_dtype));
      Array inner = mul(add(f1_t, beta_bracket),
                        full(t_work.shape(), 2.0 * M_PI, work_dtype));
      phase = add(inner, full(inner.shape(), phi, work_dtype));
    }
    break;
  }
  case ChirpMethod::Logarithmic: {
    // beta = t1 / ln(f1/f0)
    // phase = 2*pi*beta*f0*((f1/f0)^(t/t1) - 1) + phi
    double beta = t1 / std::log(f1 / f0);
    double ratio = f1 / f0;
    // (f1/f0)^(t/t1) = exp((t/t1) * ln(f1/f0))
    Array exponent =
        mul(t_work, full(t_work.shape(), std::log(ratio) / t1, work_dtype));
    Array power = exp(exponent);
    Array minus_one = sub(power, full(power.shape(), 1.0, work_dtype));
    Array inner = mul(
        minus_one, full(minus_one.shape(), 2.0 * M_PI * beta * f0, work_dtype));
    phase = add(inner, full(inner.shape(), phi, work_dtype));
    break;
  }
  case ChirpMethod::Hyperbolic: {
    // sing = -f1*t1/(f0-f1)
    // phase = 2*pi*(-sing*f0)*ln(|1 - t/sing|) + phi
    double sing = -f1 * t1 / (f0 - f1);
    // 1 - t/sing
    Array t_over_sing = div(t_work, full(t_work.shape(), sing, work_dtype));
    Array one_minus = sub(full(t_work.shape(), 1.0, work_dtype), t_over_sing);
    Array abs_one_minus = abs(one_minus);
    Array log_val = log(abs_one_minus);
    Array inner = mul(
        log_val, full(log_val.shape(), 2.0 * M_PI * (-sing) * f0, work_dtype));
    phase = add(inner, full(inner.shape(), phi, work_dtype));
    break;
  }
  }

  Array result = cos(phase);

  if (result.dtype() != t.dtype()) {
    result = result.to(t.dtype());
  }
  return result;
}

// ============================================================================
// unit_impulse
// ============================================================================

Array unit_impulse(const Shape &shape, int64_t idx, DType dtype,
                   const Place &place) {
  // Always create on CPU first (need direct pointer access to set the impulse)
  Array result = zeros(shape, dtype, CPUPlace());

  if (idx < 0) {
    idx = shape.numel() / 2;
  }

  INS_CHECK(idx >= 0 && idx < shape.numel(),
            "unit_impulse: idx out of range [0, ", shape.numel(), ")");

  switch (dtype) {
  case DType::F32:
    result.data<float>()[idx] = 1.0f;
    break;
  case DType::F64:
    result.data<double>()[idx] = 1.0;
    break;
  case DType::I32:
    result.data<int32_t>()[idx] = 1;
    break;
  case DType::I64:
    result.data<int64_t>()[idx] = 1;
    break;
  default:
    result.data<double>()[idx] = 1.0;
    break;
  }

  // Transfer to target device if needed
  if (place.kind() != DeviceKind::CPU) {
    result = result.to(place);
  }
  return result;
}

} // namespace signal
} // namespace ins
