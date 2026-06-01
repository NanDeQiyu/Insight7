#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

namespace insight {

// IEEE 754 half-precision (float16) conversion
// Format: 1 sign bit, 5 exponent bits, 10 mantissa bits
inline float f16_to_f32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;

  uint32_t f;
  if (exp == 0) {
    if (mantissa == 0) {
      // Zero
      f = sign << 31;
    } else {
      // Denormalized — normalize it
      exp = 1;
      while (!(mantissa & 0x400)) {
        mantissa <<= 1;
        exp--;
      }
      mantissa &= 0x3FF;
      f = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exp == 31) {
    // Inf or NaN
    f = (sign << 31) | 0x7F800000 | (mantissa << 13);
  } else {
    // Normalized
    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &f, 4);
  return result;
}

inline uint16_t f32_to_f16(float val) {
  uint32_t f;
  std::memcpy(&f, &val, 4);

  uint32_t sign = (f >> 31) & 1;
  int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = (f >> 13) & 0x3FF;

  if (exp <= 0) {
    if (exp < -10)
      return static_cast<uint16_t>(sign << 15); // underflow to zero
    mantissa = (mantissa | 0x400) >> (1 - exp);
    return static_cast<uint16_t>((sign << 15) | mantissa);
  }
  if (exp >= 31) {
    // Overflow to inf
    return static_cast<uint16_t>((sign << 15) | 0x7C00);
  }
  return static_cast<uint16_t>((sign << 15) | (exp << 10) | mantissa);
}

// bfloat16 conversion
// Format: 1 sign bit, 8 exponent bits, 7 mantissa bits (truncated float32)
inline float bf16_to_f32(uint16_t b) {
  uint32_t f = static_cast<uint32_t>(b) << 16;
  float result;
  std::memcpy(&result, &f, 4);
  return result;
}

inline uint16_t f32_to_bf16(float val) {
  uint32_t f;
  std::memcpy(&f, &val, 4);
  // Round to nearest even
  uint32_t rounding_bias = 0x7FFF + ((f >> 16) & 1);
  return static_cast<uint16_t>((f + rounding_bias) >> 16);
}

} // namespace insight
