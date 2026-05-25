// tests/test_dtype.cpp
#include "insight/c_api/dtype.h"
#include "insight/core/dtype.h"
#include <gtest/gtest.h>

using namespace ins;

// ========== DType 大小测试 ==========
TEST(DTypeTest, Size) {
  EXPECT_EQ(dtype_size(DType::BOOL), sizeof(bool));
  EXPECT_EQ(dtype_size(DType::U8), sizeof(uint8_t));
  EXPECT_EQ(dtype_size(DType::I8), sizeof(int8_t));
  EXPECT_EQ(dtype_size(DType::I16), sizeof(int16_t));
  EXPECT_EQ(dtype_size(DType::I32), sizeof(int32_t));
  EXPECT_EQ(dtype_size(DType::I64), sizeof(int64_t));
  EXPECT_EQ(dtype_size(DType::F16), sizeof(uint16_t));  // stored as uint16_t
  EXPECT_EQ(dtype_size(DType::BF16), sizeof(uint16_t)); // stored as uint16_t
  EXPECT_EQ(dtype_size(DType::F32), sizeof(float));
  EXPECT_EQ(dtype_size(DType::F64), sizeof(double));
  EXPECT_EQ(dtype_size(DType::C32), sizeof(std::complex<float>));
  EXPECT_EQ(dtype_size(DType::C64), sizeof(std::complex<double>));
  EXPECT_EQ(dtype_size(DType::F8_E4M3), sizeof(uint8_t));
  EXPECT_EQ(dtype_size(DType::F8_E5M2), sizeof(uint8_t));
  EXPECT_EQ(dtype_size(DType::U16), sizeof(uint16_t));
  EXPECT_EQ(dtype_size(DType::U32), sizeof(uint32_t));
  EXPECT_EQ(dtype_size(DType::U64), sizeof(uint64_t));
  EXPECT_EQ(dtype_size(DType::UNKNOWN), 0);

  // 验证边界
  EXPECT_EQ(dtype_size(static_cast<DType>(99)), 0);
}

// ========== DType 名称测试 ==========
TEST(DTypeTest, Name) {
  EXPECT_STREQ(dtype_name(DType::UNKNOWN), "unknown");
  EXPECT_STREQ(dtype_name(DType::BOOL), "bool");
  EXPECT_STREQ(dtype_name(DType::U8), "uint8");
  EXPECT_STREQ(dtype_name(DType::I8), "int8");
  EXPECT_STREQ(dtype_name(DType::I16), "int16");
  EXPECT_STREQ(dtype_name(DType::I32), "int32");
  EXPECT_STREQ(dtype_name(DType::I64), "int64");
  EXPECT_STREQ(dtype_name(DType::F16), "float16");
  EXPECT_STREQ(dtype_name(DType::BF16), "bfloat16");
  EXPECT_STREQ(dtype_name(DType::F32), "float32");
  EXPECT_STREQ(dtype_name(DType::F64), "float64");
  EXPECT_STREQ(dtype_name(DType::C32), "complex64");
  EXPECT_STREQ(dtype_name(DType::C64), "complex128");
  EXPECT_STREQ(dtype_name(DType::F8_E4M3), "float8_e4m3");
  EXPECT_STREQ(dtype_name(DType::F8_E5M2), "float8_e5m2");
  EXPECT_STREQ(dtype_name(DType::U16), "uint16");
  EXPECT_STREQ(dtype_name(DType::U32), "uint32");
  EXPECT_STREQ(dtype_name(DType::U64), "uint64");

  // 验证边界
  EXPECT_STREQ(dtype_name(static_cast<DType>(99)), "unknown");
}

// ========== dtype_from_name 测试 ==========
TEST(DTypeTest, FromName) {
  EXPECT_EQ(dtype_from_name("unknown"), DType::UNKNOWN);
  EXPECT_EQ(dtype_from_name("bool"), DType::BOOL);
  EXPECT_EQ(dtype_from_name("uint8"), DType::U8);
  EXPECT_EQ(dtype_from_name("int8"), DType::I8);
  EXPECT_EQ(dtype_from_name("int16"), DType::I16);
  EXPECT_EQ(dtype_from_name("int32"), DType::I32);
  EXPECT_EQ(dtype_from_name("int64"), DType::I64);
  EXPECT_EQ(dtype_from_name("float16"), DType::F16);
  EXPECT_EQ(dtype_from_name("bfloat16"), DType::BF16);
  EXPECT_EQ(dtype_from_name("float32"), DType::F32);
  EXPECT_EQ(dtype_from_name("float64"), DType::F64);
  EXPECT_EQ(dtype_from_name("complex64"), DType::C32);
  EXPECT_EQ(dtype_from_name("complex128"), DType::C64);
  EXPECT_EQ(dtype_from_name("float8_e4m3"), DType::F8_E4M3);
  EXPECT_EQ(dtype_from_name("float8_e5m2"), DType::F8_E5M2);
  EXPECT_EQ(dtype_from_name("uint16"), DType::U16);
  EXPECT_EQ(dtype_from_name("uint32"), DType::U32);
  EXPECT_EQ(dtype_from_name("uint64"), DType::U64);

  // 无效名称
  EXPECT_EQ(dtype_from_name("invalid_type"), DType::UNKNOWN);
}

// ========== DType 浮点判断测试 ==========
TEST(DTypeTest, IsFloatingPoint) {
  // 浮点类型应该返回 true
  EXPECT_TRUE(is_floating_point(DType::F16));
  EXPECT_TRUE(is_floating_point(DType::BF16));
  EXPECT_TRUE(is_floating_point(DType::F32));
  EXPECT_TRUE(is_floating_point(DType::F64));
  EXPECT_TRUE(is_floating_point(DType::F8_E4M3));
  EXPECT_TRUE(is_floating_point(DType::F8_E5M2));

  // 复数类型不算浮点（is_complex 单独判断）
  EXPECT_FALSE(is_floating_point(DType::C32));
  EXPECT_FALSE(is_floating_point(DType::C64));

  // 整数类型返回 false
  EXPECT_FALSE(is_floating_point(DType::BOOL));
  EXPECT_FALSE(is_floating_point(DType::U8));
  EXPECT_FALSE(is_floating_point(DType::I8));
  EXPECT_FALSE(is_floating_point(DType::I16));
  EXPECT_FALSE(is_floating_point(DType::I32));
  EXPECT_FALSE(is_floating_point(DType::I64));
  EXPECT_FALSE(is_floating_point(DType::U16));
  EXPECT_FALSE(is_floating_point(DType::U32));
  EXPECT_FALSE(is_floating_point(DType::U64));

  EXPECT_FALSE(is_floating_point(DType::UNKNOWN));
  EXPECT_FALSE(is_floating_point(static_cast<DType>(99)));
}

// ========== DType 整数判断测试 ==========
TEST(DTypeTest, IsInteger) {
  // 整数类型返回 true
  EXPECT_TRUE(is_integer(DType::U8));
  EXPECT_TRUE(is_integer(DType::I8));
  EXPECT_TRUE(is_integer(DType::I16));
  EXPECT_TRUE(is_integer(DType::I32));
  EXPECT_TRUE(is_integer(DType::I64));
  EXPECT_TRUE(is_integer(DType::U16));
  EXPECT_TRUE(is_integer(DType::U32));
  EXPECT_TRUE(is_integer(DType::U64));

  // BOOL 不是整数（按 NumPy 习惯）
  EXPECT_FALSE(is_integer(DType::BOOL));

  // 浮点类型返回 false
  EXPECT_FALSE(is_integer(DType::F16));
  EXPECT_FALSE(is_integer(DType::BF16));
  EXPECT_FALSE(is_integer(DType::F32));
  EXPECT_FALSE(is_integer(DType::F64));
  EXPECT_FALSE(is_integer(DType::F8_E4M3));
  EXPECT_FALSE(is_integer(DType::F8_E5M2));

  // 复数类型返回 false
  EXPECT_FALSE(is_integer(DType::C32));
  EXPECT_FALSE(is_integer(DType::C64));

  EXPECT_FALSE(is_integer(DType::UNKNOWN));
}

// ========== DType 复数判断测试 ==========
TEST(DTypeTest, IsComplex) {
  EXPECT_TRUE(is_complex(DType::C32));
  EXPECT_TRUE(is_complex(DType::C64));

  EXPECT_FALSE(is_complex(DType::F32));
  EXPECT_FALSE(is_complex(DType::F64));
  EXPECT_FALSE(is_complex(DType::I32));
  EXPECT_FALSE(is_complex(DType::I64));
  EXPECT_FALSE(is_complex(DType::U8));
  EXPECT_FALSE(is_complex(DType::BOOL));
  EXPECT_FALSE(is_complex(DType::UNKNOWN));
}

// ========== DType 有符号判断测试 ==========
TEST(DTypeTest, IsSigned) {
  // 有符号类型
  EXPECT_TRUE(is_signed(DType::I8));
  EXPECT_TRUE(is_signed(DType::I16));
  EXPECT_TRUE(is_signed(DType::I32));
  EXPECT_TRUE(is_signed(DType::I64));
  EXPECT_TRUE(is_signed(DType::F16));
  EXPECT_TRUE(is_signed(DType::BF16));
  EXPECT_TRUE(is_signed(DType::F32));
  EXPECT_TRUE(is_signed(DType::F64));
  EXPECT_TRUE(is_signed(DType::F8_E4M3));
  EXPECT_TRUE(is_signed(DType::F8_E5M2));
  EXPECT_TRUE(is_signed(DType::C32));
  EXPECT_TRUE(is_signed(DType::C64));

  // 无符号类型
  EXPECT_FALSE(is_signed(DType::U8));
  EXPECT_FALSE(is_signed(DType::U16));
  EXPECT_FALSE(is_signed(DType::U32));
  EXPECT_FALSE(is_signed(DType::U64));

  // BOOL 视为无符号
  EXPECT_FALSE(is_signed(DType::BOOL));

  EXPECT_FALSE(is_signed(DType::UNKNOWN));
}

// ========== 类型转换测试 (static_cast 验证) ==========
TEST(DTypeTest, CastToInt) {
  EXPECT_EQ(static_cast<int>(DType::UNKNOWN), 0);
  EXPECT_EQ(static_cast<int>(DType::BOOL), 1);
  EXPECT_EQ(static_cast<int>(DType::U8), 2);
  EXPECT_EQ(static_cast<int>(DType::I8), 3);
  EXPECT_EQ(static_cast<int>(DType::I16), 4);
  EXPECT_EQ(static_cast<int>(DType::I32), 5);
  EXPECT_EQ(static_cast<int>(DType::I64), 6);
  EXPECT_EQ(static_cast<int>(DType::F16), 7);
  EXPECT_EQ(static_cast<int>(DType::BF16), 8);
  EXPECT_EQ(static_cast<int>(DType::F32), 9);
  EXPECT_EQ(static_cast<int>(DType::F64), 10);
  EXPECT_EQ(static_cast<int>(DType::C32), 11);
  EXPECT_EQ(static_cast<int>(DType::C64), 12);
  EXPECT_EQ(static_cast<int>(DType::F8_E4M3), 13);
  EXPECT_EQ(static_cast<int>(DType::F8_E5M2), 14);
  EXPECT_EQ(static_cast<int>(DType::U16), 15);
  EXPECT_EQ(static_cast<int>(DType::U32), 16);
  EXPECT_EQ(static_cast<int>(DType::U64), 17);
  EXPECT_EQ(static_cast<int>(DType::DTYPE_COUNT), 18);
}

// ========== 枚举值与 C API 一致性测试 ==========
TEST(DTypeTest, CAPICompatibility) {
  // 验证 C API 的字符串返回与 C++ 一致
  EXPECT_STREQ(insight_dtype_name(static_cast<int32_t>(DType::F32)), "float32");
  EXPECT_STREQ(insight_dtype_name(static_cast<int32_t>(DType::I64)), "int64");
  EXPECT_STREQ(insight_dtype_name(static_cast<int32_t>(DType::BOOL)), "bool");
  EXPECT_STREQ(insight_dtype_name(static_cast<int32_t>(DType::C64)),
               "complex128");

  // 验证大小
  EXPECT_EQ(insight_dtype_size(static_cast<int32_t>(DType::F32)),
            sizeof(float));
  EXPECT_EQ(insight_dtype_size(static_cast<int32_t>(DType::C64)),
            sizeof(std::complex<double>));

  // 验证类型判断
  EXPECT_EQ(insight_dtype_is_float(static_cast<int32_t>(DType::F32)), 1);
  EXPECT_EQ(insight_dtype_is_float(static_cast<int32_t>(DType::I32)), 0);
  EXPECT_EQ(insight_dtype_is_int(static_cast<int32_t>(DType::I32)), 1);
  EXPECT_EQ(insight_dtype_is_int(static_cast<int32_t>(DType::F32)), 0);
  EXPECT_EQ(insight_dtype_is_complex(static_cast<int32_t>(DType::C32)), 1);
  EXPECT_EQ(insight_dtype_is_complex(static_cast<int32_t>(DType::F32)), 0);
  EXPECT_EQ(insight_dtype_is_signed(static_cast<int32_t>(DType::I32)), 1);
  EXPECT_EQ(insight_dtype_is_signed(static_cast<int32_t>(DType::U32)), 0);
}