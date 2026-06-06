---
name: add-python-dunder-operators
description: Adding all Python dunder operators to pybind11 Array binding — arithmetic, comparison, bitwise, unary, conversion
source: auto-skill
extracted_at: '2026-06-05T19:13:01.355Z'
---

# Python Dunder Operators for pybind11 Array

## Complete operator set

Add to `bindings/python/insight_py.cpp` in the Array class definition:

```cpp
// --- Arithmetic ---
.def("__add__", [](const Array &a, const Array &b) { return a + b; })
.def("__add__", [](const Array &a, double b) { return a + b; })
.def("__radd__", [](const Array &a, double b) { return b + a; })
.def("__sub__", [](const Array &a, const Array &b) { return a - b; })
.def("__sub__", [](const Array &a, double b) { return a - b; })
.def("__rsub__", [](const Array &a, double b) { return b - a; })
.def("__mul__", [](const Array &a, const Array &b) { return a * b; })
.def("__mul__", [](const Array &a, double b) { return a * b; })
.def("__rmul__", [](const Array &a, double b) { return b * a; })
.def("__truediv__", [](const Array &a, const Array &b) { return a / b; })
.def("__truediv__", [](const Array &a, double b) { return a / b; })
.def("__rtruediv__", [](const Array &a, double b) { return b / a; })
.def("__floordiv__", [](const Array &a, const Array &b) { return floor(a / b); })
.def("__floordiv__", [](const Array &a, double b) { return floor(a / b); })
.def("__rfloordiv__", [](const Array &a, double b) { return floor(b / a); })
.def("__mod__", [](const Array &a, const Array &b) { return a % b; })
.def("__mod__", [](const Array &a, double b) { return a % b; })
.def("__pow__", [](const Array &a, const Array &b) { return pow(a, b); })
.def("__pow__", [](const Array &a, double b) {
    return pow(a, full(a.shape(), b, a.dtype(), a.place())); })
.def("__rpow__", [](const Array &a, double b) {
    return pow(full(a.shape(), b, a.dtype(), a.place()), a); })
.def("__matmul__", [](const Array &a, const Array &b) { return matmul(a, b); })

// --- Unary ---
.def("__neg__", [](const Array &a) { return -a; })
.def("__pos__", [](const Array &a) { return +a; })
.def("__abs__", [](const Array &a) { return ins::abs(a); })
.def("__invert__", [](const Array &a) { return ~a; })

// --- Bitwise ---
.def("__and__", [](const Array &a, const Array &b) { return a & b; })
.def("__and__", [](const Array &a, int64_t b) { return a & b; })
.def("__rand__", [](const Array &a, int64_t b) { return b & a; })
.def("__or__", [](const Array &a, const Array &b) { return a | b; })
.def("__or__", [](const Array &a, int64_t b) { return a | b; })
.def("__ror__", [](const Array &a, int64_t b) { return b | a; })
.def("__xor__", [](const Array &a, const Array &b) { return a ^ b; })
.def("__xor__", [](const Array &a, int64_t b) { return a ^ b; })
.def("__rxor__", [](const Array &a, int64_t b) { return b ^ a; })
.def("__lshift__", [](const Array &a, const Array &b) { return a << b; })
.def("__lshift__", [](const Array &a, int64_t b) { return a << b; })
.def("__rshift__", [](const Array &a, const Array &b) { return a >> b; })
.def("__rshift__", [](const Array &a, int64_t b) { return a >> b; })

// --- Comparison ---
.def("__eq__", [](const Array &a, const Array &b) { return equal(a, b); })
.def("__ne__", [](const Array &a, const Array &b) { return not_equal(a, b); })
.def("__lt__", [](const Array &a, const Array &b) { return less(a, b); })
.def("__le__", [](const Array &a, const Array &b) { return less_equal(a, b); })
.def("__gt__", [](const Array &a, const Array &b) { return greater(a, b); })
.def("__ge__", [](const Array &a, const Array &b) { return greater_equal(a, b); })

// --- Conversion ---
.def("__bool__", [](const Array &a) { return static_cast<bool>(a); })
.def("__len__", [](const Array &a) {
    INS_CHECK(a.defined(), "Array is not initialized");
    if (a.shape().ndim() == 0) throw py::type_error("len() of unsized object");
    return a.shape().dim(0); })
.def("__int__", [](const Array &a) {
    INS_CHECK(a.numel() == 1, "int() only works for scalar arrays");
    Array cpu = a.to(CPUPlace());
    switch (cpu.dtype()) {
    case DType::F64: return py::cast(static_cast<int64_t>(cpu.data<double>()[0]));
    case DType::F32: return py::cast(static_cast<int64_t>(cpu.data<float>()[0]));
    case DType::I64: return py::cast(cpu.data<int64_t>()[0]);
    case DType::I32: return py::cast(static_cast<int64_t>(cpu.data<int32_t>()[0]));
    case DType::BOOL: return py::cast(static_cast<int64_t>(cpu.data<bool>()[0]));
    default: throw py::type_error("unsupported dtype for int()"); } })
.def("__float__", [](const Array &a) {
    INS_CHECK(a.numel() == 1, "float() only works for scalar arrays");
    Array cpu = a.to(CPUPlace());
    switch (cpu.dtype()) {
    case DType::F64: return py::cast(cpu.data<double>()[0]);
    case DType::F32: return py::cast(static_cast<double>(cpu.data<float>()[0]));
    case DType::I64: return py::cast(static_cast<double>(cpu.data<int64_t>()[0]));
    case DType::I32: return py::cast(static_cast<double>(cpu.data<int32_t>()[0]));
    default: throw py::type_error("unsupported dtype for float()"); } })
```

## How to apply

1. Add all operators to `bindings/python/insight_py.cpp` Array class
2. Build: `cmake --build . --target insight_python`
3. Test: `python3 -m pytest tests/cpu/python/test_operators.py -v`
4. The test file covers all operators with CPU+GPU, Array+scalar, broadcasting, edge cases

## Notes

- `__matmul__` needs `matmul()` from linalg ops
- `__floordiv__` uses `floor(a/b)` — no native C++ floor division operator
- `__pow__` scalar variant uses `full(a.shape(), b, ...)` to broadcast scalar
- `__int__`/`__float__` must copy to CPU first (`.to(CPUPlace())`)
- Bitwise operators use `int64_t` for scalar (not double)
