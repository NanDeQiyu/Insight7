---
name: add-array-setitem
description: "Adding __setitem__ to Array class — view-based assignment with fill_/copy_from_ via HAL dispatch, operator= semantics, non-contiguous output handling"
source: auto-skill
extracted_at: '2026-06-06T13:36:12.088Z'
---

# Adding `__setitem__` to Insight7 Array

## Architecture Overview

`__setitem__` is implemented through 3 layers:

1. **C++ Array class**: `fill_(double)`, `copy_from_(const Array&)`, `operator=(double)`, non-const `operator[]`
2. **Backend kernels**: Reuses existing `full` (fill) and `contiguous_copy` (copy) kernels via HAL dispatch
3. **Python binding**: `__setitem__` overloads for int, string, tuple, py::slice

## Step 1: C++ Array methods (`include/insight/core/array.h` + `src/core/array.cpp`)

### Header declarations (array.h)

```cpp
// In-place Mutation section
void fill_(double value);
void copy_from_(const Array &src);

// Assignment section
Array &operator=(double value);  // scalar fill

// Non-const indexing (returns view for assignment-through)
Array operator[](int64_t index);
Array operator[](const std::string &spec);
Array operator[](const Slice &slice);
```

### Implementation (array.cpp) — MUST use HAL dispatch, NEVER raw data<T>() pointers

```cpp
void Array::fill_(double value) {
  INS_CHECK(defined(), "Array is not initialized");
  if (numel() == 0) return;
  if (is_contiguous()) {
    double val = value;
    ops().launch("full", place(), dtype(), {layout_ptr(), &val}, {layout_ptr()});
  } else {
    // Non-contiguous: create contiguous filled array, then copy
    Array filled = full(shape(), value, dtype(), place());
    copy_from_(filled);
  }
}

void Array::copy_from_(const Array &src) {
  INS_CHECK(defined(), "...");
  INS_CHECK(src.defined(), "...");
  INS_CHECK(shape() == src.shape(), "copy_from_(): shape mismatch...");
  if (numel() == 0) return;
  if (dtype() == src.dtype()) {
    Array src_dev = src.to(place());
    ops().launch("contiguous_copy", place(), dtype(),
                 {(void*)src_dev.layout_ptr()}, {layout_ptr()});
  } else {
    Array converted = src.to(dtype()).to(place());
    ops().launch("contiguous_copy", place(), dtype(),
                 {(void*)converted.layout_ptr()}, {layout_ptr()});
  }
}
```

### operator= semantics

```cpp
Array &Array::operator=(const Array &other) {
  if (this == &other) return *this;
  // View with matching shape/dtype: write through shared memory
  if (layout_.is_view && defined() && other.defined() &&
      shape_ == other.shape_ && dtype() == other.dtype()) {
    copy_from_(other);
    return *this;
  }
  // Rebind (non-view, different shape/dtype, or uninitialized)
  // ... existing rebind logic ...
}

Array &Array::operator=(double value) {
  fill_(value);
  return *this;
}
```

**CRITICAL**: View `operator=` only writes through when shape AND dtype match. Otherwise it rebinds. This prevents breaking existing code that does `view = other_shape_array` (rebind pattern).

### Non-const operator[] (array.cpp)

```cpp
Array Array::operator[](int64_t index) {
  return static_cast<const Array&>(*this)[index];
}
Array Array::operator[](const std::string &spec) {
  return static_cast<const Array&>(*this)[spec];
}
Array Array::operator[](const Slice &slice) {
  return static_cast<const Array&>(*this)[slice];
}
```

## Step 2: Python binding (`bindings/python/insight_py.cpp`)

Add `__setitem__` overloads for all index types. Each gets the view via `__getitem__`, then calls `fill_` or `copy_from_`:

```cpp
// After __getitem__ and at() definitions, before arithmetic operators:

// int index + scalar/array
.def("__setitem__", [](Array &a, int64_t idx, double val) {
  Array view = a[idx]; view.fill_(val); })
.def("__setitem__", [](Array &a, int64_t idx, const Array &src) {
  Array view = a[idx]; view.copy_from_(src); })

// string spec + scalar/array
.def("__setitem__", [](Array &a, const std::string &spec, double val) {
  Array view = a[spec]; view.fill_(val); })
.def("__setitem__", [](Array &a, const std::string &spec, const Array &src) {
  Array view = a[spec]; view.copy_from_(src); })

// tuple index + scalar/array (build spec string, same logic as __getitem__)
.def("__setitem__", [](Array &a, py::tuple tup, double val) {
  std::string spec = build_spec_from_tuple(tup);
  Array view = a[spec]; view.fill_(val); })
.def("__setitem__", [](Array &a, py::tuple tup, const Array &src) {
  std::string spec = build_spec_from_tuple(tup);
  Array view = a[spec]; view.copy_from_(src); })

// py::slice + scalar/array (convert to Slice object)
.def("__setitem__", [](Array &a, py::slice pyslice, double val) {
  Array view = a[py_slice_to_insight_slice(pyslice)]; view.fill_(val); })
.def("__setitem__", [](Array &a, py::slice pyslice, const Array &src) {
  Array view = a[py_slice_to_insight_slice(pyslice)]; view.copy_from_(src); })
```

## Step 3: Lua binding (`bindings/lua/insight_lua.cpp`)

Add to Array usertype:
```cpp
array_type["fill_"] = &Array::fill_;
array_type["copy_from_"] = &Array::copy_from_;
```

## Step 4: Julia binding (`bindings/julia/insight_julia_capi.cpp`)

Add C ABI wrappers:
```cpp
extern "C" void insight_jl_fill(Array *arr, double value) { arr->fill_(value); }
extern "C" void insight_jl_copy_from(Array *dst, const Array *src) { dst->copy_from_(*src); }
```

## C++ Assignment on Temporary Views

`arr[":"] = 7` **compiles and works in C++**. Although `operator[]` returns a temporary (rvalue), `operator=` is a regular member function that CAN be called on rvalues. The temporary view shares memory with the parent array, so `operator=(double)` calls `fill_()` which writes through the shared pointer before the temporary is destroyed.

```cpp
arr[":"] = 7.0;           // ✅ temporary view → fill_() → shared memory modified
arr[Slice(1,3)] = 42.0;   // ✅ same mechanism
arr[":"] = other_arr;     // ✅ temporary view → copy_from_() → shared memory modified
arr["0:2,1:3"] = src;     // ✅ 2D string slice assignment
```

The key insight: the temporary's lifetime extends through the full expression, so `operator=` executes `fill_()`/`copy_from_()` before the temporary is destroyed. The data is already written to shared memory.

Python also works via `__setitem__` method call: `a[":"] = 7` → `a.__setitem__(":", 7)`.

## How to apply

1. Add `fill_()`, `copy_from_()`, `operator=(double)`, non-const `operator[]` to C++ Array class
2. Add `__setitem__` overloads to Python binding (all 5 index types × 2 value types)
3. Add `fill_`/`copy_from_` to Lua and Julia bindings
4. Build and run full CPU test suite: `OMP_NUM_THREADS=1 ./tests/insight_tests_cpu`
5. Test `__setitem__` with 1D and 2D slices, scalar and array values

## Critical: contiguous_copy kernel must handle non-contiguous output

See `fix-contiguous-copy-noncontiguous-output` skill. Without that fix, 2D slice assignment writes to wrong positions.

## Critical: full kernel must handle out->offset

See `fix-full-kernel-view-offset` skill. Without that fix, scalar view assignment `a["1,1"] = 99` writes to position [0,0] instead of [1,1].

## Lua __newindex — sol2 workaround

sol2's `sol::meta_function::new_index` doesn't work for usertypes (conflicts with property system). The workaround is to override `__newindex` from Lua in `init.lua`:

```lua
local tmp = native.zeros({1}, native.float64, native.CPUPlace())
local mt = getmetatable(tmp)
mt.__newindex = function(self, key, value)
  if type(key) == "number" then
    local idx = key > 0 and (key - 1) or key  -- 1-based → 0-based
    local view = self:at(idx)
    if type(value) == "number" then view:fill_(value) else view:copy_from_(value) end
  elseif type(key) == "string" then
    local view = self[key]  -- uses __index for string slicing
    if type(value) == "number" then view:fill_(value) else view:copy_from_(value) end
  end
end
```

Note: `Array()` constructor doesn't work in sol2 (table, not callable). Use `native.zeros({1}, ...)` to get a temporary for metatable access.

### Lua Array{1,2,3} constructor — __call metamethod on usertype table

sol2 usertypes are tables, not callable. To support `ins.Array{1,2,3}`, set `__call` on the usertype table's metatable:

```cpp
// In insight_lua.cpp, after sol2 usertype setup:
sol::stack::push(lua, m);
lua_getfield(L, -1, "Array");
if (!lua_isnil(L, -1)) {
  if (!lua_getmetatable(L, -1)) {
    lua_newtable(L);
    lua_setmetatable(L, -2);
    lua_getmetatable(L, -1);
  }
  lua_pushcfunction(L, [](lua_State *L) -> int {
    sol::table t(L, 2);
    sol::optional<Place> place;
    if (lua_gettop(L) > 2) place = sol::stack::get<Place>(L, 3);
    Array result = from_lua_table(t, place);
    sol::stack::push(L, std::move(result));
    return 1;
  });
  lua_setfield(L, -2, "__call");
  lua_pop(L, 1);
}
lua_pop(L, 2);
```

## Julia binding

```cpp
// insight_julia_capi.cpp
extern "C" void insight_jl_fill(Array *arr, double value) { arr->fill_(value); }
extern "C" void insight_jl_copy_from(Array *dst, const Array *src) { dst->copy_from_(*src); }
```

```julia
# Insight.jl
function fill_!(arr::InsightArray, value::Float64)
    ccall((:insight_jl_fill, LIB_INSIGHT), Cvoid, (Ptr{Cvoid}, Float64), arr, value)
end
function copy_from_!(dst::InsightArray, src::InsightArray)
    ccall((:insight_jl_copy_from, LIB_INSIGHT), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), dst, src)
end
```
