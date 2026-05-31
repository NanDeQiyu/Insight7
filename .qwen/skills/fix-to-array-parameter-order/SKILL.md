---
name: fix-to-array-parameter-order
description: to_array with Shape requires specific parameter ordering — Shape must come before DType
source: auto-skill
extracted_at: '2026-05-30T02:00:00.000Z'
---

# Fix to_array Parameter Order

## Problem

The `to_array` function has multiple overloads with different parameter orderings.
Using the wrong order causes compilation errors like:
```
invalid initialization of reference of type 'const ins::Shape&' from expression of type 'std::vector<long int>'
```
or ambiguous overload resolution.

## Available Overloads

```cpp
// 1. Auto-deduced dtype, optional place
template <typename T>
Array to_array(const std::vector<T> &data, const Place &place = get_device());

// 2. Explicit dtype, optional place
template <typename T>
Array to_array(const std::vector<T> &data, DType dtype,
               const Place &place = get_device());

// 3. With shape (CORRECT ORDER: vector, Shape, DType, Place)
template <typename T>
Array to_array(const std::vector<T> &data, const Shape &shape,
               DType dtype = dtype_of<T>(), const Place &place = get_device());
```

## Common Mistakes

### ❌ Wrong: DType before Shape
```cpp
to_array(data, DType::F64, cpu, Shape(out_shape))  // COMPILE ERROR
to_array(data, DType::F64, Shape({m, 1}), cpu)     // COMPILE ERROR (DType before Shape)
```

### ❌ Wrong: Initializer list as shape
```cpp
to_array(data, DType::F64, cpu, {m, 1})  // COMPILE ERROR: {m,1} ambiguous
to_array(data, cpu, {m, m})              // COMPILE ERROR: {m,m} not a Place
```

### ✅ Correct
```cpp
to_array(data, Shape(out_shape), DType::F64, cpu)   // vector, Shape, DType, Place
to_array(data, Shape({m, 1}), DType::F64, cpu)       // explicit Shape construction
to_array(data, DType::F64, cpu)                       // no shape (1D auto)
to_array(data, cpu)                                    // auto dtype, explicit place
```

## Also Applies To Other Creation Functions

### `full()`, `zeros()`, `ones()` — Shape is always first
```cpp
// ✅ Correct
full(Shape({n}), 1.0, DType::F64, cpu)
zeros(Shape({m, n}), DType::F64, cpu)
ones({n}, DType::F64, cpu)                    // initializer_list OK for these

// ❌ Wrong
full({n}, 1.0, DType::F64)                    // Works (initializer_list overload)
full(t_shape, 1.0, DType::F64)                // FAILS if t_shape is std::vector
```

### `reshape()` — takes Shape, not vector
```cpp
// ✅ Correct
reshape(arr, Shape(out_shape))

// ❌ Wrong
reshape(arr, out_shape)  // FAILS if out_shape is std::vector<int64_t>
```

## Rule of Thumb

When passing a `std::vector<int64_t>` as a shape to ANY creation function:
- Always wrap it: `Shape(vec)`
- For `to_array` with shape: `to_array(data, Shape(vec), DType, Place)`
- For `full`/`zeros`/`ones`: prefer `Shape(vec)` over bare `vec`
- For `reshape`: always `reshape(arr, Shape(vec))`

When using initializer lists `{n, m}`:
- `full({n}, val)` works (has initializer_list overload)
- `to_array(data, {n, m})` does NOT work (no matching overload)
- Always use `Shape({n, m})` for to_array
