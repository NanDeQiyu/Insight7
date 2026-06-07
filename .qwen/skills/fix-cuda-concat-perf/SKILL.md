---
name: fix-cuda-concat-perf
description: Replace element-by-element cudaMemcpy loop with cudaMemcpy2D for batched strided copy (2390x speedup on typical sizes)
source: auto-skill
extracted_at: '2026-06-06T21:25:49.785Z'
---

# Concat CUDA Kernel: 逐元素 cudaMemcpy → cudaMemcpy2D

## 诊断方法

当发现某个 CUDA 操作异常慢（比合理预期慢 1000x+），用 **逐段计时** 定位瓶颈：

```python
t0 = time.time()
result = some_operation(a, b)
print(f"op: {(time.time() - t0)*1000:.2f} ms")
```

然后在微基准测试中隔离各个子操作：

```python
# 测试 concat 本身
t0 = time.time()
for _ in range(20):
    c = ins.concat([a, b], -1)
print(f"concat: {(time.time()-t0)/20*1000:.3f} ms per call")
```

## 根因

**`cudaMemcpy` 有 ~5μs 的驱动级开销。** 如果在 for 循环中逐元素调用：

```cpp
for (int64_t linear = 0; linear < src_size; ++linear) {
    cudaMemcpy(dst + offset, src + offset, element_size, cudaMemcpyDeviceToDevice);
}
```

对 32000 个元素，32000 × 5μs ≈ 160ms。这个时间与数据量无关，纯粹是调用次数开销。

## 解决方案：`cudaMemcpy2D`

用 `cudaMemcpy2D` 做批量 strided copy，将 O(N) 次 `cudaMemcpy` 降为 O(num_inputs) 次：

```cpp
// For each input in concat:
// inner = product of dims after concat axis
// outer = product of dims before concat axis

size_t width = src_axis_dim * inner * elem_size;   // bytes per "row"
size_t height = outer;                               // number of rows
size_t src_pitch = width;                            // contiguous source
size_t dst_pitch = out_axis_dim * inner * elem_size; // strided output
size_t dst_off = axis_offset * inner * elem_size;    // offset in bytes

cudaMemcpy2D(
    (char*)out->data + dst_off, dst_pitch,
    (const char*)src->data + src_off, src_pitch,
    width, height, cudaMemcpyDeviceToDevice);
```

### 关键参数推导

对 concat 沿 axis A 操作的数组 `[d0, d1, ..., dA, ..., d_{n-1}]`：

- `inner = d_{A+1} × ... × d_{n-1}`（axis 后面的维度乘积）
- `outer = d_0 × ... × d_{A-1}`（axis 前面的维度乘积）
- `width = src->dims[A] × inner × elem_size`（每个 "chunk" 的字节数）
- `height = outer`（chunk 的数量）
- `src_pitch = width`（源数据连续）
- `dst_pitch = out->dims[A] × inner × elem_size`（输出中的 stride）
- `dst_off = axis_offset × inner × elem_size`（该输入在输出中的偏移）

该参数化对所有 axis 都成立。

### 对比

| 方式 | 调用次数 | 耗时 ([32,1000,1]×2 f64) | 加速比 |
|------|---------|--------------------------|--------|
| 逐元素 cudaMemcpy | 32000 | ~196ms | 1x |
| cudaMemcpy2D | 2 | ~0.082ms | **2390x** |

## 搜索模式

在 CUDA kernel 代码中搜索以下反模式：

```cpp
for (int64_t linear = 0; linear < src_size; ++linear) {
    // ... compute indices ...
    cudaMemcpy(...);  // ← 逐元素 cudaMemcpy 在循环中!
}
```

应当用批量的 `cudaMemcpy` / `cudaMemcpy2D` / `cudaMemcpy3D` 替代。

## 适用场景

- `concat` / `stack` 等需要批量内存拷贝的 kernel
- 任何在 for 循环中逐元素调用 `cudaMemcpy` 的代码
- `cudaMemcpy2D` 的 pitch 参数支持任意 stride，适用于所有 axis 的 concat
