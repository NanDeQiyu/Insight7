"""GPU transfer demo for Insight7 Python bindings.

Demonstrates: GPU device management, CPU-GPU data transfer,
GPU arithmetic, reductions, and zero-copy views.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "build", "bindings", "python")
)

import insight as ins


def separator(title):
    print(f"\n{'=' * 40}")
    print(f"  {title}")
    print(f"{'=' * 40}")


def main():
    ins.init()

    if not ins.has_device("gpu"):
        return

    print("Insight7 GPU Transfer Demo (Python)")

    # --- CPU to GPU transfer ---
    separator("CPU -> GPU Transfer")

    cpu_arr = ins.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"CPU array:\n{cpu_arr.numpy()}")

    gpu_arr = cpu_arr.to(ins.GPUPlace(0))
    print(
        f"GPU array shape=({gpu_arr.shape[0]},{gpu_arr.shape[1]}) place={'gpu' if gpu_arr.place.is_gpu() else 'cpu'}"
    )

    # --- GPU to CPU transfer ---
    separator("GPU -> CPU Transfer")

    back = gpu_arr.to(ins.CPUPlace())
    print(f"Round-tripped back to CPU:\n{back.numpy()}")

    # --- GPU arithmetic ---
    separator("GPU Arithmetic (F64)")

    a = ins.from_array([1.0, 2.0, 3.0]).to(ins.GPUPlace(0))
    b = ins.from_array([4.0, 5.0, 6.0]).to(ins.GPUPlace(0))

    sum_ab = a + b
    mul_ab = a * b
    print(f"GPU [1,2,3] + [4,5,6] = {sum_ab.to(ins.CPUPlace()).numpy()}")
    print(f"GPU [1,2,3] * [4,5,6] = {mul_ab.to(ins.CPUPlace()).numpy()}")

    # --- GPU arithmetic F32 ---
    separator("GPU Arithmetic (F32)")

    a32 = ins.from_array([1.0, 2.0, 3.0]).to(ins.float32).to(ins.GPUPlace(0))
    b32 = ins.from_array([4.0, 5.0, 6.0]).to(ins.float32).to(ins.GPUPlace(0))
    c32 = a32 * b32 + a32
    print(f"GPU F32: [1,2,3]*[4,5,6]+[1,2,3] = {c32.to(ins.CPUPlace()).numpy()}")

    # --- GPU reductions ---
    separator("GPU Reductions")

    data = ins.from_array([1.0, 2.0, 3.0, 4.0, 5.0]).to(ins.GPUPlace(0))
    print(f"GPU sum:  {ins.sum(data).to(ins.CPUPlace()).numpy()}")
    print(f"GPU mean: {ins.mean(data).to(ins.CPUPlace()).numpy()}")
    print(f"GPU max:  {ins.max(data).to(ins.CPUPlace()).numpy()}")

    # --- Zero-copy views ---
    separator("Zero-Copy Views (GPU)")

    base = ins.arange(12, dtype=ins.float32).reshape(ins.Shape([3, 4]))
    gpu_base = base.to(ins.GPUPlace(0))

    # Reshape (view, no copy)
    reshaped = gpu_base.reshape(ins.Shape([4, 3]))
    print(f"Reshape 3x4 -> 4x3: shape=({reshaped.shape[0]},{reshaped.shape[1]})")

    # Transpose (view, no copy)
    transposed = gpu_base.transpose()
    print(f"Transpose 3x4 -> 4x3: shape=({transposed.shape[0]},{transposed.shape[1]})")

    # Slice (view, no copy)
    sliced = gpu_base["0:2"]
    print(f"Slice rows 0-1: shape=({sliced.shape[0]},{sliced.shape[1]})")
    print(f"Sliced data: {sliced.to(ins.CPUPlace()).numpy()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
