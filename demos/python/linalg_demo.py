"""Linear algebra demo for Insight7 Python bindings.

Demonstrates: matrix multiplication, determinant, inverse, SVD,
and linear system solving on CPU (and GPU if available).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "build", "bindings", "python")
)

import insight as ins
import numpy as np


def separator(title):
    print(f"\n{'=' * 40}")
    print(f"  {title}")
    print(f"{'=' * 40}")


def gpu_available():
    try:
        ins.load_backend("cuda")
        return True
    except Exception:
        return False


def run_cpu_linalg():
    separator("CPU Linear Algebra")

    # MatMul F64
    A = ins.from_array([[1.0, 2.0], [3.0, 4.0]])
    B = ins.from_array([[5.0, 6.0], [7.0, 8.0]])
    C = ins.matmul(A, B)
    print(f"MatMul F64:\n{C}")

    # MatMul F32
    A32 = A.to(ins.float32)
    B32 = B.to(ins.float32)
    C32 = ins.matmul(A32, B32)
    print(f"MatMul F32:\n{C32}")

    # Determinant
    try:
        print(f"det([[1,2],[3,4]]) = {ins.det(A)}")
    except Exception:
        print("det: skipped (requires OpenBLAS)")

    # Inverse
    try:
        A_inv = ins.inv(A)
        print(f"inv([[1,2],[3,4]]):\n{A_inv}")
        identity_check = ins.matmul(A, A_inv)
        print(f"A * A_inv (should be identity):\n{identity_check}")
    except Exception:
        print("inv: skipped (requires OpenBLAS)")

    # SVD
    try:
        D = ins.from_array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        U, S, VT = ins.svd(D, full_matrices=False)
        print(f"SVD singular values: {S}")
    except Exception:
        print("SVD: skipped (requires OpenBLAS)")

    # Solve linear system
    try:
        A3 = ins.from_array([[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]])
        b = ins.from_array([1.0, -2.0, 0.0])
        x = ins.solve(A3, b)
        print(f"Ax=b solution: {x}")
    except Exception:
        print("solve: skipped (requires OpenBLAS)")


def run_gpu_linalg():
    separator("GPU Linear Algebra")

    # MatMul F64 on GPU
    A = ins.from_array([[1.0, 2.0], [3.0, 4.0]]).to(ins.GPUPlace(0))
    B = ins.from_array([[5.0, 6.0], [7.0, 8.0]]).to(ins.GPUPlace(0))
    C = ins.matmul(A, B)
    print(f"GPU MatMul F64:\n{C.to(ins.CPUPlace())}")

    # MatMul F32 on GPU
    A32 = ins.from_array([[1.0, 2.0], [3.0, 4.0]]).to(ins.float32).to(ins.GPUPlace(0))
    B32 = ins.from_array([[5.0, 6.0], [7.0, 8.0]]).to(ins.float32).to(ins.GPUPlace(0))
    C32 = ins.matmul(A32, B32)
    print(f"GPU MatMul F32:\n{C32.to(ins.CPUPlace())}")

    # GPU det + inv
    A = ins.from_array([[1.0, 2.0], [3.0, 4.0]]).to(ins.GPUPlace(0))
    try:
        print(f"GPU det = {ins.det(A).to(ins.CPUPlace())}")
    except Exception:
        print("GPU det: skipped (requires OpenBLAS)")
    try:
        A_inv = ins.inv(A).to(ins.CPUPlace())
        print(f"GPU inv:\n{A_inv}")
    except Exception:
        print("GPU inv: skipped (requires OpenBLAS)")

    # GPU SVD (F32)
    try:
        D = (
            ins.from_array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
            .to(ins.float32)
            .to(ins.GPUPlace(0))
        )
        U, S, VT = ins.svd(D, full_matrices=False)
        print(f"GPU SVD singular values (F32): {S.to(ins.CPUPlace())}")
    except Exception:
        print("GPU SVD: skipped (requires OpenBLAS)")


def main():
    ins.init(["cpu"])

    print("Insight7 Linear Algebra Demo (Python)")

    run_cpu_linalg()

    if gpu_available():
        try:
            run_gpu_linalg()
        except Exception as e:
            print(f"\n[GPU not available: {e}]")
    else:
        print("\n[GPU not available, skipping GPU linalg demo]")

    print("\nDone!")


if __name__ == "__main__":
    main()
