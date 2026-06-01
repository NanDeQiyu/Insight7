# modules/linalg.jl
# Linear algebra functions.

"""
    matmul(a::InsightArray, b::InsightArray) -> InsightArray

Matrix multiplication of two arrays.
"""
function matmul end

"""
    det(x::InsightArray) -> InsightArray

Compute the determinant of a square matrix.
"""
function det end

"""
    inv(x::InsightArray) -> InsightArray

Compute the inverse of a square matrix.
"""
function inv end

"""
    solve(a::InsightArray, b::InsightArray) -> InsightArray

Solve the linear system Ax = b.
"""
function solve end

"""
    svd(x::InsightArray) -> (U, S, Vt)

Compute the singular value decomposition. Returns a tuple `(U, S, Vt)`.
"""
function svd end

"""
    cholesky(x::InsightArray; lower::Bool=true) -> InsightArray

Compute the Cholesky decomposition of a positive-definite matrix.
"""
function cholesky end

"""
    lstsq(a::InsightArray, b::InsightArray) -> InsightArray

Compute the least-squares solution to Ax = b.
"""
function lstsq end

"""
    cond_fn(x::InsightArray; p::Float64=2.0) -> InsightArray

Compute the condition number of a matrix.
"""
function cond_fn end

"""
    matrix_rank(x::InsightArray) -> InsightArray

Compute the rank of a matrix.
"""
function matrix_rank end
