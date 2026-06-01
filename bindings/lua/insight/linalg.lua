--- Linear algebra operations.
--
-- Provides matrix multiplication, decompositions, eigenvalue
-- solvers, and other standard linear algebra routines.
--
-- @module insight.linalg

local native = require("_insight")
local M = {}

--- Matrix multiplication.
-- @tparam Array a First input matrix.
-- @tparam Array b Second input matrix.
-- @treturn Array Matrix product.
function M.matmul(a, b)
  return native.matmul(a, b)
end

--- Dot product of two arrays.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Dot product result.
function M.dot(a, b)
  return native.dot(a, b)
end

--- Outer product of two 1-D arrays.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Outer product matrix.
function M.outer(a, b)
  return native.outer(a, b)
end

--- Determinant of a square matrix.
-- @tparam Array x Input square matrix.
-- @number Determinant value.
function M.det(x)
  return native.det(x)
end

--- Matrix inverse.
-- @tparam Array x Input square matrix.
-- @treturn Array Inverse matrix.
function M.inv(x)
  return native.inv(x)
end

--- Solve linear system Ax = b.
-- @tparam Array a Coefficient matrix.
-- @tparam Array b Right-hand side.
-- @treturn Array Solution x.
function M.solve(a, b)
  return native.solve(a, b)
end

--- Singular value decomposition.
-- @tparam Array x Input matrix.
-- @treturn Array U, S, Vt (or compact equivalents).
function M.svd(x)
  return native.svd(x)
end

--- Eigenvalue decomposition for symmetric/Hermitian matrices.
-- @tparam Array x Symmetric input matrix.
-- @string[opt="L"] uplo Upper or lower triangular ("U" or "L").
-- @treturn Array Eigenvalues, eigenvectors.
function M.eigh(x, uplo)
  return native.eigh(x, uplo or "L")
end

--- Cholesky decomposition.
-- @tparam Array x Symmetric positive-definite matrix.
-- @bool[opt=true] lower If true, return lower triangular factor.
-- @treturn Array Cholesky factor.
function M.cholesky(x, lower)
  return native.cholesky(x, lower == nil and true or lower)
end

--- QR decomposition.
-- @tparam Array x Input matrix.
-- @string[opt="reduced"] mode "reduced" or "complete".
-- @treturn Array Q, R matrices.
function M.qr(x, mode)
  return native.qr(x, mode or "reduced")
end

--- Matrix or vector norm.
-- @tparam Array x Input array.
-- @number[opt=2] ord Order of the norm.
-- @number Norm value.
function M.norm(x, ord)
  return native.norm(x, ord or 2.0)
end

--- Sum of diagonal elements (trace).
-- @tparam Array x Input square matrix.
-- @number Trace value.
function M.trace(x)
  return native.trace(x)
end

--- Least-squares solution to a linear system.
-- @tparam Array a Coefficient matrix.
-- @tparam Array b Target values.
-- @number[opt=-1] rcond Cut-off ratio for small singular values.
-- @treturn Array Least-squares solution.
function M.lstsq(a, b, rcond)
  return native.lstsq(a, b, rcond or -1.0)
end

--- Condition number of a matrix.
-- @tparam Array x Input matrix.
-- @number[opt=2] p Order of the norm.
-- @number Condition number.
function M.cond(x, p)
  return native.cond(x, p or 2.0)
end

--- Rank of a matrix.
-- @tparam Array x Input matrix.
-- @number[opt=-1] tol Tolerance threshold for singular values.
-- @int Matrix rank.
function M.matrix_rank(x, tol)
  return native.matrix_rank(x, tol or -1.0)
end

return M
