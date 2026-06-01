--- Linear algebra operations.
--
-- Provides matrix multiplication, decompositions, eigenvalue
-- solvers, and other standard linear algebra routines.
--
-- @module insight.linalg

local native = require("_insight")
local M = {}

local _wrap = require("insight._wrap")

--- Matrix multiplication.
-- @tparam Array a First input matrix.
-- @tparam Array b Second input matrix.
-- @treturn Array Matrix product.
M.matmul = _wrap({ "a", "b" }, function(a, b)
  return native.matmul(a, b)
end)

--- Dot product of two arrays.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Dot product result.
M.dot = _wrap({ "a", "b" }, function(a, b)
  return native.dot(a, b)
end)

--- Outer product of two 1-D arrays.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Outer product matrix.
M.outer = _wrap({ "a", "b" }, function(a, b)
  return native.outer(a, b)
end)

--- Determinant of a square matrix.
-- @tparam Array x Input square matrix.
-- @number Determinant value.
M.det = _wrap({ "x" }, function(x)
  return native.det(x)
end)

--- Matrix inverse.
-- @tparam Array x Input square matrix.
-- @treturn Array Inverse matrix.
M.inv = _wrap({ "x" }, function(x)
  return native.inv(x)
end)

--- Solve linear system Ax = b.
-- @tparam Array a Coefficient matrix.
-- @tparam Array b Right-hand side.
-- @treturn Array Solution x.
M.solve = _wrap({ "a", "b" }, function(a, b)
  return native.solve(a, b)
end)

--- Singular value decomposition.
-- @tparam Array x Input matrix.
-- @treturn Array U, S, Vt (or compact equivalents).
M.svd = _wrap({ "x" }, function(x)
  return native.svd(x)
end)

--- Eigenvalue decomposition for symmetric/Hermitian matrices.
-- @tparam Array x Symmetric input matrix.
-- @string[opt="L"] uplo Upper or lower triangular ("U" or "L").
-- @treturn Array Eigenvalues, eigenvectors.
M.eigh = _wrap({ "x", "uplo" }, function(x, uplo)
  return native.eigh(x, uplo or "L")
end)

--- Cholesky decomposition.
-- @tparam Array x Symmetric positive-definite matrix.
-- @bool[opt=true] lower If true, return lower triangular factor.
-- @treturn Array Cholesky factor.
M.cholesky = _wrap({ "x", "lower" }, function(x, lower)
  return native.cholesky(x, lower == nil and true or lower)
end)

--- QR decomposition.
-- @tparam Array x Input matrix.
-- @string[opt="reduced"] mode "reduced" or "complete".
-- @treturn Array Q, R matrices.
M.qr = _wrap({ "x", "mode" }, function(x, mode)
  return native.qr(x, mode or "reduced")
end)

--- Matrix or vector norm.
-- @tparam Array x Input array.
-- @number[opt=2] ord Order of the norm.
-- @number Norm value.
M.norm = _wrap({ "x", "ord" }, function(x, ord)
  return native.norm(x, ord or 2.0)
end)

--- Sum of diagonal elements (trace).
-- @tparam Array x Input square matrix.
-- @number Trace value.
M.trace = _wrap({ "x" }, function(x)
  return native.trace(x)
end)

--- Least-squares solution to a linear system.
-- @tparam Array a Coefficient matrix.
-- @tparam Array b Target values.
-- @number[opt=-1] rcond Cut-off ratio for small singular values.
-- @treturn Array Least-squares solution.
M.lstsq = _wrap({ "a", "b", "rcond" }, function(a, b, rcond)
  return native.lstsq(a, b, rcond or -1.0)
end)

--- Condition number of a matrix.
-- @tparam Array x Input matrix.
-- @number[opt=2] p Order of the norm.
-- @number Condition number.
M.cond = _wrap({ "x", "p" }, function(x, p)
  return native.cond(x, p or 2.0)
end)

--- Rank of a matrix.
-- @tparam Array x Input matrix.
-- @number[opt=-1] tol Tolerance threshold for singular values.
-- @int Matrix rank.
M.matrix_rank = _wrap({ "x", "tol" }, function(x, tol)
  return native.matrix_rank(x, tol or -1.0)
end)

return M
