--- Indexing and searching operations.
--
-- Provides functions for taking, sorting, gathering, scattering,
-- and searching elements in arrays.
--
-- @module insight.indexing

local native = require("_insight")
local M = {}

--- Take elements from an array along an axis by index.
-- @tparam Array x Input array.
-- @tparam Array indices Indices of elements to take.
-- @int[opt] axis Axis along which to take. If nil, operates on flattened array.
-- @treturn Array Gathered elements.
function M.take(x, indices, axis)
  return native.take(x, indices, axis)
end

--- Return indices of non-zero elements.
-- @tparam Array x Input array.
-- @treturn Array Indices of non-zero elements.
function M.nonzero(x)
  return native.nonzero(x)
end

--- Return indices of non-zero elements (flattened).
-- @tparam Array x Input array.
-- @treturn Array Flat indices of non-zero elements.
function M.flatnonzero(x)
  return native.flatnonzero(x)
end

--- Return indices that would sort the array.
-- @tparam Array x Input array.
-- @int[opt=-1] axis Axis along which to sort. -1 is last axis.
-- @treturn Array Index array for sorting.
function M.argsort(x, axis)
  return native.argsort(x, axis or -1)
end

--- Return a sorted copy of the array.
-- @tparam Array x Input array.
-- @int[opt=-1] axis Axis along which to sort. -1 is last axis.
-- @treturn Array Sorted array.
function M.sort(x, axis)
  return native.sort(x, axis or -1)
end

--- Return unique elements of the array.
-- @tparam Array x Input array.
-- @treturn UniqueResult Table with fields: values, indices, inverse, counts.
function M.unique(x)
  return native.unique(x)
end

--- Return the k largest or smallest elements.
-- @tparam Array x Input array.
-- @int k Number of elements to return.
-- @int[opt=-1] axis Axis along which to search. -1 is last axis.
-- @bool[opt=true] largest If true, return largest; otherwise smallest.
-- @treturn Array Top-k values.
function M.topk(x, k, axis, largest)
  return native.topk(x, k, axis or -1, largest == nil and true or largest)
end

--- Gather values along an axis by index.
-- @tparam Array x Input array.
-- @tparam Array indices Indices to gather.
-- @int[opt=0] axis Axis along which to gather.
-- @treturn Array Gathered values.
function M.gather(x, indices, axis)
  return native.gather(x, indices, axis or 0)
end

--- Scatter values into an array by index.
-- @tparam Array x Input array.
-- @tparam Array indices Indices at which to scatter.
-- @tparam Array src Source values to scatter.
-- @int[opt=0] axis Axis along which to scatter.
-- @treturn Array Updated array.
function M.scatter(x, indices, src, axis)
  return native.scatter(x, indices, src, axis or 0)
end

--- Scatter-add values into an array by index.
-- @tparam Array x Input array.
-- @tparam Array indices Indices at which to add.
-- @tparam Array src Source values to add.
-- @int[opt=0] axis Axis along which to scatter-add.
-- @treturn Array Updated array.
function M.scatter_add(x, indices, src, axis)
  return native.scatter_add(x, indices, src, axis or 0)
end

--- Scatter with reduction into an array by index.
-- @tparam Array x Input array.
-- @tparam Array indices Indices at which to reduce.
-- @tparam Array src Source values.
-- @string reduce Reduction type ("add", "mul", "mean", "amax", "amin").
-- @int[opt=0] axis Axis along which to scatter-reduce.
-- @treturn Array Updated array.
function M.scatter_reduce(x, indices, src, reduce, axis)
  return native.scatter_reduce(x, indices, src, reduce, axis or 0)
end

--- 1-D linear interpolation.
-- @tparam Array x Points at which to evaluate.
-- @tparam Array xp Known x-coordinates (must be sorted).
-- @tparam Array fp Known y-coordinates.
-- @treturn Array Interpolated values.
function M.interp(x, xp, fp)
  return native.interp(x, xp, fp)
end

--- Return an array of indices (meshgrid-like).
-- @tparam table dimensions Shape of the output grid.
-- @treturn Array Index arrays.
function M.indices(dimensions)
  return native.indices(dimensions)
end

--- Construct open mesh from sequences (for advanced indexing).
-- @tparam table args Sequence of 1-D arrays.
-- @treturn table List of open mesh arrays.
function M.ix_(...)
  return native.ix_(...)
end

return M
