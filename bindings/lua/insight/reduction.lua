--- Reduction operations.
--
-- Provides functions that reduce array dimensions by aggregating
-- values along one or more axes (sum, mean, max, min, etc.).
--
-- @module insight.reduction

local native = require("_insight")
local M = {}

--- Sum of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to sum. Defaults to all elements.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Sum result.
function M.sum(x, axis, keepdims)
  return native.sum(x, axis, keepdims or false)
end

--- Mean of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute mean.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Mean result.
function M.mean(x, axis, keepdims)
  return native.mean(x, axis, keepdims or false)
end

--- Maximum of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to find maximum.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Maximum result.
function M.max(x, axis, keepdims)
  return native.max(x, axis, keepdims or false)
end

--- Minimum of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to find minimum.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Minimum result.
function M.min(x, axis, keepdims)
  return native.min(x, axis, keepdims or false)
end

--- Product of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute product.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Product result.
function M.prod(x, axis, keepdims)
  return native.prod(x, axis, keepdims or false)
end

--- Indices of maximum values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to find argmax.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Indices of maximum values.
function M.argmax(x, axis, keepdims)
  return native.argmax(x, axis, keepdims or false)
end

--- Indices of minimum values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to find argmin.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Indices of minimum values.
function M.argmin(x, axis, keepdims)
  return native.argmin(x, axis, keepdims or false)
end

--- Test whether any element is true.
-- @tparam Array x Input boolean array.
-- @int[opt] axis Axis along which to test.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Boolean result.
function M.any(x, axis, keepdims)
  return native.any(x, axis, keepdims or false)
end

--- Test whether all elements are true.
-- @tparam Array x Input boolean array.
-- @int[opt] axis Axis along which to test.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Boolean result.
function M.all(x, axis, keepdims)
  return native.all(x, axis, keepdims or false)
end

--- Variance of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute variance.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @int[opt=0] ddof Delta degrees of freedom.
-- @treturn Array Variance result.
function M.var(x, axis, keepdims, ddof)
  return native.var(x, axis, keepdims or false, ddof or 0)
end

--- Standard deviation of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute std.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @int[opt=0] ddof Delta degrees of freedom.
-- @treturn Array Standard deviation result.
function M.std(x, axis, keepdims, ddof)
  return native.std(x, axis, keepdims or false, ddof or 0)
end

--- Cumulative sum along an axis.
-- @tparam Array x Input array.
-- @int axis Axis along which to compute cumulative sum.
-- @treturn Array Cumulative sum result.
function M.cumsum(x, axis)
  return native.cumsum(x, axis)
end

--- Cumulative product along an axis.
-- @tparam Array x Input array.
-- @int axis Axis along which to compute cumulative product.
-- @treturn Array Cumulative product result.
function M.cumprod(x, axis)
  return native.cumprod(x, axis)
end

--- Cumulative maximum along an axis.
-- @tparam Array x Input array.
-- @int axis Axis along which to compute cumulative maximum.
-- @treturn Array Cumulative maximum result.
function M.cummax(x, axis)
  return native.cummax(x, axis)
end

--- Cumulative minimum along an axis.
-- @tparam Array x Input array.
-- @int axis Axis along which to compute cumulative minimum.
-- @treturn Array Cumulative minimum result.
function M.cummin(x, axis)
  return native.cummin(x, axis)
end

--- Standard error of the mean.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute SEM.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @int[opt=0] ddof Delta degrees of freedom.
-- @treturn Array Standard error of the mean.
function M.sem(x, axis, keepdims, ddof)
  return native.sem(x, axis, keepdims or false, ddof or 0)
end

--- Count non-zero elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to count.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Count of non-zero elements.
function M.count_nonzero(x, axis, keepdims)
  return native.count_nonzero(x, axis, keepdims or false)
end

--- Median of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute median.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Median result.
function M.median(x, axis, keepdims)
  return native.median(x, axis, keepdims or false)
end

--- Compute the q-th quantile.
-- @tparam Array x Input array.
-- @number q Quantile to compute, between 0 and 1.
-- @int[opt] axis Axis along which to compute.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Quantile result.
function M.quantile(x, q, axis, keepdims)
  return native.quantile(x, q, axis, keepdims or false)
end

--- Compute the q-th percentile.
-- @tparam Array x Input array.
-- @number q Percentile to compute, between 0 and 100.
-- @int[opt] axis Axis along which to compute.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Percentile result.
function M.percentile(x, q, axis, keepdims)
  return native.percentile(x, q, axis, keepdims or false)
end

--- Sum of array elements, ignoring NaN values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to sum.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array NaN-safe sum result.
function M.nansum(x, axis, keepdims)
  return native.nansum(x, axis, keepdims or false)
end

--- Mean of array elements, ignoring NaN values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute mean.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array NaN-safe mean result.
function M.nanmean(x, axis, keepdims)
  return native.nanmean(x, axis, keepdims or false)
end

--- Maximum of array elements, ignoring NaN values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to find maximum.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array NaN-safe maximum result.
function M.nanmax(x, axis, keepdims)
  return native.nanmax(x, axis, keepdims or false)
end

--- Minimum of array elements, ignoring NaN values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to find minimum.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array NaN-safe minimum result.
function M.nanmin(x, axis, keepdims)
  return native.nanmin(x, axis, keepdims or false)
end

--- Standard deviation, ignoring NaN values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute std.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @int[opt=0] ddof Delta degrees of freedom.
-- @treturn Array NaN-safe standard deviation.
function M.nanstd(x, axis, keepdims, ddof)
  return native.nanstd(x, axis, keepdims or false, ddof or 0)
end

--- Variance, ignoring NaN values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute variance.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @int[opt=0] ddof Delta degrees of freedom.
-- @treturn Array NaN-safe variance.
function M.nanvar(x, axis, keepdims, ddof)
  return native.nanvar(x, axis, keepdims or false, ddof or 0)
end

return M
