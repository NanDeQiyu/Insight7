--- Array manipulation operations.
--
-- Provides functions for reshaping, transposing, concatenating,
-- splitting, tiling, and other array manipulation routines.
--
-- @module insight.manipulation

local native = require("_insight")
local M = {}

--- Concatenate arrays along an axis.
-- @tparam table arrays List of arrays to concatenate.
-- @int[opt=0] axis Axis along which to concatenate.
-- @treturn Array Concatenated array.
function M.concat(arrays, axis)
  return native.concat(arrays, axis or 0)
end

--- Stack arrays along a new axis.
-- @tparam table arrays List of arrays to stack.
-- @int[opt=0] axis Axis along which to stack.
-- @treturn Array Stacked array.
function M.stack(arrays, axis)
  return native.stack(arrays, axis or 0)
end

--- Split an array into sub-arrays.
-- @tparam Array x Input array.
-- @int sections Number of equal sections to split into.
-- @int[opt=0] axis Axis along which to split.
-- @treturn table List of sub-arrays.
function M.split(x, sections, axis)
  return native.split(x, sections, axis or 0)
end

--- Construct an array by tiling.
-- @tparam Array x Input array.
-- @tparam table reps Number of repetitions per axis.
-- @treturn Array Tiled array.
function M.tile(x, reps)
  return native.tile(x, reps)
end

--- Repeat elements of an array.
-- @tparam Array x Input array.
-- @int repeats Number of repetitions.
-- @int[opt] axis Axis along which to repeat. If nil, array is flattened first.
-- @treturn Array Repeated array.
function M.repeat_elements(x, repeats, axis)
  return native["repeat"](x, repeats, axis)
end

--- Reverse array along given axis.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to flip. If nil, flip all axes.
-- @treturn Array Flipped array.
function M.flip(x, axis)
  return native.flip(x, axis)
end

--- Pad an array.
-- @tparam Array x Input array.
-- @tparam table pad_width Padding widths per axis {before, after, ...}.
-- @number[opt=0] constant_value Value used for padding.
-- @treturn Array Padded array.
function M.pad(x, pad_width, constant_value)
  return native.pad(x, pad_width, constant_value or 0.0)
end

--- Flatten an array to 1-D.
-- @tparam Array x Input array.
-- @treturn Array Flattened array.
function M.flatten(x)
  return native.flatten(x)
end

--- Flatten an array to 1-D (contiguous).
-- @tparam Array x Input array.
-- @treturn Array Flattened contiguous array.
function M.ravel(x)
  return native.ravel(x)
end

--- Roll array elements along an axis.
-- @tparam Array x Input array.
-- @int shift Number of positions to shift.
-- @int[opt] axis Axis along which to roll. If nil, flattened first.
-- @treturn Array Rolled array.
function M.roll(x, shift, axis)
  return native.roll(x, shift, axis)
end

--- Permute the axes of an array.
-- @tparam Array x Input array.
-- @tparam table axes Permutation of axis indices.
-- @treturn Array Array with permuted axes.
function M.permute(x, axes)
  return native.permute(x, axes)
end

--- Swap two axes of an array.
-- @tparam Array x Input array.
-- @int axis1 First axis to swap.
-- @int axis2 Second axis to swap.
-- @treturn Array Array with swapped axes.
function M.swapaxes(x, axis1, axis2)
  return native.swapaxes(x, axis1, axis2)
end

--- Move axes of an array to new positions.
-- @tparam Array x Input array.
-- @int source Original position of the axis.
-- @int destination Destination position of the axis.
-- @treturn Array Array with moved axes.
function M.moveaxis(x, source, destination)
  return native.moveaxis(x, source, destination)
end

--- Flip array in left/right direction (along axis 1).
-- @tparam Array x Input array.
-- @treturn Array Left-right flipped array.
function M.fliplr(x)
  return native.fliplr(x)
end

--- Flip array in up/down direction (along axis 0).
-- @tparam Array x Input array.
-- @treturn Array Up-down flipped array.
function M.flipud(x)
  return native.flipud(x)
end

--- Rotate array by 90 degrees in the plane specified by axes.
-- @tparam Array x Input array.
-- @int[opt=1] k Number of 90-degree rotations.
-- @tparam[opt={0,1}] table axes The two axes that define the plane of rotation.
-- @treturn Array Rotated array.
function M.rot90(x, k, axes)
  return native.rot90(x, k or 1, axes or { 0, 1 })
end

--- Extract diagonal or construct a diagonal array.
-- @tparam Array x Input array.
-- @int[opt=0] k Diagonal offset (0 for main, >0 for upper, <0 for lower).
-- @treturn Array Diagonal array.
function M.diag(x, k)
  return native.diag(x, k or 0)
end

--- Return specified diagonals.
-- @tparam Array x Input array.
-- @int[opt=0] offset Diagonal offset.
-- @int[opt=0] axis1 First axis for the 2-D sub-arrays.
-- @int[opt=1] axis2 Second axis for the 2-D sub-arrays.
-- @treturn Array Diagonal values.
function M.diagonal(x, offset, axis1, axis2)
  return native.diagonal(x, offset or 0, axis1 or 0, axis2 or 1)
end

--- Lower triangular part of an array.
-- @tparam Array x Input array.
-- @int[opt=0] k Diagonal above which to zero elements.
-- @treturn Array Lower triangular array.
function M.tril(x, k)
  return native.tril(x, k or 0)
end

--- Upper triangular part of an array.
-- @tparam Array x Input array.
-- @int[opt=0] k Diagonal below which to zero elements.
-- @treturn Array Upper triangular array.
function M.triu(x, k)
  return native.triu(x, k or 0)
end

--- N-th discrete difference along an axis.
-- @tparam Array x Input array.
-- @int[opt=1] n Number of times to differ.
-- @int[opt=-1] axis Axis along which to compute the difference.
-- @treturn Array Difference array.
function M.diff(x, n, axis)
  return native.diff(x, n or 1, axis or -1)
end

return M
