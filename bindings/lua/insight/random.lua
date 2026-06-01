--- Random number generation.
--
-- Provides functions for generating random arrays from various
-- distributions, plus seed control.
--
-- @module insight.random

local native = require("_insight")
local M = {}

--- Generate uniform random values in [0, 1).
-- @tparam table shape Array dimensions, e.g. {2, 3}.
-- @tparam[opt] DType dtype Data type (default float32).
-- @tparam[opt] Place place Device placement.
-- @treturn Array Random array.
function M.rand(shape, dtype, place)
  return native.rand(shape, dtype, place)
end

--- Generate standard normal random values.
-- @tparam table shape Array dimensions, e.g. {2, 3}.
-- @tparam[opt] DType dtype Data type (default float32).
-- @tparam[opt] Place place Device placement.
-- @treturn Array Random array from N(0,1).
function M.randn(shape, dtype, place)
  return native.randn(shape, dtype, place)
end

--- Generate random integers in [low, high).
-- @int low Lower bound (inclusive).
-- @int high Upper bound (exclusive).
-- @tparam table shape Array dimensions.
-- @tparam[opt] DType dtype Data type (default int64).
-- @tparam[opt] Place place Device placement.
-- @treturn Array Random integer array.
function M.randint(low, high, shape, dtype, place)
  return native.randint(low, high, shape, dtype, place)
end

--- Generate a random permutation of integers [0, n).
-- @int n Upper bound (exclusive).
-- @tparam[opt] DType dtype Data type (default int64).
-- @tparam[opt] Place place Device placement.
-- @treturn Array Permutation array.
function M.randperm(n, dtype, place)
  return native.randperm(n, dtype, place)
end

--- Set the random seed.
-- @int s Seed value.
function M.seed(s)
  native.seed(s)
end

--- Get the current random seed.
-- @int Current seed value.
function M.get_seed()
  return native.get_seed()
end

--- Generate random values with the same shape and dtype as input.
-- @tparam Array arr Reference array.
-- @treturn Array Uniform random array with same shape/dtype.
function M.rand_like(arr)
  return native.rand_like(arr)
end

--- Generate standard normal values with the same shape and dtype as input.
-- @tparam Array arr Reference array.
-- @treturn Array Normal random array with same shape/dtype.
function M.randn_like(arr)
  return native.randn_like(arr)
end

--- Generate exponential random values.
-- @number scale Scale parameter (1/rate).
-- @tparam table shape Array dimensions.
-- @tparam[opt] DType dtype Data type (default float32).
-- @tparam[opt] Place place Device placement.
-- @treturn Array Exponential random array.
function M.exponential(scale, shape, dtype, place)
  return native.exponential(scale, shape, dtype, place)
end

--- Generate gamma random values.
-- @number alpha Shape parameter.
-- @tparam table shape Array dimensions.
-- @tparam[opt] DType dtype Data type (default float32).
-- @tparam[opt] Place place Device placement.
-- @treturn Array Gamma random array.
function M.gamma(alpha, shape, dtype, place)
  return native.gamma(alpha, shape, dtype, place)
end

--- Generate beta random values.
-- @number a Alpha shape parameter.
-- @number b Beta shape parameter.
-- @tparam table shape Array dimensions.
-- @tparam[opt] DType dtype Data type (default float32).
-- @tparam[opt] Place place Device placement.
-- @treturn Array Beta random array.
function M.beta(a, b, shape, dtype, place)
  return native.beta(a, b, shape, dtype, place)
end

--- Generate binomial random values.
-- @int n Number of trials.
-- @number p Probability of success.
-- @tparam table shape Array dimensions.
-- @tparam[opt] DType dtype Data type (default float64).
-- @tparam[opt] Place place Device placement.
-- @treturn Array Binomial random array.
function M.binomial(n, p, shape, dtype, place)
  return native.binomial(n, p, shape, dtype, place)
end

--- Generate Poisson random values.
-- @number lam Expected number of events.
-- @tparam table shape Array dimensions.
-- @tparam[opt] DType dtype Data type (default float64).
-- @tparam[opt] Place place Device placement.
-- @treturn Array Poisson random array.
function M.poisson(lam, shape, dtype, place)
  return native.poisson(lam, shape, dtype, place)
end

return M
