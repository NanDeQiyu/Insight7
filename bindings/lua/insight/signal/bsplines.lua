--- B-spline basis functions.
-- Provides Gaussian, cubic, and quadratic B-spline functions.
-- @module insight.signal.bsplines

local native = require("_insight")
local sig = native.signal

local M = {}

local function _wrap(names, fn)
  return function(...)
    if select("#", ...) == 1 and type(select(1, ...)) == "table" then
      local t = select(1, ...)
      local has_names = false
      for k, _ in pairs(t) do
        if type(k) ~= "number" then
          has_names = true
          break
        end
      end
      if has_names then
        local pos = {}
        for i, name in ipairs(names) do
          pos[i] = t[name]
          if pos[i] == nil then
            pos[i] = t[i]
          end
        end
        return fn(table.unpack(pos, 1, #names))
      end
    end
    return fn(...)
  end
end

--- Return a Gaussian B-spline basis function.
-- @int n Order of the spline.
-- @number x Evaluation points.
-- @treturn Array B-spline values.
M.gauss_spline = _wrap({ "n", "x" }, function(n, x)
  return sig.gauss_spline(n, x)
end)

--- Return a cubic B-spline.
-- @number x Evaluation points.
-- @treturn Array Cubic B-spline values.
M.cubic = _wrap({ "x" }, function(x)
  return sig.cubic(x)
end)

--- Return a quadratic B-spline.
-- @number x Evaluation points.
-- @treturn Array Quadratic B-spline values.
M.quadratic = _wrap({ "x" }, function(x)
  return sig.quadratic(x)
end)

return M
