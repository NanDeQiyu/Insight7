--- Peak finding functions.
-- Provides functions to find local extrema (maxima and minima)
-- in signal arrays.
-- @module insight.signal.peak_finding

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

--- Find indices of local extrema.
-- @tparam Array x Input signal array.
-- @int order Number of points on each side to compare.
-- @treturn Array Indices of local extrema.
M.argrelextrema = _wrap({ "x", "order" }, function(x, order)
  return sig.argrelextrema(x, order)
end)

--- Find indices of local maxima.
-- @tparam Array x Input signal array.
-- @int order Number of points on each side to compare.
-- @treturn Array Indices of local maxima.
M.argrelmax = _wrap({ "x", "order" }, function(x, order)
  return sig.argrelmax(x, order)
end)

--- Find indices of local minima.
-- @tparam Array x Input signal array.
-- @int order Number of points on each side to compare.
-- @treturn Array Indices of local minima.
M.argrelmin = _wrap({ "x", "order" }, function(x, order)
  return sig.argrelmin(x, order)
end)

return M
