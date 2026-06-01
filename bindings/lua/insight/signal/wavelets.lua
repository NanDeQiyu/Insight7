--- Wavelet functions.
-- Provides Morlet, Ricker (Mexican hat), and Morlet2 wavelet functions.
-- @module insight.signal.wavelets

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

--- Compute a Morlet wavelet.
-- @number w Frequency parameter.
-- @treturn Array Complex Morlet wavelet.
M.morlet = _wrap({ "w" }, function(w)
  return sig.morlet(w)
end)

--- Compute a Ricker (Mexican hat) wavelet.
-- @int points Number of points.
-- @number a Width parameter.
-- @treturn Array Ricker wavelet.
M.ricker = _wrap({ "points", "a" }, function(points, a)
  return sig.ricker(points, a)
end)

--- Compute a Morlet2 wavelet.
-- @number w Frequency parameter.
-- @number s Scale parameter.
-- @treturn Array Complex Morlet2 wavelet.
M.morlet2 = _wrap({ "w", "s" }, function(w, s)
  return sig.morlet2(w, s)
end)

return M
