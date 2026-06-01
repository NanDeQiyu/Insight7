--- Complex number operations.
--
-- Provides functions for creating, querying, and manipulating
-- complex-valued arrays.
--
-- @module insight.complex

local native = require("_insight")
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

--- Check if an array has a complex dtype.
-- @tparam Array x Input array.
-- @bool True if the array has a complex dtype (complex64 or complex128).
M.is_complex = _wrap({ "x" }, function(x)
  return native.is_complex(x)
end)

--- Check if an array uses legacy complex storage (last dim = 2).
-- @tparam Array x Input array.
-- @bool True if the array uses the legacy real-imag layout.
M.has_complex_shape = _wrap({ "x" }, function(x)
  return native.has_complex_shape(x)
end)

--- Convert a real array to complex (imaginary part = 0).
-- @tparam Array real Real-valued input array.
-- @tparam[opt] Array imag Imaginary part. If nil, zeros are used.
-- @treturn Array Complex-valued array.
M.to_complex = _wrap({ "real", "imag" }, function(real, imag)
  if imag then
    return native.to_complex(real, imag)
  end
  return native.to_complex(real)
end)

--- View a real array with last dim=2 as complex (zero-copy).
-- @tparam Array x Real input array whose last dimension is 2.
-- @treturn Array Complex view.
M.as_complex = _wrap({ "x" }, function(x)
  return native.as_complex(x)
end)

--- View a complex array as real with last dim=2 (zero-copy).
-- @tparam Array x Complex input array.
-- @treturn Array Real view with doubled last dimension.
M.as_real = _wrap({ "x" }, function(x)
  return native.as_real(x)
end)

--- Extract the real part of a complex array.
-- @tparam Array z Complex input array.
-- @treturn Array Real-valued array.
M.real = _wrap({ "z" }, function(z)
  return native.real(z)
end)

--- Extract the imaginary part of a complex array.
-- @tparam Array z Complex input array.
-- @treturn Array Real-valued array.
M.imag = _wrap({ "z" }, function(z)
  return native.imag(z)
end)

return M
