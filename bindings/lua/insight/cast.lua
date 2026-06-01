--- Type casting operations.
--
-- Provides functions for converting arrays between different data types.
--
-- @module insight.cast

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

--- Cast an array to a different data type.
-- @tparam Array x Input array.
-- @tparam DType dtype Target data type (e.g. ins.float64).
-- @treturn Array Array with the new data type.
M.cast = _wrap({ "x", "dtype" }, function(x, dtype)
  return native.cast(x, dtype)
end)

return M
