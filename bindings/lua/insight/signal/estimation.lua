--- State estimation functions.
-- Provides the KalmanFilter class for linear dynamic system estimation.
-- @module insight.signal.estimation

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

--- KalmanFilter class for linear dynamic system state estimation.
-- The Kalman filter is initialized with the state dimension (dim_x)
-- and measurement dimension (dim_z).
-- @type KalmanFilter
M.KalmanFilter = sig.KalmanFilter

return M
