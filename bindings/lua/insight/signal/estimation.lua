--- State estimation functions.
-- Provides the KalmanFilter class for linear dynamic system estimation.
-- @module insight.signal.estimation

local native = require("_insight")
local sig = native.signal

local M = {}

local _wrap = require("insight._wrap")

--- KalmanFilter class for linear dynamic system state estimation.
-- The Kalman filter is initialized with the state dimension (dim_x)
-- and measurement dimension (dim_z).
-- @type KalmanFilter
M.KalmanFilter = sig.KalmanFilter

return M
