--- Signal processing submodule for Insight7.
-- Provides window functions, filtering, convolution, spectral analysis,
-- and other signal processing routines.
-- @module insight.signal
-- @see insight.signal.windows
-- @see insight.signal.filtering

local native = require("_insight")
local sig = native.signal

local M = {}

-- Load sub-modules
local windows = require("insight.signal.windows")
local waveforms = require("insight.signal.waveforms")
local bsplines = require("insight.signal.bsplines")
local filter_design = require("insight.signal.filter_design")
local convolution = require("insight.signal.convolution")
local filtering = require("insight.signal.filtering")
local spectral = require("insight.signal.spectral")
local wavelets = require("insight.signal.wavelets")
local acoustics = require("insight.signal.acoustics")
local demod = require("insight.signal.demod")
local peak_finding = require("insight.signal.peak_finding")
local radar = require("insight.signal.radar")
local estimation = require("insight.signal.estimation")
local io_mod = require("insight.signal.io")

-- Merge all sub-modules
for k, v in pairs(windows) do
  M[k] = v
end
for k, v in pairs(waveforms) do
  M[k] = v
end
for k, v in pairs(bsplines) do
  M[k] = v
end
for k, v in pairs(filter_design) do
  M[k] = v
end
for k, v in pairs(convolution) do
  M[k] = v
end
for k, v in pairs(filtering) do
  M[k] = v
end
for k, v in pairs(spectral) do
  M[k] = v
end
for k, v in pairs(wavelets) do
  M[k] = v
end
for k, v in pairs(acoustics) do
  M[k] = v
end
for k, v in pairs(demod) do
  M[k] = v
end
for k, v in pairs(peak_finding) do
  M[k] = v
end
for k, v in pairs(radar) do
  M[k] = v
end
for k, v in pairs(estimation) do
  M[k] = v
end
for k, v in pairs(io_mod) do
  M[k] = v
end

-- Re-export types from native signal submodule
M.ChirpMethod = sig.ChirpMethod

return M
