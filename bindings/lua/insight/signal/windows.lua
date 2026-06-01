--- Window functions for signal processing.
-- Provides standard window functions used in spectral analysis,
-- filter design, and other signal processing tasks.
-- @module insight.signal.windows

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

--- Return a boxcar (rectangular) window.
-- @int M Number of points in the output window.
-- @treturn Array Boxcar window of length M.
M.boxcar = _wrap({ "M" }, function(M)
  return sig.boxcar(M)
end)

--- Return a triangular window.
-- @int M Number of points in the output window.
-- @treturn Array Triangular window of length M.
M.triang = _wrap({ "M" }, function(M)
  return sig.triang(M)
end)

--- Return a Parzen window.
-- @int M Number of points in the output window.
-- @treturn Array Parzen window of length M.
M.parzen = _wrap({ "M" }, function(M)
  return sig.parzen(M)
end)

--- Return a Bohman window.
-- @int M Number of points in the output window.
-- @treturn Array Bohman window of length M.
M.bohman = _wrap({ "M" }, function(M)
  return sig.bohman(M)
end)

--- Return a Bartlett window.
-- @int M Number of points in the output window.
-- @treturn Array Bartlett window of length M.
M.bartlett = _wrap({ "M" }, function(M)
  return sig.bartlett(M)
end)

--- Return a cosine window.
-- @int M Number of points in the output window.
-- @treturn Array Cosine window of length M.
M.cosine = _wrap({ "M" }, function(M)
  return sig.cosine(M)
end)

--- Return an exponential (Poisson) window.
-- @int M Number of points in the output window.
-- @number[opt=1.0] tau Decay parameter.
-- @number[opt] center Center of the window, defaults to (M-1)/2.
-- @treturn Array Exponential window of length M.
M.exponential = _wrap({ "M", "tau", "center" }, function(M, tau, center)
  tau = tau or 1.0
  if center == nil then
    center = (M - 1) / 2.0
  end
  return sig.exponential(M, tau, center)
end)

--- Return a Blackman window.
-- @int M Number of points in the output window.
-- @treturn Array Blackman window of length M.
M.blackman = _wrap({ "M" }, function(M)
  return sig.blackman(M)
end)

--- Return a Nuttall minimum 4-term Blackman-Harris window.
-- @int M Number of points in the output window.
-- @treturn Array Nuttall window of length M.
M.nuttall = _wrap({ "M" }, function(M)
  return sig.nuttall(M)
end)

--- Return a Blackman-Harris window.
-- @int M Number of points in the output window.
-- @treturn Array Blackman-Harris window of length M.
M.blackmanharris = _wrap({ "M" }, function(M)
  return sig.blackmanharris(M)
end)

--- Return a flat-top window.
-- @int M Number of points in the output window.
-- @treturn Array Flat-top window of length M.
M.flattop = _wrap({ "M" }, function(M)
  return sig.flattop(M)
end)

--- Return a Hann window.
-- @int M Number of points in the output window.
-- @treturn Array Hann window of length M.
M.hann = _wrap({ "M" }, function(M)
  return sig.hann(M)
end)

--- Return a generalized Hamming window.
-- @int M Number of points in the output window.
-- @number alpha The window coefficient.
-- @treturn Array Generalized Hamming window of length M.
M.general_hamming = _wrap({ "M", "alpha" }, function(M, alpha)
  return sig.general_hamming(M, alpha)
end)

--- Return a Hamming window.
-- @int M Number of points in the output window.
-- @treturn Array Hamming window of length M.
M.hamming = _wrap({ "M" }, function(M)
  return sig.hamming(M)
end)

--- Return a Tukey (tapered cosine) window.
-- @int M Number of points in the output window.
-- @number[opt=0.5] alpha Shape parameter.
-- @treturn Array Tukey window of length M.
M.tukey = _wrap({ "M", "alpha" }, function(M, alpha)
  return sig.tukey(M, alpha or 0.5)
end)

--- Return a Bartlett-Hann window.
-- @int M Number of points in the output window.
-- @treturn Array Bartlett-Hann window of length M.
M.barthann = _wrap({ "M" }, function(M)
  return sig.barthann(M)
end)

--- Return a Gaussian window.
-- @int M Number of points in the output window.
-- @number std The standard deviation.
-- @treturn Array Gaussian window of length M.
M.gaussian = _wrap({ "M", "std" }, function(M, std)
  return sig.gaussian(M, std)
end)

--- Return a generalized Gaussian window.
-- @int M Number of points in the output window.
-- @number p Shape parameter.
-- @number sig Standard deviation.
-- @treturn Array Generalized Gaussian window of length M.
M.general_gaussian = _wrap({ "M", "p", "sig_val" }, function(M, p, sig_val)
  return sig.general_gaussian(M, p, sig_val)
end)

--- Return a Kaiser window.
-- @int M Number of points in the output window.
-- @number beta Shape parameter.
-- @treturn Array Kaiser window of length M.
M.kaiser = _wrap({ "M", "beta" }, function(M, beta)
  return sig.kaiser(M, beta)
end)

--- Return a Dolph-Chebyshev window.
-- @int M Number of points in the output window.
-- @number at Attenuation in dB.
-- @treturn Array Dolph-Chebyshev window of length M.
M.chebwin = _wrap({ "M", "at" }, function(M, at)
  return sig.chebwin(M, at)
end)

--- Return a Taylor window.
-- @int M Number of points in the output window.
-- @int[opt=11] nbar Number of nearly constant level sidelobes.
-- @number[opt=-30] sll Desired sidelobe level in dB.
-- @bool[opt=true] norm Whether to normalize.
-- @treturn Array Taylor window of length M.
M.taylor = _wrap({ "M", "nbar", "sll", "norm" }, function(M, nbar, sll, norm)
  return sig.taylor(M, nbar or 11, sll or -30, norm == nil and true or norm)
end)

--- Return a window with arbitrary cosine coefficients.
-- @int M Number of points in the output window.
-- @tparam table coeffs Sequence of cosine coefficients.
-- @treturn Array Window of length M.
M.general_cosine = _wrap({ "M", "coeffs" }, function(M, coeffs)
  return sig.general_cosine(M, coeffs)
end)

--- Return a window of a given type and length.
-- @string window Window type string (e.g., 'hann', 'hamming', 'kaiser').
-- @int Nx Number of points in the output window.
-- @bool[opt=true] fftbins If true, create a periodic window for FFT.
-- @treturn Array Window array of length Nx.
M.get_window = _wrap({ "window", "Nx", "fftbins" }, function(window, Nx, fftbins)
  return sig.get_window(window, Nx, fftbins == nil and true or fftbins)
end)

return M
