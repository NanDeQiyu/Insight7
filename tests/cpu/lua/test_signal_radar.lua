-- Signal radar CPU binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;bindings/lua/?.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_signal_radar.lua

describe("Signal Radar CPU Tests", function()
  local ins
  setup(function()
    ins = require("insight")
  end)

  it("pulse_compression", function()
    local n = 64
    local tpl_t = {}
    local row = {}
    for i = 1, n do
      tpl_t[i] = math.sin(2 * math.pi * i / n)
      row[i] = tpl_t[i]
    end
    for i = n + 1, 2 * n do
      row[i] = 0.0
    end
    -- pulse_compression requires 2D input [num_pulses, samples_per_pulse]
    local tpl = ins.from_table(tpl_t)
    local sig = ins.from_table({ row })
    local result = ins.signal.pulse_compression(sig, tpl)
    assert.is_not_nil(result)
    assert.is_true(result.numel > 0)
  end)

  it("pulse_doppler", function()
    local rows = {}
    for r = 1, 16 do
      local row = {}
      for c = 1, 32 do
        row[c] = math.sin(2 * math.pi * r * c / 512)
      end
      rows[r] = row
    end
    local data = ins.from_table(rows)
    local result = ins.signal.pulse_doppler(data)
    assert.is_not_nil(result)
    assert.is_true(result.numel > 0)
  end)

  it("cfar_alpha", function()
    local alpha = ins.signal.cfar_alpha(1e-3, 20)
    assert.is_not_nil(alpha)
    assert.is_true(alpha > 0)
  end)

  it("cfar_alpha_different_pfa", function()
    local alpha1 = ins.signal.cfar_alpha(1e-2, 20)
    local alpha2 = ins.signal.cfar_alpha(1e-6, 20)
    -- Larger pfa => larger alpha
    assert.is_true(alpha1 < alpha2)
  end)

  -- Note: ca_cfar requires std::vector<int> args which need special binding.
  -- Note: mvdr requires well-conditioned covariance (may fail with certain inputs).
  -- Note: ambgfun requires complex-valued input.

  it("mvdr_square_input", function()
    -- Use square input to avoid singular covariance issues
    local n = 4
    local rows = {}
    for r = 1, n do
      local row = {}
      for c = 1, n do
        row[c] = math.sin(2 * math.pi * r * c / (n * n))
      end
      rows[r] = row
    end
    local x = ins.from_table(rows)
    local sv = ins.from_table({ 1.0, 1.0, 1.0, 1.0 })
    local result = ins.signal.mvdr(x, sv)
    assert.is_not_nil(result)
    assert.is_true(result.numel > 0)
  end)
end)
