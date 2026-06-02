-- Signal demod CPU binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_signal_demod.lua

describe("Signal Demod CPU Tests", function()
  local ins
  setup(function()
    ins = require("insight")
  end)

  -- Helper: build a complex FM signal
  local function make_fm(n, fs, fc, fm, dev)
    local re_t = {}
    local im_t = {}
    local phase = 0
    local dt = 1.0 / fs
    for i = 0, n - 1 do
      local message = math.sin(2 * math.pi * fm * i * dt)
      phase = phase + 2 * math.pi * (fc + dev * message) * dt
      re_t[i + 1] = math.cos(phase)
      im_t[i + 1] = math.sin(phase)
    end
    local re = ins.from_table(re_t)
    local im = ins.from_table(im_t)
    return ins.to_complex(re, im)
  end

  it("fm_demod", function()
    local x = make_fm(512, 256.0, 20.0, 2.0, 5.0)
    local result = ins.signal.fm_demod(x)
    assert.is_not_nil(result)
    assert.is_true(result.numel > 0)
  end)

  it("fm_demod_output_length", function()
    local x = make_fm(256, 256.0, 20.0, 2.0, 5.0)
    local result = ins.signal.fm_demod(x)
    assert.is_not_nil(result)
    -- fm_demod uses diff, so output is 1 shorter along axis
    assert.are.equal(255, result.numel)
  end)

  it("fm_demod_short_signal", function()
    local x = make_fm(128, 256.0, 10.0, 1.0, 3.0)
    local result = ins.signal.fm_demod(x)
    assert.is_not_nil(result)
    assert.are.equal(127, result.numel)
  end)
end)
