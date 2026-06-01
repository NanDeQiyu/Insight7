-- Signal wavelets CPU binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_signal_wavelets.lua

describe("Signal Wavelets CPU Tests", function()
  local ins
  setup(function()
    ins = require("insight")
  end)

  it("morlet", function()
    local w = ins.signal.morlet(5.0)
    assert.is_not_nil(w)
    assert.is_true(w.numel > 0)
  end)

  it("morlet_high_freq", function()
    local w = ins.signal.morlet(10.0)
    assert.is_not_nil(w)
    assert.is_true(w.numel > 0)
  end)

  it("ricker", function()
    local w = ins.signal.ricker(100, 4.0)
    assert.is_not_nil(w)
    assert.are.equal(100, w.numel)
  end)

  it("ricker_narrow", function()
    local w = ins.signal.ricker(50, 2.0)
    assert.is_not_nil(w)
    assert.are.equal(50, w.numel)
  end)

  it("ricker_wide", function()
    local w = ins.signal.ricker(200, 10.0)
    assert.is_not_nil(w)
    assert.are.equal(200, w.numel)
  end)

  it("morlet2", function()
    local w = ins.signal.morlet2(5.0, 1.0)
    assert.is_not_nil(w)
    assert.is_true(w.numel > 0)
  end)

  it("morlet2_scaled", function()
    local w = ins.signal.morlet2(5.0, 2.0)
    assert.is_not_nil(w)
    assert.is_true(w.numel > 0)
  end)
end)
