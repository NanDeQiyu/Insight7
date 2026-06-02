-- Signal Waveforms CPU binding tests.
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;bindings/lua/?.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_signal_waveforms.lua

describe("Signal Waveforms CPU Tests", function()
  local ins
  setup(function()
    ins = require("insight")
  end)

  -- ========================================================================
  -- Sawtooth (3 tests)
  -- ========================================================================

  it("sawtooth_basic", function()
    local t = ins.from_table({ 0.0, math.pi, 2.0 * math.pi - 1e-10 })
    local y = ins.signal.sawtooth(t, 1.0)
    assert.is_not_nil(y)
    assert.are.equal(3, y.numel)
    assert.near(-1.0, y:get(0), 1e-6)
    assert.near(0.0, y:get(1), 1e-6)
    assert.near(1.0, y:get(2), 1e-6)
  end)

  it("sawtooth_triangle", function()
    local t = ins.from_table({ 0.0, math.pi / 2.0, math.pi })
    local y = ins.signal.sawtooth(t, 0.5)
    assert.is_not_nil(y)
    assert.are.equal(3, y.numel)
    assert.near(-1.0, y:get(0), 1e-6)
    assert.near(0.0, y:get(1), 1e-6)
    assert.near(1.0, y:get(2), 1e-6)
  end)

  it("sawtooth_periodicity", function()
    local t = ins.from_table({ 0.5, 0.5 + 2.0 * math.pi, 0.5 + 4.0 * math.pi })
    local y = ins.signal.sawtooth(t, 1.0)
    assert.is_not_nil(y)
    assert.are.equal(3, y.numel)
    assert.near(y:get(0), y:get(1), 1e-6)
    assert.near(y:get(0), y:get(2), 1e-6)
  end)

  -- ========================================================================
  -- Square (2 tests)
  -- ========================================================================

  it("square_basic", function()
    local t = ins.from_table({ 0.0, math.pi, 2.0 * math.pi - 1e-10 })
    local y = ins.signal.square(t, 0.5)
    assert.is_not_nil(y)
    assert.are.equal(3, y.numel)
    assert.near(1.0, y:get(0), 1e-6)
    assert.near(-1.0, y:get(1), 1e-6)
    assert.near(-1.0, y:get(2), 1e-6)
  end)

  it("square_periodicity", function()
    local t = ins.from_table({ 1.0, 1.0 + 2.0 * math.pi, 1.0 + 4.0 * math.pi })
    local y = ins.signal.square(t, 0.5)
    assert.is_not_nil(y)
    assert.are.equal(3, y.numel)
    assert.near(y:get(0), y:get(1), 1e-6)
    assert.near(y:get(0), y:get(2), 1e-6)
  end)

  -- ========================================================================
  -- Gausspulse (2 tests)
  -- ========================================================================

  it("gausspulse_peak", function()
    local t = ins.from_table({ 0.0 })
    local y = ins.signal.gausspulse(t, 0, 1.0)
    assert.is_not_nil(y)
    assert.are.equal(1, y.numel)
    assert.near(1.0, y:get(0), 1e-6)
  end)

  it("gausspulse_output", function()
    local t = ins.from_table({ 0.0, 0.5, 1.0 })
    local y = ins.signal.gausspulse(t, 0, 1.0)
    assert.is_not_nil(y)
    assert.are.equal(3, y.numel)
    -- Peak at t=0
    assert.near(1.0, y:get(0), 1e-6)
    -- All values should be valid numbers
    for i = 0, 2 do
      assert.is_false(y:get(i) ~= y:get(i)) -- NaN check
    end
  end)

  -- ========================================================================
  -- Chirp (2 tests)
  -- ========================================================================

  it("chirp_linear", function()
    local t = ins.from_table({ 0.0 })
    local y = ins.signal.chirp(t, 6.0, 1.0, 10.0, "linear", 0)
    assert.is_not_nil(y)
    assert.are.equal(1, y.numel)
    assert.near(1.0, y:get(0), 1e-6)
  end)

  it("chirp_with_phase", function()
    local t = ins.from_table({ 0.0 })
    local y = ins.signal.chirp(t, 6.0, 1.0, 10.0, "linear", math.pi / 2.0)
    assert.is_not_nil(y)
    assert.are.equal(1, y.numel)
    assert.near(0.0, y:get(0), 1e-6)
  end)

  -- ========================================================================
  -- Unit Impulse (1 test)
  -- ========================================================================

  it("unit_impulse_center", function()
    local y = ins.signal.unit_impulse({ 11 })
    assert.is_not_nil(y)
    assert.are.equal(11, y.numel)
    assert.near(1.0, y:get(5), 1e-10)
    assert.near(0.0, y:get(0), 1e-10)
    assert.near(0.0, y:get(10), 1e-10)
  end)
end)
