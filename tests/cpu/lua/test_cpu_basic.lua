-- CPU binding tests — 35 tests aligned with Python and Julia
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_cpu_basic.lua

describe("CPU Binding Tests", function()
  local ins
  setup(function()
    ins = require("_insight")
    ins.set_device(ins.CPUPlace())
  end)

  -- ========================================================================
  -- Creation (7 tests)
  -- ========================================================================

  it("zeros", function()
    local a = ins.zeros({ 2, 3 }, ins.float32)
    assert.is_not_nil(a)
    assert.are.equal(6, a.numel)
  end)

  it("ones", function()
    local a = ins.ones({ 2, 3 }, ins.float32)
    assert.is_not_nil(a)
    assert.are.equal(6, a.numel)
  end)

  it("full", function()
    local a = ins.full({ 2, 3 }, 7.0, ins.float32)
    assert.is_not_nil(a)
    assert.are.equal(6, a.numel)
  end)

  it("eye", function()
    local a = ins.eye(3)
    assert.is_not_nil(a)
    assert.are.equal(9, a.numel)
  end)

  it("arange", function()
    local a = ins.arange(10, ins.float32)
    assert.is_not_nil(a)
    assert.are.equal(10, a.numel)
  end)

  it("linspace", function()
    local a = ins.linspace(0.0, 1.0, 5, ins.float64)
    assert.is_not_nil(a)
    assert.are.equal(5, a.numel)
  end)

  it("from_table", function()
    local a = ins.from_table({ 1.5, 2.5, 3.5 })
    assert.is_not_nil(a)
    assert.are.equal(3, a.numel)
  end)

  -- ========================================================================
  -- Arithmetic (5 tests)
  -- ========================================================================

  it("add", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local b = ins.from_table({ 4.0, 5.0, 6.0 })
    local c = ins.add(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("sub", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0 })
    local b = ins.from_table({ 1.0, 2.0, 3.0 })
    local c = ins.sub(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("mul", function()
    local a = ins.from_table({ 2.0, 3.0, 4.0 })
    local b = ins.from_table({ 5.0, 6.0, 7.0 })
    local c = ins.mul(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("div", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0 })
    local b = ins.from_table({ 2.0, 4.0, 5.0 })
    local c = ins.div(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("neg", function()
    local a = ins.from_table({ 1.0, -2.0, 3.0 })
    local c = ins.negative(a)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  -- ========================================================================
  -- Unary (5 tests)
  -- ========================================================================

  it("abs", function()
    local a = ins.from_table({ -3.0, -1.0, 0.0, 2.0, 4.0 })
    local b = ins.abs(a)
    assert.is_not_nil(b)
    assert.are.equal(5, b.numel)
  end)

  it("sqrt", function()
    local a = ins.from_table({ 1.0, 4.0, 9.0, 16.0 })
    local b = ins.sqrt(a)
    assert.is_not_nil(b)
    assert.are.equal(4, b.numel)
  end)

  it("exp", function()
    local a = ins.from_table({ 0.0, 1.0, 2.0 })
    local b = ins.exp(a)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  it("log", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local b = ins.log(a)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  it("sin", function()
    local a = ins.from_table({ 0.0, 0.5, 1.0 })
    local b = ins.sin(a)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  -- ========================================================================
  -- Reduction (5 tests)
  -- ========================================================================

  it("sum", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0 })
    local s = ins.sum(a)
    assert.is_not_nil(s)
    assert.are.equal(1, s.numel)
  end)

  it("mean", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0 })
    local m = ins.mean(a)
    assert.is_not_nil(m)
    assert.are.equal(1, m.numel)
  end)

  it("max", function()
    local a = ins.from_table({ 3.0, 1.0, 4.0, 1.0, 5.0 })
    local m = ins.max(a)
    assert.is_not_nil(m)
    assert.are.equal(1, m.numel)
  end)

  it("min", function()
    local a = ins.from_table({ 3.0, 1.0, 4.0, 1.0, 5.0 })
    local m = ins.min(a)
    assert.is_not_nil(m)
    assert.are.equal(1, m.numel)
  end)

  it("argmax", function()
    local a = ins.from_table({ 1.0, 5.0, 3.0, 2.0 })
    local m = ins.argmax(a)
    assert.is_not_nil(m)
    assert.are.equal(1, m.numel)
  end)

  -- ========================================================================
  -- Comparison (4 tests)
  -- ========================================================================

  it("equal", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local b = ins.from_table({ 1.0, 0.0, 3.0 })
    local c = ins.equal(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("greater", function()
    local a = ins.from_table({ 3.0, 1.0, 5.0 })
    local b = ins.from_table({ 2.0, 4.0, 5.0 })
    local c = ins.greater(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("less", function()
    local a = ins.from_table({ 1.0, 5.0, 3.0 })
    local b = ins.from_table({ 2.0, 4.0, 5.0 })
    local c = ins.less(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("logical_and", function()
    local a = ins.from_table({ 1, 1, 0 }, ins.bool)
    local b = ins.from_table({ 1, 0, 0 }, ins.bool)
    local c = ins.logical_and(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  -- ========================================================================
  -- Manipulation (3 tests)
  -- ========================================================================

  it("flatten", function()
    local a = ins.zeros({ 2, 3 }, ins.float64)
    local b = ins.flatten(a)
    assert.is_not_nil(b)
    assert.are.equal(6, b.numel)
  end)

  it("concat", function()
    local a = ins.from_table({ 1.0, 2.0 })
    local b = ins.from_table({ 3.0, 4.0 })
    local c = ins.concat({ a, b })
    assert.is_not_nil(c)
    assert.are.equal(4, c.numel)
  end)

  it("flip", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local b = ins.flip(a)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  -- ========================================================================
  -- Linalg (3 tests)
  -- ========================================================================

  it("matmul", function()
    local a = ins.from_table({ { 1.0, 2.0 }, { 3.0, 4.0 } })
    local b = ins.from_table({ { 5.0, 6.0 }, { 7.0, 8.0 } })
    local c = ins.matmul(a, b)
    assert.is_not_nil(c)
    assert.are.equal(4, c.numel)
  end)

  it("det", function()
    local a = ins.from_table({ { 1.0, 2.0 }, { 3.0, 4.0 } })
    local d = ins.det(a)
    assert.is_not_nil(d)
    assert.are.equal(1, d.numel)
    assert.near(-2.0, d:item(), 1e-6)
  end)

  it("inv", function()
    local a = ins.from_table({ { 1.0, 2.0 }, { 3.0, 4.0 } })
    local b = ins.inv(a)
    assert.is_not_nil(b)
    assert.are.equal(4, b.numel)
  end)

  -- ========================================================================
  -- FFT (2 tests)
  -- ========================================================================

  it("fft", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0 })
    local y = ins.fft(x)
    assert.is_not_nil(y)
    assert.are.equal(4, y.numel)
  end)

  it("ifft", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0 })
    local y = ins.fft(x)
    local z = ins.ifft(y)
    assert.is_not_nil(z)
    assert.are.equal(4, z.numel)
  end)

  -- ========================================================================
  -- Signal (1 test)
  -- ========================================================================

  it("hann", function()
    local w = ins.signal.hann(16)
    assert.is_not_nil(w)
    assert.are.equal(16, w.numel)
  end)

  -- ========================================================================
  -- GPU not available: must throw
  -- ========================================================================

  it("gpu_throws_without_backend", function()
    if ins.has_device("gpu") then
      pending("GPU available")
    end
    local a = ins.ones({ 3 }, ins.float32)
    assert.has_error(function()
      a:to(ins.GPUPlace(0))
    end)
  end)

  it("set_device_gpu_throws", function()
    if ins.has_device("gpu") then
      pending("GPU available")
    end
    assert.has_error(function()
      ins.set_device(ins.GPUPlace(0))
    end)
  end)
end)
