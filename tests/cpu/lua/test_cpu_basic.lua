-- CPU binding tests — 35 tests aligned with Python and Julia
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_cpu_basic.lua

describe("CPU Binding Tests", function()
    local ins
    setup(function()
        ins = require("_insight")
    end)

    -- ========================================================================
    -- Creation (7 tests)
    -- ========================================================================

    it("zeros", function()
        local a = ins.zeros({2, 3}, ins.float32)
        assert.are.equal(6, a.numel)
    end)

    it("ones", function()
        local a = ins.ones({2, 3}, ins.float32)
        assert.are.equal(6, a.numel)
    end)

    it("full", function()
        local a = ins.full({2, 3}, 7.0, ins.float32)
        assert.are.equal(6, a.numel)
        local v = a:get({1, 1})
        assert.is_true(math.abs(v - 7.0) < 1e-5)
    end)

    it("eye", function()
        local a = ins.eye(3)
        assert.are.equal(9, a.numel)
    end)

    it("arange", function()
        local a = ins.arange(10, ins.float32)
        assert.are.equal(10, a.numel)
    end)

    it("linspace", function()
        local a = ins.linspace(0.0, 1.0, 5, ins.float64)
        assert.are.equal(5, a.numel)
    end)

    it("from_table", function()
        local data = {1.5, 2.5, 3.5}
        local a = ins.from_table(data)
        assert.are.equal(3, a.numel)
        local v = a:get(1)
        assert.is_true(math.abs(v - 1.5) < 1e-5)
    end)

    -- ========================================================================
    -- Arithmetic (5 tests)
    -- ========================================================================

    it("add", function()
        local a = ins.from_table({1.0, 2.0, 3.0})
        local b = ins.from_table({4.0, 5.0, 6.0})
        local c = ins.add(a, b)
        assert.are.equal(3, c.numel)
        local v = c:get(1)
        assert.is_true(math.abs(v - 5.0) < 1e-5)
    end)

    it("sub", function()
        local a = ins.from_table({10.0, 20.0, 30.0})
        local b = ins.from_table({1.0, 2.0, 3.0})
        local c = ins.sub(a, b)
        assert.are.equal(3, c.numel)
        local v = c:get(1)
        assert.is_true(math.abs(v - 9.0) < 1e-5)
    end)

    it("mul", function()
        local a = ins.from_table({2.0, 3.0, 4.0})
        local b = ins.from_table({5.0, 6.0, 7.0})
        local c = ins.mul(a, b)
        assert.are.equal(3, c.numel)
        local v = c:get(1)
        assert.is_true(math.abs(v - 10.0) < 1e-5)
    end)

    it("div", function()
        local a = ins.from_table({10.0, 20.0, 30.0})
        local b = ins.from_table({2.0, 4.0, 5.0})
        local c = ins.div(a, b)
        assert.are.equal(3, c.numel)
        local v = c:get(1)
        assert.is_true(math.abs(v - 5.0) < 1e-5)
    end)

    it("neg", function()
        local a = ins.from_table({1.0, -2.0, 3.0})
        local c = ins.negative(a)
        assert.are.equal(3, c.numel)
        local v = c:get(1)
        assert.is_true(math.abs(v - (-1.0)) < 1e-5)
    end)

    -- ========================================================================
    -- Unary (5 tests)
    -- ========================================================================

    it("abs", function()
        local a = ins.from_table({-3.0, -1.0, 0.0, 2.0, 4.0})
        local b = ins.abs(a)
        assert.are.equal(5, b.numel)
        local v = b:get(1)
        assert.is_true(math.abs(v - 3.0) < 1e-5)
    end)

    it("sqrt", function()
        local a = ins.from_table({1.0, 4.0, 9.0, 16.0})
        local b = ins.sqrt(a)
        assert.are.equal(4, b.numel)
        local v = b:get(2)
        assert.is_true(math.abs(v - 2.0) < 1e-5)
    end)

    it("exp", function()
        local a = ins.from_table({0.0, 1.0, 2.0})
        local b = ins.exp(a)
        assert.are.equal(3, b.numel)
        local v = b:get(1)
        assert.is_true(math.abs(v - 1.0) < 1e-5)
    end)

    it("log", function()
        local a = ins.from_table({1.0, 2.0, 3.0})
        local b = ins.log(a)
        assert.are.equal(3, b.numel)
        local v = b:get(1)
        assert.is_true(math.abs(v - 0.0) < 1e-5)
    end)

    it("sin", function()
        local a = ins.from_table({0.0, 0.5, 1.0})
        local b = ins.sin(a)
        assert.are.equal(3, b.numel)
        local v = b:get(1)
        assert.is_true(math.abs(v - 0.0) < 1e-5)
    end)

    -- ========================================================================
    -- Reduction (5 tests)
    -- ========================================================================

    it("sum", function()
        local a = ins.from_table({1.0, 2.0, 3.0, 4.0})
        local s = ins.sum(a)
        assert.are.equal(1, s.numel)
        local v = s:get(1)
        assert.is_true(math.abs(v - 10.0) < 1e-5)
    end)

    it("mean", function()
        local a = ins.from_table({1.0, 2.0, 3.0, 4.0})
        local m = ins.mean(a)
        assert.are.equal(1, m.numel)
        local v = m:get(1)
        assert.is_true(math.abs(v - 2.5) < 1e-5)
    end)

    it("max", function()
        local a = ins.from_table({3.0, 1.0, 4.0, 1.0, 5.0})
        local m = ins.max(a)
        assert.are.equal(1, m.numel)
        local v = m:get(1)
        assert.is_true(math.abs(v - 5.0) < 1e-5)
    end)

    it("min", function()
        local a = ins.from_table({3.0, 1.0, 4.0, 1.0, 5.0})
        local m = ins.min(a)
        assert.are.equal(1, m.numel)
        local v = m:get(1)
        assert.is_true(math.abs(v - 1.0) < 1e-5)
    end)

    it("argmax", function()
        local a = ins.from_table({1.0, 5.0, 3.0, 2.0})
        local m = ins.argmax(a)
        assert.are.equal(1, m.numel)
    end)

    -- ========================================================================
    -- Comparison (4 tests)
    -- ========================================================================

    it("equal", function()
        local a = ins.from_table({1.0, 2.0, 3.0})
        local b = ins.from_table({1.0, 0.0, 3.0})
        local c = ins.equal(a, b)
        assert.are.equal(3, c.numel)
    end)

    it("greater", function()
        local a = ins.from_table({3.0, 1.0, 5.0})
        local b = ins.from_table({2.0, 4.0, 5.0})
        local c = ins.greater(a, b)
        assert.are.equal(3, c.numel)
    end)

    it("less", function()
        local a = ins.from_table({1.0, 5.0, 3.0})
        local b = ins.from_table({2.0, 4.0, 5.0})
        local c = ins.less(a, b)
        assert.are.equal(3, c.numel)
    end)

    it("logical_and", function()
        local a = ins.from_table({1, 1, 0}, ins.bool)
        local b = ins.from_table({1, 0, 0}, ins.bool)
        local c = ins.logical_and(a, b)
        assert.are.equal(3, c.numel)
    end)

    -- ========================================================================
    -- Manipulation (3 tests)
    -- ========================================================================

    it("reshape", function()
        local a = ins.arange(6, ins.float64)
        local b = ins.reshape(a, {2, 3})
        assert.are.equal(6, b.numel)
        assert.are.equal(2, b.ndim)
    end)

    it("transpose", function()
        local a = ins.from_table({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}})
        local b = ins.transpose(a)
        assert.are.equal(6, b.numel)
    end)

    it("squeeze", function()
        local a = ins.zeros({1, 3, 1}, ins.float64)
        local b = ins.squeeze(a)
        assert.are.equal(3, b.numel)
    end)

    -- ========================================================================
    -- Linalg (3 tests)
    -- ========================================================================

    it("matmul", function()
        local a = ins.from_table({{1.0, 2.0}, {3.0, 4.0}})
        local b = ins.from_table({{5.0, 6.0}, {7.0, 8.0}})
        local c = ins.matmul(a, b)
        assert.are.equal(4, c.numel)
    end)

    it("det", function()
        local a = ins.from_table({{1.0, 2.0}, {3.0, 4.0}})
        local d = ins.det(a)
        assert.are.equal(1, d.numel)
        local v = d:get(1)
        assert.is_true(math.abs(v - (-2.0)) < 1e-5)
    end)

    it("inv", function()
        local a = ins.from_table({{1.0, 2.0}, {3.0, 4.0}})
        local b = ins.inv(a)
        assert.are.equal(4, b.numel)
    end)

    -- ========================================================================
    -- FFT (2 tests)
    -- ========================================================================

    it("fft", function()
        local x = ins.from_table({1.0, 2.0, 3.0, 4.0})
        local y = ins.fft(x)
        assert.are.equal(4, y.numel)
    end)

    it("ifft", function()
        local x = ins.from_table({1.0, 2.0, 3.0, 4.0})
        local y = ins.fft(x)
        local z = ins.ifft(y)
        assert.are.equal(4, z.numel)
    end)

    -- ========================================================================
    -- Signal (1 test)
    -- ========================================================================

    it("hann", function()
        local w = ins.signal.hann(16)
        assert.are.equal(16, w.numel)
        local v0 = w:get(1)
        assert.is_true(math.abs(v0) < 1e-5)
    end)
end)
