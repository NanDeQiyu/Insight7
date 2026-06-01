-- Creation CPU binding tests — aligned with C++ test_creation.cpp
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;"
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_creation.lua

describe("Creation CPU Tests", function()
  local ins
  setup(function()
    ins = require("_insight")
  end)

  it("zeros_1d", function()
    local a = ins.zeros({ 5 }, ins.float32)
    assert.is_not_nil(a)
    assert.are.equal(5, a.numel)
  end)

  it("zeros_2d", function()
    local a = ins.zeros({ 3, 4 }, ins.float64)
    assert.is_not_nil(a)
    assert.are.equal(12, a.numel)
    assert.are.equal(2, a.ndim)
  end)

  it("ones_1d", function()
    local a = ins.ones({ 4 }, ins.float32)
    assert.is_not_nil(a)
    assert.are.equal(4, a.numel)
  end)

  it("ones_2d", function()
    local a = ins.ones({ 2, 5 }, ins.float64)
    assert.is_not_nil(a)
    assert.are.equal(10, a.numel)
  end)

  it("full", function()
    local a = ins.full({ 3, 3 }, 7.0, ins.float32)
    assert.is_not_nil(a)
    assert.are.equal(9, a.numel)
  end)

  it("full_negative", function()
    local a = ins.full({ 2, 2 }, -3.5, ins.float64)
    assert.is_not_nil(a)
    assert.are.equal(4, a.numel)
  end)

  it("eye", function()
    local a = ins.eye(4)
    assert.is_not_nil(a)
    assert.are.equal(16, a.numel)
  end)

  it("arange", function()
    local a = ins.arange(10, ins.float32)
    assert.is_not_nil(a)
    assert.are.equal(10, a.numel)
  end)

  it("linspace", function()
    local a = ins.linspace(0.0, 1.0, 11, ins.float64)
    assert.is_not_nil(a)
    assert.are.equal(11, a.numel)
  end)

  it("from_table_1d", function()
    local a = ins.from_table({ 1.5, 2.5, 3.5 })
    assert.is_not_nil(a)
    assert.are.equal(3, a.numel)
  end)

  it("from_table_2d", function()
    local a = ins.from_table({ { 1, 2, 3 }, { 4, 5, 6 } })
    assert.is_not_nil(a)
    assert.are.equal(6, a.numel)
    assert.are.equal(2, a.ndim)
  end)

  it("zeros_like", function()
    local a = ins.ones({ 3, 3 }, ins.float32)
    local b = ins.zeros_like(a)
    assert.is_not_nil(b)
    assert.are.equal(9, b.numel)
  end)

  it("ones_like", function()
    local a = ins.zeros({ 2, 4 }, ins.float64)
    local b = ins.ones_like(a)
    assert.is_not_nil(b)
    assert.are.equal(8, b.numel)
  end)

  it("dtype_float64", function()
    local a = ins.zeros({ 3 }, ins.float64)
    assert.are.equal(ins.float64, a.dtype)
  end)

  it("dtype_int32", function()
    local a = ins.ones({ 3 }, ins.int32)
    assert.are.equal(ins.int32, a.dtype)
  end)
end)
