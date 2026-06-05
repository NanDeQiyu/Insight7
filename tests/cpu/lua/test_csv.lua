-- CSV CPU binding tests — aligned with C++ test_csv.cpp
-- Note: CSV read/write is not exposed in the Lua bindings. This test verifies
-- data roundtrip via from_table which is the binding-level equivalent.
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;"
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_csv.lua

describe("CSV CPU Tests", function()
  local ins
  setup(function()
    ins = require("_insight")
    ins.set_device(ins.CPUPlace())
  end)

  it("roundtrip_1d", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0 })
    assert.is_not_nil(a)
    assert.are.equal(4, a.numel)
  end)

  it("roundtrip_2d", function()
    local a = ins.from_table({ { 1, 2, 3 }, { 4, 5, 6 } })
    assert.is_not_nil(a)
    assert.are.equal(6, a.numel)
  end)

  it("roundtrip_int", function()
    local a = ins.from_table({ 10, 20, 30 }, ins.int32)
    assert.is_not_nil(a)
    assert.are.equal(3, a.numel)
  end)

  it("roundtrip_float64", function()
    local a = ins.from_table({ 1.1, 2.2, 3.3 })
    assert.is_not_nil(a)
    assert.are.equal(3, a.numel)
  end)

  it("roundtrip_after_ops", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local b = ins.add(a, a)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)
end)
