-- Print CPU binding tests — aligned with C++ test_print.cpp
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;"
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_print.lua

describe("Print CPU Tests", function()
  local ins
  setup(function()
    ins = require("_insight")
    ins.set_device(ins.CPUPlace())
  end)

  it("tostring_1d", function()
    local a = ins.zeros({ 5 }, ins.float32)
    local s = tostring(a)
    assert.is_not_nil(s)
    assert.is_true(#s > 0)
  end)

  it("tostring_2d", function()
    local a = ins.ones({ 3, 4 }, ins.float64)
    local s = tostring(a)
    assert.is_not_nil(s)
    assert.is_true(#s > 0)
  end)

  it("tostring_int", function()
    local a = ins.zeros({ 3 }, ins.int32)
    local s = tostring(a)
    assert.is_not_nil(s)
    assert.is_true(#s > 0)
  end)

  it("tostring_from_table", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local s = tostring(a)
    assert.is_not_nil(s)
    assert.is_true(#s > 0)
  end)

  it("tostring_2d_table", function()
    local a = ins.from_table({ { 1, 2 }, { 3, 4 } })
    local s = tostring(a)
    assert.is_not_nil(s)
    assert.is_true(#s > 0)
  end)

  it("repr_does_not_crash_large", function()
    local a = ins.ones({ 100, 100 }, ins.float32)
    local s = tostring(a)
    assert.is_not_nil(s)
  end)

  it("repr_empty", function()
    local a = ins.zeros({ 0 }, ins.float32)
    local s = tostring(a)
    assert.is_not_nil(s)
  end)

  it("repr_after_cast", function()
    local a = ins.ones({ 3 }, ins.float32)
    local b = ins.cast(a, ins.int32)
    local s = tostring(b)
    assert.is_not_nil(s)
  end)
end)
