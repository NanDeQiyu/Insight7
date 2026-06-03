-- Audio/I/O CPU binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_audio.lua

describe("Audio CPU Tests", function()
  local ins
  setup(function()
    ins = require("_insight")
  end)

  -- Note: Audio WAV read/write is C++ only (audio::read/audio::write).
  -- These tests verify signal I/O functions available in bindings.

  it("write creates file", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local path = "/tmp/insight_lua_audio_test.bin"
    ins.signal.write_bin(path, a)
    local f = io.open(path, "r")
    assert.is_not_nil(f)
    f:close()
    os.remove(path)
  end)

  it("write read roundtrip float64", function()
    local a = ins.from_table({ 1.0, 2.5, 3.7, -1.2, 0.0 })
    local path = "/tmp/insight_lua_audio_f64.bin"
    ins.signal.write_bin(path, a)
    local result = ins.signal.read_bin(path, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(5, result.numel)
    os.remove(path)
  end)

  it("write read roundtrip float32", function()
    local a = ins.zeros({ 4 }, ins.float32)
    local path = "/tmp/insight_lua_audio_f32.bin"
    ins.signal.write_bin(path, a)
    local result = ins.signal.read_bin(path, ins.float32)
    assert.is_not_nil(result)
    assert.are.equal(4, result.numel)
    os.remove(path)
  end)

  it("write read roundtrip int32", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0 })
    a = ins.cast(a, ins.int32)
    local path = "/tmp/insight_lua_audio_i32.bin"
    ins.signal.write_bin(path, a)
    local result = ins.signal.read_bin(path, ins.int32)
    assert.is_not_nil(result)
    assert.are.equal(3, result.numel)
    os.remove(path)
  end)

  it("write read zeros", function()
    local a = ins.zeros({ 100 }, ins.float64)
    local path = "/tmp/insight_lua_audio_zeros.bin"
    ins.signal.write_bin(path, a)
    local result = ins.signal.read_bin(path, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(100, result.numel)
    os.remove(path)
  end)

  it("write read ones", function()
    local a = ins.ones({ 50 }, ins.float64)
    local path = "/tmp/insight_lua_audio_ones.bin"
    ins.signal.write_bin(path, a)
    local result = ins.signal.read_bin(path, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(50, result.numel)
    os.remove(path)
  end)

  it("write read negative values", function()
    local a = ins.from_table({ -100.5, -200.3, -300.1 })
    local path = "/tmp/insight_lua_audio_neg.bin"
    ins.signal.write_bin(path, a)
    local result = ins.signal.read_bin(path, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(3, result.numel)
    os.remove(path)
  end)

  it("write read single element", function()
    local a = ins.from_table({ 42.0 })
    local path = "/tmp/insight_lua_audio_single.bin"
    ins.signal.write_bin(path, a)
    local result = ins.signal.read_bin(path, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(1, result.numel)
    os.remove(path)
  end)

  it("write read large signal", function()
    local t = {}
    for i = 1, 1000 do
      t[i] = i * 0.1
    end
    local a = ins.from_table(t)
    local path = "/tmp/insight_lua_audio_large.bin"
    ins.signal.write_bin(path, a)
    local result = ins.signal.read_bin(path, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(1000, result.numel)
    os.remove(path)
  end)

  it("write file size correct", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 })
    local path = "/tmp/insight_lua_audio_size.bin"
    ins.signal.write_bin(path, a)
    local f = io.open(path, "r")
    local size = f:seek("end")
    f:close()
    assert.are.equal(40, size) -- 5 doubles * 8 bytes
    os.remove(path)
  end)

  it("write read overwrite", function()
    local a1 = ins.from_table({ 1.0, 2.0, 3.0 })
    local a2 = ins.from_table({ 4.0, 5.0, 6.0, 7.0 })
    local path = "/tmp/insight_lua_audio_overwrite.bin"
    os.remove(path)
    ins.signal.write_bin(path, a1, false)
    ins.signal.write_bin(path, a2, false)
    local result = ins.signal.read_bin(path, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(4, result.numel)
    os.remove(path)
  end)
end)
