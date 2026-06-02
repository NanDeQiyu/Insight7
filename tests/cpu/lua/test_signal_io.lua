-- Signal io CPU binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;bindings/lua/?.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_signal_io.lua

describe("Signal IO CPU Tests", function()
  local ins
  local tmpfiles
  setup(function()
    ins = require("insight")
    tmpfiles = {}
  end)

  local function make_tmp(ext)
    local f = os.tmpname() .. (ext or ".bin")
    -- Remove any pre-existing file to avoid append issues
    os.remove(f)
    tmpfiles[#tmpfiles + 1] = f
    return f
  end

  teardown(function()
    for _, f in ipairs(tmpfiles) do
      os.remove(f)
    end
  end)

  it("write_read_bin_roundtrip", function()
    local data = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 })
    local tmpfile = make_tmp(".bin")
    ins.signal.write_bin(data, tmpfile)
    local result = ins.signal.read_bin(tmpfile, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(5, result.numel)
  end)

  it("write_read_bin_float32", function()
    local data = ins.cast(ins.from_table({ 10.0, 20.0, 30.0 }), ins.float32)
    local tmpfile = make_tmp(".bin")
    ins.signal.write_bin(data, tmpfile)
    local result = ins.signal.read_bin(tmpfile, ins.float32)
    assert.is_not_nil(result)
    assert.are.equal(3, result.numel)
  end)

  it("write_read_bin_int16", function()
    local data = ins.from_table({ 100, 200, 300, 400 })
    local tmpfile = make_tmp(".bin")
    ins.signal.write_bin(data, tmpfile)
    local result = ins.signal.read_bin(tmpfile, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(4, result.numel)
  end)

  it("write_read_bin_large", function()
    local t = {}
    for i = 1, 1024 do
      t[i] = math.sin(i)
    end
    local data = ins.from_table(t)
    local tmpfile = make_tmp(".bin")
    ins.signal.write_bin(data, tmpfile)
    local result = ins.signal.read_bin(tmpfile, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(1024, result.numel)
  end)

  it("pack_bin", function()
    local data = ins.from_table({ 1.0, 2.0, 3.0 })
    local packed = ins.signal.pack_bin(data)
    assert.is_not_nil(packed)
    assert.is_true(packed.numel > 0)
  end)

  it("unpack_bin", function()
    local data = ins.from_table({ 1.0, 2.0, 3.0, 4.0 })
    local tmpfile = make_tmp(".bin")
    ins.signal.write_bin(data, tmpfile)
    local raw = ins.signal.read_bin(tmpfile, ins.uint8)
    local result = ins.signal.unpack_bin(raw, ins.float64)
    assert.is_not_nil(result)
    assert.is_true(result.numel > 0)
  end)

  it("write_sigmf", function()
    local data = ins.from_table({ 1.0, 2.0, 3.0, 4.0 })
    local tmpfile = make_tmp(".sigmf-data")
    ins.signal.write_sigmf(data, tmpfile, {})
    -- File should exist
    local f = io.open(tmpfile, "rb")
    assert.is_not_nil(f)
    if f then
      f:close()
    end
  end)
end)
