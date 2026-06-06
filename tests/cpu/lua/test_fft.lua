-- FFT CPU binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_fft.lua

describe("FFT CPU Tests", function()
  local ins
  setup(function()
    ins = require("_insight")
    ins.set_device(ins.CPUPlace())
  end)

  it("fft 1d", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0 })
    local y = ins.fft(x)
    assert.is_not_nil(y)
    assert.are.equal(4, y.numel)
  end)

  it("ifft 1d", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0 })
    local y = ins.fft(x)
    local z = ins.ifft(y)
    assert.is_not_nil(z)
    assert.are.equal(4, z.numel)
  end)

  it("fft ifft roundtrip", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 })
    local X = ins.fft(x)
    local x_back = ins.ifft(X)
    assert.is_not_nil(x_back)
    assert.are.equal(8, x_back.numel)
  end)

  it("rfft", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0 })
    local y = ins.rfft(x)
    assert.is_not_nil(y)
    assert.are.equal(3, y.numel)
  end)

  it("irfft", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0 })
    local y = ins.rfft(x)
    local z = ins.irfft(y)
    assert.is_not_nil(z)
    assert.are.equal(4, z.numel)
  end)

  it("fftfreq", function()
    local f = ins.fftfreq(8, 1.0)
    assert.is_not_nil(f)
    assert.are.equal(8, f.numel)
  end)

  it("rfftfreq", function()
    local f = ins.rfftfreq(8, 1.0)
    assert.is_not_nil(f)
    assert.are.equal(5, f.numel)
  end)

  it("rfft2", function()
    local x = ins.from_table({ { 1.0, 2.0 }, { 3.0, 4.0 } })
    local y = ins.rfft2(x)
    assert.is_not_nil(y)
  end)

  it("irfft2", function()
    local x = ins.from_table({ { 1.0, 2.0 }, { 3.0, 4.0 } })
    local y = ins.rfft2(x)
    local z = ins.irfft2(y)
    assert.is_not_nil(z)
    assert.are.equal(4, z.numel)
  end)

  it("fft longer signal", function()
    local t = {}
    for i = 1, 64 do
      t[i] = math.sin(2 * math.pi * i / 64)
    end
    local x = ins.from_table(t)
    local y = ins.fft(x)
    assert.is_not_nil(y)
    assert.are.equal(64, y.numel)
  end)

  it("fft ones", function()
    local t = {}
    for i = 1, 8 do
      t[i] = 1.0
    end
    local x = ins.from_table(t)
    local y = ins.fft(x)
    assert.is_not_nil(y)
    assert.are.equal(8, y.numel)
  end)

  it("fft non power of two", function()
    local t = {}
    for i = 1, 10 do
      t[i] = i * 1.0
    end
    local x = ins.from_table(t)
    local y = ins.fft(x)
    assert.is_not_nil(y)
    assert.are.equal(10, y.numel)
  end)

  it("fftshift", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 })
    local y = ins.fftshift(x)
    assert.are.equal(8, y.numel)
    -- fftshift([1..8]) = [5,6,7,8,1,2,3,4]
    assert.are.equal(5.0, y[1])
    assert.are.equal(6.0, y[2])
    assert.are.equal(7.0, y[3])
    assert.are.equal(8.0, y[4])
    assert.are.equal(1.0, y[5])
    assert.are.equal(2.0, y[6])
    assert.are.equal(3.0, y[7])
    assert.are.equal(4.0, y[8])
  end)

  it("ifftshift", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 })
    local y = ins.ifftshift(x)
    assert.are.equal(8, y.numel)
    -- ifftshift([1..8]) = [5,6,7,8,1,2,3,4]
    assert.are.equal(5.0, y[1])
    assert.are.equal(6.0, y[2])
    assert.are.equal(7.0, y[3])
    assert.are.equal(8.0, y[4])
    assert.are.equal(1.0, y[5])
    assert.are.equal(2.0, y[6])
    assert.are.equal(3.0, y[7])
    assert.are.equal(4.0, y[8])
  end)

  it("next_fast_len", function()
    local n = ins.next_fast_len(100)
    assert.is_true(n >= 100)
  end)

  it("fftfreq custom spacing", function()
    local f = ins.fftfreq(16, 0.5)
    assert.is_not_nil(f)
    assert.are.equal(16, f.numel)
  end)
end)
