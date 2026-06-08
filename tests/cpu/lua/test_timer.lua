-- test_timer.lua — Timer tests for Insight7 Lua binding
local ins = require("insight")

describe("Timer (CPU)", function()
  it("creates and destroys timer", function()
    local t = ins.Timer(0, 0)
    assert.is_not_nil(t)
    assert.is_not_nil(t._handle)
  end)

  it("measures elapsed time", function()
    local t = ins.Timer(0, 0)
    t:start()
    -- Busy wait for ~10ms
    local start = os.clock()
    while os.clock() - start < 0.01 do
    end
    t:stop()
    local ms = t:elapsed()
    assert.is_true(ms >= 0.0)
    assert.is_true(ms < 100.0)
  end)

  it("supports manual start/stop", function()
    local t = ins.Timer(0, 0)
    t:start()
    local start = os.clock()
    while os.clock() - start < 0.005 do
    end
    t:stop()
    local ms = t:elapsed()
    assert.is_true(ms >= 0.0)
  end)

  it("raises error on nil arguments", function()
    assert.has_error(function()
      ins.Timer(nil, nil)
    end)
    assert.has_error(function()
      ins.Timer(0, nil)
    end)
  end)

  it("raises error on non-numeric arguments", function()
    assert.has_error(function()
      ins.Timer("cpu", "0")
    end)
  end)
end)
