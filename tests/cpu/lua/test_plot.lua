-- Plot CPU binding tests — 13 tests aligned with C++ plot test suite.
--
-- Tests that plot binding functions exist and are callable.
-- Does NOT require gnuplot — only checks that the binding layer works.
-- Skipped entirely when Insight is built without plot support.

describe("Plot CPU Tests", function()
  local ins

  setup(function()
    ins = require("_insight")
    ins.set_device(ins.CPUPlace())
  end)

  -- Skip all plot tests when plot module is not available
  local has_plot = nil
  local function require_plot()
    if has_plot == nil then
      has_plot = (ins.plot ~= nil)
    end
    if not has_plot then
      pending("plot module not available (INSIGHT_USE_MATPLOT=OFF)")
    end
    return has_plot
  end

  it("plot module exists", function()
    assert.is_not_nil(ins.plot)
  end)

  it("plot function exists", function()
    if not require_plot() then
      return
    end
    assert.is_function(ins.plot.plot)
  end)

  it("scatter function exists", function()
    if not require_plot() then
      return
    end
    assert.is_function(ins.plot.scatter)
  end)

  it("bar function exists", function()
    if not require_plot() then
      return
    end
    assert.is_function(ins.plot.bar)
  end)

  it("hist function exists", function()
    if not require_plot() then
      return
    end
    assert.is_function(ins.plot.hist)
  end)

  it("imshow function exists", function()
    if not require_plot() then
      return
    end
    assert.is_function(ins.plot.imshow)
  end)

  it("contour function exists", function()
    if not require_plot() then
      return
    end
    assert.is_function(ins.plot.contour)
  end)

  it("subplot function exists", function()
    if not require_plot() then
      return
    end
    assert.is_function(ins.plot.subplot)
  end)

  it("title function exists", function()
    if not require_plot() then
      return
    end
    assert.is_function(ins.plot.title)
  end)

  it("xlabel function exists", function()
    if not require_plot() then
      return
    end
    assert.is_function(ins.plot.xlabel)
  end)

  it("ylabel function exists", function()
    if not require_plot() then
      return
    end
    assert.is_function(ins.plot.ylabel)
  end)

  it("legend function exists", function()
    if not require_plot() then
      return
    end
    assert.is_function(ins.plot.legend)
  end)

  it("savefig function exists", function()
    if not require_plot() then
      return
    end
    assert.is_function(ins.plot.save)
  end)

  it("clf function exists", function()
    if not require_plot() then
      return
    end
    assert.is_function(ins.plot.clf)
  end)
end)
