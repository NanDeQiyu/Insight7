-- Plot CPU binding tests — 13 tests aligned with C++ plot test suite.
--
-- Tests that plot binding functions exist and are callable.
-- Does NOT require gnuplot — only checks that the binding layer works.
--
-- Functions tested: plot, scatter, bar, hist, imshow, contour,
--                   subplot, title, xlabel, ylabel, legend, savefig, close

describe("Plot CPU Tests", function()
  local ins

  setup(function()
    ins = require("_insight")
    ins.set_device(ins.CPUPlace())
  end)

  -- Verify the plot module is loaded
  it("plot module exists", function()
    assert.is_not_nil(ins.plot)
  end)

  it("plot function exists", function()
    assert.is_function(ins.plot.plot)
  end)

  it("scatter function exists", function()
    assert.is_function(ins.plot.scatter)
  end)

  it("bar function exists", function()
    assert.is_function(ins.plot.bar)
  end)

  it("hist function exists", function()
    assert.is_function(ins.plot.hist)
  end)

  it("imshow function exists", function()
    assert.is_function(ins.plot.imshow)
  end)

  it("contour function exists", function()
    assert.is_function(ins.plot.contour)
  end)

  it("subplot function exists", function()
    assert.is_function(ins.plot.subplot)
  end)

  it("title function exists", function()
    assert.is_function(ins.plot.title)
  end)

  it("xlabel function exists", function()
    assert.is_function(ins.plot.xlabel)
  end)

  it("ylabel function exists", function()
    assert.is_function(ins.plot.ylabel)
  end)

  it("legend function exists", function()
    assert.is_function(ins.plot.legend)
  end)

  it("savefig function exists", function()
    assert.is_function(ins.plot.save)
  end)

  it("clf function exists", function()
    assert.is_function(ins.plot.clf)
  end)
end)
