-- Plot CPU binding tests — 13 tests aligned with C++ plot test suite.
--
-- Tests that plot functions exist and can be called without crashing.
-- Plotting is hard to verify numerically, so we only check smoke behavior.
--
-- Functions tested: plot, scatter, bar, hist, imshow, contour,
--                   subplot, title, xlabel, ylabel, legend, savefig, close
--
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_plot.lua

describe("Plot CPU Tests", function()
  local ins
  local gnuplot_available = false

  setup(function()
    ins = require("_insight")
    -- Check if gnuplot is available (required by matplotplusplus)
    local ok, exit_type, code = os.execute("gnuplot --version > /dev/null 2>&1")
    gnuplot_available = (ok == true) or (ok == 0)
  end)

  local function require_gnuplot()
    if not gnuplot_available then
      pending("gnuplot not installed, skipping plot tests")
    end
  end

  it("plot: line plot without crashing", function()
    require_gnuplot()
    local y = ins.from_table({ 1.0, 3.0, 2.0, 4.0 })
    ins.plot.plot(y)
    ins.plot.clf()
  end)

  it("scatter: scatter plot without crashing", function()
    require_gnuplot()
    local x = ins.from_table({ 1.0, 2.0, 3.0 })
    local y = ins.from_table({ 4.0, 5.0, 6.0 })
    ins.plot.scatter(x, y)
    ins.plot.clf()
  end)

  it("bar: bar chart without crashing", function()
    require_gnuplot()
    local y = ins.from_table({ 3.0, 1.0, 4.0, 1.0, 5.0 })
    ins.plot.bar(y)
    ins.plot.clf()
  end)

  it("hist: histogram without crashing", function()
    require_gnuplot()
    local data = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 })
    ins.plot.hist(data, 4)
    ins.plot.clf()
  end)

  it("imshow: image display without crashing", function()
    require_gnuplot()
    local data = ins.from_table({ { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } })
    ins.plot.imshow(data)
    ins.plot.clf()
  end)

  it("contour: contour plot without crashing", function()
    require_gnuplot()
    local x = ins.from_table({ { 0.0, 1.0 }, { 0.0, 1.0 } })
    local y = ins.from_table({ { 0.0, 0.0 }, { 1.0, 1.0 } })
    local z = ins.from_table({ { 0.0, 1.0 }, { 1.0, 2.0 } })
    ins.plot.contour(x, y, z)
    ins.plot.clf()
  end)

  it("subplot: subplot layout without crashing", function()
    require_gnuplot()
    ins.plot.subplot(2, 1, 1)
    ins.plot.clf()
  end)

  it("title: set title without crashing", function()
    require_gnuplot()
    ins.plot.title("Test Title")
    ins.plot.clf()
  end)

  it("xlabel: set x-axis label without crashing", function()
    require_gnuplot()
    ins.plot.xlabel("X Axis")
    ins.plot.clf()
  end)

  it("ylabel: set y-axis label without crashing", function()
    require_gnuplot()
    ins.plot.ylabel("Y Axis")
    ins.plot.clf()
  end)

  it("legend: set legend without crashing", function()
    require_gnuplot()
    local y = ins.from_table({ 1.0, 2.0, 3.0 })
    ins.plot.plot(y)
    ins.plot.legend({ "data" })
    ins.plot.clf()
  end)

  it("savefig: save figure to file without crashing", function()
    require_gnuplot()
    local y = ins.from_table({ 1.0, 2.0, 3.0 })
    ins.plot.plot(y)
    local tmpfile = "/tmp/insight_plot_test_lua.png"
    ins.plot.save(tmpfile)
    -- Check file was created
    local f = io.open(tmpfile, "r")
    assert.is_not_nil(f)
    if f then
      f:close()
    end
    os.remove(tmpfile)
    ins.plot.clf()
  end)

  it("close: clear figure without crashing", function()
    require_gnuplot()
    ins.plot.figure(1)
    ins.plot.clf()
  end)
end)
