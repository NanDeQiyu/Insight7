-- bindings/lua/insight.lua
--
-- Lua loader for the Insight7 native module.
-- Delegates to insight/init.lua for the full wrapper.
--
-- Usage:
--   local ins = require("insight")
--   local a = ins.zeros({2, 3}, ins.float32)
--   local b = ins.ones({2, 3}, ins.float32)
--   local c = a + b
--   print(c)

return require("insight.init")
