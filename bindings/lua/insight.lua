-- bindings/lua/insight.lua
--
-- Lua loader for the Insight7 native module.
-- This file is installed by luarocks and handles finding the native library.
--
-- Usage:
--   local ins = require("insight")
--   local a = ins.zeros({2, 3}, ins.float32)
--   local b = ins.ones({2, 3}, ins.float32)
--   local c = a + b
--   print(c)

-- Try to load the native C++ module (insight.so / insight.dll)
local ok, native = pcall(require, "insight_native")
if not ok then
    -- Fallback: try loading directly (standalone build)
    ok, native = pcall(require, "insight")
    if not ok then
        error("Failed to load Insight native module: " .. tostring(native) ..
              "\nMake sure libinsight.so is in your LUA_CPATH or LD_LIBRARY_PATH.")
    end
end

return native
