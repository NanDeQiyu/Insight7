--- Dual-mode wrapper: supports both positional and named-table arguments.
-- func(a, b)  or  func{a=x, b=y}  or  func{x, y}
-- @tparam table names Ordered argument names
-- @tparam function fn The underlying function to call
-- @treturn function A wrapper supporting both calling conventions
return function(names, fn)
  return function(...)
    if select("#", ...) == 1 and type(select(1, ...)) == "table" then
      local t = select(1, ...)
      local has_names = false
      for k, _ in pairs(t) do
        if type(k) ~= "number" then
          has_names = true
          break
        end
      end
      if has_names then
        local pos = {}
        for i, name in ipairs(names) do
          pos[i] = t[name]
          if pos[i] == nil then
            pos[i] = t[i]
          end
        end
        return fn(table.unpack(pos, 1, #names))
      end
    end
    return fn(...)
  end
end
