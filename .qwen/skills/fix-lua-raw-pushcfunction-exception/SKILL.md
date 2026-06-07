---
name: fix-lua-raw-pushcfunction-exception
description: Lua sol2 lambdas and raw lua_pushcfunction must catch C++ exceptions — sol2 does NOT always auto-catch
source: auto-skill
extracted_at: '2026-06-07T01:46:20.559Z'
---

# Lua Exception Handling in Bindings

## Problem

C++ exceptions that cross the Lua FFI boundary call `std::terminate()`, crashing
the process. `pcall` cannot catch this — the process is already dead.

**Key discovery (2026-06-07)**: sol2 does NOT always auto-catch exceptions in
lambdas. Even regular `sol::table` lambdas can crash if the exception type is
a custom class (like `ins::Exception`) rather than `std::runtime_error`.

**Symptom**:
```
terminate called after throwing an instance of 'ins::Exception'
what():  reshape(): shape.numel() mismatch. Expected 6, got 8
Aborted (core dumped)
```

## When to Catch

| Binding method | Exception handling |
|---|---|
| `sol::usertype` method (function pointer) | ✅ sol2 auto-catches `std::exception` |
| `sol::table` lambda (simple) | ⚠️ May NOT catch custom exception types |
| `lua_pushcfunction` raw | ❌ **MUST catch manually** |
| Lambda with `sol::this_state` | ⚠️ May NOT catch — always add try-catch |

**Rule of thumb**: If the C++ code can throw ANY exception, add try-catch in
the Lua binding lambda. Don't rely on sol2 to catch it.

## Fix Pattern

```cpp
// For regular sol2 lambdas:
array_type["reshape"] = [](const ins::Array &a, sol::table new_shape,
                           sol::this_state L) {
    try {
        return a.reshape(ins::Shape(table_to_shape(new_shape)));
    } catch (const std::exception &e) {
        return luaL_error(L, "%s", e.what()), ins::Array();
    }
};

// For raw lua_pushcfunction:
lua_pushcfunction(L, [](lua_State *L) -> int {
    try {
        ins::Array result = from_lua_table(t, place);
        sol::stack::push(L, std::move(result));
        return 1;
    } catch (const std::exception &e) {
        return luaL_error(L, "%s", e.what());
    }
});

// For module-level functions:
m["from_table"] = [](sol::table t,
                     sol::optional<ins::Place> place) -> ins::Array {
    try {
        return from_lua_table(t, place);
    } catch (const std::exception &e) {
        luaL_error(t.lua_state(), "%s", e.what());
        return ins::Array();  // unreachable, luaL_error longjmps
    }
};
```

## Testing

```lua
it("error_ragged_table", function()
    assert.has_error(function()
        ins.Array({ { 3, 3 }, { 3 } })
    end)
end)

it("process_survives_error", function()
    pcall(function() ins.Array({ { 1, 2 }, { 3 } }) end)
    local a = ins.Array({ 5, 6, 7 })
    assert.is_not_nil(a)
end)
```

## Also See

- `feedback_load_backend_exception`: `load_backend` must catch exceptions
- `fix-lua-binding-api-gotchas`: `to()` try/catch pattern
