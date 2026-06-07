---
name: fix-lua-raw-pushcfunction-exception
description: Lua raw lua_pushcfunction lambdas must catch C++ exceptions manually — sol2 exception handling does NOT apply
source: auto-skill
extracted_at: '2026-06-07T01:10:00.000Z'
---

# Lua Raw `lua_pushcfunction` Exception Handling

## Problem

When binding C++ functions to Lua via `lua_pushcfunction` (raw C API), sol2's
built-in exception-to-Lua-error conversion does **NOT** apply. Any uncaught
C++ exception crosses the FFI boundary and calls `std::terminate()`, crashing
the Lua process with `Aborted (core dumped)`.

This is different from sol2's regular function bindings (which DO catch
exceptions automatically). The issue only affects raw `lua_pushcfunction` lambdas.

**Symptom**:
```
> ins.Array({{3,3},{3}})
terminate called after throwing an instance of 'std::runtime_error'
what():  Ragged nested table at root[2]: dimension 1 has inconsistent sizes (2 vs 1)
Aborted (core dumped)
```

`pcall` cannot catch this — the process is already dead.

## Affected Patterns

1. **Metamethods bound via `lua_pushcfunction`** — e.g., `__call` on Array usertype
2. **Module-level functions bound via `lua_pushcfunction`** — e.g., `from_table`
3. **Any sol2 lambda that can throw** — sol2 DOES catch for regular bindings,
   but NOT for raw `lua_pushcfunction`

## Fix

Wrap the C++ call in `try-catch` and convert to `luaL_error`:

```cpp
// ❌ WRONG — exception escapes, crashes process
lua_pushcfunction(L, [](lua_State *L) -> int {
    sol::table t(L, 2);
    ins::Array result = from_lua_table(t, place);  // throws on ragged table
    sol::stack::push(L, std::move(result));
    return 1;
});

// ✅ CORRECT — catch and convert to Lua error
lua_pushcfunction(L, [](lua_State *L) -> int {
    sol::table t(L, 2);
    try {
        ins::Array result = from_lua_table(t, place);
        sol::stack::push(L, std::move(result));
        return 1;
    } catch (const std::exception &e) {
        return luaL_error(L, "%s", e.what());
    }
});
```

For module-level functions bound as lambdas:

```cpp
// ❌ WRONG — bare function pointer, no exception handling
m["from_table"] = &from_lua_table;

// ✅ CORRECT — lambda wrapper with try-catch
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

Use `assert.has_error` in busted to verify errors are raised as Lua errors:

```lua
it("error_ragged_table", function()
    assert.has_error(function()
        ins.Array({ { 3, 3 }, { 3 } })
    end)
end)

it("process_survives_error", function()
    -- Verify process doesn't crash after error
    pcall(function()
        ins.Array({ { 1, 2 }, { 3 } })
    end)
    local a = ins.Array({ 5, 6, 7 })
    assert.is_not_nil(a)
    assert.are.equal(3, a.numel)
end)
```

## Key Distinction

| Binding method | Exception handling |
|---|---|
| `sol::usertype` method | ✅ sol2 auto-catches |
| `sol::table` lambda | ✅ sol2 auto-catches |
| `lua_pushcfunction` raw | ❌ **MUST catch manually** |
| `sol::overload` lambda | ✅ sol2 auto-catches |

## Also See

- `fix-lua-binding-api-gotchas` item 13: `to()` try/catch pattern (same root cause)
- `feedback_load_backend_exception`: `load_backend` must catch exceptions
