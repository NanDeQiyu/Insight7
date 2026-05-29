# bindings/CMakeLists.txt
#
# Language bindings for Insight7.
# Each binding is optional and controlled by INSIGHT_BUILD_*_BINDING options.

option(INSIGHT_BUILD_PYTHON_BINDING "Build Python bindings (pybind11)" ON)
option(INSIGHT_BUILD_LUA_BINDING "Build Lua/LuaJIT bindings (sol2)" ON)
option(INSIGHT_BUILD_JULIA_BINDING "Build Julia bindings (C ABI wrapper)" ON)

if(INSIGHT_BUILD_PYTHON_BINDING)
    add_subdirectory(python)
endif()

if(INSIGHT_BUILD_LUA_BINDING)
    add_subdirectory(lua)
endif()

if(INSIGHT_BUILD_JULIA_BINDING)
    add_subdirectory(julia)
endif()
