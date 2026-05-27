# cmake/FetchDependency.cmake
#
# Function to fetch a dependency with local path priority.
#
# Usage:
#   auto_fetch_dependency(<name> <git_repo> <git_tag>)
#
# Behavior:
#   1. If third_party/<name>/CMakeLists.txt exists, use it directly.
#   2. Otherwise, download from Git repository using FetchContent.
#
function(auto_fetch_dependency NAME GIT_REPO GIT_TAG)
    set(LOCAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/${NAME}")

    if(EXISTS "${LOCAL_PATH}/CMakeLists.txt")
        message(STATUS "Using local ${NAME} from ${LOCAL_PATH}")
        add_subdirectory("${LOCAL_PATH}" "${CMAKE_CURRENT_BINARY_DIR}/${NAME}")
    else()
        message(STATUS "Local ${NAME} not found, fetching from ${GIT_REPO} (tag: ${GIT_TAG})...")
        include(FetchContent)
        FetchContent_Declare(
            "${NAME}"
            GIT_REPOSITORY "${GIT_REPO}"
            GIT_TAG        "${GIT_TAG}"
        )
        FetchContent_MakeAvailable("${NAME}")
    endif()
endfunction()