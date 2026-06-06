# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/aistudio/plum/Insight7/build.luarocks/_deps/ogg-src")
  file(MAKE_DIRECTORY "/home/aistudio/plum/Insight7/build.luarocks/_deps/ogg-src")
endif()
file(MAKE_DIRECTORY
  "/home/aistudio/plum/Insight7/build.luarocks/_deps/ogg-build"
  "/home/aistudio/plum/Insight7/build.luarocks/_deps/ogg-subbuild/ogg-populate-prefix"
  "/home/aistudio/plum/Insight7/build.luarocks/_deps/ogg-subbuild/ogg-populate-prefix/tmp"
  "/home/aistudio/plum/Insight7/build.luarocks/_deps/ogg-subbuild/ogg-populate-prefix/src/ogg-populate-stamp"
  "/home/aistudio/plum/Insight7/build.luarocks/_deps/ogg-subbuild/ogg-populate-prefix/src"
  "/home/aistudio/plum/Insight7/build.luarocks/_deps/ogg-subbuild/ogg-populate-prefix/src/ogg-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/aistudio/plum/Insight7/build.luarocks/_deps/ogg-subbuild/ogg-populate-prefix/src/ogg-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/aistudio/plum/Insight7/build.luarocks/_deps/ogg-subbuild/ogg-populate-prefix/src/ogg-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
