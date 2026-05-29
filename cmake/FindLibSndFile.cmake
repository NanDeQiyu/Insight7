# cmake/FindLibSndFile.cmake
#
# Find libsndfile (https://github.com/libsndfile/libsndfile)
#
# This module defines:
#   SndFile_FOUND        - True if libsndfile was found
#   SndFile_INCLUDE_DIRS - Include directories
#   SndFile_LIBRARIES    - Libraries to link
#
# And imports the target:
#   SndFile::sndfile         - Shared library
#   SndFile::sndfile-static  - Static library
#

find_path(SndFile_INCLUDE_DIR
    NAMES sndfile.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/libsndfile/include
        $ENV{SNDFILE_HOME}/include
        ${CMAKE_PREFIX_PATH}/include
)

find_library(SndFile_LIBRARY
    NAMES sndfile libsndfile
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/libsndfile/lib
        $ENV{SNDFILE_HOME}/lib
        ${CMAKE_PREFIX_PATH}/lib
        ${CMAKE_PREFIX_PATH}/lib64
)

# Windows: also look for the import library
if(WIN32)
    find_library(SndFile_IMPLIB
        NAMES sndfile libsndfile
        PATHS
            /usr/lib
            /usr/local/lib
            $ENV{SNDFILE_HOME}/lib
            ${CMAKE_PREFIX_PATH}/lib
        PATH_SUFFIXES
            lib
    )
endif()

if(SndFile_INCLUDE_DIR AND SndFile_LIBRARY)
    set(SndFile_FOUND TRUE)
    set(SndFile_INCLUDE_DIRS ${SndFile_INCLUDE_DIR})
    set(SndFile_LIBRARIES ${SndFile_LIBRARY})
    message(STATUS "Found libsndfile: ${SndFile_LIBRARY}")

    if(NOT TARGET SndFile::sndfile)
        add_library(SndFile::sndfile UNKNOWN IMPORTED)
        set_target_properties(SndFile::sndfile PROPERTIES
            IMPORTED_LOCATION "${SndFile_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${SndFile_INCLUDE_DIR}"
        )
        if(WIN32 AND SndFile_IMPLIB)
            set_target_properties(SndFile::sndfile PROPERTIES
                IMPORTED_IMPLIB "${SndFile_IMPLIB}"
            )
        endif()
    endif()
else()
    set(SndFile_FOUND FALSE)
    message(STATUS "libsndfile not found via find_path/find_library")
endif()

mark_as_advanced(SndFile_INCLUDE_DIR SndFile_LIBRARY SndFile_IMPLIB)
