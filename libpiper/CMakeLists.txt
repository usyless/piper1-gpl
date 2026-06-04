# Builds libpiper
cmake_minimum_required(VERSION 3.26)
project(piper LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBPIPER_FULL_AUDIOCHUNK
    ON
    CACHE BOOL
    "Enable the full audio chunk struct"
)

set(LIBPIPER_USE_BUNDLED_ONNXRUNTIME
    ON
    CACHE BOOL
    "Use the bundled onnxruntime, which is downloaded from microsoft or synesthesiam for arm32.
    Mainly only disable this if trying to link onnxruntime statically, if dynamic just define ONNXRUNTIME_DIR.
    If disabled, make sure to set LIBPIPER_ONNXRUNTIME_INCLUDE_DIRS, LIBPIPER_ONNXRUNTIME_TARGET_LINK_DIRS and LIBPIPER_ONNXRUNTIME_TARGET_LINK_FILES"
)

if (NOT LIBPIPER_USE_BUNDLED_ONNXRUNTIME)
    set(LIBPIPER_ONNXRUNTIME_INCLUDE_DIRS
        ""
        CACHE STRING
        "Include directories for non-bundled onnxruntime, make sure to be semicolon separated"
    )

    set(LIBPIPER_ONNXRUNTIME_TARGET_LINK_DIRS
        ""
        CACHE STRING
        "Library directories for non-bundled onnxruntime, make sure to be semicolon separated"
    )

    set(LIBPIPER_ONNXRUNTIME_TARGET_LINK_FILES
        ""
        CACHE STRING
        "Files to link for non-bundled onnxruntime, make sure to be semicolon separated"
    )

    if (DEFINED CACHE{LIBPIPER_ONNXRUNTIME_INCLUDE_DIRS} AND
        DEFINED CACHE{LIBPIPER_ONNXRUNTIME_TARGET_LINK_DIRS} AND
        DEFINED CACHE{LIBPIPER_ONNXRUNTIME_TARGET_LINK_FILES})

        if ("${LIBPIPER_ONNXRUNTIME_INCLUDE_DIRS}" STREQUAL "" OR
            "${LIBPIPER_ONNXRUNTIME_TARGET_LINK_DIRS}" STREQUAL "" OR
            "${LIBPIPER_ONNXRUNTIME_TARGET_LINK_FILES}" STREQUAL "")
            
            message(FATAL_ERROR "Non-bundled ONNXRuntime requires LIBPIPER_ONNXRUNTIME_INCLUDE_DIRS, LIBPIPER_ONNXRUNTIME_TARGET_LINK_DIRS, and LIBPIPER_ONNXRUNTIME_TARGET_LINK_FILES to be set and non-empty!")
        endif()
    else()
        message(FATAL_ERROR "Missing one or more required non-bundled ONNXRuntime cache variables.")
    endif()
endif()

include(FetchContent)

FetchContent_Declare(
    espeak_ng_external
    GIT_REPOSITORY https://github.com/espeak-ng/espeak-ng.git
    GIT_TAG fbe4b3764285c35b1f035cb8d09ad9fc19f71c30 # 10 April 2026
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
set(USE_ASYNC OFF CACHE BOOL "" FORCE)
set(USE_MBROLA OFF CACHE BOOL "" FORCE)
set(USE_LIBSONIC OFF CACHE BOOL "" FORCE)
set(USE_LIBPCAUDIO OFF CACHE BOOL "" FORCE)
set(USE_KLATT OFF CACHE BOOL "" FORCE)
set(USE_SPEECHPLAYER OFF CACHE BOOL "" FORCE)
set(EXTRA_cmn ON CACHE BOOL "" FORCE)
set(EXTRA_ru ON CACHE BOOL "" FORCE)
set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(espeak_ng_external)

add_compile_definitions(_FILE_OFFSET_BITS=64)

include_directories(
    "${espeak_ng_external_SOURCE_DIR}/src/include"
    "${espeak_ng_external_SOURCE_DIR}/src/ucd-tools/src/include"
)

# ---- libpiper ------

add_library(${PROJECT_NAME} STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/piper.cpp"
)

if (LIBPIPER_FULL_AUDIOCHUNK)
    message(STATUS "Using the full piper::AudioChunk as set by LIBPIPER_FULL_AUDIOCHUNK")
    target_compile_definitions(${PROJECT_NAME} PUBLIC 
        LIBPIPER_FULL_AUDIOCHUNK
    )
else()
    message(STATUS "Ignoring the full piper::AudioChunk as set by LIBPIPER_FULL_AUDIOCHUNK")
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    espeak-ng
    ucd
)

# ---- onnxruntime ---

if (LIBPIPER_USE_BUNDLED_ONNXRUNTIME)

# Look for onnxruntime files in <root>/lib
if(NOT DEFINED ONNXRUNTIME_DIR)
    if(NOT DEFINED ONNXRUNTIME_VERSION)
        set(ONNXRUNTIME_VERSION "1.25.1")
    endif()

    if(WIN32)
        # Windows x86-64
        set(ONNXRUNTIME_PREFIX "onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
        set(ONNXRUNTIME_EXT "zip")
    elseif (APPLE)
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
            # MacOS x86-64
            set(ONNXRUNTIME_PREFIX "onnxruntime-osx-x86_64-${ONNXRUNTIME_VERSION}")
        elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL arm64)
            # MacOS Apple Silicon
            set(ONNXRUNTIME_PREFIX "onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}")
        else()
            message(FATAL_ERROR "Unsupported architecture for onnxruntime")
        endif()

        set(ONNXRUNTIME_EXT "tgz")
    else()
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
            # Linux x86-64
            set(ONNXRUNTIME_PREFIX "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
        elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
            # Linux ARM 64-bit
            set(ONNXRUNTIME_PREFIX "onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}")
        elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL armv7l)
            # Linux ARM 32-bit
            set(ONNXRUNTIME_PREFIX "onnxruntime-linux-arm32-${ONNXRUNTIME_VERSION}")
            set(ONNXRUNTIME_URL "https://github.com/synesthesiam/prebuilt-apps/releases/download/v1.0/onnxruntime-linux-arm32-${ONNXRUNTIME_VERSION}.tgz")
        else()
            message(FATAL_ERROR "Unsupported architecture for onnxruntime")
        endif()

        set(ONNXRUNTIME_EXT "tgz")
    endif()

    if(NOT DEFINED ONNXRUNTIME_URL)
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_PREFIX}.${ONNXRUNTIME_EXT}")
    endif()

    set(ONNXRUNTIME_FILENAME "${ONNXRUNTIME_PREFIX}.${ONNXRUNTIME_EXT}")
    set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_LIST_DIR}/lib/${ONNXRUNTIME_PREFIX}")

    if(NOT EXISTS "${ONNXRUNTIME_DIR}")
        if(NOT EXISTS "download/${ONNXRUNTIME_FILENAME}")
            # Download onnxruntime release
            message("Downloading ${ONNXRUNTIME_URL}")
            file(DOWNLOAD "${ONNXRUNTIME_URL}" "download/${ONNXRUNTIME_FILENAME}")
        endif()

        # Extract .zip or .tgz to a directory like lib/onnxruntime-linux-x64-1.22.0/
        file(ARCHIVE_EXTRACT INPUT "download/${ONNXRUNTIME_FILENAME}" DESTINATION "${CMAKE_CURRENT_LIST_DIR}/lib")
    endif()
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
    "${ONNXRUNTIME_DIR}/include"
)

target_link_directories(${PROJECT_NAME} PUBLIC
    "${ONNXRUNTIME_DIR}/lib"
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    onnxruntime
)

else()

target_include_directories(${PROJECT_NAME} PUBLIC ${LIBPIPER_ONNXRUNTIME_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME} PUBLIC ${LIBPIPER_ONNXRUNTIME_TARGET_LINK_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC
    -Wl,--start-group
    ${LIBPIPER_ONNXRUNTIME_TARGET_LINK_FILES}
    -Wl,--end-group
)

endif()