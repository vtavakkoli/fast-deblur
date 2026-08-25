cmake_minimum_required(VERSION 3.22)
project(fast_deblur VERSION 0.1.0 LANGUAGES CXX)

option(FAST_DEBLUR_NATIVE "Enable -march=native for local CPU builds" OFF)
option(FAST_DEBLUR_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs photo)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
find_package(nlohmann_json 3.10 REQUIRED)
find_package(OpenMP QUIET)

set(FAST_DEBLUR_SOURCES
    src/fft.cpp
    src/boundary.cpp
    src/dark_channel.cpp
    src/kernel.cpp
    src/optimization.cpp
    src/quality.cpp
    src/refinement.cpp
    src/saturation.cpp
    src/deblur.cpp
)

add_library(fast_deblur ${FAST_DEBLUR_SOURCES})
add_library(fast_deblur::fast_deblur ALIAS fast_deblur)
target_include_directories(fast_deblur
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(fast_deblur
    PUBLIC ${OpenCV_LIBS}
    PRIVATE PkgConfig::FFTW3
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fast_deblur PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(fast_deblur PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow;-O3>
)
if(FAST_DEBLUR_WARNINGS_AS_ERRORS)
    target_compile_options(fast_deblur PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Werror>)
endif()
if(FAST_DEBLUR_NATIVE)
    target_compile_options(fast_deblur PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native>)
endif()

add_executable(fast-deblur apps/fast-deblur.cpp)
target_link_libraries(fast-deblur PRIVATE fast_deblur::fast_deblur)

add_executable(fast-deblur-benchmark apps/benchmark.cpp)
target_link_libraries(fast-deblur-benchmark PRIVATE fast_deblur::fast_deblur nlohmann_json::nlohmann_json)

enable_testing()
add_executable(fast-deblur-tests tests/test_core.cpp)
target_include_directories(fast-deblur-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(fast-deblur-tests PRIVATE fast_deblur::fast_deblur)
add_test(NAME fast-deblur-core COMMAND fast-deblur-tests)
