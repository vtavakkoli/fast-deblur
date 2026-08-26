cmake_minimum_required(VERSION 3.22)
project(fast_deblur VERSION 0.2.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

option(FAST_DEBLUR_NATIVE "Enable -march=native for local CPU builds" OFF)
option(FAST_DEBLUR_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(FAST_DEBLUR_BUILD_CLI "Build the fast-deblur CLI" ON)
option(FAST_DEBLUR_BUILD_BENCHMARKS "Build the benchmark executable" ON)
option(FAST_DEBLUR_BUILD_TESTS "Build fast-deblur tests when BUILD_TESTING is enabled" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs photo)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
find_package(OpenMP QUIET)
if(FAST_DEBLUR_BUILD_BENCHMARKS)
    find_package(nlohmann_json 3.10 REQUIRED)
endif()

set(FAST_DEBLUR_SOURCES
    src/fft.cpp
    src/boundary.cpp
    src/dark_channel.cpp
    src/kernel.cpp
    src/trajectory.cpp
    src/trajectory_api.cpp
    src/optimization.cpp
    src/quality.cpp
    src/refinement.cpp
    src/saturation.cpp
    src/deblur.cpp
)

add_library(fast_deblur ${FAST_DEBLUR_SOURCES})
add_library(fast_deblur::fast_deblur ALIAS fast_deblur)
target_compile_features(fast_deblur PUBLIC cxx_std_20)
target_include_directories(fast_deblur
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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

if(FAST_DEBLUR_BUILD_CLI)
    add_executable(fast-deblur apps/fast-deblur.cpp)
    target_link_libraries(fast-deblur PRIVATE fast_deblur::fast_deblur)
endif()

if(FAST_DEBLUR_BUILD_BENCHMARKS)
    add_executable(fast-deblur-benchmark apps/benchmark.cpp)
    target_link_libraries(
        fast-deblur-benchmark
        PRIVATE fast_deblur::fast_deblur nlohmann_json::nlohmann_json
    )
endif()

if(BUILD_TESTING AND FAST_DEBLUR_BUILD_TESTS)
    add_executable(fast-deblur-tests tests/test_core.cpp)
    target_include_directories(fast-deblur-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(fast-deblur-tests PRIVATE fast_deblur::fast_deblur)
    add_test(NAME fast-deblur-core COMMAND fast-deblur-tests)
endif()

install(
    TARGETS fast_deblur
    EXPORT fast_deblurTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if(FAST_DEBLUR_BUILD_CLI)
    install(TARGETS fast-deblur RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(FAST_DEBLUR_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/fast_deblur")
set(FAST_DEBLUR_CONFIG_NEEDS_OPENMP ${OpenMP_CXX_FOUND})
configure_package_config_file(
    cmake/fast_deblurConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/fast_deblurConfig.cmake"
    INSTALL_DESTINATION "${FAST_DEBLUR_CMAKE_INSTALL_DIR}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/fast_deblurConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    EXPORT fast_deblurTargets
    FILE fast_deblurTargets.cmake
    NAMESPACE fast_deblur::
    DESTINATION "${FAST_DEBLUR_CMAKE_INSTALL_DIR}"
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/fast_deblurConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/fast_deblurConfigVersion.cmake"
    DESTINATION "${FAST_DEBLUR_CMAKE_INSTALL_DIR}"
)
