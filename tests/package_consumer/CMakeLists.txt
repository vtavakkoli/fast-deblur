cmake_minimum_required(VERSION 3.22)
project(fast_deblur_package_consumer LANGUAGES CXX)

find_package(fast_deblur CONFIG REQUIRED)

add_executable(fast-deblur-package-consumer main.cpp)
target_link_libraries(fast-deblur-package-consumer PRIVATE fast_deblur::fast_deblur)
