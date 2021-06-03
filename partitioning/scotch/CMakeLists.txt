cmake_minimum_required(VERSION 3.2)
project(SCOTCHWrapper CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
list(APPEND CMAKE_PREFIX_PATH $ENV{SCOTCH_DIR} $ENV{SCOTCH_ROOT})
find_package(SCOTCH)

add_library(SCOTCHWrapper SHARED SCOTCHWrapper.cpp)
target_link_libraries(SCOTCHWrapper PUBLIC SCOTCH::scotch)
