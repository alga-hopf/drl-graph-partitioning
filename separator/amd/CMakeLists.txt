cmake_minimum_required(VERSION 3.2)
project(AMDWrapper CXX Fortran)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)

add_library(AMDWrapper SHARED amdbar.F AMDReordering.cpp)
