cmake_minimum_required(VERSION 3.10)

# Project name
project(Lab6)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find OpenCL library
find_package(OpenCL REQUIRED)

# Add executables
add_executable(image_simulation src/image_simulation.cpp)
target_link_libraries(image_simulation OpenCL)

add_executable(gpu_balancer src/gpu_balancer.cpp)
target_link_libraries(gpu_balancer OpenCL)

# Add CImg library
include_directories(${CMAKE_SOURCE_DIR}/include/CImg)
