cmake_minimum_required(VERSION 3.16)

project(04_Smart_pointers VERSION 1.0 LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(to_fix_memory_leaks src/to_fix_memory_leaks.cpp)

add_executable(shared_ptr src/shared_ptr.cpp)

