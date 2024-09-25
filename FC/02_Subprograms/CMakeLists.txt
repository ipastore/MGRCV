cmake_minimum_required(VERSION 3.30)

project (Triangles VERSION 1.0 LANGUAGES CXX)

include_directories(include)

add_executable(triangles src/functions.cpp src/triangle.cpp)

add_executable(tests src/functions.cpp src/tests.cpp)
