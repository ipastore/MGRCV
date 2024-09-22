cmake_minimum_required(VERSION 3.16)

project (data_structure VERSION 1.0 LANGUAGES CXX)

include_directories(include)

add_executable(rational-struct src/rational‐struct/rational.cpp src/rational‐struct/calculator.cpp)

add_executable(rational‐class src/rational‐class/rational_class.cpp src/rational‐class/calculator.cpp)