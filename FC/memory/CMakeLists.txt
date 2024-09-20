cmake_minimum_required(VERSION 3.30)

project (global_local VERSION 1.0 LANGUAGES CXX)

include_directories(include)

add_executable(global_local src/global_local.cpp)

add_executable(write_through_ptr_ref src/write_through_ptr_ref.cpp)

add_executable(to_fix_double_delete src/to_fix_double_delete.cpp)

add_executable(to_fix_missing_new src/to_fix_missing_new.cpp)

add_executable(variables src/variables.cpp)

add_executable(variables_wout_pointers src/variables_wout_pointers.cpp)

add_executable(max_min_global_2d_matrix src/max_min_global_2d_matrix.cpp)

