# Minimum CMake version required
cmake_minimum_required(VERSION 3.5)

# Add the executable for Laboratory 5
add_executable(Assignment5 basic_environ.c)

# Link against OpenCL library
find_package(OpenCL REQUIRED)
target_link_libraries(Assignment5 PRIVATE OpenCL::OpenCL)

# Set compilation options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
target_compile_options(Assignment5 PRIVATE -Wall -Wextra)