# Minimum CMake version required
cmake_minimum_required(VERSION 3.5)

# Project name
project(Assignment5)

# Add the executable for Laboratory 5
add_executable(basic_environ basic_environ.c)
add_executable(test_sobel_image_generator test_sobel_image_generator.cpp)
add_executable(sobel_gt sobel_gt.cpp)
add_executable(sobel_global_local sobel_global_local.cpp)

# Add the include directories
include_directories(${CMAKE_SOURCE_DIR}/CImg)
include_directories(/opt/X11/include)
include_directories(/opt/homebrew/include)

link_directories(/opt/X11/lib)

# Link against OpenCL library
find_package(OpenCL REQUIRED)
find_package(X11 REQUIRED)
find_package(OpenGL REQUIRED)

#Link libraries
target_link_libraries(basic_environ PRIVATE OpenCL::OpenCL X11)
target_link_libraries(test_sobel_image_generator PRIVATE X11 OpenGL::GL /opt/homebrew/lib/libjpeg.dylib)
target_link_libraries(sobel_gt PRIVATE OpenCL::OpenCL X11 OpenGL::GL /opt/homebrew/lib/libjpeg.dylib)
target_link_libraries(sobel_global_local PRIVATE OpenCL::OpenCL X11 OpenGL::GL /opt/homebrew/lib/libjpeg.dylib)

# Set compilation options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set compilation flags
target_compile_options(basic_environ PRIVATE -Wall -Wextra)
target_compile_options(sobel PRIVATE -Wall -Wextra)
target_compile_options(gaussian PRIVATE -Wall -Wextra)
target_compile_options(test_sobel_image_generator PRIVATE -Wall -Wextra)
target_compile_options(sobel_gt PRIVATE -Wall -Wextra)
target_compile_options(sobel_global_local PRIVATE -Wall -Wextra)