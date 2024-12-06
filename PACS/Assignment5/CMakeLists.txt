# Minimum CMake version required
cmake_minimum_required(VERSION 3.5)

# Project name
project(Assignment5)

# Add the executable for Laboratory 5
add_executable(basic_environ basic_environ.c)
add_executable(CImg_prueba CImg_prueba.cpp)
add_executable(sobel sobel.cpp)

# Add the include directories
include_directories(${CMAKE_SOURCE_DIR}/CImg)

# Add the include and library paths for X11
include_directories(/opt/X11/include)
link_directories(/opt/X11/lib)

# Include the jpeg-turbo headers
include_directories(/opt/homebrew/include)

# Link against OpenCL library
find_package(OpenCL REQUIRED)
find_package(X11 REQUIRED)
find_package(OpenGL REQUIRED)

#Link libraries
target_link_libraries(basic_environ PRIVATE OpenCL::OpenCL X11)
# target_link_libraries(CImg_prueba PRIVATE OpenGL::GL X11)
target_link_libraries(sobel PRIVATE OpenCL::OpenCL X11 OpenGL::GL /opt/homebrew/lib/libjpeg.dylib)  
target_link_libraries(CImg_prueba PRIVATE X11 OpenGL::GL /opt/homebrew/lib/libjpeg.dylib)

# Set compilation options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set compilation flags
target_compile_options(basic_environ PRIVATE -Wall -Wextra)
target_compile_options(CImg_prueba PRIVATE -Wall -Wextra)