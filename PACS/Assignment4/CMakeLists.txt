cmake_minimum_required(VERSION 3.0.2)
project(lab1)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


add_executable(smallpt_thread_pool smallpt_thread_pool.cpp)
target_include_directories(smallpt_thread_pool
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

