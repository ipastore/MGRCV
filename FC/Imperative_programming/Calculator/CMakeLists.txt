cmake_minimum_required(VERSION 3.30)

project (Calculator VERSION 1.0 LANGUAGES CXX)

# Set compiler flags
#Debug
################ SWITCH MACOS AND WINDOWS ################
# Set compiler flags
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Windows-specific settings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX /Od /Zi")
else()
    # macOS and other Unix-like systems
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -O0 -g")
endif()
################ SWITCH MACOS AND WINDOWS ################

#Release
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -03 -s")

add_executable(calculator calculator.cpp)


