cmake_minimum_required(VERSION 3.15)
project(MiniGimp VERSION 1.0 LANGUAGES CXX)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(X11 REQUIRED)

find_package(PNG)
if(PNG_FOUND)
    include_directories(${PNG_INCLUDE_DIR})
    list(APPEND cimg_libs ${PNG_LIBRARY})
    list(APPEND cimg_defs "cimg_use_png")
endif(PNG_FOUND)

find_package(JPEG)
if(JPEG_FOUND)
    include_directories(${JPEG_INCLUDE_DIR})
    list(APPEND cimg_libs ${JPEG_LIBRARY})
    list(APPEND cimg_defs "cimg_use_jpeg")
endif(JPEG_FOUND)

find_package(TIFF)
if(TIFF_FOUND)
    include_directories(${TIFF_INCLUDE_DIR})
    list(APPEND cimg_libs ${TIFF_LIBRARY})
    list(APPEND image_defs "cimg_use_tiff")
endif(TIFF_FOUND)

if(FFTW3_FOUND)
    include_directories(${FFTW3_INCLUDE_DIR})
    list(APPEND cimg_libs ${FFTW3_LIBRARY})
    list(APPEND cimg_defs "cimg_use_fftw3")
endif(FFTW3_FOUND)

if(X11_FOUND)
    include_directories(${X11_INCLUDE_DIR})
    list(APPEND cimg_libs ${X11_LIBRARIES})
    list(APPEND cimgdisplay_libs ${X11_LIBRARIES})
    list(APPEND cimgdisplay_defs cimg_display=1)
    list(APPEND cimg_defs cimg_OS=1)
endif(X11_FOUND)

include_directories(${CMAKE_CURRENT_LIST_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")

add_executable(minigimp brushes.cpp color.cpp grayscale.cpp main.cpp boke.cpp)
target_compile_definitions(minigimp PRIVATE ${cimg_defs} ${cimgdisplay_defs})
target_link_libraries(minigimp ${cimg_libs} ${cimgdisplay_libs})


include_directories(/opt/homebrew/include)
link_directories(/opt/homebrew/lib)
target_link_libraries(minigimp /opt/homebrew/lib/libX11.dylib /opt/homebrew/lib/libXext.dylib)