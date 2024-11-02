# Install script for directory: /Users/ignaciopastorebenaim/Documents/MGRCV/TPs/MSAP/NORI-2024_mac/ext/openexr/IlmBase/Iex

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Library/Developer/CommandLineTools/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/MSAP/NORI-2024_mac/build/ext_build/openexr/IlmBase/Iex/libIex.a")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libIex.a" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libIex.a")
    execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/ranlib" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libIex.a")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/MSAP/NORI-2024_mac/build/ext_build/openexr/IlmBase/Iex/CMakeFiles/Iex.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/OpenEXR" TYPE FILE FILES
    "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/MSAP/NORI-2024_mac/ext/openexr/IlmBase/Iex/IexBaseExc.h"
    "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/MSAP/NORI-2024_mac/ext/openexr/IlmBase/Iex/IexMathExc.h"
    "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/MSAP/NORI-2024_mac/ext/openexr/IlmBase/Iex/IexThrowErrnoExc.h"
    "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/MSAP/NORI-2024_mac/ext/openexr/IlmBase/Iex/IexErrnoExc.h"
    "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/MSAP/NORI-2024_mac/ext/openexr/IlmBase/Iex/IexMacros.h"
    "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/MSAP/NORI-2024_mac/ext/openexr/IlmBase/Iex/Iex.h"
    "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/MSAP/NORI-2024_mac/ext/openexr/IlmBase/Iex/IexNamespace.h"
    "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/MSAP/NORI-2024_mac/ext/openexr/IlmBase/Iex/IexExport.h"
    "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/MSAP/NORI-2024_mac/ext/openexr/IlmBase/Iex/IexForward.h"
    )
endif()

