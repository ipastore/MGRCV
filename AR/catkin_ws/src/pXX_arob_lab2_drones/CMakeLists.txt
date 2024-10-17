cmake_minimum_required(VERSION 3.0.2)
project(lab2_drones)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Add dependencies here (look for them in imported libraries in the header files)
# REMOVE dependencies!!
find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
  tf2_ros
  roslib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES 
  CATKIN_DEPENDS roscpp geometry_msgs nav_msgs tf2_ros roslib
  DEPENDS
)

###########
## Build ##
###########

# NOTE: Everytime catkin_SOMETHING gets called, catkin takes care of adding
# dependencies, includes and links detected by the package manager. That is
# why catkin is easier than pure cmake. catkin_SOMETHING is generated when
# catkin_package() is called.



# NOTE: There are two options to add a .cpp into a executable node
# We recommend option 2 but option 1 is better for larger projects.

# 1: Declare a C++ library containing parts of the code that
# are more general and can be used in multiple executables (nodes)
# add_library(${PROJECT_NAME}_lib
#   src/follow_targets_3d_sol.cpp
# )
## And link catkin libraries against it
# target_link_libraries(${PROJECT_NAME}_lib
#   ${catkin_LIBRARIES}
# )
# Then you will have to target the executables against this library

# 2: Declare a C++ executable and add all the source files to it
# (RECOMMENDED FOR THIS LAB)


## Build the executable
add_executable(follow_targets_3d src/follow_targets_3d.cpp src/follow_targets_3d_node.cpp)
target_link_libraries(follow_targets_3d ${catkin_LIBRARIES})


#############
## Install ##
#############

# Not for now