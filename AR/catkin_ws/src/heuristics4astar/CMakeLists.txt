cmake_minimum_required(VERSION 3.0.2)
project(heuristics4astar)


find_package(catkin REQUIRED COMPONENTS
  costmap_2d
  nav_core
  pluginlib
  roscpp
  global_planner
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES custom_astar_lib
  CATKIN_DEPENDS roscpp pluginlib nav_core global_planner costmap_2d
  # DEPENDS system_lib
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)


add_library(custom_astar_lib src/custom_astar.cpp)
target_link_libraries(custom_astar_lib ${catkin_LIBRARIES})


install(TARGETS custom_astar_lib
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
install(FILES custom_astar_plugin.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")

