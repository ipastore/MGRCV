cmake_minimum_required(VERSION 3.0.2)
project(arob_lab5)

find_package(catkin REQUIRED COMPONENTS
  nav_core
  pluginlib
  geometry_msgs
  std_msgs
  nav_msgs
  roscpp
  rospy
  tf2
  tf2_ros
  tf2_geometry_msgs
  base_local_planner
  message_generation
)

find_package(Eigen3 REQUIRED)


## Generate messages in the 'msg' folder
#add_message_files(
#  FILES
#  my_msg.msg
#)

## Generate services in the 'srv' folder
#  add_service_files(
#    FILES
#    my_srv.srv
# )

#generate_messages(DEPENDENCIES std_msgs )
## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES arob_lab5
  CATKIN_DEPENDS nav_core pluginlib geometry_msgs std_msgs nav_msgs roscpp tf2 tf2_ros tf2_geometry_msgs base_local_planner message_runtime
#  DEPENDS system_lib
  DEPENDS EIGEN3
)

include_directories(
	include 
	${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIRS}
)

add_library(rrt_global_planner_lib src/rrt_global_planner.cpp src/TreeNode.cpp)
target_link_libraries(rrt_global_planner_lib ${catkin_LIBRARIES})

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
