cmake_minimum_required(VERSION 3.0.2)
project(second_package)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES second_package
)

include_directories(include ${catkin_INCLUDE_DIRS})


add_executable(helloWorld src/helloWorld.cpp)


target_link_libraries(helloWorld ${catkin_LIBRARIES})


