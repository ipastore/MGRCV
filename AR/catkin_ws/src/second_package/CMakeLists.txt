cmake_minimum_required(VERSION 3.0.2)
project(second_package)

## Encuentra los paquetes de catkin y cualquier otra dependencia
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

## Declara el paquete de catkin
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs
)

## Incluir directorios
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Declara un ejecutable
add_executable(helloWorld src/helloWorld.cpp)

## Especifica las bibliotecas que deben enlazarse al ejecutable
target_link_libraries(helloWorld
  ${catkin_LIBRARIES}
)

## Añade dependencias para garantizar el orden correcto de compilación
add_dependencies(helloWorld ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
