# Laboratory 2 - lab2_drones

## Instructions

Place this package in your workspace.
You have to fill missing content in several files:
- Start by understanding the included "_node" file in "src/" and the header file.
- Fill in dependencies used in this package in the "CMakeLists.txt" and "package.xml".
- Complete the code in the header and source files to add the required functionality.

Once all the code is added, compile the package with `catkin build`remember to source the workspace to find the node. In order to run the node, you will have to pass the path to one of the "targetsN.txt" files in "data". You can pass a parameter when using rosrun with_
```
rosrun package_name node_name _parameter_name:=parameter_value
```

**NOTE:** Notice the "_" before the parameter name!