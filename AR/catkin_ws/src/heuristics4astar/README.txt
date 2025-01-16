========================================================
  README - heuristics4astar
========================================================

CONTENTS:
  1) Overview
  2) Launching the Code
  3) Changing the Heuristic at Runtime
  4) Sending a Navigation Goal
  5) Example Workflow


1) OVERVIEW
-----------
This package implements an A* global planner plugin for the ROS
Navigation Stack with runtime-switchable heuristics (Manhattan, Euclidean,
Chebyshev). It extends nav_core::BaseGlobalPlanner, so it can be loaded by
move_base like any standard global planner plugin.

The plugin is defined as "heuristics4astar/GlobalPlanner" and is loaded
via pluginlib. In the 'heuristics4astar.launch' file, you can set which map is
loaded and set other move_base parameters.


2) LAUNCHING THE CODE
---------------------
To start the simulation (or real robot setup) and load the heuristics4astar
planner, run:

    roslaunch heuristics4astar heuristics4astar.launch


If you want to change the map, edit the launch file parameter that points to
the map YAML and .world files. 


3) CHANGING THE HEURISTIC AT RUNTIME
------------------------------------
We expose the parameter 'heuristic_type' via dynamic_reconfigure. You can
choose among:
  0 => Manhattan
  1 => Euclidean
  2 => Chebyshev

Method A) Using the command line:

    rosrun dynamic_reconfigure dynparam set /move_base/GlobalPlanner heuristic_type 2

Method B) Using RQT:

    rosrun rqt_reconfigure rqt_reconfigure

In the GUI, expand '/move_base/GlobalPlanner' and change 'heuristic_type' among
(0,1,2).


4) SENDING A NAVIGATION GOAL
----------------------------
To make the robot plan to a new goal, you can do a single command:

    rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped "header:
      seq: 0
      stamp: {secs: 0, nsecs: 0}
      frame_id: 'map'
    pose:
      position: {x: 5.5, y: 6.5, z: 0.0}
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}"

This sets a new goal at (5.5, 6.5) in the 'map' coordinate frame. The
heuristics4astar plugin will compute the plan. You should see the robot
move (in Stage/Gazebo or physically) to that goal if everything is configured.


5) EXAMPLE WORKFLOW
-------------------
Assuming you have a standard setup with the 'heuristics4astar' package built:

  a) Launch:
       roslaunch heuristics4astar heuristics4astar.launch
  b) (Optional) Switch heuristic at runtime:
       rosrun dynamic_reconfigure dynparam set /move_base/GlobalPlanner heuristic_type 1
  c) Send a goal:
       rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped ...
  d) Observe the plan and motion in RViz and the simulator.

That’s it! You should now see the robot plan using A* with your chosen heuristic.
Check the logs or CSV outputs (if enabled) for planning times, path length, etc.

-----------------------------------------------------
For more details on code structure and experiments,
please see the full documentation or contact the authors.