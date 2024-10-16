#ifndef FOLLOW_TARGETS_3D_HPP
#define FOLLOW_TARGETS_3D_HPP

#include <vector>
#include <math.h>
#include <fstream>

#include <iostream>
#include <sstream>
#include <stdio.h> 

// EXERCISE: Add the dependencies to the CMakeLists.txt and package.xml files
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h> // The added dependency should be tf2

class FollowTargets3D {

public:
	FollowTargets3D(ros::NodeHandle nh );

    ~FollowTargets3D() {}

private:
    // INFO: It is a convention to use _ for private variables and methods
    ros::NodeHandle nh_;

    // Declare your publisher and subscribers

    std::string targets_file_path_;
    std::vector<geometry_msgs::Pose> targets_; //List of goal targets
    int current_goal_index_;
    geometry_msgs::PoseStamped current_goal_;

    bool readTargets_(std::string file);
    
    //  EXERCISE: Implement the logic to publish new goals when the previous one is reached
    //  (more info in the constructor of this class).
    void odometryCallback_(const nav_msgs::Odometry& msg);
};

#endif