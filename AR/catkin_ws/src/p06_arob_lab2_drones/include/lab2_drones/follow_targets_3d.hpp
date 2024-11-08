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
#include <tf2_ros/transform_broadcaster.h> // The added dependency should be tf2 ----> DONE

class FollowTargets3D {

public:
	FollowTargets3D(ros::NodeHandle nh );
    ~FollowTargets3D() {}

private:
    // INFO: It is a convention to use _ for private variables and methods
    // VARIABLES
    ros::NodeHandle nh_;

    // Publisher and subscribers
    ros::Publisher goal_pub_;
    ros::Subscriber odometry_sub_;

    // Timer for continuous goal publishing
    ros::Timer goal_timer_;

    bool odometry_received_;
    std::string targets_file_path_;
    std::vector<geometry_msgs::Pose> targets_; //List of goal targets
    geometry_msgs::PoseStamped current_goal_;
    geometry_msgs::Point current_position_;
    int current_goal_index_;

    // METHODS
    bool readTargets_(std::string file);
    //  EXERCISE: Implement the logic to publish new goals when the previous one is reached
    void goalTimerCallback_(const ros::TimerEvent& event);
    void odometryCallback_(const nav_msgs::Odometry& msg);
};

#endif