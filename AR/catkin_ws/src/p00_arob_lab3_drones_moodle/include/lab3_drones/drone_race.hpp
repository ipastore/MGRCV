#ifndef DRONE_RACE_HPP
#define DRONE_RACE_HPP

#include <iostream>
#include <sstream>
#include <stdio.h> 
#include <math.h>
#include <fstream>

#include <eigen3/Eigen/Core>

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/MarkerArray.h>

#include <mav_trajectory_generation/polynomial_optimization_linear.h>
#include <mav_trajectory_generation/trajectory_sampling.h>
#include <mav_trajectory_generation_ros/ros_visualization.h>
#include <mav_trajectory_generation_ros/ros_conversions.h>
#include <mav_trajectory_generation_ros/ros_visualization.h>

class DroneRace {

public:
	DroneRace(ros::NodeHandle nh );

    ~DroneRace() {};

private:
    // INFO: It is a convention to use _ for private variables and methods
    ros::NodeHandle nh_;

    //ROS publishers-suscribers
    ros::Publisher pub_traj_markers_;
    ros::Publisher pub_traj_vectors_;
    ros::Publisher pub_gate_markers_;
    ros::Publisher pub_goal_;
    ros::Publisher pub_cmd_vel_;

    ros::Timer cmd_timer_;

    // Mission description
    std::string targets_file_path_;
    std::vector<geometry_msgs::Pose> gates_;
    std::vector<Eigen::Vector3d> gate_normals_;

    // Trajectory
    mav_trajectory_generation::Trajectory trajectory_;
    std::vector<Eigen::Vector3f> goal_list_;
    std::vector<Eigen::Vector3f> goal_vel_list_;
    
    // Control
    int current_goal_idx_;
    bool is_pose_control_; // Specifies if the control is in pose or vel

    geometry_msgs::PoseStamped goal_;
    geometry_msgs::Twist goal_vel_;

    //Id markers
    int id_marker = 0;

    bool readGates_(std::string file_name);
    void getGateNormal();

    void generateTrajectoryExample_(); 
    void generateTrajectory_();
    void generate_gates_vertex_(mav_trajectory_generation::Vertex::Vector &vertices);

    void commandTimerCallback_(const ros::TimerEvent& event);


    Eigen::Matrix<double, 3, 3> RPYtoRMatrix_(double roll, double pitch, double yaw);
    Eigen::Matrix<double, 3, 3> quatToRMatrix_(geometry_msgs::Quaternion q);
    geometry_msgs::Quaternion RPYToQuat_(double roll, double pitch, double yaw);

    void drawGoalMarker_(mav_trajectory_generation::Vertex goal);
    void drawGates_();
    void drawGateMarkers_(geometry_msgs::Pose gate, int &id);
    void drawTrajectoryMarkers_();
};

#endif