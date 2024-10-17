#include "lab2_drones/follow_targets_3d.hpp"
#include <ros/package.h>

FollowTargets3D::FollowTargets3D(ros::NodeHandle nh) : nh_(nh)
{
    // Read from parameters the path for the targets,
    // otherwise use a default value.
    if (!nh_.getParam("targets_file_path", targets_file_path_))
    {
        ROS_WARN("There is no 'targets_file_path' parameter. Using default value.");
        targets_file_path_ = "/home/padidavid/Documents/unizar/master/repositorios/MGRCV/AR/catkin_ws/src/pXX_arob_lab2_drones/data/targets.txt";
    }
    // Try to open the targets file.
    if (!readTargets_(targets_file_path_))
    {
        ROS_ERROR("Could not read targets from file: %s", targets_file_path_.c_str());
        ros::shutdown();
    }

    // Assign a first goal and save the current index. In this case, the goal is the first target.
    current_goal_index_ = 0;
    current_goal_.pose.position = targets_[current_goal_index_].position;
    // Even if we only want the position, it is a good idea to initialize rotation to zero.
    current_goal_.pose.orientation = geometry_msgs::Quaternion();
    
    
    // ADDITIONAL EXERCISE: Usually, it is a good practice to wait until odometry is received before sending
    // any goal. Implement the logic to achieve this.
    odometry_received_ = false; // Flag for domometry reception
    // Subscriber to odometry
    odometry_sub_ = nh_.subscribe("/ground_truth/state", 10, &FollowTargets3D::odometryCallback_, this);

    // EXERCISE: Define your publishers and subscribers
    // Find the topic to send pose commands and the type of message it receives
    // Publisher for the goal
    goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/command/pose", 10);

    // ADDITIONAL EXERCISE: In some cases, robot controllers might need to receive the goal continuously
    // (e.g. at a frequency of 100Hz).
    // You will have to save the next_goal_id and send the goal in an interval.
    // Find how this is commonly done in ROS.
    // Timer to periodically send the goal if odometry is received
    goal_timer_ = nh_.createTimer(ros::Duration(0.01), &FollowTargets3D::goalTimerCallback_, this);
   
    // INFO: In order to pass as a callback a member of a function, we have to pass a reference 
    // to the callback method and a reference to the object of the class:
    // https://wiki.ros.org/roscpp_tutorials/Tutorials/UsingClassMethodsAsCallbacks
    // (THE LAST SENTENCE IN SUBSCRIPTIONS IS REALLY IMPORTANT)

    ROS_INFO("FollowTargets3D initialized");
}

bool FollowTargets3D::readTargets_(std::string file_path)
{
    //Open the file
    std::ifstream input_file;
    input_file.open(file_path, std::ifstream::in);
    if (!input_file) {
        return false;
    }
    targets_.clear();
    geometry_msgs::Pose tempPose;
    std::string line;

    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        std::string value;

        // Read and parse x, y, z values separated by semicolons
        std::getline(iss, value, ';');
        tempPose.position.x = std::stod(value);
        
        std::getline(iss, value, ';');
        tempPose.position.y = std::stod(value);
        
        std::getline(iss, value, ';');
        tempPose.position.z = std::stod(value);

        targets_.push_back(tempPose);
    }

    // Close the file
    input_file.close();

    // Show the targets for debugging
    for (const auto& target : targets_) {
        std::cout << "Goal " << target.position.x << " "
                  << target.position.y << " "
                  << target.position.z << std::endl;
    }

    return true;
}

void FollowTargets3D::odometryCallback_(const nav_msgs::Odometry& msg)
{
    if (!odometry_received_) {
        // Odometry flag is set to FALSE by default
        // When the odometry is received for the first time, the flag is set to TRUE
        // Once it's TRUE will never get here again and we'ĺl start calculating the distance to the goal
        odometry_received_ = true;
        ROS_INFO("Odometry received. Ready to start sending goals.");
    } else {
        // EXERCISE: Find a topic to listen to the position of the quadrotor in order to send a new goal
        // once the previous is reached.
        // We get the current position of the drone
        current_position_ = msg.pose.pose.position;

        // We define a distance thresholde to know if we have reach (aprox) the goal
        const double goal_threshold = 0.5; 

        // We calculate the euclidean distance to the goal to check if we have reached it
        double distance_to_goal = std::sqrt(
            std::pow(current_goal_.pose.position.x - current_position_.x, 2) +
            std::pow(current_goal_.pose.position.y - current_position_.y, 2) +
            std::pow(current_goal_.pose.position.z - current_position_.z, 2)
        );

        // WE check if goal is reached
        if (distance_to_goal < goal_threshold) {
            // If we have more goals, we set the next goal in case there is one
            if (current_goal_index_ < targets_.size() - 1) {
            current_goal_index_++;
            current_goal_.pose.position = targets_[current_goal_index_].position;
            current_goal_.pose.orientation = geometry_msgs::Quaternion(); // Reset orientation
            ROS_INFO("Reached goal. Sending next goal [%.2f, %.2f, %.2f]",
                     current_goal_.pose.position.x, 
                     current_goal_.pose.position.y, 
                     current_goal_.pose.position.z);
            } else {
                // If we have reached all goals, we stop the drone
                ROS_INFO("All goals reached!");
            }
        }
        
    }

}

void FollowTargets3D::goalTimerCallback_(const ros::TimerEvent& event)
{
    // If odometry is not received, we do not send goals
    if (!odometry_received_) {
        odometry_received_ = true;
        ROS_INFO("Waiting for odometry data before sending goals.");
        return;
    }

    // Publish the goal
    goal_pub_.publish(current_goal_);
    ROS_INFO_THROTTLE(1, "Current position: [%.2f, %.2f, %.2f]", 
                      current_position_.x, 
                      current_position_.y, 
                      current_position_.z);
    ROS_INFO_THROTTLE(1, "Publishing goal: [%.2f, %.2f, %.2f]", 
                      current_goal_.pose.position.x, 
                      current_goal_.pose.position.y, 
                      current_goal_.pose.position.z);
}
