#include "lab2_drones/follow_targets_3d.hpp"

FollowTargets3D::FollowTargets3D(ros::NodeHandle nh) : nh_(nh)
{
    // Read from parameters the path for the targets,
    // otherwise use a default value.
    if (!nh_.getParam("targets_file_path", targets_file_path_))
    {
        ROS_WARN("There is no 'targets_file_path' parameter. Using default value.");
        targets_file_path_ = "/home/arob/catkin_ws/src/pXX_arob_lab2_drones/data/targets.txt";
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
    // ADDITIONAL EXERCISE: : Usually, it is a good practice to wait until odometry is received before sending
    // any goal. Implement the logic to achieve this.

    // EXERCISE: Define your publishers and subscribers
    // Find the topic to send pose commands and the type of message it receives

    // EXERCISE: Find a topic to listen to the position of the quadrotor in order to send a new goal
    // once the previous is reached. 
    // INFO: In order to pass as a callback a member of a function, we have to pass a reference 
    // to the callback method and a reference to the object of the class:
    // https://wiki.ros.org/roscpp_tutorials/Tutorials/UsingClassMethodsAsCallbacks
    // (THE LAST SENTENCE IN SUBSCRIPTIONS IS REALLY IMPORTANT)

    // ADDITIONAL EXERCISE: In some cases, robot controllers might need to receive the goal continuously
    // (e.g. at a frequency of 100Hz).
    // You will have to save the next_goal_id and send the goal in an interval.
    // Find how this is commonly done in ROS.

    ROS_INFO("FollowTargets3D initialized");
}

bool FollowTargets3DSol::readTargets_(std::string file_path)
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

}
