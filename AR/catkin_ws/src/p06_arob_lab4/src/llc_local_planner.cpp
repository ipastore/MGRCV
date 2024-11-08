#include <pluginlib/class_list_macros.h>

#include "AROB_lab4/llc_local_planner.h"



//register this planner as a BaseLocalPlanner plugin
PLUGINLIB_EXPORT_CLASS(p06_llc_local_planner::LLCLocalPlanner,
nav_core::BaseLocalPlanner)

using namespace std;

namespace p06_llc_local_planner{

	double euclideanDistance(const geometry_msgs::Pose pose1, const geometry_msgs::Pose pose2){
		double ex = pose2.position.x - pose1.position.x;
		double ey = pose2.position.y - pose1.position.y;

		return std::sqrt(ex*ex+ey*ey);
	}

	LLCLocalPlanner::LLCLocalPlanner(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros){
		initialize(name, tf, costmap_ros);
	}

	void LLCLocalPlanner::initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros){

		if(!initialized_){

			// std::cout << "Initialize ..." << std::endl;

			costmap_ros_ = costmap_ros;
			costmap_ = costmap_ros_->getCostmap();
			tf_ = tf;

			ros::NodeHandle nh("~/" + name);
			ros::NodeHandle nh_local("~/local_costmap/");

			nh.param("kalpha", kalpha_, 0.0);
			nh.param("krho", krho_, 0.0);
			nh.param("kbeta", kbeta_, 0.0);
			nh.param("rho_th", rho_th_, 0.0);
			nh.param("yaw_th", yaw_th_, 0.0);

			nh_local.getParam("robot_radius", robot_radius_);

			std::cout << "Parameters: kalpha: " << kalpha_ << ", krho: " << krho_ << ", kbeta: " << kbeta_;
			std::cout << ", rho_th: " << rho_th_ << std::endl;
			std::cout << ", yaw_th: " << yaw_th_ << std::endl;
			std::cout << "Robot radius: " << robot_radius_ << std::endl;

			initialized_ = true;

		}else{
			ROS_WARN("This planner has already been initialized, doing nothing.");
		}

	}

	bool LLCLocalPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped>& plan){

		// std:cout << "setPlan ..." << std::endl;
		
		if(!initialized_){
			ROS_ERROR("THe planner has not been initialized.");
			return false;
		}

		//reset the global plan
		global_plan_.clear();

		// update obstacle info from costmap
		costmap_ = costmap_ros_->getCostmap();

		//Prune the plan to store the set of points within the local costmap
		for (auto it = plan.begin(); it != plan.end(); ++it){

			unsigned mx, my;
			if (costmap_->worldToMap((*it).pose.position.x, (*it).pose.position.y, mx, my))
				global_plan_.push_back((*it));
		}

		if (global_plan_.empty()){
			ROS_WARN("Global plan empty");
			return false;
		}

	   	return true;
	}

	bool LLCLocalPlanner::computeVelocityCommands(geometry_msgs::Twist& cmd_vel){
	    //Compute the velocity command (v,w) for a differential-drive robot

		// std::cout << "ComputeVelocityCommands ..." << std::endl;

		if(!initialized_){
			ROS_ERROR("THe planner has not been initialized.");
			return false;
		}

		//Get robot and goal pose in the global frame of the local costmap
		geometry_msgs::PoseStamped robot_pose;
		costmap_ros_->getRobotPose(robot_pose);
		geometry_msgs::PoseStamped goal = global_plan_.back();

		//Read obstacle information from the costmap
		costmap_ = costmap_ros_->getCostmap();
		for (unsigned int i=0; i<costmap_->getSizeInCellsX()-1; i++){
			for (unsigned int j=0; j<costmap_->getSizeInCellsY()-1; j++){
				if (costmap_->getCost(i,j) == costmap_2d::LETHAL_OBSTACLE){
					double obs_wx, obs_wy;
					geometry_msgs::Pose obs_pose;					
					costmap_->mapToWorld(i, j, obs_pose.position.x, obs_pose.position.y); //mapToWorld returns coordinates in the global frame of the local costmap
					// std::cout << "Obs coordinates: " << obs_pose.position.x << ", " << obs_pose.position.y << std::endl;
					
					if (euclideanDistance(robot_pose.pose, obs_pose) < 1.5*robot_radius_){
						ROS_ERROR("Imminent collision");
						return false;
					}

				}
			}
		}
		

		// Compute polar coordinates
        double delta_x = goal.pose.position.x - robot_pose.pose.position.x;
        double delta_y = goal.pose.position.y - robot_pose.pose.position.y;
        double rho = std::sqrt(delta_x * delta_x + delta_y * delta_y);
        double theta = tf::getYaw(robot_pose.pose.orientation);
        double alpha = -theta + std::atan2(delta_y, delta_x);
        double beta = -theta - alpha;

		// Normalize alpha and beta to [-pi, pi]
		alpha = angles::normalize_angle(alpha);
		beta = angles::normalize_angle(beta);

		// Apply control law
        cmd_vel.linear.x = krho_ * rho;
        cmd_vel.angular.z = kalpha_ * alpha + kbeta_ * beta;

        // // Stop the robot if the goal is near
        // if (rho < rho_th_) {
        //     cmd_vel.linear.x = 0.0;
        //     cmd_vel.angular.z = 0.0;
        //     return true;
        // }


		return true;
	}

	bool LLCLocalPlanner::isGoalReached(){
		//Check if the robot has reached the position and orientation of the goal

		// std::cout << "isGoalReached ..." << std::endl;

		if (! initialized_) {
			ROS_ERROR("This planner has not been initialized.");
			return false;
		}

		//Get robot and goal pose in the global frame of the local costmap
		geometry_msgs::PoseStamped robot_pose;
		costmap_ros_->getRobotPose(robot_pose);
		const geometry_msgs::PoseStamped goal = global_plan_.back();

		// Check distance to goal
        double delta_x = goal.pose.position.x - robot_pose.pose.position.x;
        double delta_y = goal.pose.position.y - robot_pose.pose.position.y;
        double rho = std::sqrt(delta_x * delta_x + delta_y * delta_y);

		// Compute yaw difference
		double robot_yaw = tf::getYaw(robot_pose.pose.orientation);
		double goal_yaw = tf::getYaw(goal.pose.orientation);
		double yaw_error = angles::normalize_angle(goal_yaw - robot_yaw);

    	return (rho < rho_th_ && std::abs(yaw_error) < yaw_th_);
	}

};
