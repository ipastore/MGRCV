 bool LLCLocalPlanner::computeVelocityCommands(geometry_msgs::Twist& cmd_vel) {
        if (!initialized_) {
            ROS_ERROR("The planner has not been initialized.");
            return false;
        }

        // Get robot and goal pose
        geometry_msgs::PoseStamped robot_pose;
        costmap_ros_->getRobotPose(robot_pose);
        geometry_msgs::PoseStamped goal = global_plan_.back();

        // Compute polar coordinates
        double delta_x = goal.pose.position.x - robot_pose.pose.position.x;
        double delta_y = goal.pose.position.y - robot_pose.pose.position.y;
        double rho = std::sqrt(delta_x * delta_x + delta_y * delta_y);
        double theta = tf::getYaw(robot_pose.pose.orientation);
        double alpha = -theta + std::atan2(delta_y, delta_x);
        double beta = -theta - alpha;

        // Apply control law
        cmd_vel.linear.x = krho_ * rho;
        cmd_vel.angular.z = kalpha_ * alpha + kbeta_ * beta;

        // Stop the robot if the goal is near
        if (rho < rho_th_) {
            cmd_vel.linear.x = 0.0;
            cmd_vel.angular.z = 0.0;
            return true;
        }

        // Collision avoidance
        for (unsigned int i = 0; i < costmap_->getSizeInCellsX(); i++) {
            for (unsigned int j = 0; j < costmap_->getSizeInCellsY(); j++) {
                if (costmap_->getCost(i, j) == costmap_2d::LETHAL_OBSTACLE) {
                    geometry_msgs::Pose obs_pose;
                    costmap_->mapToWorld(i, j, obs_pose.position.x, obs_pose.position.y);

                    if (euclideanDistance(robot_pose.pose, obs_pose) < 1.5 * robot_radius_) {
                        ROS_ERROR("Imminent collision detected");
                        cmd_vel.linear.x = 0.0;
                        cmd_vel.angular.z = 0.0;
                        return false;
                    }
                }
            }
        }

        return true;
    }