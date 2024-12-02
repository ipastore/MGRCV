#include <pluginlib/class_list_macros.h>
#include "AROB_lab5/rrt_global_planner.h"

//register this planner as a BaseGlobalPlanner plugin
PLUGINLIB_EXPORT_CLASS(rrt_planner::RRTPlanner, nav_core::BaseGlobalPlanner)

//Default Constructor
namespace rrt_planner {

double distance(const unsigned int x0, const unsigned int y0, const unsigned int x1, const unsigned int y1){
    return std::sqrt((int)(x1-x0)*(int)(x1-x0) + (int)(y1-y0)*(int)(y1-y0));
}

RRTPlanner::RRTPlanner() : costmap_ros_(NULL), initialized_(false),
                            max_samples_(0.0){}

RRTPlanner::RRTPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros){
    initialize(name, costmap_ros);
}

void RRTPlanner::initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros){

    if (!initialized_){
        ros::NodeHandle nh("~/" + name);
        ros::NodeHandle nh_local("~/local_costmap/");
        ros::NodeHandle nh_global("~/global_costmap/");
        marker_pub_ = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);


        //to make sure one of the nodes in the plan lies in the local costmap
        double width, height;
        nh_local.param("width", width, 3.0);
        nh_local.param("height", height, 3.0);
        nh.param("maxsamples", max_samples_, 30000);
        nh.param("max_dist", max_dist_, 0.0);
        nh.param("resolution", resolution_, 0.00);
        

        // std::cout << "Parameters: " << max_samples_ << ", " << dist_th_ << ", " << visualize_markers_ << ", " << max_dist_ << std::endl;
        std::cout << "Max distance: " << max_dist_ << std::endl;
        std::cout << "Local costmap size: " << width << ", " << height << std::endl;
        std::cout << "Global costmap resolution: " << resolution_ << std::endl;

        costmap_ros_ = costmap_ros;
        costmap_ = costmap_ros->getCostmap();
        global_frame_id_ = costmap_ros_->getGlobalFrameID();

        initialized_ = true;
    }
	else{
	    ROS_WARN("This planner has already been initialized... doing nothing.");
    }
}

bool RRTPlanner::makePlan(const geometry_msgs::PoseStamped& start, const geometry_msgs::PoseStamped& goal, 
                            std::vector<geometry_msgs::PoseStamped>& plan ){

    // std::cout << "RRTPlanner::makePlan" << std::endl;
    
    if (!initialized_){
        ROS_ERROR("The planner has not been initialized.");
        return false;
    }

	if (start.header.frame_id != costmap_ros_->getGlobalFrameID()){
		ROS_ERROR("The start pose must be in the %s frame, but it is in the %s frame.",
				  global_frame_id_.c_str(), start.header.frame_id.c_str());
		return false;
	}

	if (goal.header.frame_id != costmap_ros_->getGlobalFrameID()){
		ROS_ERROR("The goal pose must be in the %s frame, but it is in the %s frame.",
				  global_frame_id_.c_str(), goal.header.frame_id.c_str());
		return false;
	}
    
    plan.clear();
    costmap_ = costmap_ros_->getCostmap();  // Update information from costmap
    
    // Get start and goal poses in map coordinates
    unsigned int goal_mx, goal_my, start_mx, start_my;
    if (!costmap_->worldToMap(goal.pose.position.x, goal.pose.position.y, goal_mx, goal_my)){
        ROS_WARN("Goal position is out of map bounds.");
        return false;
    }    
    costmap_->worldToMap(start.pose.position.x, start.pose.position.y, start_mx, start_my);

    std::vector<int> point_start{(int)start_mx,(int)start_my};
    std::vector<int> point_goal{(int)goal_mx,(int)goal_my};    
  	std::vector<std::vector<int>> solRRT;
    bool computed = computeRRT(point_start, point_goal, solRRT);
    if (computed){        
        getPlan(solRRT, plan);
        // add goal
        plan.push_back(goal);
        publishLineMarker(solRRT, global_frame_id_);


        // 
    }else{
        ROS_WARN("No plan computed");
    }

    return computed;
}

bool RRTPlanner::straightLine(const std::vector<int> start, const std::vector<int> goal, std::vector<std::vector<int>>& sol){

    double dist_to_goal = distance(start[0], start[1], goal[0], goal[1]);
    bool goal_achived = false;
    std::vector<int> current = start;

    while (!goal_achived) {

        if (dist_to_goal >= (max_dist_ / resolution_)){
            double theta = atan2(goal[1] - current[1], goal[0] - current[0]);
            int new_x = static_cast<int>(current[0] + max_dist_ / resolution_ * cos(theta));
            int new_y = static_cast<int>(current[1] + max_dist_ / resolution_ * sin(theta));
            current = {new_x, new_y};
            dist_to_goal = distance(current[0], current[1], goal[0], goal[1]);

            sol.push_back(current);
            publishLineMarker(sol, global_frame_id_);
        // TO FIX: 
        // Points should not be empty for specified marker type. 
        // At least two points are required for a LINE_STRIP marker.
        // Done with the if in publishLineMarker method?
        // Just with a warning

        }else{
            goal_achived = true;
            return goal_achived;
        }
    }
}

void RRTPlanner::publishLineMarker(const std::vector<std::vector<int>>& path, const std::string& frame_id) {
    if (path.size() < 2) {
        ROS_WARN("Path is empty or contains fewer than two points. Unable to create LINE_STRIP marker.");
        return;
    }
    
     visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = ros::Time::now();
    marker.ns = "rrt_path";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;

    // Set the scale and color of the marker
    marker.scale.x = 0.05;  
    marker.color.r = 1.0;   
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 0.5;

    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;  

    // Add points to the marker
    for (const auto& point : path) {
        geometry_msgs::Point p;
        costmap_->mapToWorld(point[0], point[1], p.x, p.y);
        p.z = 0.0;  
        marker.points.push_back(p);
    }

    // Publish the marker
    marker_pub_.publish(marker);
}


bool RRTPlanner::computeRRT(const std::vector<int> start, const std::vector<int> goal, 
                            std::vector<std::vector<int>>& sol){
    
    // Parameters to use    
    bool finished = false;
    srand(time(NULL));  //Initialize random number generator

    // Initialize the tree with the starting point in map coordinates
    TreeNode *itr_node = new TreeNode(start); 

    // If goal visible from start, return an straight line
    if (obstacleFree(start[0], start[1], goal[0], goal[1])){
        ROS_INFO("The goal is reachable from the start, returning straight line.");
        return straightLine(start, goal, sol);
    }

    // Start the RRT algorithm
    for (size_t i = 0; i < max_samples_; i++){

        // Generate a random point  within the map limits
        int x_rand = (int)((double) rand() / (RAND_MAX) * costmap_->getSizeInCellsX());
        int y_rand = (int)((double) rand() / (RAND_MAX) * costmap_->getSizeInCellsY());

        // With previous find the nearest node in the tree
        std::vector<int> new_point = {(int)x_rand, (int)y_rand};
        TreeNode *new_node =  new TreeNode(new_point);
        TreeNode *near = new_node->neast(itr_node);
        delete new_node;

        
        // Try to find a valid point given some checks
        // TODO add attempt as a parameter
        bool valid_point = false;
        for (int attempt = 0; attempt < 10; attempt++){
            // 10% bias towards the goal
            if(rand() % 10 == 0){
                x_rand = goal[0];
                y_rand = goal[1];
                ROS_INFO("Bias towards the goal");
            } else {
                // Generate a random point within a circle of radius max_dist_ centered at neareast node
                double angle = (rand() % 360) * M_PI / 180.0;
                double radius = ((double)rand() / RAND_MAX) * max_dist_ / resolution_;
                x_rand = near->getNode()[0] + (int)(radius * cos(angle));
                y_rand = near->getNode()[1] + (int)(radius * sin(angle));
            }

            // Check 1: is the new point free?
            if (costmap_->getCost(x_rand, y_rand) != costmap_2d::FREE_SPACE) continue;
            
            // Check2 2: is the new point obstacle free?
            if (!obstacleFree(near->getNode()[0], near->getNode()[1], x_rand, y_rand)) continue;

            // Check 3: is the new point closer to the goal than the parent node?

            // // Calculate the total distance from start to goal through the new point
            // double parent_to_new = distance(near->getNode()[0], near->getNode()[1], x_rand, y_rand);
            // double new_to_goal = distance(x_rand, y_rand, goal[0], goal[1]);
            // double parent_to_start = distance (near->getNode()[0], near->getNode()[1], start[0], start[1]);
            
            // double total_distance_new = parent_to_start + parent_to_new + new_to_goal;

            // // Calculate the total distance from start to goal through the parent
            // double parent_to_goal = distance(near->getNode()[0], near->getNode()[1], goal[0], goal[1]);
            // double total_distance_parent = parent_to_start + parent_to_goal;       

            // ROS_INFO("Parent to new: %f", parent_to_new);
            // ROS_INFO("New to goal: %f", new_to_goal);
            // ROS_INFO("Parent to start: %f", parent_to_start);
            // ROS_INFO("Total distance new: %f", total_distance_new);
            // ROS_INFO("Parent to goal: %f", parent_to_goal);
            // ROS_INFO("Total distance parent: %f", total_distance_parent);
            
            // Check if the new point increases the total distance
            
            // Add a threshold because the euclidian distance is to the goal is always the shortest path
            // TODO: add this as a parameter
            // double distance_threshold = 50;

            // Just check for the 2 firsts point of the tree
            // if (total_distance_new >= total_distance_parent + distance_threshold) continue;



            // If all checks are passed, the new point is valid
            valid_point = true;
            new_point = {(int)x_rand, (int)y_rand};
            break;
        }
        
        // Continue for the outter for if no valid point was found
        if (!valid_point) {
            ROS_INFO("No valid point found after retries.");
            // delete near; IDK why is giving -11 segmentation error or -6 bad_alloc
            continue;
        }

        // Add the valid point to the tree
        // FIX I am getting lines over obstacles!!!!!
        // new_node = new TreeNode(new_point);
        // near->appendChild(new_node);
        // sol.push_back(new_point);

        // delete new_node; IDK why is giving -11 segmentation error or -6 bad_alloc
        // delete near; IDK why is giving -11 segmentation error or -6 bad_alloc

        // // Visualize the tree growth
        // std::vector<std::vector<int>> current_path = itr_node->returnSolution();
        // publishLineMarker(current_path, global_frame_id_);
        ROS_INFO("Point [%d, %d] added to the tree.", new_point[0], new_point[1]);

        // Check if the goal is reachable
        if (obstacleFree(new_point[0], new_point[1], goal[0], goal[1])) {
            ROS_INFO("Goal visible from last node.");
            ROS_INFO("RRT has elaborated a feasible path after %zu iterations", i);
            straightLine(new_point, goal, sol);

            finished = true;
            
            // Shoudl be here these two deletes? 
            delete near;
            delete new_node;
            
            break;
        }
    }
    
    delete itr_node;

    if (finished)
    {
        // ROS_INFO("RRT has elaborated a feasible path after %zu iterations", i);
    } else {
        ROS_INFO("RRT could not elaborate a feaseible path after %d iterations", max_samples_);
    }
    
    // TODO: 2024-11-30
    // Don t know: getting and error -11 (segmentation error) when puting the goal in the upper right corner
    // TODO 2024-12-01
    // Enhance performance of chosing a new point. If the selected goal is too far away, the global planners is throwing too much random new valid
    // Fix the path getting over obstacles.
    // COMPUTE THE PATH IN REVERSE ORDER
    
    return finished;
}

bool RRTPlanner::obstacleFree(const unsigned int x0, const unsigned int y0, 
                            const unsigned int x1, const unsigned int y1){
    //Bresenham algorithm to check if the line between points (x0,y0) - (x1,y1) is free of collision

    int dx = x1 - x0;
    int dy = y1 - y0;

    int incr_x = (dx > 0) ? 1.0 : -1.0;
    int incr_y = (dy > 0) ? 1.0 : -1.0;

    unsigned int da, db, incr_x_2, incr_y_2;
    if (abs(dx) >= abs(dy)){
        da = abs(dx); db = abs(dy);
        incr_x_2 = incr_x; incr_y_2 = 0;
    }else{
        da = abs(dy); db = abs(dx);
        incr_x_2 = 0; incr_y_2 = incr_y;
    }

    int p = 2*db - da;
    unsigned int a = x0; 
    unsigned int b = y0;
    unsigned int end = da;
    for (unsigned int i=0; i<end; i++){
        if (costmap_->getCost(a, b) != costmap_2d::FREE_SPACE){  // to include cells with inflated cost
            return false;
        }else{
            if (p >= 0){
                a += incr_x;
                b += incr_y;
                p -= 2*da;
            }else{
                a += incr_x_2;
                b += incr_y_2;
            }
            p += 2*db;
        }
    }

    return true;
}

void RRTPlanner::getPlan(const std::vector<std::vector<int>> sol, std::vector<geometry_msgs::PoseStamped>& plan){

    for (auto it = sol.rbegin(); it != sol.rend(); it++){
        std::vector<int> point = (*it);
        geometry_msgs::PoseStamped pose;

        costmap_->mapToWorld((unsigned int)point[0], (unsigned int)point[1], 
                            pose.pose.position.x, pose.pose.position.y);
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = global_frame_id_;
        pose.pose.orientation.w = 1;
        plan.push_back(pose);

    }
}

};
