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

        }else{
            goal_achived = true;
            return goal_achived;
        }

        // FIXED: Adding the last goal is done in makePlan method
        // sol.push_back(current);
    }
    // FIXED: Adding the last goal is done in makePlan method
    // return goal_achived;
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
    marker.color.a = 1.0;

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

    // CHECK IF THE GOAL IS REACHABLE FROM THE START
    if (obstacleFree(start[0], start[1], goal[0], goal[1])){
        ROS_INFO("The goal is reachable from the start");
        return straightLine(start, goal, sol);

        //TODO: add this points to the nodes to be computed for the path
        // DONE inside the straightLine function?
    }
    // IF NOT, WE HAVE TO ITERATE RRT
    //TODO: Add parameters
    // Done in rrt_global_planner_params.yaml?

    for (size_t i = 0; i < max_samples_; i++){

        // Now we must sample a new vertex. We have to select a random cell in the map
        int x_rand = (int)((double) rand() / (RAND_MAX) * costmap_->getSizeInCellsX());
        int y_rand = (int)((double) rand() / (RAND_MAX) * costmap_->getSizeInCellsY());

        // // Check if the cell is free, if not make the point reacheable
        // if(!costmap_->getCost(x_rand, y_rand) != costmap_2d::FREE_SPACE){
        //     ROS_INFO("Point [%d, %d] rejected: not in free space.", x_rand, y_rand);
        //     continue;
        // }

        // Find randomly nearest node in the tree
        std::vector<int> new_point = {(int)x_rand, (int)y_rand};
        TreeNode *new_node =  new TreeNode(new_point);
        TreeNode *near = new_node->neast(itr_node);
        delete new_node;

        bool valid_point = false;
        for (int attempt = 0; attempt < 5; attempt++){
            // 10% bias towards the goal
            if(rand() % 10 == 0){
                x_rand = goal[0];
                y_rand = goal[1];
                ROS_INFO("Bias towards the goal");
            } else {
                double angle = (rand() % 360) * M_PI / 180.0;
                double radius = ((double)rand() / RAND_MAX) * max_dist_ / resolution_;
                x_rand = near->getNode()[0] + (int)(radius * cos(angle));
                y_rand = near->getNode()[1] + (int)(radius * sin(angle));
            }

            if (costmap_->getCost(x_rand, y_rand) != costmap_2d::FREE_SPACE) continue;
            if (!obstacleFree(near->getNode()[0], near->getNode()[1], x_rand, y_rand)) continue;

            valid_point = true;
            new_point = {(int)x_rand, (int)y_rand};
            break;
        }
        
        if (!valid_point) {
            ROS_INFO("No valid point found after retries.");
            continue;
        }

        // Add the valid point to the tree
        new_node = new TreeNode(new_point);
        near->appendChild(new_node);
        sol.push_back(new_point);

        // Visualize the tree growth
        std::vector<std::vector<int>> current_path = itr_node->returnSolution();
        publishLineMarker(current_path, global_frame_id_);
        ROS_INFO("Point [%d, %d] added to the tree.", new_point[0], new_point[1]);

        // Check if the goal is reachable
        if (obstacleFree(new_point[0], new_point[1], goal[0], goal[1])) {
            ROS_INFO("Goal reached.");
            finished = true;
            break;
        }

    }

    // itr_node->~TreeNode();
    // FIXED? correct way to delete the tree?
    delete itr_node;

    if (finished)
    {
        ROS_INFO("The goal has been reached");
    } else {
        ROS_INFO("The goal has not been reached after %d iterations", max_samples_);
    }
    
    // TODO: 2024-11-30
    // Half DONE: drawing the straight line (but getting a new goal if the goal is reachable)
    // NOT DONE: Getting reachables path but getting new plans again as previously, so it getting crazy
    // Don t know: getting and error -11 (segmentation error) when puting the goal in the upper right corner
    // Maybe was a lot of iterations and segmentation fault for the growing tree? change iterations form 300000 to 30000

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
