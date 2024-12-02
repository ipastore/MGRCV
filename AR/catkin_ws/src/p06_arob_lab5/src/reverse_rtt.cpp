bool RRTPlanner::computeRRT(const std::vector<int> start, const std::vector<int> goal, 
                            std::vector<std::vector<int>>& sol) {
    // Parameters to use    
    bool finished = false;
    srand(time(NULL));  // Initialize random number generator

    // Initialize the tree with the starting point in map coordinates
    TreeNode* root = new TreeNode(start); 

    // If goal visible from start, return a straight line
    if (obstacleFree(start[0], start[1], goal[0], goal[1])) {
        ROS_INFO("The goal is reachable from the start, returning straight line.");
        return straightLine(start, goal, sol);
    }

    // Start the RRT algorithm
    TreeNode* goal_node = nullptr;  // Pointer to the goal node when found
    for (size_t i = 0; i < max_samples_; i++) {
        // Generate a random point within the map limits
        int x_rand = (int)((double) rand() / (RAND_MAX) * costmap_->getSizeInCellsX());
        int y_rand = (int)((double) rand() / (RAND_MAX) * costmap_->getSizeInCellsY());

        // Find the nearest node in the tree
        std::vector<int> random_point = {(int)x_rand, (int)y_rand};
        TreeNode* random_node = new TreeNode(random_point);
        TreeNode* nearest = random_node->neast(root);
        delete random_node;

        // Try to find a valid point given some checks
        bool valid_point = false;
        std::vector<int> new_point;
        for (int attempt = 0; attempt < 10; attempt++) {
            // 10% bias towards the goal
            if (rand() % 10 == 0) {
                x_rand = goal[0];
                y_rand = goal[1];
                ROS_INFO("Bias towards the goal");
            } else {
                // Generate a random point within a circle of radius max_dist_ centered at nearest node
                double angle = (rand() % 360) * M_PI / 180.0;
                double radius = ((double)rand() / RAND_MAX) * max_dist_ / resolution_;
                x_rand = nearest->getNode()[0] + (int)(radius * cos(angle));
                y_rand = nearest->getNode()[1] + (int)(radius * sin(angle));
            }

            // Check 1: Is the new point free?
            if (costmap_->getCost(x_rand, y_rand) != costmap_2d::FREE_SPACE) continue;

            // Check 2: Is the new point obstacle-free?
            if (!obstacleFree(nearest->getNode()[0], nearest->getNode()[1], x_rand, y_rand)) continue;

            valid_point = true;
            new_point = {(int)x_rand, (int)y_rand};
            break;
        }

        // Continue if no valid point was found
        if (!valid_point) {
            ROS_INFO("No valid point found after retries.");
            continue;
        }

        // Add the valid point to the tree
        TreeNode* new_node = new TreeNode(new_point);
        nearest->appendChild(new_node);

        // Check if the goal is reachable
        if (obstacleFree(new_point[0], new_point[1], goal[0], goal[1])) {
            ROS_INFO("Goal visible from last node.");
            goal_node = new TreeNode(goal);  // Create a node for the goal
            new_node->appendChild(goal_node);  // Attach the goal node to the tree
            finished = true;
            break;
        }
    }

    // If the goal is reached, trace back the solution
    if (finished) {
        ROS_INFO("RRT has elaborated a feasible path.");
        TreeNode* current = goal_node;
        while (current != nullptr) {
            sol.push_back(current->getNode());  // Push nodes into solution
            current = current->getParent();    // Move to the parent node
        }
    } else {
        ROS_WARN("RRT could not elaborate a feasible path after %d iterations", max_samples_);
    }

    // Clean up the tree
    delete root;

    return finished;
}