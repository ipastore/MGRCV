bool RRTPlanner::computeRRT(const std::vector<int> start, const std::vector<int> goal, 
                            std::vector<std::vector<int>>& sol) {
    bool finished = false;
    srand(time(NULL));  // Initialize random number generator

    // Initialize the tree with the starting point
    TreeNode *itr_node = new TreeNode(start);

    // Check if the goal is directly reachable from the start
    if (obstacleFree(start[0], start[1], goal[0], goal[1])) {
        ROS_INFO("The goal is reachable from the start.");
        return straightLine(start, goal, sol);
    }

    // Iterative RRT loop
    for (size_t i = 0; i < max_samples_; i++) {
        // Generate a random point
        int x_rand = (int)((double)rand() / (RAND_MAX) * costmap_->getSizeInCellsX());
        int y_rand = (int)((double)rand() / (RAND_MAX) * costmap_->getSizeInCellsY());

        // Find the nearest node
        std::vector<int> new_point = {(int)x_rand, (int)y_rand};
        TreeNode *new_node = new TreeNode(new_point);  // Temporary node
        TreeNode *near = new_node->neast(itr_node);    // Nearest node
        delete new_node;  // Clean up temporary node

        // Retry-based sampling logic
        bool valid_point = false;
        for (int attempt = 0; attempt < 5; attempt++) {
            if (rand() % 10 == 0) {  // Goal bias
                x_rand = goal[0];
                y_rand = goal[1];
                ROS_INFO("Goal bias applied.");
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

    delete itr_node;  // Clean up the tree
    return finished;
}