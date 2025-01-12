#include <heuristics4astar/custom_astar.h>
#include <global_planner/astar.h>
#include <cmath>
#include <pluginlib/class_list_macros.h>


namespace custom_astar {

CustomAStarExpansion::CustomAStarExpansion(): global_planner::AStarExpansion(nullptr, 0, 0) {}

CustomAStarExpansion::CustomAStarExpansion(global_planner::PotentialCalculator* p_calc, int xs, int ys)
    : global_planner::AStarExpansion(p_calc, xs, ys) {
    ros::NodeHandle nh("~GlobalPlanner");
    nh.param<std::string>("heuristic_type", heuristic_type_, "manhattan");
    ROS_INFO("Custom A* heuristic type: %s", heuristic_type_.c_str());
}

void CustomAStarExpansion::add(unsigned char* costs, float* potential, float prev_potential, int next_i, int end_x, int end_y) {
    if (next_i < 0 || next_i >= ns_) return;
    if (potential[next_i] < POT_HIGH) return;
    if (costs[next_i] >= lethal_cost_ && !(unknown_ && costs[next_i] == costmap_2d::NO_INFORMATION)) return;

    potential[next_i] = p_calc_->calculatePotential(potential, costs[next_i] + neutral_cost_, next_i, prev_potential);

    int x = next_i % nx_, y = next_i / nx_;
    float distance;

    // Select heuristics for distance
    if (heuristic_type_ == "manhattan") {
        distance = abs(end_x - x) + abs(end_y - y);
    } else if (heuristic_type_ == "euclidean") {
        distance = std::sqrt(std::pow(end_x - x, 2) + std::pow(end_y - y, 2));
    } else {
        distance = 0.0;  // Default heuristic (Dijkstra behavior)
    }

    custom_queue_.push_back(global_planner::Index(next_i, potential[next_i] + distance * neutral_cost_));
    std::push_heap(custom_queue_.begin(), custom_queue_.end(), global_planner::greater1());
}

} // namespace custom_astar
PLUGINLIB_EXPORT_CLASS(custom_astar::CustomAStarExpansion, nav_core::BaseGlobalPlanner)

