#include <global_planner/astar.h>
#include <ros/ros.h>
#include <string>
#include <vector>

#ifndef CUSTOM_ASTAR_H
#define CUSTOM_ASTAR_H

namespace custom_astar {

class CustomAStarExpansion : public global_planner::AStarExpansion {
public:
    
    CustomAStarExpansion(global_planner::PotentialCalculator* p_calc, int xs, int ys);

protected:
    void add(unsigned char* costs, float* potential, float prev_potential, int next_i, int end_x, int end_y);

private:
    std::string heuristic_type_;  // Stores the selected heuristic
    std::vector<global_planner::Index> custom_queue_;  // Custom queue for A* expansion

    };

};
#endif

