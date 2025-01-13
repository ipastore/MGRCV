/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, 2013, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Eitan Marder-Eppstein
 *         David V. Lu!!
 *********************************************************************/
// #include<global_planner/astar.h>
#include<heuristics4astar/custom_astar.h>
#include<costmap_2d/cost_values.h>
#include <ros/ros.h>

// namespace global_planner {
namespace heuristics4astar {

CustomAStarExpansion::CustomAStarExpansion(PotentialCalculator* p_calc, int xs, int ys) :
        Expander(p_calc, xs, ys)
        , heuristic_type_("manhattan") // default 
        {}
        
// For dynamic parameter
void CustomAStarExpansion::setHeuristicType(int type)
{
  // Convert the integer enum into a string
  switch (type) {
    case 0: heuristic_type_ = "manhattan"; break;
    case 1: heuristic_type_ = "euclidean"; break;
    case 2: heuristic_type_ = "chebyshev"; break;
    default: heuristic_type_ = "manhattan"; // fallback
  }
    ROS_INFO("Using heuristic: %s", heuristic_type_.c_str());

}
// For static parameter initialization
void CustomAStarExpansion::setHeuristicTypeString(const std::string &heuristic_str)
{
    if (heuristic_str == "manhattan") {
        heuristic_type_ = "manhattan";
    } else if (heuristic_str == "euclidean") {
        heuristic_type_ = "euclidean";
    } else if (heuristic_str == "chebyshev") {
        heuristic_type_ = "chebyshev";
    } else {
        heuristic_type_ = "manhattan"; // fallback
    }
    
    ROS_INFO("Using heuristic: %s", heuristic_type_.c_str());

}

// Getter for the heuristic type
std::string CustomAStarExpansion::getHeuristicTypeName() const
{
    return heuristic_type_; 
}


bool CustomAStarExpansion::calculatePotentials(unsigned char* costs, double start_x, double start_y, double end_x, double end_y,
                                        int cycles, float* potential) {
    queue_.clear();
    int start_i = toIndex(start_x, start_y);
    queue_.push_back(Index(start_i, 0));

    std::fill(potential, potential + ns_, POT_HIGH);
    potential[start_i] = 0;

    int goal_i = toIndex(end_x, end_y);
    int cycle = 0;

    while (queue_.size() > 0 && cycle < cycles) {
        Index top = queue_[0];
        std::pop_heap(queue_.begin(), queue_.end(), greater1());
        queue_.pop_back();

        int i = top.i;
        if (i == goal_i)
            return true;

        add(costs, potential, potential[i], i + 1, end_x, end_y);
        add(costs, potential, potential[i], i - 1, end_x, end_y);
        add(costs, potential, potential[i], i + nx_, end_x, end_y);
        add(costs, potential, potential[i], i - nx_, end_x, end_y);

        cycle++;
    }

    return false;
}

void CustomAStarExpansion::add(unsigned char* costs, float* potential, float prev_potential, int next_i, int end_x,
                         int end_y) {
    if (next_i < 0 || next_i >= ns_)
        return;

    if (potential[next_i] < POT_HIGH)
        return;

    if(costs[next_i]>=lethal_cost_ && !(unknown_ && costs[next_i]==costmap_2d::NO_INFORMATION))
        return;

    potential[next_i] = p_calc_->calculatePotential(potential, costs[next_i] + neutral_cost_, next_i, prev_potential);
    int x = next_i % nx_, y = next_i / nx_;
    // float distance = abs(end_x - x) + abs(end_y - y);
    
    // TODO: add selection of heuristic managed in a config file
    // Decide which distance to use, based on heuristic_type_
    float distance = 0.0f;

    if (heuristic_type_ == "manhattan") {
        // L1 norm
        distance = std::fabs(end_x - x) + std::fabs(end_y - y);
    }
    else if (heuristic_type_ == "euclidean") {
        // L2 norm
        float dx = static_cast<float>(end_x - x);
        float dy = static_cast<float>(end_y - y);
        distance = std::sqrt(dx * dx + dy * dy);
    }
    else if (heuristic_type_ == "chebyshev") {
        // Linf norm
        float dx = std::fabs(end_x - x);
        float dy = std::fabs(end_y - y);
        distance = std::max(dx, dy);
    }
    // else fallback is 0 => effectively Dijkstra's algorithm

    queue_.push_back(Index(next_i, potential[next_i] + distance * neutral_cost_));
    std::push_heap(queue_.begin(), queue_.end(), greater1());
}

} //end namespace heuristics4astar
