#include "lab2_drones/follow_targets_3d.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "move_drone");
    ros::NodeHandle nh("~");

    FollowTargets3D follow_targets(nh);

    while (ros::ok()) {
        ros::spinOnce();
    }
    return 0;
}