#include "lab3_drones/drone_race.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "drone_race");
    ros::NodeHandle nh("~");

    DroneRace drone_race(nh);

    while (ros::ok())
    {
        ros::spinOnce();
    }
	return 0;
}