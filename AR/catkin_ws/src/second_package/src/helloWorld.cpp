#include <ros/ros.h>

int main(int argc, char** argv) //includes ROS main header file
{
  ros::init(argc, argv, "helloWorld"); //should be called before calling other ROS functions
  ros::NodeHandle nodeHandle;          //The node handle is the access point for communications
                                       // with the ROS system (topics, services, parameters)
  ros::Rate loopRate(10);              //ros::Rate is a helper class to run loops at a desired frequency
  unsigned int count = 0;
  while (ros::ok()) { //ros::ok() checks if a node should continue running. Returns false if Ctrl + C is received or ros::shutdown() has been called
    ROS_INFO_STREAM("Hello World " << count); //ROS_INFO() logs messages to the filesystem
    ros::spinOnce();                          //ros::spinOnce() processes incoming messages via callbacks
    loopRate.sleep();
    count++;
  }
  return 0;
}