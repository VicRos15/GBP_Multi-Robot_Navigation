cmake_minimum_required(VERSION 3.0.2)
project(gbp_system)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  RobotState.msg
  RobotsState.msg
  RobotTrajectory.msg
  RobotsTrajectory.msg
)

generate_messages(
  DEPENDENCIES
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/gbp_planner_node.py
    scripts/pb_sim_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)