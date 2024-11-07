#include "lab3_drones/drone_race.hpp"

using namespace std;

DroneRace::DroneRace(ros::NodeHandle nh) : nh_(nh)
{
    // Read from parameters the path for the targets,
    // otherwise use a default value.
    if (!nh_.getParam("targets_file_path", targets_file_path_))
    {
        ROS_WARN("There is no 'targets_file_path' parameter. Using default value.");
        targets_file_path_ = "/home/padidavid/Documents/unizar/master/repositorios/MGRCV/AR/catkin_ws/src/p00_arob_lab3_drones_moodle/data/gates.txt";
    }
    // Try to open the targets file.
    if (!readGates_(targets_file_path_))
    {
        ROS_ERROR("Could not read targets from file: %s", targets_file_path_.c_str());
        ros::shutdown();
        return;
    }
    current_goal_idx_ = 0;

    // This variable will control if we are in pose or cmd_vel control mode
    is_pose_control_ = false;

    if (is_pose_control_) {
        cout << "Position control selected" << endl;
    } else {
        cout << "velocity control selected" << endl;
    }

    pub_goal_ = nh_.advertise<geometry_msgs::PoseStamped>("/command/pose", 1000);
    pub_cmd_vel_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1000);

    // create publisher for RVIZ markers
    pub_traj_markers_ =
        nh_.advertise<visualization_msgs::MarkerArray>("/trajectory_markers", 1000);
    pub_traj_vectors_ =
        nh_.advertise<visualization_msgs::MarkerArray>("/trajectory_vectors", 1000);
    pub_gate_markers_ =
        nh_.advertise<visualization_msgs::MarkerArray>("/gate_markers", 1000);

    ros::Duration sleeptime(1.0);
    sleeptime.sleep(); // Sleep for a moment before trying to draw

    drawGates_();
    // generateTrajectoryExample_();
    generateTrajectory_();

    cmd_timer_ = nh_.createTimer(ros::Duration(0.01), &DroneRace::commandTimerCallback_, this);
    ROS_INFO("DroneRace initialized");
}

void DroneRace::getGateNormal() {

    gate_normals_.clear(); // Clear the normals vector

    for (const auto& gate : gates_) {
        // Convert the gate's orientation (quaternion) to a rotation matrix
        Eigen::Quaterniond quat(gate.orientation.w, gate.orientation.x, gate.orientation.y, gate.orientation.z);
        Eigen::Matrix3d rotation_matrix = quat.toRotationMatrix();

        // The normal vector to the gate plane is aligned with the local Y axis
        Eigen::Vector3d normal = rotation_matrix.col(0); // Use the Y axis for the normal
        gate_normals_.emplace_back(normal);
    }

    ROS_INFO("Calculated normals for %lu gates.", gate_normals_.size());
}


void DroneRace::getGateAngles() {
    gate_angles_.clear(); // Clear the angles vector

    if (gate_normals_.size() < 2) {
        ROS_WARN("Not enough gates to calculate angles. At least 2 gates are required.");
        return;
    }

    for (size_t i = 0; i < gate_normals_.size() - 1; ++i) {
        Eigen::Vector3d n1 = gate_normals_[i];
        Eigen::Vector3d n2 = gate_normals_[i + 1];

        // Calculate the dot product and clamp to [-1, 1]
        double dot_product = n1.dot(n2) / (n1.norm() * n2.norm());
        dot_product = std::max(-1.0, std::min(1.0, dot_product)); // Clamp to avoid acos errors

        // Compute the angle
        double angle = std::acos(dot_product);
        gate_angles_.push_back(angle);

        // Debug: Print each angle and the normals
        ROS_INFO("Angle between Gate %lu and Gate %lu: %.3f radians (%.3f degrees)",
                 i, i + 1, angle, angle * 180.0 / M_PI);
        ROS_DEBUG("Normal 1: [%.3f, %.3f, %.3f], Normal 2: [%.3f, %.3f, %.3f]",
                  n1.x(), n1.y(), n1.z(), n2.x(), n2.y(), n2.z());
    }

    ROS_INFO("Calculated angles for %lu gates.", gate_normals_.size());
}


bool DroneRace::readGates_(string file_name) {
    //Open the file
    ifstream input_file;
    input_file.open(file_name, ifstream::in);
    if (!input_file) {
        cerr << "Error opening the file." << endl;
        return false;
    }
    gates_.clear();

    geometry_msgs::Pose temp_pose;
    double yaw = 0;
    std::string line;
    while (std::getline(input_file, line))
    {
        std::istringstream iss(line);
        iss >> temp_pose.position.x;
        iss >> temp_pose.position.y;
        iss >> temp_pose.position.z;
        iss >> yaw;
        temp_pose.orientation = RPYToQuat_(0, 0, yaw);
        gates_.push_back(temp_pose);
    }

    // Close the file
    input_file.close();

    getGateNormal();
    getGateAngles();

    return true;
}

void DroneRace::commandTimerCallback_(const ros::TimerEvent& event) {
    if (goal_list_.empty()) {
        ROS_WARN("No trajectory generated or trajectory is empty.");
        return; 

    
    }

    if (is_pose_control_) {
        // We are in pose control mode
        // We publish the current goal
        if (current_goal_idx_ < goal_list_.size() - 1) {
            pub_goal_.publish(goal_);

            // We update the next goal
            goal_.pose.position.x = goal_list_[current_goal_idx_].x();
            goal_.pose.position.y = goal_list_[current_goal_idx_].y();
            goal_.pose.position.z = goal_list_[current_goal_idx_].z();
        } else {
            // If we have reached all goals, we stop the drone
            goal_.pose.position.x = 0;
            goal_.pose.position.y = 0;
            goal_.pose.position.z = 0;
            pub_goal_.publish(goal_);
            ROS_INFO("All goals reached!");
        }
    } else {
        if (current_goal_idx_ < goal_list_.size() - 1) {
            // We are in velocity control mode
            // We publish the current goal
            pub_cmd_vel_.publish(goal_vel_);

            // We update the next goal
            goal_vel_.linear.x = goal_vel_list_[current_goal_idx_].x();
            goal_vel_.linear.y = goal_vel_list_[current_goal_idx_].y();
            goal_vel_.linear.z = goal_vel_list_[current_goal_idx_].z();
        } else {
            // If we have reached all goals, we stop the drone
            goal_vel_.linear.x = 0;
            goal_vel_.linear.y = 0;
            goal_vel_.linear.z = 0;
            pub_cmd_vel_.publish(goal_vel_);
            ROS_INFO("All goals reached!");
        }
    }

    // Update the current goal index
    current_goal_idx_++;
}

void DroneRace::generate_gates_vertex_(mav_trajectory_generation::Vertex::Vector &vertices){

    for (size_t i = 0; i < gates_.size(); ++i) {
        mav_trajectory_generation::Vertex gate_vertex(3);

        // Centro del gate
        Eigen::Vector3d pos_gate(gates_[i].position.x, gates_[i].position.y, gates_[i].position.z);
        gate_vertex.addConstraint(mav_trajectory_generation::derivative_order::POSITION, pos_gate);

        // Gate velocity (aligned with the normal)
        Eigen::Vector3d normal = gate_normals_[i]; // Use precomputed normals
        
        // 
        double velocity_magnitude = 2.0; // 
        if (i > 0) { 
            double angle = gate_angles_[i - 1]; 
            if (angle < M_PI / 4) { 
                velocity_magnitude = 0.5; 
            } else if (angle < M_PI / 2) { 
                velocity_magnitude = 0.75; 
            }
        }
        
        Eigen::Vector3d velocity = velocity_magnitude * normal; // Ensure the type matches
        gate_vertex.addConstraint(mav_trajectory_generation::derivative_order::VELOCITY, velocity);

        // Debug: Print the normal
        ROS_INFO("Gate %lu Normal: [%.3f, %.3f, %.3f]", i, normal.x(), normal.y(), normal.z());


        vertices.push_back(gate_vertex);
    }
}

void DroneRace::generateTrajectory_() {
    //constants
    const int dimension = 3; //we only compute the trajectory in x, y and z
    const int derivative_to_optimize = mav_trajectory_generation::derivative_order::SNAP; //POSITION, VELOCITY, ACCELERATION, JERK, SNAP
    const float vel_gate_mod = 2.0; // desired velocity to cross the gate. Adjust for max performance

    // Let's initialize the vertices vector that will contain the vertices of the trajectory
    mav_trajectory_generation::Vertex::Vector vertices;

    // NEXT STEP: Create the vertices for the trajectory and its constraints

    // STARTING POINT: (0,0,0)
    // Constraints for the START vertice to have zero derivatives up to snap
    // For this exercise, the starting and ending point is the origin of coordinates
    // We create the start vertice with its constraints and push it to the vertices vector
    mav_trajectory_generation::Vertex start(dimension);
    start.makeStartOrEnd(Eigen::Vector3d(0,0,0), derivative_to_optimize);
    vertices.push_back(start);

    // INTERMEDIATE VERTICES: Gates
    generate_gates_vertex_(vertices);

    // ENDING POINT: (0,0,0)
    // Constraints for the ENDING vertice to have zero derivatives up to snap
    // For this exercise, the starting and ending point is the origin of coordinates
    // We create the start vertice with its constraints and push it to the vertices vector
    mav_trajectory_generation::Vertex  end(dimension);
    end.makeStartOrEnd(Eigen::Vector3d(0,0,0), derivative_to_optimize);
    vertices.push_back(end);

    // Provide the time constraints on the vertices
    std::vector<double> segment_times;
    // INCLUDE YOUR CODE HERE
    const double v_max = 2.0;
    const double a_max = 2.0;
    // Here we are calculating the constraints using v_max and a_max, but we could do it manually
    segment_times = estimateSegmentTimes(vertices, v_max, a_max);
    cout << "Segment times = " << segment_times.size() << endl;
    // INCLUDE YOUR CODE HERE

    // Solve the optimization problem
    const int N = 10; //Degree of the polynomial, even number at least two times the highest derivative
    mav_trajectory_generation::PolynomialOptimization<N> opt(dimension);
    opt.setupFromVertices(vertices, segment_times, derivative_to_optimize);
    opt.solveLinear();

    //Obtain the trajectory
    trajectory_.clear();
    opt.getTrajectory(&trajectory_);

    //Sample the trajectory (to obtain positions, velocities, etc.)
    mav_msgs::EigenTrajectoryPoint::Vector states;
    double sampling_interval = 0.01; //How much time between intermediate points
    bool success = mav_trajectory_generation::sampleWholeTrajectory(trajectory_, sampling_interval, &states);
    // Example to access the data
    cout << "Trajectory time = " << trajectory_.getMaxTime() << endl;
    cout << "Number of states = " << states.size() << endl;
    cout << "Position (world frame) " << 3 << " X = " << states[2].position_W[0] << endl;
    cout << "Velocity (world frame) " << 3 << " X = " << states[2].velocity_W[0] << endl;

    // Default Visualization
    visualization_msgs::MarkerArray markers;
    double distance = 0.5; // Distance by which to seperate additional markers. Set 0.0 to disable.
    std::string frame_id = "world";
    mav_trajectory_generation::drawMavTrajectory(trajectory_, distance, frame_id, &markers);
    pub_traj_vectors_.publish(markers);

    //AROB visualization
    drawTrajectoryMarkers_();

    // Generate list of commands to publish to the drone
    // INCLUDE YOUR CODE HERE
    for (const auto& state : states) {
        // Almacenar posición en goal_list_
        goal_list_.push_back( Eigen::Vector3f(
                                state.position_W.x(),
                                state.position_W.y(),
                                state.position_W.z()));
        // Almacenar velocidad en goal_vel_list_
        goal_vel_list_.push_back(Eigen::Vector3f(
                                    state.velocity_W.x(),
                                    state.velocity_W.y(),
                                    state.velocity_W.z()));
    }
}

void DroneRace::generateTrajectoryExample_() {
    //constants
    const int dimension = 3; //we only compute the trajectory in x, y and z
    const int derivative_to_optimize = mav_trajectory_generation::derivative_order::SNAP; //POSITION, VELOCITY, ACCELERATION, JERK, SNAP

    // Definition of the trajectory beginning, end and intermediate constraints
    mav_trajectory_generation::Vertex::Vector vertices;
    mav_trajectory_generation::Vertex start(dimension), middle(dimension), end(dimension);
    start.makeStartOrEnd(Eigen::Vector3d(0,0,1), derivative_to_optimize);
    vertices.push_back(start);

    //Position constraint
    middle.addConstraint(mav_trajectory_generation::derivative_order::POSITION, Eigen::Vector3d(1,2,3));
    middle.addConstraint(mav_trajectory_generation::derivative_order::VELOCITY, Eigen::Vector3d(1,0,0));
    vertices.push_back(middle);

    end.makeStartOrEnd(Eigen::Vector3d(2,1,5), derivative_to_optimize);
    vertices.push_back(end);
    
    // Provide the time constraints on the vertices
    std::vector<double> segment_times;
    const double v_max = 2.0;
    const double a_max = 2.0;
    // Here we are calculating the constraints using v_max and a_max, but we could do it manually
    segment_times = estimateSegmentTimes(vertices, v_max, a_max);
    cout << "Segment times = " << segment_times.size() << endl;
    
    // Solve the optimization problem
    const int N = 10; //Degree of the polynomial, must be an EVEN number at least TWO TIMES the degree of the highest derivative
    mav_trajectory_generation::PolynomialOptimization<N> opt(dimension);
    opt.setupFromVertices(vertices, segment_times, derivative_to_optimize);
    opt.solveLinear();

    //Obtain the trajectory
    opt.getTrajectory(&trajectory_);
    //Sample the trajectory (to obtain positions, velocities, etc.)
    mav_msgs::EigenTrajectoryPoint::Vector states;
    double sampling_interval = 0.01; //How much time between intermediate points
    bool success = mav_trajectory_generation::sampleWholeTrajectory(trajectory_, sampling_interval, &states);
    // Example to access the data
    cout << "Trajectory time = " << trajectory_.getMaxTime() << endl;
    cout << "Number of states = " << states.size() << endl;
    cout << "Position (world frame) " << 3 << " X = " << states[2].position_W[0] << endl;
    cout << "Velocity (world frame) " << 3 << " X = " << states[2].velocity_W[0] << endl;

    // Default Visualization
    visualization_msgs::MarkerArray markers;
    double distance = 0.5; // Distance by which to seperate additional markers. Set 0.0 to disable.
    std::string frame_id = "world";
    mav_trajectory_generation::drawMavTrajectory(trajectory_, distance, frame_id, &markers);
    pub_traj_vectors_.publish(markers);

    //AROB visualization
    drawTrajectoryMarkers_();
}

Eigen::Matrix<double, 3, 3> DroneRace::RPYtoRMatrix_(double roll, double pitch, double yaw) {
    Eigen::AngleAxis<double> rollAngle(roll, Eigen::Matrix<double, 1, 3>::UnitX());
    Eigen::AngleAxis<double> pitchAngle(pitch, Eigen::Matrix<double, 1, 3>::UnitY());
    Eigen::AngleAxis<double> yawAngle(yaw, Eigen::Matrix<double, 1, 3>::UnitZ());

    Eigen::Matrix<double, 3, 3> R;

    Eigen::Quaternion<double> q = yawAngle * rollAngle * pitchAngle;

    R = q.matrix();

    return (R);
}

Eigen::Matrix<double, 3, 3> DroneRace::quatToRMatrix_(geometry_msgs::Quaternion q) {
    double roll, pitch, yaw;
    tf2::Quaternion quat_tf;
    tf2::fromMsg(q, quat_tf);
    Eigen::Matrix<double, 3, 3> mat_res;
    tf2::Matrix3x3(quat_tf).getRPY(roll, pitch, yaw);

    return RPYtoRMatrix_(roll, pitch, yaw);
}

geometry_msgs::Quaternion DroneRace::RPYToQuat_(double roll, double pitch, double yaw) {
    tf2::Quaternion quaternion_tf2;
    quaternion_tf2.setRPY(roll, pitch, yaw);
    geometry_msgs::Quaternion quaternion = tf2::toMsg(quaternion_tf2);
    return quaternion;
}

void DroneRace::drawGates_() {
    int id = 0;
    for (geometry_msgs::Pose gate : gates_) {
        drawGateMarkers_(gate,id);
    }
}

void DroneRace::drawGateMarkers_(geometry_msgs::Pose gate, int &id){
    visualization_msgs::Marker marker;
        visualization_msgs::Marker marker_sphere;
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker line_marker;
    //std::vector<visualization_msgs::Marker> line_marker_vector;
    
    Eigen::Matrix<double, 3, 3> rotate_gate = quatToRMatrix_(gate.orientation);
    Eigen::Matrix<double, 3, 1> pos_gate(gate.position.x, gate.position.y, gate.position.z);

    marker.header.frame_id = "world";  // Change this frame_id according to your setup
    marker.header.stamp = ros::Time::now();
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.ns = "corner";
    marker.scale.x = 0.2;
    marker.scale.y = 0.2;
    marker.scale.z = 0.2;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.lifetime = ros::Duration();

    marker_sphere.header.frame_id = "world";  // Change this frame_id according to your setup
    marker_sphere.header.stamp = ros::Time::now();
    marker_sphere.type = visualization_msgs::Marker::SPHERE;
    marker_sphere.action = visualization_msgs::Marker::ADD;
    marker_sphere.ns = "corner";
    marker_sphere.scale.x = 0.25;
    marker_sphere.scale.y = 0.25;
    marker_sphere.scale.z = 0.25;
    marker_sphere.color.a = 1.0;
    marker_sphere.color.r = 1.0;
    marker_sphere.color.g = 1.0;
    marker_sphere.color.b = 0.0;
    marker_sphere.pose.orientation.w = 1.0;
    marker_sphere.lifetime = ros::Duration();

    line_marker.header.frame_id = "world";  // Change this frame_id according to your setup
    line_marker.header.stamp = ros::Time::now();
    line_marker.ns = "line";
    line_marker.id = id;
    line_marker.type = visualization_msgs::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::Marker::ADD;
    line_marker.scale.x = 0.05;  // Line width
    line_marker.pose.orientation.w = 1.0;
    line_marker.lifetime = ros::Duration();

    // Set the color (green in this case)
    line_marker.color.r = 0.0;
    line_marker.color.g = 1.0;
    line_marker.color.b = 0.0;
    line_marker.color.a = 1.0;

    float gate_size = 0.75;

    //Generate the gate corners and edges
    // POSITION 1: UP-LEFT
    Eigen::Matrix<double, 3, 1> move_gate;
    move_gate << 0.0, gate_size, gate_size;
    Eigen::Matrix<double, 3, 1> position = pos_gate + rotate_gate * move_gate;
    marker_sphere.pose.position.x = position(0);
    marker_sphere.pose.position.y = position(1);
    marker_sphere.pose.position.z = position(2);
    marker_sphere.id = id + 1;
    line_marker.points.push_back(marker_sphere.pose.position);
    marker_array.markers.push_back(marker_sphere);

    // POSITION 2: UP-RIGHT
    move_gate << 0.0, -gate_size, gate_size;
    Eigen::Matrix<double, 3, 1> position2 = pos_gate + rotate_gate * move_gate;
    marker.pose.position.x = position2(0);
    marker.pose.position.y = position2(1);
    marker.pose.position.z = position2(2);
    marker.id = id + 2;
    line_marker.points.push_back(marker.pose.position);
    marker_array.markers.push_back(marker);

    // POSITION 3: DOWN-RIGHT
    move_gate << 0.0, -gate_size, -gate_size;
    Eigen::Matrix<double, 3, 1> position3 = pos_gate + rotate_gate * move_gate;
    marker.pose.position.x = position3(0);
    marker.pose.position.y = position3(1);
    marker.pose.position.z = position3(2);
    marker.id = id + 3;
    line_marker.points.push_back(marker.pose.position);
    marker_array.markers.push_back(marker);

    // POSITION 4: DOWN-LEFT
    move_gate << 0.0, gate_size, -gate_size;
    Eigen::Matrix<double, 3, 1> position4 = pos_gate + rotate_gate * move_gate;
    marker_sphere.pose.position.x = position4(0);
    marker_sphere.pose.position.y = position4(1);
    marker_sphere.pose.position.z = position4(2);
    marker_sphere.id = id + 4;
    marker_array.markers.push_back(marker_sphere);
    line_marker.points.push_back(marker_sphere.pose.position);

    // Same first point to close the gate
    marker_sphere.pose.position.x = position(0);
    marker_sphere.pose.position.y = position(1);
    marker_sphere.pose.position.z = position(2);
    line_marker.points.push_back(marker_sphere.pose.position);
    id+=5;
    marker_array.markers.push_back(line_marker);
    pub_gate_markers_.publish(marker_array);
}

void DroneRace::drawGoalMarker_(mav_trajectory_generation::Vertex goal){
    Eigen::VectorXd pos;
    goal.getConstraint(mav_trajectory_generation::derivative_order::POSITION,&pos);
    visualization_msgs::Marker marker_aux;
    marker_aux.header.frame_id = "world";
    marker_aux.header.stamp = ros::Time(0);
    marker_aux.id = id_marker;
    id_marker++;
    marker_aux.ns = "point";
    marker_aux.type = visualization_msgs::Marker::CUBE;
    marker_aux.pose.position.x = pos(0);
    marker_aux.pose.position.y = pos(1);
    marker_aux.pose.position.z = pos(2);
    marker_aux.pose.orientation.x = 0;
    marker_aux.pose.orientation.y = 0;
    marker_aux.pose.orientation.z = 0;
    marker_aux.pose.orientation.w = 1;
    marker_aux.scale.x = 0.1;
    marker_aux.scale.y = 0.1;
    marker_aux.scale.z = 0.1;
    marker_aux.color.r = 1.0f;
    marker_aux.color.g = 0.0f;
    marker_aux.color.b = 0.0f;
    marker_aux.color.a = 1.0;
    marker_aux.lifetime = ros::Duration();
    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.push_back(marker_aux);
    pub_traj_markers_.publish(marker_array);
}

void DroneRace::drawTrajectoryMarkers_(){
    visualization_msgs::MarkerArray markers;
    int derivative_order = mav_trajectory_generation::derivative_order::POSITION;
    double sampling_time = 0.1;
    mav_msgs::EigenTrajectoryPoint::Vector states;
    double sampling_interval = 0.1;
    bool success = mav_trajectory_generation::sampleWholeTrajectory(trajectory_, sampling_interval, &states);   
    for(int i=0; i< states.size(); i++) {
        visualization_msgs::Marker marker_aux;
        marker_aux.header.frame_id = "world";
        //marker_aux.header.stamp = ros::Time::now();
        marker_aux.header.stamp = ros::Time(0);
        marker_aux.id = 1000+i;
        marker_aux.ns = "point";
        marker_aux.type = visualization_msgs::Marker::CUBE;
        marker_aux.pose.position.x = states[i].position_W[0] ;
        marker_aux.pose.position.y = states[i].position_W[1] ;
        marker_aux.pose.position.z = states[i].position_W[2] ;
        marker_aux.pose.orientation.x = 0;
        marker_aux.pose.orientation.y = 0;
        marker_aux.pose.orientation.z = 0;
        marker_aux.pose.orientation.w = 1;
        marker_aux.scale.x = 0.03;
        marker_aux.scale.y = 0.03;
        marker_aux.scale.z = 0.03;
        marker_aux.color.r = 0.0f;
        marker_aux.color.g = 0.0f;
        marker_aux.color.b = 1.0f;
        marker_aux.color.a = 1.0;
        marker_aux.lifetime = ros::Duration();
        markers.markers.push_back(marker_aux);
    }
    pub_traj_markers_.publish(markers);
}

