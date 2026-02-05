// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
// Make sure you have read through the header file as well

#include "rrt/rrt.h"
#include <cmath>

// Destructor of the RRT class
RRT::~RRT() {
    // Do something in here, free up used memory, print message, etc.
    RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "RRT shutting down");
}

// Constructor of the RRT class
RRT::RRT(): rclcpp::Node("rrt_node"), gen((std::random_device())()) {

    // ROS publishers
    // TODO: create publishers for the the drive topic, and other topics you might need
    // Drive command publisher
    drive_pub_ = this->create_publisher<
        ackermann_msgs::msg::AckermannDriveStamped>(
        "/drive", 10);

    // Path visualization publisher
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
        "/rrt_path", 10);

    // Tree visualization publisher
    tree_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/rrt_tree", 10);


    // ROS subscribers
    // TODO: create subscribers as you need
        std::string pose_topic = "ego_racecar/odom";
        std::string scan_topic = "/scan";
        pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            pose_topic, 1, std::bind(&RRT::pose_callback, this, std::placeholders::_1));
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            scan_topic, 1, std::bind(&RRT::scan_callback, this, std::placeholders::_1));

        cell_size = 0.05; // in meters
        grid_width = 2.0; // in meters
        this->occupancy_grid = std::vector<std::vector<bool>>(static_cast<size_t>(grid_width / cell_size), std::vector<bool>(static_cast<size_t>(grid_width / cell_size), 0));

    RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "Created new RRT Object.");
}

void RRT::scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
    // The scan callback, update your occupancy grid here
    // Args:
    //    scan_msg (*LaserScan): pointer to the incoming scan message
    // Returns:
    //

    // TODO: update your occupancy grid
    for (auto &row : occupancy_grid) {
        std::fill(row.begin(), row.end(), false);
    }
    
    double angle = scan_msg->angle_min;

    for (size_t i = 0; i < scan_msg->ranges.size(); i++) {

        double r = scan_msg->ranges[i];

        // Ignore invalid readings
        if (r < scan_msg->range_min || r > scan_msg->range_max) {
            angle += scan_msg->angle_increment;
            continue;
        }

        // Convert polar → Cartesian (car frame)
        double x = r * std::cos(angle);
        double y = r * std::sin(angle);

        // Convert to grid coordinates
        int grid_x = static_cast<int>((x + grid_width / 2.0) / cell_size);
        int grid_y = static_cast<int>((y + grid_width / 2.0) / cell_size);

        // Check bounds
        if (grid_x >= 0 && grid_x < occupancy_grid.size() &&
            grid_y >= 0 && grid_y < occupancy_grid[0].size()) {

            occupancy_grid[grid_x][grid_y] = true; // occupied
        }

        angle += scan_msg->angle_increment;
    }

}

void RRT::pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg) {
    // The pose callback when subscribed to particle filter's inferred pose
    // The RRT main loop happens here
    // Args:
    //    pose_msg (*PoseStamped): pointer to the incoming pose message
    // Returns:
    //

    // -----------------------------
    // ------- RRT MAIN LOOP -------
    // -----------------------------
    bool RRT_star = false;
    std::vector<RRT_Node> tree;  // tree as std::vector

    RRT_Node root = {0, 0, 0, -1, true}; // initialize the root node at the car's local position
    tree.push_back(root);
    
    RRT_Node latest_added_node = root;
    while (! is_goal(latest_added_node, 0.0, 1.0)){
        // sample
        std::vector<double> sampled_point = sample();

        // nearest
        int nearest_node_indx = nearest(tree, sampled_point);

        // steer
        RRT_Node new_node = steer(tree[nearest_node_indx], sampled_point);

        if (!RRT_star){
            new_node.parent = nearest_node_indx;
            new_node.cost = cost(tree, new_node);
        }
        else{
            // RRT*
            std::vector<int> neighborhood = near(tree, new_node);
            // choose parent
            new_node.parent = neighborhood[0];
            double min_cost = cost(tree, new_node);
            
            double current_cost;
            for (int i = 1; i < neighborhood.size(); i++) {
                current_cost = cost(tree, tree[neighborhood[i]]) + line_cost(new_node, tree[neighborhood[i]]);
                if (current_cost < min_cost) {
                    min_cost = current_cost;
                    new_node.parent = neighborhood[i];
                }
            }
            new_node.cost = min_cost;
        }

        // If the steer operation returned a node at the exact same location
        // as its parent (zero-length expansion), treat this as a collision /
        // invalid expansion and skip adding the node.
        if (std::fabs(new_node.x - tree[new_node.parent].x) < 1e-6 &&
            std::fabs(new_node.y - tree[new_node.parent].y) < 1e-6) {
            continue;
        }

        // check collision
        if (! check_collision(tree[new_node.parent], new_node)) {
            // add to tree
            tree.push_back(new_node);
            latest_added_node = new_node;
        }
    }

    std::vector<RRT_Node> path = find_path(tree, latest_added_node);

    // -----------------------------------
    // ------- PUBLISHING THE PATH -------
    // -----------------------------------
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->get_clock()->now();
    path_msg.header.frame_id = "map";
    path_msg.poses.reserve(path.size());

    for (const auto &node : path) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header = path_msg.header;

        pose.pose.position.x = node.x;
        pose.pose.position.y = node.y;
        pose.pose.position.z = 0.0;

        pose.pose.orientation.x = 0.0;
        pose.pose.orientation.y = 0.0;
        pose.pose.orientation.z = 0.0;
        pose.pose.orientation.w = 1.0;  // neutral orientation temporarily *****

        path_msg.poses.push_back(pose);
    }

    path_pub_->publish(path_msg);

    // path found as Path message

}

std::vector<double> RRT::sample() {
    // This method returns a sampled point from the free space
    // You should restrict so that it only samples a small region
    // of interest around the car's current position
    // Args:
    // Returns:
    //     sampled_point (std::vector<double>): the sampled point in free space

    std::vector<double> sampled_point;
    // look up the documentation on how to use std::mt19937 devices with a distribution
    // the generator and the distribution is created for you (check the header file)
    
    double r = 0.5; // radius of the sampling region
    x_dist = std::uniform_real_distribution<double>(-r, r); // set the range for x
    y_dist = std::uniform_real_distribution<double>(-r, r); // set the range for y

    sampled_point.push_back(x_dist(gen));
    sampled_point.push_back(y_dist(gen));
    
    return sampled_point;
}


int RRT::nearest(std::vector<RRT_Node> &tree, std::vector<double> &sampled_point) {
    // This method returns the nearest node on the tree to the sampled point
    // Args:
    //     tree (std::vector<RRT_Node>): the current RRT tree
    //     sampled_point (std::vector<double>): the sampled point in free space
    // Returns:
    //     nearest_node (int): index of nearest node on the tree

    int nearest_node = 0;
    double dist_current_sq, dist_nearest_sq;
    for (int i = 1; i < tree.size(); i++) {
        dist_current_sq = pow((tree[i].x - sampled_point[0]), 2) + pow((tree[i].y - sampled_point[1]), 2);
        dist_nearest_sq = pow((tree[nearest_node].x - sampled_point[0]), 2) + pow((tree[nearest_node].y - sampled_point[1]), 2);
        if (dist_current_sq < dist_nearest_sq) { //if two nodes are equidistant, keep the first one
            nearest_node = i;
        }
    }

    return nearest_node;
}

RRT_Node RRT::steer(RRT_Node &nearest_node, std::vector<double> &sampled_point) {
    // The function steer:(x,y)->z returns a point such that z is “closer” 
    // to y than x is. The point z returned by the function steer will be 
    // such that z minimizes ||z−y|| while at the same time maintaining 
    //||z−x|| <= max_expansion_dist, for a prespecified max_expansion_dist > 0

    // basically, expand the tree towards the sample point (within a max dist)

    // Args:
    //    tree (std::vector<RRT_Node>): the current RRT tree
    //    nearest_node (int): index of nearest node on the tree to the sampled point
    //    sampled_point (std::vector<double>): the sampled point in free space
    // Returns:
    //    new_node (RRT_Node): new node created from steering

    RRT_Node new_node;
    new_node.is_root = false;

    double dx = sampled_point[0] - nearest_node.x;
    double dy = sampled_point[1] - nearest_node.y;
    double dist = std::sqrt(dx*dx + dy*dy);

    if (dist == 0.0) {
        new_node.x = nearest_node.x;
        new_node.y = nearest_node.y;
    } else if (dist <= max_expansion_dist) {
        new_node.x = sampled_point[0];
        new_node.y = sampled_point[1];
    } else {
        new_node.x = nearest_node.x + max_expansion_dist * (dx / dist);
        new_node.y = nearest_node.y + max_expansion_dist * (dy / dist);
    }

    // ****This function returns the new node, we must check for collision before adding to the tree
    return new_node;
}

bool RRT::check_collision(RRT_Node &nearest_node, RRT_Node &new_node) {
    // This method returns a boolean indicating if the path between the 
    // nearest node and the new node created from steering is collision free
    // Args:
    //    nearest_node (RRT_Node): nearest node on the tree to the sampled point
    //    new_node (RRT_Node): new node created from steering
    // Returns:
    //    collision (bool): true if in collision, false otherwise
    // TODO: fill in this method

    // Step size for checking along the edge
    double step = cell_size / 2.0;

    double dx = new_node.x - nearest_node.x;
    double dy = new_node.y - nearest_node.y;
    double distance = line_cost(nearest_node, new_node);

    int num_steps = static_cast<int>(distance / step);
    if (num_steps <= 0) num_steps = 1;

    for (int i = 0; i <= num_steps; i++) {
        double t = static_cast<double>(i) / num_steps;

        // Interpolated point along the line
        double x = nearest_node.x + t * dx;
        double y = nearest_node.y + t * dy;

        // Convert to grid coordinates
        int grid_x = static_cast<int>((x + grid_width / 2.0) / cell_size);
        int grid_y = static_cast<int>((y + grid_width / 2.0) / cell_size);

        // Check bounds
        if (grid_x < 0 || grid_x >= occupancy_grid.size() ||
            grid_y < 0 || grid_y >= occupancy_grid[0].size()) {
            continue;
        }

        // If occupied → collision
        if (occupancy_grid[grid_x][grid_y]) {
            return true;
        }
    }

    // No collision detected
    return false;
}

bool RRT::is_goal(RRT_Node &latest_added_node, double goal_x, double goal_y) {
    // This method checks if the latest node added to the tree is close
    // enough (defined by goal_threshold) to the goal so we can terminate
    // the search and find a path
    // Args:
    //   latest_added_node (RRT_Node): latest addition to the tree
    //   goal_x (double): x coordinate of the current goal
    //   goal_y (double): y coordinate of the current goal
    // Returns:
    //   close_enough (bool): true if node close enough to the goal

    bool close_enough = false;
    // if (x-x_goal)^2 + (y - y_goal)^2 < goal_threshold^2
    if (pow((latest_added_node.x - goal_x), 2) + pow((latest_added_node.y - goal_y), 2) < goal_threshold * goal_threshold) {
        close_enough = true;
    }

    return close_enough;
}

std::vector<RRT_Node> RRT::find_path(std::vector<RRT_Node> &tree, RRT_Node &latest_added_node) {
    // This method traverses the tree from the node that has been determined
    // as goal
    // Args:
    //   latest_added_node (RRT_Node): latest addition to the tree that has been
    //      determined to be close enough to the goal
    // Returns:
    //   path (std::vector<RRT_Node>): the vector that represents the order of
    //      of the nodes traversed as the found path
    
    std::vector<RRT_Node> found_path = {latest_added_node};
    RRT_Node current_node = latest_added_node;
    
    while (!current_node.is_root) {
        found_path.push_back(tree[current_node.parent]);
        current_node = tree[current_node.parent];
    }

    std::reverse(found_path.begin(), found_path.end());
    return found_path;
}

// RRT* methods
double RRT::cost(std::vector<RRT_Node> &tree, RRT_Node &node) {
    // This method returns the cost associated with a node
    // Args:
    //    tree (std::vector<RRT_Node>): the current tree
    //    node (RRT_Node): the node the cost is calculated for
    // Returns:
    //    cost (double): the cost value associated with the node

    if (node.is_root) {
        return 0;
    }

    if (node.is_root) return 0.0; // safeguard
    double total_cost = line_cost(node, tree[node.parent]) + cost(tree, tree[node.parent]);
    return total_cost;
}

double RRT::line_cost(RRT_Node &n1, RRT_Node &n2) {
    // This method returns the cost of the straight line path between two nodes
    // Args:
    //    n1 (RRT_Node): the RRT_Node at one end of the path
    //    n2 (RRT_Node): the RRT_Node at the other end of the path
    // Returns:
    //    cost (double): the cost value associated with the path

    double cost = 0;
    cost = sqrt(pow((n1.x - n2.x), 2) + pow((n1.y - n2.y), 2));
    return cost;
}

std::vector<int> RRT::near(std::vector<RRT_Node> &tree, RRT_Node &node) {
    // This method returns the set of Nodes in the neighborhood of a 
    // node.
    // Args:
    //   tree (std::vector<RRT_Node>): the current tree
    //   node (RRT_Node): the node to find the neighborhood for
    // Returns:
    //   neighborhood (std::vector<int>): the index of the nodes in the neighborhood

    std::vector<int> neighborhood;
    double dist_sq;
    for (int i = 0; i < tree.size(); i++) {
        dist_sq = pow((tree[i].x - node.x), 2) + pow((tree[i].y - node.y), 2);
        if (dist_sq <= neighborhood_threshold * neighborhood_threshold) {
            neighborhood.push_back(i);
        }
    }

    return neighborhood;
}