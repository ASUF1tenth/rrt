"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid

# TODO: import as you need

# class def for tree nodes
# It's up to you if you want to use this
class Node(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.parent = None
        self.cost = None # only used in RRT*
        self.is_root = False

# class def for RRT
class RRT(Node):
    def __init__(self):
        # initialize ROS2 node
        super().__init__('rrt_node')

        # topics, not saved as attributes
        pose_topic = "ego_racecar/odom"
        scan_topic = "/scan"

        # RRT parameters (defaults, can be overridden via params/yaml)
        self.max_expansion_dist = 0.1
        self.goal_threshold = 0.1
        self.neighborhood_threshold = 0.5

        # occupancy grid parameters
        self.cell_size = 0.05  # in meters
        self.grid_width = 2.0  # in meters

        # random generator and distributions
        self.r = 0.5
        self.gen = np.random.default_rng()
        self.x_dist = lambda: self.gen.uniform(-self.r, self.r)
        self.y_dist = lambda: self.gen.uniform(-self.r, self.r)

        # create subscribers
        # pose can be Odometry messages from ego_racecar/odom
        self.pose_sub_ = self.create_subscription(
            Odometry,
            pose_topic,
            self.pose_callback,
            1)

        self.scan_sub_ = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            1)

        # publishers: drive, path, and optional tree visualization
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, 'drive', 1)
        self.path_pub_ = self.create_publisher(Path, 'path', 1)
        try:
            from visualization_msgs.msg import MarkerArray
            self.tree_pub_ = self.create_publisher(MarkerArray, 'tree', 1)
        except Exception:
            self.tree_pub_ = None

        # occupancy_grid indexed as [grid_x][grid_y]
        self.occupancy_grid = [[False for _ in range(int(self.grid_width / self.cell_size))] for _ in range(int(self.grid_width / self.cell_size))]

    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        # Clear occupancy grid
        for row in self.occupancy_grid:
            for i in range(len(row)):
                row[i] = False

        angle = scan_msg.angle_min

        for r in scan_msg.ranges:

            # Ignore invalid readings
            if r < scan_msg.range_min or r > scan_msg.range_max:
                angle += scan_msg.angle_increment
                continue

            # Convert polar → Cartesian (car frame)
            x = r * math.cos(angle)
            y = r * math.sin(angle)

            # Convert to grid coordinates
            grid_x = int((x + self.grid_width / 2.0) / self.cell_size)
            grid_y = int((y + self.grid_width / 2.0) / self.cell_size)

            # Check bounds and mark occupied
            if 0 <= grid_x < len(self.occupancy_grid) and 0 <= grid_y < len(self.occupancy_grid[0]):
                self.occupancy_grid[grid_x][grid_y] = True

            angle += scan_msg.angle_increment

    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """
        # -----------------------------
        # ------- RRT MAIN LOOP -------
        # -----------------------------
        RRT_star = False
        tree = []  # tree as list

        root = Node()
        root.x = 0
        root.y = 0
        root.cost = 0
        root.parent = -1
        root.is_root = True
        tree.append(root)

        latest_added_node = root
        while (not self.is_goal(latest_added_node, 0.0, 1.0)):
            # sample
            sampled_point = self.sample()

            # nearest
            nearest_node_indx = self.nearest(tree, sampled_point)

            # steer (pass nearest node object to steer)
            new_node = self.steer(tree[nearest_node_indx], sampled_point)

            if (not RRT_star):
                new_node.parent = nearest_node_indx
                new_node.cost = self.cost(tree, new_node)
            else:
                # RRT*
                neighborhood = self.near(tree, new_node)
                # choose parent
                new_node.parent = neighborhood[0]
                min_cost = self.cost(tree, new_node)

                for i in range(1, len(neighborhood)):
                    current_cost = self.cost(tree, tree[neighborhood[i]]) + self.line_cost(new_node, tree[neighborhood[i]])
                    if (current_cost < min_cost):
                        min_cost = current_cost
                        new_node.parent = neighborhood[i]
                new_node.cost = min_cost

            # check collision
            if (not self.check_collision(tree[new_node.parent], new_node)):
                # add to tree
                tree.append(new_node)
                latest_added_node = new_node

        path = self.find_path(tree, latest_added_node)

        # -----------------------------------
        # ------- PUBLISHING THE PATH -------
        # -----------------------------------
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now()
        path_msg.header.frame_id = "map"

        for node in path:
            pose = PoseStamped()
            pose.header = path_msg.header

            pose.pose.position.x = node.x
            pose.pose.position.y = node.y
            pose.pose.position.z = 0.0

            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        # publish path (assumes path_pub_ publisher exists)
        try:
            self.path_pub_.publish(path_msg)
        except Exception:
            pass

    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """
        # radius of the sampling region
        r = 0.5

        # random generator
        gen = np.random.default_rng()

        # set the range for x and y (translate uniform_real_distribution)
        x_dist = gen.uniform(-r, r)
        y_dist = gen.uniform(-r, r)

        sampled_point = []
        sampled_point.append(x_dist)
        sampled_point.append(y_dist)

        return (sampled_point[0], sampled_point[1])

    def nearest(self, tree, sampled_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """
        nearest_node = 0
        for i in range(1, len(tree)):
            dist_current_sq = (tree[i].x - sampled_point[0])**2 + (tree[i].y - sampled_point[1])**2
            dist_nearest_sq = (tree[nearest_node].x - sampled_point[0])**2 + (tree[nearest_node].y - sampled_point[1])**2
            if dist_current_sq < dist_nearest_sq:  # if two nodes are equidistant, keep the first one
                nearest_node = i

        return nearest_node

    def steer(self, nearest_node, sampled_point):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """
        new_node = Node()
        new_node.is_root = False

        dx = sampled_point[0] - nearest_node.x
        dy = sampled_point[1] - nearest_node.y
        denom = math.sqrt(dx * dx + dy * dy)

        if denom == 0:
            new_node.x = nearest_node.x
            new_node.y = nearest_node.y
        else:
            new_node.x = nearest_node.x + self.max_expansion_dist * dx / denom
            new_node.y = nearest_node.y + self.max_expansion_dist * dy / denom

        return new_node

    def check_collision(self, nearest_node, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest_node (Node): nearest node on the tree
            new_node (Node): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """
        step = self.cell_size / 2.0

        dx = new_node.x - nearest_node.x
        dy = new_node.y - nearest_node.y
        distance = self.line_cost(nearest_node, new_node)

        num_steps = int(distance / step)
        if num_steps <= 0:
            num_steps = 1

        for i in range(num_steps + 1):
            t = float(i) / num_steps

            # Interpolated point along the line
            x = nearest_node.x + t * dx
            y = nearest_node.y + t * dy

            # Convert to grid coordinates
            grid_x = int((x + self.grid_width / 2.0) / self.cell_size)
            grid_y = int((y + self.grid_width / 2.0) / self.cell_size)

            # Check bounds
            if grid_x < 0 or grid_x >= len(self.occupancy_grid) or \
               grid_y < 0 or grid_y >= len(self.occupancy_grid[0]):
                continue

            # If occupied → collision
            if self.occupancy_grid[grid_x][grid_y]:
                return True

        # No collision detected
        return False

    def is_goal(self, latest_added_node, goal_x, goal_y):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enoughg to the goal
        """
        # use a goal threshold attribute; default must be set elsewhere

        dx = latest_added_node.x - goal_x
        dy = latest_added_node.y - goal_y
        return (dx * dx + dy * dy) < (self.goal_threshold * self.goal_threshold)

    def find_path(self, tree, latest_added_node):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        found_path = [latest_added_node]
        current_node = latest_added_node

        while not current_node.is_root:
            parent_idx = current_node.parent
            # protect against invalid parent
            if parent_idx is None or parent_idx < 0 or parent_idx >= len(tree):
                break
            parent_node = tree[parent_idx]
            found_path.append(parent_node)
            current_node = parent_node

        found_path.reverse()
        return found_path



    # The following methods are needed for RRT* and not RRT
    def cost(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        if node.is_root:
            return 0.0

        parent_idx = node.parent
        if parent_idx is None or parent_idx < 0:
            return 0.0

        return self.line_cost(node, tree[parent_idx]) + self.cost(tree, tree[parent_idx])

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (Node): node at one end of the straight line
            n2 (Node): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """
        dx = n1.x - n2.x
        dy = n1.y - n2.y
        return math.sqrt(dx * dx + dy * dy)

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
        """
        neighborhood = []

        for i in range(len(tree)):
            dx = tree[i].x - node.x
            dy = tree[i].y - node.y
            dist_sq = dx * dx + dy * dy
            if dist_sq <= self.neighborhood_threshold * self.neighborhood_threshold:
                neighborhood.append(i)

        return neighborhood

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()