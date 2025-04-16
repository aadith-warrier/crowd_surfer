import rospy
import tf2_ros

from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path

import torch
import math as m
import numpy as np

from priest.priest import State, Obstacles, Priest
from priest.crowd_surfer_priest import Planner
from utils.trajectory import visualise_trajectory


class OpenLoopBag():
    def __init__(self):
        rospy.init_node('open_loop_bag', anonymous=True)

        #init and load the planner
        self.planner = Planner()

        #subscribers
        rospy.Subscriber('/scan', LaserScan, self.laser_scan_callback)
        rospy.Subscriber('/marker', MarkerArray, self.marker_callback)
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)

        #publishers
        self.occupancy_grid_publisher = rospy.Publisher('/grid_map', OccupancyGrid)
        self.trajectory_publisher = rospy.Publisher('/pred_trajectory', Path)
        self.sampled_trajectory_publisher_1 = rospy.Publisher('/sampled_trajectory_1', Path)
        self.sampled_trajectory_publisher_2 = rospy.Publisher('/sampled_trajectory_2', Path)
        self.sampled_trajectory_publisher_3 = rospy.Publisher('/sampled_trajectory_3', Path)
        self.sampled_trajectory_publisher_4 = rospy.Publisher('/sampled_trajectory_4', Path)
        self.sampled_trajectory_publisher_5 = rospy.Publisher('/sampled_trajectory_5', Path)

        self.optimized_trajectory_publisher = rospy.Publisher('/optimized_trajectory', Path)

        self.sampled_trajectory_publishers = [self.sampled_trajectory_publisher_1, self.sampled_trajectory_publisher_2, self.sampled_trajectory_publisher_3, self.sampled_trajectory_publisher_4, self.sampled_trajectory_publisher_5]

        self.marker_positions = {}

        #timer
        while not rospy.is_shutdown():
            self.infer_trajectories()   

    def laser_scan_to_grid(self, scan, grid_size=60, resolution=0.1, max_range=30.0):

        self.static_obstacles = []
        grid = -1 * np.ones((grid_size, grid_size), dtype=np.int8)
        center = grid_size // 2
        
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        
        for r, theta in zip(scan.ranges, angles):
            if 0 < r < max_range:
                x = int(center + (r * np.cos(theta)) / resolution)
                y = int(center + (r * np.sin(theta)) / resolution)
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    grid[y, x] = 100
                    self.static_obstacles.append([x, y])

        self.static_obstacles = np.asarray(self.static_obstacles)

        self.occupancy_grid = grid

        # Convert to OccupancyGrid message
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header = scan.header
        occupancy_grid.info.resolution = resolution
        occupancy_grid.info.width = grid_size
        occupancy_grid.info.height = grid_size
        occupancy_grid.info.origin = Pose()
        occupancy_grid.info.origin.position.x = -grid_size * resolution / 2
        occupancy_grid.info.origin.position.y = -grid_size * resolution / 2
        occupancy_grid.data = grid.flatten().tolist()
        
        return occupancy_grid

    def laser_scan_callback(self, msg_data):
        self.time = msg_data.header.stamp
        occupancy_grid = self.laser_scan_to_grid(scan=msg_data, grid_size=60, resolution=0.1, max_range=msg_data.range_max)
        self.occupancy_grid_publisher.publish(occupancy_grid)
    
    def marker_callback(self, msg_data):

        #read and store marker data
        num_markers = len(msg_data.markers)
        for i in range(num_markers):
            marker = msg_data.markers[i]
            time = marker.header.stamp
            id = marker.id
            x = marker.pose.position.x
            y = marker.pose.position.y

            if str(id) in self.marker_positions.keys():
                marker_data = self.marker_positions[str(id)]
                prev_time = marker_data[-1][0]
                prev_x = marker_data[-1][1]
                prev_y = marker_data[-1][2]
                delta_t = time - prev_time
                delta_t = delta_t.secs + delta_t.nsecs*1e-9
                u = (x-prev_x)/delta_t
                v = (y-prev_y)/delta_t
                if len(marker_data) < 5:
                    marker_data.append((time, x, y, u, v))
                if len(marker_data) == 5:
                    marker_data.pop(0)
                    marker_data.append((time, x, y, u, v))
            else:
                self.marker_positions[str(id)] = [(time, x, y, None, None)]

        #process stored marker data and get the markers with the closest to the robot
        distances = {}
        for i in self.marker_positions.keys():
            distances[i] = m.sqrt(self.marker_positions[i][-1][1]**2 + self.marker_positions[i][-1][2]**2)
        distances = torch.tensor(list(distances.values()))
        _, idx = torch.topk(distances, k=10, largest=False)
        dynamic_obstacles = []
        for i in idx:
            dynamic_obstacles.append([data[1:] for data in self.marker_positions[str(i.item())]])
        self.dynamic_obstacles = torch.tensor(dynamic_obstacles).permute(1, 2, 0)
        return

    def infer_trajectories(self):

        if not hasattr(self, 'occupancy_grid'):
            rospy.logerr("Not recieved occupancy map")
            return
        if not hasattr(self, 'dynamic_obstacles'):
            rospy.logerr("Not recieved dynamic obstacles")
            return 

        state_initial = State(0, 0, 0, 0, 0, 0)
        state_goal = State(5, 5)
        
        obstacles = Obstacles(self.static_obstacles[:, 0], self.static_obstacles[:, 1], self.dynamic_obstacles[4, 0, :].numpy(), self.dynamic_obstacles[4, 1, :].numpy(), self.dynamic_obstacles[4, 2, :].numpy(), self.dynamic_obstacles[4, 3, :].numpy())
        c_x, c_y, x, y, x_vqvae, y_vqvae = self.planner.generate_trajectory(self.occupancy_grid, state_initial, state_goal, obstacles)

        x = np.asarray(x)
        y = np.asarray(y)
        x_vqvae = np.asarray(x_vqvae)
        y_vqvae = np.asarray(y_vqvae)

        for i in range(5):
            self.publish_trajectory(x_vqvae[i, :], y_vqvae[i, :], self.sampled_trajectory_publishers[i])

        self.publish_trajectory(x, y, self.optimized_trajectory_publisher)
  
    def publish_trajectory(self, x, y, publisher):

        # Create Path message
        path_msg = Path()
        path_msg.header.frame_id = "base_link"  # Set to appropriate frame
        path_msg.header.stamp = rospy.Time.now()

        for x, y in zip(x, y):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0  # Assume 2D trajectory

            path_msg.poses.append(pose)

        publisher.publish(path_msg)

def main(args=None):
    OpenLoopBag()

if __name__ == '__main__':
    main()