import rclpy
from rclpy.node import Node

from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path

import torch
import numpy as np

from models.vq_vae import VQVAE
from models.fused import FusedModel
from utils.trajectory import visualise_trajectory


class OpenLoopBag(Node):
    def __init__(self):
        super().__init__('open_loop_bag')

        #init and load models
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vqvae = VQVAE()
        self.vqvae = self.vqvae.to(self.device)
        self.vqvae.load_state_dict(torch.load('src/crowd_surfer/checkpoints/state_dict/vqvae.pth', map_location=self.device))
        self.vqvae.eval()

        self.pixelcnn = FusedModel()
        self.pixelcnn = self.pixelcnn.to(self.device)
        self.pixelcnn.load_state_dict(torch.load('src/crowd_surfer/checkpoints/state_dict/pixelcnn.pth', map_location=self.device))
        self.pixelcnn.eval()

        #subscribers
        self.laserscan_subscription = self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, 10)
        self.marker_subscription = self.create_subscription(MarkerArray, '/marker', self.marker_callback, 10)
        # self.tf_subscription = self.create_subscription()

        #publishers
        self.occupancy_grid_publisher = self.create_publisher(OccupancyGrid, '/grid_map', 10)
        self.trajectory_publisher = self.create_publisher(Path, '/pred_trajectory', 10)
        self.sampled_trajectory_publisher_1 = self.create_publisher(Path, '/sampled_trajectory_1', 10)
        self.sampled_trajectory_publisher_2 = self.create_publisher(Path, '/sampled_trajectory_2', 10)
        self.sampled_trajectory_publisher_3 = self.create_publisher(Path, '/sampled_trajectory_3', 10)
        self.sampled_trajectory_publisher_4 = self.create_publisher(Path, '/sampled_trajectory_4', 10)
        self.sampled_trajectory_publisher_5 = self.create_publisher(Path, '/sampled_trajectory_5', 10)
        self.sampled_trajectory_publishers = [self.sampled_trajectory_publisher_1, self.sampled_trajectory_publisher_2, self.sampled_trajectory_publisher_3, self.sampled_trajectory_publisher_4, self.sampled_trajectory_publisher_5]

        #timer
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        if hasattr(self, 'occupancy_grid'):
            self.infer_trajectories(self.occupancy_grid, torch.zeros([5, 4, 10]), torch.zeros([1]))
        else:
            print("No Occucpancy Grid. Cannot generate trajectory")
            

    def laser_scan_to_grid(self, scan, grid_size=60, resolution=0.1, max_range=30.0):
        grid = -1 * np.ones((grid_size, grid_size), dtype=np.int8)
        center = grid_size // 2
        
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        
        for r, theta in zip(scan.ranges, angles):
            if 0 < r < max_range:
                x = int(center + (r * np.cos(theta)) / resolution)
                y = int(center + (r * np.sin(theta)) / resolution)
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    grid[y, x] = 100

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
        occupancy_grid = self.laser_scan_to_grid(scan=msg_data, grid_size=60, resolution=0.1, max_range=msg_data.range_max)
        self.occupancy_grid_publisher.publish(occupancy_grid)
    
    def marker_callback(self, msg_data):
        #not working for some reason
        return

    def infer_trajectories(self, occupancy_grid, dynamic_obstacles, heading):

        occupancy_grid = torch.tensor(occupancy_grid).unsqueeze(0).unsqueeze(0).float()
        dynamic_obstacles = dynamic_obstacles.unsqueeze(0).float()
        heading = heading.unsqueeze(0).float()
        assert occupancy_grid.shape == (1, 1, 60, 60), f'Expected shape [1, 1, 60, 60] got {occupancy_grid.shape}'
        assert dynamic_obstacles.shape == (1, 5, 4, 10), f'Expected shape [1, 5, 4, 10] got {dynamic_obstacles.shape}'
        assert heading.shape == (1, 1), f'Expected shape [1, 1] got {heading.shape}'

        pixelcnn_embedding = self.pixelcnn(occupancy_grid, dynamic_obstacles, heading).permute(0, 2, 1)
        
        _, pixelcnn_idx = torch.max(pixelcnn_embedding, dim=1)
        pred_traj = self.vqvae.from_indices(pixelcnn_idx).view(2, 11)
        self.publish_trajectory(pred_traj, self.trajectory_publisher)
    
        pixelcnn_idx = torch.multinomial(torch.nn.functional.softmax(pixelcnn_embedding.squeeze().permute(1, 0)), 5).permute(1, 0)
        for i in range(5):
            pred_traj = self.vqvae.from_indices(pixelcnn_idx[i].unsqueeze(0)).view(2, 11)
            self.publish_trajectory(pred_traj, self.sampled_trajectory_publishers[i])
        return
    
    def publish_trajectory(self, trajectory, publisher):
        ''' trajectory:tensor of shape [2, 11]'''
        coefficients = trajectory.detach().numpy()  # Ensure it's on CPU
        coefficients_x = coefficients[0, :]  # First row -> X coefficients
        coefficients_y = coefficients[1, :]  # Second row -> Y coefficients

        # Compute trajectory points
        X, Y = visualise_trajectory(coefficients_x, coefficients_y)

        # Create Path message
        path_msg = Path()
        path_msg.header.frame_id = "base_link"  # Set to appropriate frame
        path_msg.header.stamp = rclpy.time.Time().to_msg()

        for x, y in zip(X, Y):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0  # Assume 2D trajectory

            path_msg.poses.append(pose)

        publisher.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)

    open_loop_bag = OpenLoopBag()

    rclpy.spin(open_loop_bag)

    open_loop_bag.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()