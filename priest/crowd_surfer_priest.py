import numpy as np 
import jax.numpy as jnp 
from priest.priest import State, Obstacles, Priest 
from priest.viz_utils import *
import priest.mpc_non_dy as mpc_non_dy
from jax import random

import rospy

import torch

from models.vq_vae import VQVAE
from models.fused import FusedModel
from utils.trajectory import visualise_trajectory

def sample_wall(x0, y0, x1, y1, spacing=0.5):
    """Returns a list of (x, y) points sampled along a wall from (x0, y0) to (x1, y1)."""
    length = np.hypot(x1 - x0, y1 - y0)
    n_points = int(length / spacing) + 1
    xs = np.linspace(x0, x1, n_points)
    ys = np.linspace(y0, y1, n_points)
    return list(zip(xs, ys))

def get_occupancy_grid(grid_size=60, cell_size=0.1, num_obstacles=10, seed=None):
    """
    Creates a 60x60 occupancy grid for a 10x10 meter environment.
    
    Returns:
        occupancy_grid (np.ndarray): grid with values
            0 = free space,
            1 = static obstacle (wall),
            2 = dynamic obstacle
    """
    if seed is not None:
        np.random.seed(seed)

    # Define walls (static obstacles)
    wall_0 = sample_wall(0, 4, 3.5, 4)
    wall_1 = sample_wall(5, 0, 5, 4)
    wall_2 = sample_wall(7, 2, 10, 2)
    wall_3 = sample_wall(8, 7, 8, 10)
    wall_4 = sample_wall(2, 6, 2, 10)
    all_walls = wall_0 + wall_1 + wall_2 + wall_3 + wall_4

    # Initialize occupancy grid
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

    # Mark static obstacles as 1
    for x, y in all_walls:
        i = int(y / cell_size)
        j = int(x / cell_size)
        if 0 <= i < grid_size and 0 <= j < grid_size:
            grid[i, j] = 1

    # Dynamic obstacle positions
    xmin, xmax = 0, 10
    ymin, ymax = 0, 10
    dynamic_obstacles_x = xmin + (xmax - xmin) * np.random.uniform(0, 1, num_obstacles)
    dynamic_obstacles_y = ymin + (ymax - ymin) * np.random.uniform(0, 1, num_obstacles)

    # Optional: dynamic velocities (not used in grid)
    dynamic_obstacles_vx = 0.5 * (2 * np.random.uniform(0, 1, num_obstacles) - 1)
    dynamic_obstacles_vy = 0.5 * (2 * np.random.uniform(0, 1, num_obstacles) - 1)

    # Mark dynamic obstacles as 2
    for x, y in zip(dynamic_obstacles_x, dynamic_obstacles_y):
        i = int(y / cell_size)
        j = int(x / cell_size)
        if 0 <= i < grid_size and 0 <= j < grid_size:
            grid[i, j] = 1

    static_obstacles_x = [x for (x,y) in all_walls]
    static_obstacles_y = [y for (x,y) in all_walls]

    return grid, all_walls, jnp.asarray(static_obstacles_x), jnp.asarray(static_obstacles_y), jnp.asarray(dynamic_obstacles_x), jnp.asarray(dynamic_obstacles_y), jnp.asarray(dynamic_obstacles_vx), jnp.asarray(dynamic_obstacles_vy)

class Planner():
    def __init__(self, ):
        
        # PRIEST configs
        self.a_obs_1 = 0.5
        self.a_obs_2 = 0.5
        self.b_obs_1 = 0.68
        self.b_obs_2 = 0.68 
        self.v_max = 1
        self.v_min = 0.2 
        self.a_max = 1
        self.t_fin = 10 
        self.num = 1000
        self.num_batch = 110
        self.maxiter = 1
        self.maxiter_cem = 10
        self.weight_track = 0.001
        self.weight_smoothness = 1
        self.way_point_shape = 1000
        self.v_des = 1

        self.num_obs_1 = 40
        self.num_obs_2 = 10

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vqvae = VQVAE()
        self.vqvae = self.vqvae.to(self.device)
        self.vqvae.load_state_dict(torch.load('src/crowd_surfer/checkpoints/state_dict/vqvae.pth', map_location=self.device))
        self.vqvae.eval()

        self.pixelcnn = FusedModel()
        self.pixelcnn = self.pixelcnn.to(self.device)
        self.pixelcnn.load_state_dict(torch.load('src/crowd_surfer/checkpoints/state_dict/pixelcnn.pth', map_location=self.device))
        self.pixelcnn.eval()

    def compute_waypoints(self, c_x, c_y, P):
        c_x = jnp.asarray(c_x.detach().numpy().T)
        c_y = jnp.asarray(c_y.detach().numpy().T)
        x = jnp.dot(P, c_x)
        y = jnp.dot(P, c_y)
        return x+5, y+5
    
    def generate_trajectory(self, occupancy_grid, state_initial: State, state_goal:State, obstacles: Obstacles):
        
        rospy.loginfo("Starting Trajectory Generation")
        x_waypoint = jnp.linspace(state_initial.x, state_goal.x, self.way_point_shape)
        y_waypoint = jnp.linspace(state_initial.y, state_goal.y, self.way_point_shape)

        dynamic_obstacles_x_t = obstacles.dynamic_x[:, None] + obstacles.dynamic_x[:, None]*torch.linspace(0, -0.5, 5)
        dynamic_obstacles_y_t = obstacles.dynamic_y[:, None] + obstacles.dynamic_y[:, None]*torch.linspace(0, -0.5, 5)
        dynamic_obstacles_vx_t = obstacles.dynamic_vx.unsqueeze(1).expand_as(dynamic_obstacles_x_t)
        dynamic_obstacles_vy_t = obstacles.dynamic_vy.unsqueeze(1).expand_as(dynamic_obstacles_y_t)
        dynamic_obstacles = torch.stack((dynamic_obstacles_x_t, 
                                          dynamic_obstacles_y_t, 
                                          dynamic_obstacles_vx_t, 
                                          dynamic_obstacles_vy_t), dim=2).permute(1, 2, 0)

        heading = torch.atan2(torch.tensor([state_goal.y - state_initial.y]), torch.tensor([state_goal.x - state_initial.x]))

        occupancy_grid = torch.tensor(occupancy_grid).unsqueeze(0).unsqueeze(0).float()
        dynamic_obstacles = dynamic_obstacles.unsqueeze(0).float()
        heading = heading.unsqueeze(0).float()
        assert occupancy_grid.shape == (1, 1, 60, 60), f'Expected shape [1, 1, 60, 60] got {occupancy_grid.shape}'
        assert dynamic_obstacles.shape == (1, 5, 4, 10), f'Expected shape [1, 5, 4, 10] got {dynamic_obstacles.shape}'
        assert heading.shape == (1, 1), f'Expected shape [1, 1] got {heading.shape}'

        pixelcnn_embedding = self.pixelcnn(occupancy_grid, dynamic_obstacles, heading).permute(0, 2, 1)
        
        _, pixelcnn_idx = torch.max(pixelcnn_embedding, dim=1)
        pred_traj = self.vqvae.from_indices(pixelcnn_idx).view(2, 11)
    
        pixelcnn_idx = torch.multinomial(torch.nn.functional.softmax(pixelcnn_embedding.squeeze().permute(1, 0)), 11).permute(1, 0)
        c = self.vqvae.from_indices(pixelcnn_idx).view(11, 2, 11)
        c = torch.stack([c]*5).reshape(-1, 2, 11)
        c_x_pred, c_y_pred = c[:, 0, :], c[:, 1, :]

        rospy.loginfo("PixelCNN-VQVAE samples generated")

        prob = mpc_non_dy.batch_crowd_nav(self.a_obs_1, self.b_obs_1, self.a_obs_2, self.b_obs_2, self.v_max, self.v_min, self.a_max, self.num_obs_1, self.num_obs_2, self.t_fin, self.num, self.num_batch, self.maxiter, self.maxiter_cem, self.weight_smoothness, self.weight_track, self.way_point_shape, self.v_des)

        key = random.PRNGKey(0)

        arc_length, arc_vec, x_diff, y_diff =prob.path_spline(x_waypoint, y_waypoint)

        initial_state = jnp.hstack(( state_initial.x, state_initial.y, state_initial.vx, state_initial.vy, state_initial.ax, state_initial.ay)) 

        x_guess_per, y_guess_per = self.compute_waypoints(c_x_pred, c_y_pred, prob.P_jax)
        x_guess_per = x_guess_per.T
        y_guess_per = y_guess_per.T

        x_vqvae = x_guess_per
        y_vqvae = y_guess_per

        initial_state = jnp.hstack(( state_initial.x, state_initial.y, state_initial.vx, state_initial.vy, state_initial.ax, state_initial.ay ))

        lamda_x = jnp.zeros((self.num_batch, prob.nvar))
        lamda_y = jnp.zeros((self.num_batch, prob.nvar))

        vx_obs = 0
        vy_obs = 0

        x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, x_obs_trajectory_dy, y_obs_trajectory_dy = prob.compute_obs_traj_prediction( jnp.asarray(obstacles.dynamic_x.numpy()).flatten(), jnp.asarray(obstacles.dynamic_y.numpy()).flatten(), obstacles.dynamic_vx.numpy(), obstacles.dynamic_vy.numpy(), jnp.asarray(obstacles.static_x.numpy()).flatten(), jnp.asarray(obstacles.static_y.numpy()).flatten(), vx_obs, vy_obs, initial_state[0], initial_state[1] ) ####### obstacle trajectory prediction
        
        sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess,c_mean, c_cov, x_fin, y_fin = prob.compute_traj_guess( initial_state, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_dy, y_obs_trajectory_dy, self.v_des, x_waypoint, y_waypoint, arc_vec, x_guess_per, y_guess_per, x_diff, y_diff)
        
        x_fin = x_fin
        y_fin = y_fin  

        rospy.loginfo("Running CEM Optimisation")
        x, y, c_x_best, c_y_best, x_best, y_best, x_guess_per , y_guess_per= prob.compute_cem(key, initial_state, x_fin, y_fin, lamda_x, lamda_y, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, x_obs_trajectory_dy, y_obs_trajectory_dy,sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess, x_waypoint,  y_waypoint, arc_vec, c_mean, c_cov )

        rospy.loginfo("Finished Trajectory Generation")
        # plot_plan(state_initial, state_goal, obstacles, x_guess_per, y_guess_per, filename='crowdsurfer_priest/best_traj.png')

        return c_x_best, c_y_best, x_best, y_best, x_vqvae, y_vqvae

if __name__ == "__main__":
    # Number of obstacles
    num_obstacles = 10

    # Define the environment boundaries
    xmin, xmax = 0, 10
    ymin, ymax = 0, 10

    state_current = State(x=5, y=5, vx=0, vy=0, ax=0, ay=0)
    state_goal = State(x=9.5, y=9.5)

    occupancy_grid, all_walls, static_obstacles_x, static_obstacles_y, dynamic_obstacles_x, dynamic_obstacles_y, dynamic_obstacles_vx, dynamic_obstacles_vy = get_occupancy_grid()
    obstacles = Obstacles(static_x=static_obstacles_x, static_y=static_obstacles_y,
                          dynamic_x=dynamic_obstacles_x, dynamic_y=dynamic_obstacles_y,
                          dynamic_vx=dynamic_obstacles_vx, dynamic_vy=dynamic_obstacles_vy)

    planner = Planner()
    planner.generate_trajectory(occupancy_grid, state_current, state_goal, obstacles) 
