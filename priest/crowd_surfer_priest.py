import numpy as np 
import jax.numpy as jnp 
from priest import State, Obstacles, Priest 
from viz_utils import *
import mpc_non_dy
from jax import random

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
        print(c.shape)
        c_x_pred, c_y_pred = c[:, 0, :], c[:, 1, :]

        a_obs_1 = 0.5
        a_obs_2 = 0.5
        b_obs_1 = 0.68
        b_obs_2 = 0.68 
        v_max = 1
        v_min = 0.2 
        a_max = 1
        t_fin = 10 
        num = 1000
        num_batch = 110
        maxiter = 1
        maxiter_cem = 20
        weight_track = 0.001
        weight_smoothness = 1
        way_point_shape = 1000
        v_des = 1

        x_waypoint = jnp.linspace(state_initial.x, state_goal.x, way_point_shape)
        y_waypoint = jnp.linspace(state_initial.y, state_goal.y, way_point_shape)

        num_obs_1 = 40
        num_obs_2 = 10

        prob = mpc_non_dy.batch_crowd_nav(a_obs_1, b_obs_1, a_obs_2, b_obs_2, v_max, v_min, a_max, num_obs_1, num_obs_2, t_fin, num, num_batch, maxiter, maxiter_cem, weight_smoothness, weight_track, way_point_shape, v_des)

        x_best, y_best = self.compute_waypoints(pred_traj[0], pred_traj[1], prob.P_jax)
        plot_plan(state_initial, state_goal, obstacles, x_best, y_best, filename="crowdsurfer_priest/best_vqvae.png")

        key = random.PRNGKey(0)

        arc_length, arc_vec, x_diff, y_diff =prob.path_spline(x_waypoint, y_waypoint)

        initial_state = jnp.hstack(( state_initial.x, state_initial.y, state_initial.vx, state_initial.vy, state_initial.ax, state_initial.ay)) 

        x_guess_per, y_guess_per = self.compute_waypoints(c_x_pred, c_y_pred, prob.P_jax)
        x_guess_per = x_guess_per.T
        y_guess_per = y_guess_per.T
        #x_guess_per, y_guess_per = prob.compute_warm_traj(initial_state, v_des, x_waypoint, y_waypoint, arc_vec, x_diff, y_diff)
        plot_plan(state_initial, state_goal, obstacles, x_guess_per, y_guess_per, filename="crowdsurfer_priest/vqvae_guess_traj.png")

        initial_state = jnp.hstack(( state_initial.x, state_initial.y, state_initial.vx, state_initial.vy, state_initial.ax, state_initial.ay ))

        lamda_x = jnp.zeros((num_batch, prob.nvar))
        lamda_y = jnp.zeros((num_batch, prob.nvar))

        vx_obs = 0
        vy_obs = 0

        x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, x_obs_trajectory_dy, y_obs_trajectory_dy = prob.compute_obs_traj_prediction( jnp.asarray(dynamic_obstacles_x).flatten(), jnp.asarray(dynamic_obstacles_y).flatten(), dynamic_obstacles_vx, dynamic_obstacles_vy, jnp.asarray(static_obstacles_x).flatten(), jnp.asarray(static_obstacles_y).flatten(), vx_obs, vy_obs, initial_state[0], initial_state[1] ) ####### obstacle trajectory prediction
        
        sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess,c_mean, c_cov, x_fin, y_fin = prob.compute_traj_guess( initial_state, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_dy, y_obs_trajectory_dy, v_des, x_waypoint, y_waypoint, arc_vec, x_guess_per, y_guess_per, x_diff, y_diff)
        
        x_fin = x_fin
        y_fin = y_fin  

        x, y, c_x_best, c_y_best, x_best, y_best, x_guess_per , y_guess_per= prob.compute_cem(key, initial_state, x_fin, y_fin, lamda_x, lamda_y, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, x_obs_trajectory_dy, y_obs_trajectory_dy,sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess, x_waypoint,  y_waypoint, arc_vec, c_mean, c_cov )

        plot_plan(state_initial, state_goal, obstacles, x_guess_per, y_guess_per, filename='crowdsurfer_priest/best_traj.png')

        return None

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
