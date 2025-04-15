from os import wait
import numpy as np 
import jax.numpy as jnp 
from priest import State, Obstacles, Priest 
from viz_utils import *
import mpc_non_dy
from jax import random

# Number of obstacles
num_obstacles = 10

# Define the environment boundaries
xmin, xmax = 0, 10
ymin, ymax = 0, 10

state_initial = State(x=5, y=5, vx=0, vy=0, ax=0, ay=0)
state_goal = State(x=9.5, y=9.5)

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
            grid[i, j] = 2

    static_obstacles_x = [x for (x,y) in all_walls]
    static_obstacles_y = [y for (x,y) in all_walls]

    return grid, all_walls, jnp.asarray(static_obstacles_x), jnp.asarray(static_obstacles_y), jnp.asarray(dynamic_obstacles_x), jnp.asarray(dynamic_obstacles_y), jnp.asarray(dynamic_obstacles_vx), jnp.asarray(dynamic_obstacles_vy)

grid, walls, static_obstacles_x, static_obstacles_y, dynamic_obstacles_x, dynamic_obstacles_y, dynamic_obstacles_vx, dynamic_obstacles_vy = get_occupancy_grid()

print("Static Obstacles X:", static_obstacles_x.shape)
print("Static Obstacles Y:", static_obstacles_y.shape)
print("Dynamic Obstacles X:", dynamic_obstacles_x.shape)
print("Dynamic Obstacles Y:", dynamic_obstacles_y.shape)
print("Dynamic Obstacles VX:", dynamic_obstacles_vx.shape)
print("Dynamic Obstacles VY:", dynamic_obstacles_vy.shape)

obstacles = Obstacles(static_x=static_obstacles_x, static_y=static_obstacles_y,
                      dynamic_x=dynamic_obstacles_x, dynamic_y=dynamic_obstacles_y,
                      dynamic_vx=dynamic_obstacles_vx, dynamic_vy=dynamic_obstacles_vy)

a_obs_1 = 0.5
a_obs_2 = 0.5
b_obs_1 = 0.68
b_obs_2 = 0.68 
v_max = 1
v_min = 0.2 
a_max = 1
t_fin = 10 
num = 100
num_batch = 110
maxiter = 1
maxiter_mpc = 1
maxiter_cem = 50
weight_track = 0.001
weight_smoothness = 1
way_point_shape = 1000
v_des = 1

x_waypoint = jnp.linspace(state_initial.x, state_goal.x, way_point_shape)
y_waypoint = jnp.linspace(state_initial.y, state_goal.y, way_point_shape)

num_obs_1 = 40
num_obs_2 = 10

prob = mpc_non_dy.batch_crowd_nav(a_obs_1, b_obs_1, a_obs_2, b_obs_2, v_max, v_min, a_max, num_obs_1, num_obs_2, t_fin, num, num_batch, maxiter, maxiter_cem, weight_smoothness, weight_track, way_point_shape, v_des)

key = random.PRNGKey(0)

arc_length, arc_vec, x_diff, y_diff =prob.path_spline(x_waypoint, y_waypoint)

initial_state = jnp.hstack(( state_initial.x, state_initial.y, state_initial.vx, state_initial.vy, state_initial.ax, state_initial.ay)) 

x_guess_per, y_guess_per = prob.compute_warm_traj(initial_state, v_des, x_waypoint, y_waypoint, arc_vec, x_diff, y_diff)
plot_plan(state_initial, state_goal, obstacles, x_guess_per, y_guess_per, filename="og/warm_traj.png")

mpc_count = 0

for i in range(0, maxiter_mpc):

    initial_state = jnp.hstack(( state_initial.x, state_initial.y, state_initial.vx, state_initial.vy, state_initial.ax, state_initial.ay ))

    lamda_x = jnp.zeros((num_batch, prob.nvar))
    lamda_y = jnp.zeros((num_batch, prob.nvar))

    vx_obs = 0
    vy_obs = 0

    x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, x_obs_trajectory_dy, y_obs_trajectory_dy = prob.compute_obs_traj_prediction( jnp.asarray(dynamic_obstacles_x).flatten(), jnp.asarray(dynamic_obstacles_y).flatten(), dynamic_obstacles_vx, dynamic_obstacles_vy, jnp.asarray(static_obstacles_x).flatten(), jnp.asarray(static_obstacles_y).flatten(), vx_obs, vy_obs, initial_state[0], initial_state[1] ) ####### obstacle trajectory prediction
    
    print("Static Obs x:", x_obs_trajectory.shape)
    print("Static Obs y:", y_obs_trajectory.shape)
    print("Dynamic Obs x:", x_obs_trajectory_dy.shape)
    print("Dynamic Obs y:", y_obs_trajectory_dy.shape)

    sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess,c_mean, c_cov, x_fin, y_fin = prob.compute_traj_guess( initial_state, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_dy, y_obs_trajectory_dy, v_des, x_waypoint, y_waypoint, arc_vec, x_guess_per, y_guess_per, x_diff, y_diff)
    
    x_fin = x_fin
    y_fin = y_fin  

    x, y, c_x_best, c_y_best, x_best, y_best, x_guess_per , y_guess_per= prob.compute_cem(key, initial_state, x_fin, y_fin, lamda_x, lamda_y, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, x_obs_trajectory_dy, y_obs_trajectory_dy,sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess, x_waypoint,  y_waypoint, arc_vec, c_mean, c_cov )

plot_plan(state_initial, state_goal, obstacles, x_guess_per, y_guess_per, filename='og/best_traj.png')
