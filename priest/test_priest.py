from os import wait
import numpy as np 
import jax.numpy as jnp 
from priest import State, Obstacles, Priest 
from viz_utils import *

# Number of obstacles
num_obstacles = 25

# Define the environment boundaries
xmin, xmax = 0, 10
ymin, ymax = 0, 10

state_initial = State(x=0.5, y=1, vx=0.1, vy=0.1, ax=0, ay=0)
state_goal = State(x=5, y=5)

def sample_wall(x0, y0, x1, y1, spacing=0.5):
    """Returns a list of (x, y) points sampled along a wall from (x0, y0) to (x1, y1)."""
    length = np.hypot(x1 - x0, y1 - y0)
    n_points = int(length / spacing) + 1
    xs = np.linspace(x0, x1, n_points)
    ys = np.linspace(y0, y1, n_points)
    return list(zip(xs, ys))

# Wall definitions (line segments)
wall_0 = sample_wall(0, 4, 3.5, 4)
wall_1 = sample_wall(5, 0, 5, 4)
wall_2 = sample_wall(7, 2, 10, 2)
wall_3 = sample_wall(6, 5, 6, 10)
wall_4 = sample_wall(2, 6, 2, 10)

# Combine all walls into one list of static obstacle positions
all_walls = wall_0 + wall_1 + wall_2 + wall_3 + wall_4

# Separate into x and y coordinates for compatibility
static_obstacles_x = jnp.asarray([x for x, y in all_walls])
static_obstacles_y = jnp.asarray([y for x, y in all_walls])

# Dynamic obstacles will be sampled similarly to static obstacles
dynamic_obstacles_x = jnp.asarray(xmin + (xmax - xmin) * np.random.uniform(0, 1, num_obstacles))

dynamic_obstacles_y = jnp.asarray(ymin + (ymax - ymin) * np.random.uniform(0, 1, num_obstacles))

# Assign random velocities to dynamic obstacles
dynamic_obstacles_vx = jnp.asarray(0.5 * (2 * np.random.uniform(0, 1, num_obstacles) - 1))  # Random velocity between -0.5 and 0.5)

dynamic_obstacles_vy = jnp.asarray(0.5 * (2 * np.random.uniform(0, 1, num_obstacles) - 1))  # Random velocity between -0.5 and 0.5)

print("Static Obstacles X:", static_obstacles_x.shape)
print("Static Obstacles Y:", static_obstacles_y.shape)
print("Dynamic Obstacles X:", dynamic_obstacles_x.shape)
print("Dynamic Obstacles Y:", dynamic_obstacles_y.shape)
print("Dynamic Obstacles VX:", dynamic_obstacles_vx.shape)
print("Dynamic Obstacles VY:", dynamic_obstacles_vy.shape)

obstacles = Obstacles(static_x=static_obstacles_x, static_y=static_obstacles_y,
                      dynamic_x=dynamic_obstacles_x, dynamic_y=dynamic_obstacles_y,
                      dynamic_vx=dynamic_obstacles_vx, dynamic_vy=dynamic_obstacles_vy)


planner = Priest()

c_mean, c_cov, c_x, c_y, x, y, xdot, ydot, xddot, yddot, state_final, x_straight_line, y_straight_line = planner.initialize_trajectories(state_initial, state_goal, obstacles)

plot_plan(state_initial, state_goal, obstacles, x_straight_line, y_straight_line, filename='straight_line.png')
plot_plan(state_initial, state_goal, obstacles, x, y, filename='warm_start.png')
print("Final State:", state_final.x, state_final.y)
print(c_x[0])
print(c_y[0])

for i in range(10):
    c_x, c_y, x, y, xdot, ydot, xddot, yddot, residual_norm = planner.optimise_trajectories(state_initial.x, state_initial.y,
                                    state_initial.vx, state_initial.vy,
                                    state_initial.ax, state_initial.ax,
                                    state_final.x, state_final.y,
                                    c_x, c_y, 
                                    x, y, 
                                    xdot, ydot, 
                                    xddot, yddot, 
                                    obstacles.obstacle_trajectory_x, obstacles.obstacle_trajectory_y, 
                                    obstacles.dynamic_obstacle_trajectory_x, obstacles.dynamic_obstacle_trajectory_y)
    plot_plan(state_initial, state_final, obstacles, x, y, filename=f'optimised_{i}.png')

    #project the trajectories
    x_project, y_project = planner.project_trajectories(x, y, x_straight_line, y_straight_line)

    #compute costs for the trajectories
    cost = planner.compute_cost(x, y,
                      xdot, ydot,
                      xddot, yddot,
                      x_project, y_project,
                      residual_norm,
                      obstacles.obstacle_trajectory_x, obstacles.obstacle_trajectory_y,
                      obstacles.dynamic_obstacle_trajectory_x, obstacles.dynamic_obstacle_trajectory_y)

    #take the best trajectories (ones with the lowest cost)
    elite_idx = jnp.argsort(cost, axis=0)
    c_elite_x = c_x[elite_idx[:10]]
    c_elite_y = c_y[elite_idx[:10]]

    #update the distribution using the best trajectories
    c_mean, c_cov = planner.update_distribution(c_elite_x, c_elite_y, c_mean, c_cov, cost[elite_idx[:10]]) 
    c_x, c_y = planner.sample_trajectories(c_mean, c_cov)
