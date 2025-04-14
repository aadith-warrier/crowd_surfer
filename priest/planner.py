import numpy as np
import jax.numpy as jnp 
from priest import State, Obstacles, Priest 

state_initial = State(x=0, y=0, vx=0, vy=0, ax=0, ay=0)
state_goal = State(x=6, y=8)

static_x = np.random.uniform(-10, 10, (25, 1)) 
static_y = np.random.uniform(-10, 10, (25, 1)) 

dynamic_x = np.random.uniform(-10, 10, (25, 1)) 
dynamic_y = np.random.uniform(-10, 10, (25, 1)) 

dynamic_vx = np.random.uniform(1, 0.4, (25, 1))**2 
dynamic_vy = np.random.uniform(1, 0.4, (25, 1))**2

obstacles = Obstacles(static_x=static_x, static_y=static_y,
                      dynamic_x=dynamic_x, dynamic_y=dynamic_y,
                      dynamic_vx=dynamic_vx, dynamic_vy=dynamic_vy)

planner = Priest()

planner.get_trajectory(state_initial, state_goal, obstacles)
