#!/bin/bash

SESSION_NAME="crowd_surfer"
WINDOW_NAME="crowd_surfer_ros"
CONDA_ENV="/scratch/aadith_warrier/crowd_surfer"

tmux kill-session -t $SESSION_NAME 2>/dev/null

# Start a new tmux session and window
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME:0 "roscore" C-m

sleep 3

tmux new-window -t $SESSION_NAME -n "rviz"
tmux send-keys -t $SESSION_NAME:1 "rviz -d src/crowd_surfer/configs/config.rviz" C-m

tmux new-window -t $SESSION_NAME -n "simulation"
tmux send-keys -t $SESSION_NAME:2 "source devel/setup.bash && conda activate $CONDA_ENV && roslaunch crowdsurfer_ros global_nav.launch" C-m

tmux new-window -t $SESSION_NAME -n "inference"
tmux send-keys -t $SESSION_NAME:3 "export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1" C-m
tmux send-keys -t $SESSION_NAME:3 "export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH" C-m
tmux send-keys -t $SESSION_NAME:3 "export PATH=/usr/local/cuda-11.8/bin:$PATH" C-m
tmux send-keys -t $SESSION_NAME:3 "conda activate $CONDA_ENV && source devel/setup.bash && python3 src/crowd_surfer/run/closed_loop_simulation.py" C-m

tmux attach -t $SESSION_NAME
