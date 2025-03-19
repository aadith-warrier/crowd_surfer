#!/bin/bash

SESSION_NAME="crowd_surfer"
WINDOW_NAME="crowd_surfer_ros"

tmux kill-session -t $SESSION_NAME 2>/dev/null

# Start a new tmux session and window
tmux new-session -d -s $SESSION_NAME

tmux new-window -t $SESSION_NAME -n "rviz"
tmux send-keys -t $SESSION_NAME:1 "rviz2 -d src/crowd_surfer/configs/config.rviz " C-m

tmux new-window -t $SESSION_NAME -n "play_bag"
tmux send-keys -t $SESSION_NAME:2 "ros2 bag play src/crowd_surfer/bags/icra_bag_1" C-m

tmux new-window -t $SESSION_NAME -n "inference"
tmux send-keys -t $SESSION_NAME:3 "source ~/miniforge3/etc/profile.d/conda.sh && conda activate crowd_surfer && python3 src/crowd_surfer/run/open_loop_bag.py" C-m

tmux attach -t $SESSION_NAME