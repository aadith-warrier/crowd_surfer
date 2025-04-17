#!/bin/bash

SESSION_NAME="crowd_surfer"
WINDOW_NAME="crowd_surfer_ros"

tmux kill-session -t $SESSION_NAME 2>/dev/null

# Start a new tmux session and window
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME:0 "roscore" C-m

sleep 3

tmux new-window -t $SESSION_NAME -n "rviz"
tmux send-keys -t $SESSION_NAME:1 "rviz -d src/crowd_surfer/configs/config.rviz" C-m

tmux new-window -t $SESSION_NAME -n "simulation"
tmux send-keys -t $SESSION_NAME:2 "source devel/setup.bash && conda activate crowdsurfer && roslaunch crowdsurfer_ros global_nav.launch" C-m

tmux new-window -t $SESSION_NAME -n "inference"
tmux send-keys -t $SESSION_NAME:3 "source ~/miniconda3/bin/activate && conda activate crowdsurfer && source devel/setup.bash && python3 src/crowd_surfer/run/closed_loop_simulation.py"

tmux attach -t $SESSION_NAME
