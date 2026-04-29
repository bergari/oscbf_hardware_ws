#!/bin/bash

# Define the grid
# counts=(1 2 3 4 5)
# radii=(0.02 0.04 0.06 0.08 0.10)
# speeds=(0.2 0.4 0.6 0.8 1.0)
counts=(1 2 3)
radii=(0.02 0.04 0.06 0.08 0.10)
speeds=(0.2 0.4 0.6 0.8 1.0)

for n in "${counts[@]}"; do
  for r in "${radii[@]}"; do
    for s in "${speeds[@]}"; do
      
      echo "------------------------------------------------"
      echo "TESTING: N=$n | R=$r | S=$s"
      
      # Set environment variables for the Python node
      export OBSTACLE_COUNT=$n
      export OBSTACLE_RADIUS=$r
      export OBSTACLE_SPEED=$s
      
      # Start all nodes in the background
      python3 traj_node.py &
      PID_TRAJ=$!
      
      python3 franka_control_node.py &
      PID_CONTROL=$!
      
      python3 pybullet_sim_node.py &
      PID_SIM=$!
      
      # Run for 30 seconds to allow startup
      sleep 30

      # Run test for 300 seconds (5 minutes)
      sleep 300
      
      # Send Interrupt signal to trigger the log_results() function
      kill -SIGINT $PID_CONTROL
      
      # Wait a moment for logging to finish, then clean up everything else
      sleep 2
      kill $PID_TRAJ $PID_SIM
      pkill -9 python3
      
      sleep 1 # Cool down period
    done
  done
done

echo "Grid search complete. Results saved to grid_search_results.csv"