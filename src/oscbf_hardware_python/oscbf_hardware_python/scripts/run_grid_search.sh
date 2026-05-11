#!/bin/bash
trap "pkill -f franka_control_node; echo 'Grid search aborted!'; exit" INT TERM

generate_seed() {
  echo $(( (RANDOM << 15) + RANDOM ))
}

# Define the grid

counts=(0)
radii=(0)
speeds=(0)
gamma=(1.5)

for n in "${counts[@]}"; do
  for r in "${radii[@]}"; do
    for s in "${speeds[@]}"; do
      for g in "${gamma[@]}"; do

        echo "------------------------------------------------"
        echo "TESTING: N=$n | R=$r | S=$s | G=$g"

        # Set environment variables for the Python node
        export OBSTACLE_COUNT=$n
        export OBSTACLE_RADIUS=$r
        export OBSTACLE_SPEED=$s
        export CBF_GAMMA=$g
        export FRANKA_RANDOM_SEED=$(generate_seed)
        export TRAJ_RANDOM_SEED=$(generate_seed)
        while [ "$TRAJ_RANDOM_SEED" = "$FRANKA_RANDOM_SEED" ]; do
          export TRAJ_RANDOM_SEED=$(generate_seed)
        done
        export GRID_SEARCH_RESULTS_FILE=${GRID_SEARCH_RESULTS_FILE:-grid_search_results_static_test.csv}

        echo "SEEDS: FRANKA=$FRANKA_RANDOM_SEED | TRAJ=$TRAJ_RANDOM_SEED"

        # ros2 bag record -o "bag_run_${n}_${r}_${s}_${FRANKA_RANDOM_SEED}_${TRAJ_RANDOM_SEED}" \
        # --compression-mode message --compression-format zstd \
        # /franka/joint_states /franka/torque_command /franka/obstacle_data \
        # /franka/robot_spheres /tracker_centroids /ee_state /ee_target_pos_array /ee_traj_seed &
        # PID_BAG=$!

        python3 franka_control_node.py &
        PID_CONTROL=$!

        # Allow startup time
        sleep 25

        python3 pybullet_sim_node.py &
        PID_SIM=$!

        sleep 5

        python3 traj_node.py &
        PID_TRAJ=$!

        # Reference changes are logged automatically by franka_control_node.
        sleep 300

        # Trigger the final active reference segment to be written.
        kill -SIGINT $PID_CONTROL

        # Stop ROS bag
        # kill -SIGINT $PID_BAG

        sleep 3

        # Kill the remaining nodes
        kill $PID_TRAJ $PID_SIM

        pkill -9 python3

        sleep 1
      done
    done
  done
done

echo "Grid search complete."
