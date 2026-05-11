#!/usr/bin/env python3
"""ROS2 Controller Node for the Franka

Publishes:
- Joint torques
Subscribes:
- Joint states (position, velocity)
- Desired end-effector state (position, orientation, velocity, angular velocity)
- Dynamic obstacle positions (from tracker centroids)
"""

import signal
import time
import sys
import os
import csv
import json
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from cbfpy import CBF

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Int64
from sensor_msgs.msg import JointState
from oscbf_msgs.msg import EEState
from visualization_msgs.msg import MarkerArray

from oscbf_hardware_python.utils.rotations_and_transforms import xyzw_to_rotation_numpy
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.core.controllers import PoseTaskTorqueController

jax.config.update("jax_enable_x64", True)

class FlyingObstacle:
    """Helper class to manage the state of a single flying obstacle."""
    def __init__(self, clock, pos_min, pos_max, radius, speed):
        self.clock = clock
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.last_switch_time = self.clock.now().nanoseconds / 1e9

        self.start_pos = self.pos_min
        self.end_pos = self.pos_max
        self.radius = radius
        self.speed = speed  # This is now the PEAK speed (v_max)

        self.is_currently_colliding = False
        self.velocity = np.zeros(3)

        # Calculate initial trajectory parameters
        self._update_trajectory_params()

    def _update_trajectory_params(self):
        """Calculates distance and total required time based on the sine-wave profile."""
        self.current_distance = np.linalg.norm(self.end_pos - self.start_pos)

        # The integral of the sine wave requires T = (pi * D) / (2 * v_max)
        if self.speed > 0 and self.current_distance > 1e-6:
            self.total_time = (np.pi * self.current_distance) / (2.0 * self.speed)
        else:
            self.total_time = 1.0 # Fallback to prevent division by zero

    def update(self):
        elapsed_t = self.clock.now().nanoseconds / 1e9 - self.last_switch_time

        # Instead of checking distance, we check if the time interval has completed
        if elapsed_t >= self.total_time:
            self.last_switch_time = self.clock.now().nanoseconds / 1e9

            # Snap start position to exact end position to avoid drift
            self.start_pos = self.end_pos.copy()

            fixed_axis = np.random.randint(0, 3)
            side = np.random.choice([self.pos_min[fixed_axis], self.pos_max[fixed_axis]])
            new_end = np.array([np.random.uniform(self.pos_min[i], self.pos_max[i]) for i in range(3)])
            new_end[fixed_axis] = side
            self.end_pos = new_end

            # Recompute total time for the new trajectory length
            self._update_trajectory_params()
            return True

        return False

    def get_current_position(self):
        if self.current_distance < 1e-6:
            self.velocity = np.zeros(3)
            return self.start_pos

        elapsed_t = self.clock.now().nanoseconds / 1e9 - self.last_switch_time

        # Clamp time to prevent overshooting if update loop lags slightly
        t = min(elapsed_t, self.total_time)

        # Progress factor from 0.0 to 1.0 using the integrated sine wave
        progress = 0.5 * (1.0 - np.cos(np.pi * t / self.total_time))

        # Calculate actual velocity vector
        current_speed = self.speed * np.sin(np.pi * t / self.total_time)
        direction = (self.end_pos - self.start_pos) / self.current_distance
        self.velocity = direction * current_speed

        # Compute exact position
        actual_position = self.start_pos + (self.end_pos - self.start_pos) * progress
        return np.clip(actual_position, self.pos_min, self.pos_max)

    def check_collision_event(self, robot_spheres):
        """
        Returns:
        - collision_triggered (bool): True ONLY on the first frame of a collision.
        - min_margin (float): The absolute closest distance between this obstacle and the robot.
        - colliding_indices (list): Which robot spheres are currently overlapping.
        """
        robot_positions, robot_radii = robot_spheres
        current_pos = self.get_current_position()

        # Calculate distance from center to center
        distances = np.linalg.norm(robot_positions - current_pos, axis=1)

        # Margin = distance - (obs_radius + robot_radius)
        margins = distances - (self.radius + robot_radii.flatten())

        min_margin = float(np.min(margins))
        touching_now = min_margin < 0

        collision_triggered = False
        colliding_indices = []

        if touching_now:
            # np.where returns a tuple, [0] gets the array of indices
            colliding_indices = np.where(margins < 0)[0].tolist()
            if not self.is_currently_colliding:
                collision_triggered = True  # First hit

        self.is_currently_colliding = touching_now

        return collision_triggered, min_margin, colliding_indices

class TrackedPointEstimator:
    """Alpha-beta tracker for one measured 3D point."""
    def __init__(
        self,
        initial_pos,
        t_sec,
        alpha=0.75,
        beta=0.25,
        max_speed=1.5,
        max_predict_dt=0.12,
        stale_timeout=0.5,
    ):
        self.pos = np.asarray(initial_pos, dtype=float)
        self.vel = np.zeros(3)
        self.last_update_t = float(t_sec)
        self.alpha = alpha
        self.beta = beta
        self.max_speed = max_speed
        self.max_predict_dt = max_predict_dt
        self.stale_timeout = stale_timeout

    def _cap_velocity(self, vel):
        speed = np.linalg.norm(vel)
        if speed > self.max_speed:
            return vel / speed * self.max_speed
        return vel

    def update(self, measured_pos, t_sec):
        measured_pos = np.asarray(measured_pos, dtype=float)
        dt = max(float(t_sec) - self.last_update_t, 1e-4)

        pred_pos = self.pos + self.vel * min(dt, self.max_predict_dt)
        residual = measured_pos - pred_pos

        self.pos = pred_pos + self.alpha * residual
        self.vel = self._cap_velocity(self.vel + (self.beta / dt) * residual)
        self.last_update_t = float(t_sec)

    def predict(self, t_sec):
        age = float(t_sec) - self.last_update_t
        if age > self.stale_timeout:
            return None, None

        pred_dt = np.clip(age, 0.0, self.max_predict_dt)
        return self.pos + self.vel * pred_dt, self.vel

class OSCBFNode(Node):
    MAX_TRACKED_OBS = 10 # Pad to this many obstacles to keep matrix sizes constant

    def __init__(
        self,
        whole_body_pos_min: ArrayLike = (-0.5, -0.6, 0.0), # Define workspace limits
        whole_body_pos_max: ArrayLike = (1.0, 0.6, 1.2),
    ):
        super().__init__("oscbf_node")
        self.get_logger().info("Initializing OSCBF Node...")

        whole_body_pos_min = np.asarray(whole_body_pos_min)
        whole_body_pos_max = np.asarray(whole_body_pos_max)
        assert whole_body_pos_min.shape == (3,)
        assert whole_body_pos_max.shape == (3,)

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.torque_pub = self.create_publisher(
            Float64MultiArray, "franka/torque_command", qos_profile
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "franka/joint_states", self.joint_state_callback, qos_profile
        )
        self.ee_state_sub = self.create_subscription(
            EEState, "ee_state", self.ee_state_callback, qos_profile
        )
        self.ee_target_pos_sub = self.create_subscription(
            Float32MultiArray,
            "ee_target_pos_array",
            self.ee_target_pos_callback,
            qos_profile,
        )
        self.ee_traj_seed_sub = self.create_subscription(
            Int64,
            "ee_traj_seed",
            self.ee_traj_seed_callback,
            qos_profile,
        )

        self.tracker_sub = self.create_subscription(
            MarkerArray, "/tracker_centroids", self.tracker_callback, qos_profile
        )

        self.sphere_pub = self.create_publisher(
            Float32MultiArray, "/franka/robot_spheres", qos_profile
        )

        self.obstacle_pub = self.create_publisher(
            Float32MultiArray, "franka/obstacle_data", qos_profile
        )

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.control_freq = 1000
        self.timer = self.create_timer(1 / self.control_freq, self.publish_control)

        self.last_torque_cmd = None
        self.last_joint_state = None
        self.last_received_target_state = None
        self.active_reference_pos = None
        self.reference_change_threshold = 1e-4
        self.reference_index = -1
        self.reference_start_q_pos = None
        self.reference_start_ee_pos = None
        self.reference_start_obstacle_positions = []
        self.reference_start_ros_time = None
        self.last_current_ee_pos = None
        self.results_file_path = os.getenv(
            "GRID_SEARCH_RESULTS_FILE", "grid_search_results_by_reference.csv"
        )

        self.tracked_joints = {}
        self.tracked_vels = {}
        self.tracker_estimators = {}

        # Tune state estimator
        self.tracker_alpha = float(os.getenv("TRACKER_ALPHA", 0.9)) # Set higher to put more weights to new updates
        self.tracker_beta = float(os.getenv("TRACKER_BETA", 0.10)) # Set higher if predicted motion is too slow
        self.tracker_max_speed = float(os.getenv("TRACKER_MAX_SPEED", 1))
        self.tracker_max_predict_dt = float(os.getenv("TRACKER_MAX_PREDICT_DT", 0.12))
        self.tracker_stale_timeout = float(os.getenv("TRACKER_STALE_TIMEOUT", 0.5))

        self.tracker_centroids = np.zeros((0, 3)) # Initialize as empty array
        self.tracker_velocities = np.zeros((0, 3)) # Initialize as empty array

        self.get_logger().info("Loading Franka model...")
        self.robot = load_panda()

        self.counter = 0
        self.counter2 = 0
        self.last_published_obstacle_count = 0

        # Replicability: Generate and apply random seed
        seed_str = os.getenv("FRANKA_RANDOM_SEED")
        self.franka_random_seed = (
            int(seed_str) if seed_str is not None else int(np.random.randint(0, 2**31 - 1))
        )
        self.traj_random_seed = os.getenv("TRAJ_RANDOM_SEED", "")
        np.random.seed(self.franka_random_seed)
        self.get_logger().info(
            f"Franka obstacle seed: {self.franka_random_seed}; "
            f"trajectory seed from env: {self.traj_random_seed or 'unknown'}"
        )
        try:
            traj_seed_int = int(self.traj_random_seed) if self.traj_random_seed else None
        except ValueError:
            traj_seed_int = None
        if traj_seed_int == self.franka_random_seed:
            self.get_logger().warn(
                "FRANKA_RANDOM_SEED and TRAJ_RANDOM_SEED are identical; "
                "grid-search runs should use different seeds."
            )

        self._reset_reference_metrics()

        # Create and manage multiple flying obstacles
        # self.num_flying_obstacles = int(os.getenv("OBSTACLE_COUNT", 2))
        # self.radius = float(os.getenv("OBSTACLE_RADIUS", 0.05))
        # self.speed = float(os.getenv("OBSTACLE_SPEED", 1.0))
        self.num_flying_obstacles = 0
        self.radius = 0.05
        self.speed = 1.0

        start_point_min = np.array([0.15+self.radius, -0.8+self.radius, 0+self.radius])
        start_point_max = np.array([0.85-self.radius, 0.8-self.radius, 1.0-self.radius])
        self.flying_obstacles = [
            FlyingObstacle(
                self.get_clock(),
                start_point_min,
                start_point_max,
                radius=self.radius,
                speed=self.speed
            ) for i in range(self.num_flying_obstacles)
        ]

        # Initialize CBF
        self.get_logger().info("Creating CBF...")
        # Set relax_cbf=Ture to enable soft constrained QP, leave at false to keep hard contraints.
        self.cbf_config = CollisionsConfig(self.robot, whole_body_pos_min, whole_body_pos_max, relax_cbf=False, cbf_relaxation_penalty=1e4)
        self.cbf = CBF.from_config(self.cbf_config)

        # Initialize the padded arrays for dynamic obstacles
        self.curr_obs_start, self.curr_obs_end, self.curr_obs_radii, self.curr_obs_v_start, self.curr_obs_v_end = self._get_padded_obstacles()

        kp_pos = 30.0 # Initial value = 50 # Match desired position
        kp_rot = 0.0 # Initial value = 20 # Match desired orientation
        kd_pos = 20.0 # Initial value = 20
        kd_rot = 0.0
        kp_joint = 10.0 # Initial value = 10 # Match desired posture
        kd_joint = 5.0 # Initial value = 5 
        self.osc_controller = PoseTaskTorqueController(
            n_joints=self.robot.num_joints,
            kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
            kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
            kp_joint=kp_joint,
            kd_joint=kd_joint,
            # Note: torque limits will be enforced via the QP. We'll set them to None here
            # because we don't want to clip the values before the QP
            tau_min=None,
            tau_max=None,
        )
        self.get_logger().info("Jit compiling OSCBF controller...")
        self._jit_compile()
        self.get_logger().info("OSCBF Node initialization complete.")

    def _reset_reference_metrics(self):
        self.tracking_errors = []
        self.reference_errors = []
        self.squared_torques = []
        self.computation_times = []
        self.cbf_interventions = 0
        self.total_control_steps = 0
        self.num_collisions = 0
        self.min_distances_to_obs = []
        self.max_torques_per_joint = np.zeros(self.robot.num_joints)
        self.colliding_spheres = set()
        self.current_segment_success = False
        self.segment_time_to_goal = None
        self.qp_infeasible_count = 0
        self.max_torque_step_diff = 0.0
        self.last_tau_cmd = None
        self.collision_locations = []
        self.collision_obstacle_position_snapshots = []
        self.impact_speeds = []
        self.sum_torque_variation = 0.0
        self.current_segment_start_time = time.perf_counter()

    def _current_q_pos(self):
        if self.last_joint_state is None:
            return None
        return np.asarray(self.last_joint_state[: self.robot.num_joints], dtype=float).copy()

    def _ee_pos_from_q(self, q_pos):
        if q_pos is None:
            return None
        return np.asarray(self.robot.ee_position(q_pos), dtype=float).copy()

    def _flying_obstacle_positions(self, refresh=False):
        positions = []
        for obs in self.flying_obstacles:
            if refresh:
                obs.update()
            positions.append(np.asarray(obs.get_current_position(), dtype=float).copy())
        return positions

    @staticmethod
    def _rounded_list(values, ndigits=5):
        if values is None:
            return None
        return np.round(np.asarray(values, dtype=float), ndigits).tolist()

    def _positions_json(self, positions):
        rounded_positions = [self._rounded_list(pos) for pos in positions]
        return json.dumps(rounded_positions)

    def _collision_snapshots_json(self):
        snapshots = []
        for snapshot in self.collision_obstacle_position_snapshots:
            snapshots.append(
                {
                    "collision_index": snapshot["collision_index"],
                    "triggered_obstacle": snapshot["triggered_obstacle"],
                    "positions": [
                        self._rounded_list(pos) for pos in snapshot["positions"]
                    ],
                }
            )
        return json.dumps(snapshots) if snapshots else "[]"

    def _start_reference_segment(self, reference_pos):
        self.reference_index += 1
        self.active_reference_pos = np.asarray(reference_pos, dtype=float).copy()
        self.reference_start_q_pos = self._current_q_pos()
        self.reference_start_ee_pos = self._ee_pos_from_q(self.reference_start_q_pos)
        self.reference_start_obstacle_positions = self._flying_obstacle_positions(
            refresh=True
        )
        self.reference_start_ros_time = self.get_clock().now().nanoseconds / 1e9
        self._reset_reference_metrics()
        # self.get_logger().info(
        #     f"Reference {self.reference_index} started at "
        #     f"{self._rounded_list(self.active_reference_pos, ndigits=4)}"
        # )

    def _get_padded_obstacles(self):
        start_list, end_list, rad_list, v_start_list, v_end_list = [], [], [], [], []
        self.counter += 1
        t_sec = self.get_clock().now().nanoseconds / 1e9

        def get_estimated_joint(joint):
            estimator = self.tracker_estimators.get(joint)
            if estimator is None:
                return None

            pos, vel = estimator.predict(t_sec)
            if pos is None:
                return None

            self.tracked_joints[joint] = pos
            self.tracked_vels[joint] = vel
            return pos, vel

        def add_capsule(j_start, j_end, radius):
            """Helper to add a capsule if both joints are currently tracked"""
            if len(start_list) >= self.MAX_TRACKED_OBS:
                return

            start_estimate = get_estimated_joint(j_start)
            end_estimate = get_estimated_joint(j_end)
            if start_estimate is not None and end_estimate is not None:
                start_pos, start_vel = start_estimate
                end_pos, end_vel = end_estimate
                start_list.append(start_pos)
                end_list.append(end_pos)
                rad_list.append(radius)
                v_start_list.append(start_vel)
                v_end_list.append(end_vel)

        def add_sphere(joint, radius):
            """A sphere is just a capsule where start and end are the same point"""
            add_capsule(joint, joint, radius)

        # Wrists (15cm Spheres)
        add_sphere("fused_left wrist", 0.15)
        add_sphere("fused_right wrist", 0.15)

        # Lower Arms (8cm Capsules)
        add_capsule("fused_left elbow", "fused_left wrist", 0.1)
        add_capsule("fused_right elbow", "fused_right wrist", 0.1)

        # Upper Arms (10cm Capsules)
        add_capsule("fused_left shoulder", "fused_left elbow", 0.10)
        add_capsule("fused_right shoulder", "fused_right elbow", 0.10)

        # Head (10cm Sphere)
        add_sphere("fused_nose", 0.10)

        # Torso Sides (15cm Capsules)
        add_capsule("fused_left shoulder", "fused_left hip", 0.15)
        add_capsule("fused_right shoulder", "fused_right hip", 0.15)

        # Add static obstacles
        # start = [np.array([0.45, 0.0, 0.0]), np.array([0.45, 0.1, 0.0]), np.array([0.45, -0.1, 0.0])]
        # end = [np.array([0.45, 0.0, 0.5]), np.array([0.45, 0.1, 0.5]), np.array([0.45, -0.1, 0.5])]
        # rad = (0.05, 0.05, 0.05)
        # start_list.extend(start)
        # end_list.extend(end)
        # rad_list.extend(rad)
        # v_start_list.append(np.zeros(3))
        # v_end_list.append(np.zeros(3))

        # Add flying obstacles
        fly_obs_positions = []
        fly_obs_radii = []
        for obs in self.flying_obstacles:
            obs.update()
            position = obs.get_current_position()
            radius = obs.radius
            fly_obs_positions.append(position)
            fly_obs_radii.append(radius)

            # A sphere is a capsule with the same start and end point
            start_list.append(position)
            end_list.append(position)
            rad_list.append(obs.radius)
            v_start_list.append(obs.velocity)
            v_end_list.append(obs.velocity)

        if self.counter % 30 == 0:
            # Get q and qdot
            if self.last_joint_state is not None:
                q = self.last_joint_state[: self.robot.num_joints]
                qdot = self.last_joint_state[self.robot.num_joints:]
            else:
                q = np.zeros(self.robot.num_joints)
                qdot = np.zeros(self.robot.num_joints)

            pos_rad_jax, vel_rad_jax = compute_robot_collision_data(self.robot, q, qdot)
            pos_rad_np = np.asarray(pos_rad_jax)
            vel_rad_np = np.asarray(vel_rad_jax)

            robot_collision_positions = pos_rad_np[:, :3]
            robot_collision_radii = pos_rad_np[:, 3, None]
            robot_collision_velocities = vel_rad_np[:, :3]

            for obs_idx, obs in enumerate(self.flying_obstacles):
                triggered, min_dist, col_indices = obs.check_collision_event(
                    robot_spheres=[robot_collision_positions, robot_collision_radii]
                )

                if self.active_reference_pos is None:
                    continue

                self.min_distances_to_obs.append(min_dist)

                # Add any colliding spheres to our set
                for idx in col_indices:
                    self.colliding_spheres.add(int(idx))

                # Collision counting
                if triggered:
                    self.num_collisions += 1
                    obs_pos = obs.get_current_position()
                    obs_vel = obs.velocity

                    # Get first colliding sphere
                    sphere_id = col_indices[0]
                    v_robot = robot_collision_velocities[sphere_id]

                    # Calculate Relative Impact Speed
                    v_rel_vector = v_robot - obs_vel
                    impact_speed_m_s = float(np.linalg.norm(v_rel_vector))
                    self.impact_speeds.append(impact_speed_m_s)

                    pos_str = f"({obs_pos[0]:.2f},{obs_pos[1]:.2f},{obs_pos[2]:.2f})"
                    self.collision_locations.append(pos_str)
                    self.collision_obstacle_position_snapshots.append(
                        {
                            "collision_index": self.num_collisions,
                            "triggered_obstacle": int(obs_idx),
                            "positions": self._flying_obstacle_positions(),
                        }
                    )

                    # self.get_logger().warn(
                    #     f"NEW Collision with obstacle {obs_idx} on sphere {sphere_id}! "
                    #     f"Rel. Impact Speed: {impact_speed_m_s:.2f} m/s | Total: {self.num_collisions}"
                    # )

            self._publish_obstacle_visuals(start_list, end_list, rad_list)

        # How many obstacles did we successfully build?
        num_tracked = len(start_list)

        # We must pad up to exactly MAX_TRACKED_OBS for JAX to stay compiled
        num_padding = self.MAX_TRACKED_OBS - num_tracked

        # If we lost tracking of everything, provide at least one dummy target
        if num_tracked == 0:
            start_list.append(np.array([100.0, 0.0, 0.0]))
            end_list.append(np.array([100.0, 0.0, 0.0]))
            rad_list.append(0.0)
            v_start_list.append(np.zeros(3))
            v_end_list.append(np.zeros(3))
            num_padding -= 1

        if num_padding > 0:
            start_list.extend([np.array([100.0, 0.0, 0.0])] * num_padding)
            end_list.extend([np.array([100.0, 0.0, 0.0])] * num_padding)
            rad_list.extend([0.0] * num_padding)
            v_start_list.extend([np.zeros(3)] * num_padding)
            v_end_list.extend([np.zeros(3)] * num_padding)

        return (
            np.vstack(start_list),
            np.vstack(end_list),
            np.array(rad_list),
            np.vstack(v_start_list),
            np.vstack(v_end_list)
        )

    def _publish_obstacle_visuals(self, start_list, end_list, rad_list):
        """Publish all currently active obstacle capsules for PyBullet visualization."""
        obstacle_msg = Float32MultiArray()

        for i, (start, end, radius) in enumerate(zip(start_list, end_list, rad_list)):
            start = np.asarray(start, dtype=float)
            end = np.asarray(end, dtype=float)
            obstacle_msg.data = (
                [float(i)]
                + start.tolist()
                + end.tolist()
                + [float(radius)]
            )
            self.obstacle_pub.publish(obstacle_msg)

        # Hide any visual obstacles that were present in a previous frame but are
        # no longer active, for example when tracker detections go stale.
        for i in range(len(rad_list), self.last_published_obstacle_count):
            obstacle_msg.data = [
                float(i),
                100.0,
                0.0,
                0.0,
                100.0,
                0.0,
                0.0,
                0.0,
            ]
            self.obstacle_pub.publish(obstacle_msg)

        self.last_published_obstacle_count = len(rad_list)

    def tracker_callback(self, msg: MarkerArray):
        t_sec = self.get_clock().now().nanoseconds / 1e9

        for marker in msg.markers:
            name = marker.ns  # e.g., "fused_left wrist", "fused_nose", etc.
            pos_base = np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])

            if name not in self.tracker_estimators:
                self.tracker_estimators[name] = TrackedPointEstimator(
                    pos_base,
                    t_sec,
                    alpha=self.tracker_alpha,
                    beta=self.tracker_beta,
                    max_speed=self.tracker_max_speed,
                    max_predict_dt=self.tracker_max_predict_dt,
                    stale_timeout=self.tracker_stale_timeout,
                )
            else:
                self.tracker_estimators[name].update(pos_base, t_sec)

            est_pos, est_vel = self.tracker_estimators[name].predict(t_sec)
            self.tracked_joints[name] = est_pos
            self.tracked_vels[name] = est_vel

    def _jit_compile(self):
            """Dummy data to trigger JIT compilation of the control function during initialization."""
            z = np.zeros(self.robot.num_joints * 2)
            q = z[: self.robot.num_joints]
            qdot = z[self.robot.num_joints:]
            z_ee_des = np.concatenate([np.ones(3), np.eye(3).ravel(), np.zeros(3), np.zeros(3)])

            # Warmup
            dummy_pos, dummy_vel = compute_robot_collision_data(self.robot, q, qdot)
            dummy_pos.block_until_ready()

            # Compile passing the padded dynamic arrays
            tau_cmd, tau_nom, current_ee_pos, is_feasible = compute_control(
                self.robot, self.osc_controller, self.cbf, z, z_ee_des,
                *self._get_padded_obstacles()
            )

            # Force JAX to finish compiling and computing before moving to the next line
            tau_cmd.block_until_ready()

    def joint_state_callback(self, msg: JointState):
        self.last_joint_state = np.array([msg.position, msg.velocity]).ravel()

    def ee_state_callback(self, msg: EEState):
        pos: Point = msg.pose.position
        quat: Quaternion = msg.pose.orientation
        vel: Vector3 = msg.twist.linear
        omega: Vector3 = msg.twist.angular
        xyzw = np.array([quat.x, quat.y, quat.z, quat.w])
        flat_rmat = xyzw_to_rotation_numpy(xyzw).flatten()
        self.last_received_target_state = np.concatenate(
            [
                np.array([pos.x, pos.y, pos.z]),
                flat_rmat,
                np.array([vel.x, vel.y, vel.z]),
                np.array([omega.x, omega.y, omega.z]),
            ]
        )

    def ee_traj_seed_callback(self, msg: Int64):
        self.traj_random_seed = int(msg.data)

    def ee_target_pos_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 3:
            return

        reference_pos = np.asarray(msg.data[:3], dtype=float)
        if self.active_reference_pos is None:
            self._start_reference_segment(reference_pos)
            return

        if (
            np.linalg.norm(reference_pos - self.active_reference_pos)
            <= self.reference_change_threshold
        ):
            return

        self.log_results()
        self._start_reference_segment(reference_pos)

    def publish_control(self):
        if self.last_joint_state is None or self.last_received_target_state is None:
            return

        self.counter2 += 1

        # Update and publish obstacle data at the control frequency
        self.curr_obs_start, self.curr_obs_end, self.curr_obs_radii, self.curr_obs_v_start, self.curr_obs_v_end = self._get_padded_obstacles()

        start_time = time.perf_counter()

        # Feed the dynamic obstacle arrays into the compiled loop
        tau_cmd, tau_nom, current_ee_pos, is_feasible = compute_control(
            self.robot,
            self.osc_controller,
            self.cbf,
            self.last_joint_state,
            self.last_received_target_state,
            self.curr_obs_start,
            self.curr_obs_end,
            self.curr_obs_radii,
            self.curr_obs_v_start,
            self.curr_obs_v_end
        )

        # Log solve time
        solve_time = time.perf_counter() - start_time
        self.computation_times.append(solve_time)

        # Track infeasibility
        if not bool(is_feasible):
            self.qp_infeasible_count += 1

        tau_cmd_np = np.asarray(tau_cmd, dtype=float)
        tau_nom_np = np.asarray(tau_nom, dtype=float)
        current_ee_pos_np = np.asarray(current_ee_pos, dtype=float)
        current_target_pos = np.asarray(self.last_received_target_state[:3], dtype=float)
        self.last_current_ee_pos = current_ee_pos_np

        if self.active_reference_pos is not None:
            pos_error = np.linalg.norm(current_ee_pos_np - current_target_pos)
            reference_error = np.linalg.norm(current_ee_pos_np - self.active_reference_pos)
            self.tracking_errors.append(float(pos_error))
            self.reference_errors.append(float(reference_error))
            self.squared_torques.append(float(np.sum(np.square(tau_cmd_np))))

            if np.linalg.norm(tau_cmd_np - tau_nom_np) > 1e-3:
                self.cbf_interventions += 1

            self.total_control_steps += 1

            self.max_torques_per_joint = np.maximum(
                self.max_torques_per_joint,
                np.abs(tau_cmd_np),
            )

            if float(reference_error) < 0.05 and not self.current_segment_success:
                self.current_segment_success = True
                self.segment_time_to_goal = (
                    time.perf_counter() - self.current_segment_start_time
                )

            if self.last_tau_cmd is not None and self.total_control_steps > self.control_freq:
                delta_tau = np.abs(tau_cmd_np - self.last_tau_cmd)
                current_max_spike = float(np.max(delta_tau))
                if current_max_spike > self.max_torque_step_diff:
                    self.max_torque_step_diff = current_max_spike

                self.sum_torque_variation += float(np.sum(delta_tau))

            self.last_tau_cmd = tau_cmd_np

        msg = Float64MultiArray()
        msg.data = tau_cmd_np.tolist()
        self.torque_pub.publish(msg)

        if self.counter2 % 20 == 0: # Publish at 50Hz to avoid flooding
            spheres_jax = self.robot.link_collision_data(self.last_joint_state[: self.robot.num_joints])
            spheres_msg = Float32MultiArray()
            spheres_msg.data = np.asarray(spheres_jax).flatten().tolist()
            self.sphere_pub.publish(spheres_msg)

    def signal_handler(self, sig, frame):
        """Handle shutdown signals by publishing zero torques."""
        # TODO: Decide if we should first slow down the robot to zero velocity before shutdown
        self.get_logger().warn("Shutdown signal received, sending zero torques...")
        self.publish_zero_torques()
        self.get_logger().warn("Zero torques sent, shutting down.")
        # Allow a brief moment for the message to be published
        time.sleep(0.1)
        sys.exit(0)

    def publish_zero_torques(self):
        """Publish zero torques to the robot."""
        msg = Float64MultiArray()
        msg.data = [0.0] * self.robot.num_joints
        # Publish multiple times to ensure delivery
        for _ in range(3):
            self.torque_pub.publish(msg)
            time.sleep(1 / self.control_freq)

    def log_results(self):
        """Save one row for the currently active reference segment."""
        if self.active_reference_pos is None or self.total_control_steps == 0:
            return

        file_path = self.results_file_path
        results_dir = os.path.dirname(file_path)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0

        segment_duration = time.perf_counter() - self.current_segment_start_time
        actual_dt = (
            segment_duration / self.total_control_steps
            if self.total_control_steps > 0
            else 0.0
        )
        control_effort = np.sum(self.squared_torques) * actual_dt
        intervention_rate = (
            self.cbf_interventions / self.total_control_steps * 100.0
            if self.total_control_steps > 0
            else 0.0
        )
        avg_comp_time_ms = (
            np.mean(self.computation_times) * 1000.0
            if self.computation_times
            else 0.0
        )
        absolute_min_dist = (
            np.min(self.min_distances_to_obs) if self.min_distances_to_obs else 0.0
        )
        colliding_parts = (
            "|".join(map(str, sorted(list(self.colliding_spheres))))
            if self.colliding_spheres
            else "None"
        )
        qp_infeasibility_rate = (
            self.qp_infeasible_count / self.total_control_steps * 100.0
            if self.total_control_steps > 0
            else 0.0
        )
        mean_tracking_error = (
            np.mean(self.tracking_errors) if self.tracking_errors else 0.0
        )
        max_tracking_error = (
            np.max(self.tracking_errors) if self.tracking_errors else 0.0
        )
        final_reference_error = (
            self.reference_errors[-1] if self.reference_errors else 0.0
        )
        min_reference_error = (
            np.min(self.reference_errors) if self.reference_errors else 0.0
        )
        max_impact_speed = np.max(self.impact_speeds) if self.impact_speeds else 0.0
        avg_jitter = (
            self.sum_torque_variation / (self.total_control_steps * self.robot.num_joints)
            if self.total_control_steps > 0
            else 0.0
        )
        q_start = (
            self._rounded_list(self.reference_start_q_pos)
            if self.reference_start_q_pos is not None
            else [""] * self.robot.num_joints
        )
        ee_start = (
            self._rounded_list(self.reference_start_ee_pos)
            if self.reference_start_ee_pos is not None
            else [""] * 3
        )
        reference_pos = self._rounded_list(self.active_reference_pos)
        reference_distance = (
            np.linalg.norm(self.active_reference_pos - self.reference_start_ee_pos)
            if self.reference_start_ee_pos is not None
            else 0.0
        )
        normalized_mean_tracking_error = (
            mean_tracking_error / reference_distance
            if reference_distance > 1e-9
            else ""
        )
        max_taus = [round(float(tau), 2) for tau in self.max_torques_per_joint]

        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "Reference_Index", "Franka_Random_Seed", "Traj_Random_Seed",
                    "N_Obs", "Radius", "Speed", "Gamma",
                    "Reference_X", "Reference_Y", "Reference_Z",
                    "Q_Start_J1", "Q_Start_J2", "Q_Start_J3", "Q_Start_J4",
                    "Q_Start_J5", "Q_Start_J6", "Q_Start_J7",
                    "EE_Start_X", "EE_Start_Y", "EE_Start_Z",
                    "Flying_Obstacle_Positions_At_Reference_Start",
                    "Collision_Flying_Obstacle_Positions_All",
                    "Total_Collisions", "Colliding_Spheres", "Collided_Obstacle_Locations",
                    "Max_Collision_Speed_Relative", "Min_Dist_To_Obs_m",
                    "Task_Success", "Time_To_Goal_s", "Segment_Duration_s",
                    "Mean_Track_Err_m", "Normalized_Mean_Track_Err",
                    "Max_Track_Err_m",
                    "Final_Reference_Err_m", "Min_Reference_Err_m",
                    "Max_Tau_J1", "Max_Tau_J2", "Max_Tau_J3", "Max_Tau_J4",
                    "Max_Tau_J5", "Max_Tau_J6", "Max_Tau_J7",
                    "Control_Effort_Nm2s", "CBF_Intervention_Pct",
                    "Avg_Comp_Time_ms", "QP_Infeasibility_Pct",
                    "Avg_Jittering", "Max_Torque_Diff"
                ])

            row_data = [
                self.reference_index,
                self.franka_random_seed,
                self.traj_random_seed,
                self.num_flying_obstacles,
                self.radius,
                self.speed,
                float(os.getenv("CBF_GAMMA", 1.0)),
                reference_pos[0],
                reference_pos[1],
                reference_pos[2],
            ]
            row_data.extend(q_start)
            row_data.extend(ee_start)
            row_data.extend([
                self._positions_json(self.reference_start_obstacle_positions),
                self._collision_snapshots_json(),
                self.num_collisions,
                colliding_parts,
                "|".join(self.collision_locations) if self.collision_locations else "None",
                round(float(max_impact_speed), 2),
                round(float(absolute_min_dist), 4),
                int(self.current_segment_success),
                (
                    round(float(self.segment_time_to_goal), 3)
                    if self.segment_time_to_goal is not None
                    else ""
                ),
                round(float(segment_duration), 3),
                round(float(mean_tracking_error), 4),
                (
                    round(float(normalized_mean_tracking_error), 4)
                    if normalized_mean_tracking_error != ""
                    else ""
                ),
                round(float(max_tracking_error), 4),
                round(float(final_reference_error), 4),
                round(float(min_reference_error), 4),
            ])
            row_data.extend(max_taus)
            row_data.extend([
                round(float(control_effort), 4),
                round(float(intervention_rate), 2),
                round(float(avg_comp_time_ms), 3),
                round(float(qp_infeasibility_rate), 3),
                round(float(avg_jitter), 3),
                round(float(self.max_torque_step_diff), 3),
            ])

            writer.writerow(row_data)

        # self.get_logger().info(
        #     f"REFERENCE {self.reference_index} LOGGED: "
        #     f"{self.num_collisions} cols | "
        #     f"Final ref err: {round(float(final_reference_error), 3)}m"
        # )

    def destroy_node(self):
        # self.log_results()
        super().destroy_node()

@jax.tree_util.register_static
class DemoConfig(OSCBFTorqueConfig):
    """CBF Config for demoing OSCBF on the Franka hardware

    Safety Constraints:
    - Joint limit avoidance
    - Singularity avoidance
    - Whole-body set containment
    """

    def __init__(
        self,
        robot: Manipulator,
        whole_body_pos_min: ArrayLike,
        whole_body_pos_max: ArrayLike,
    ):
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-2
        self.whole_body_pos_min = np.asarray(whole_body_pos_min)
        self.whole_body_pos_max = np.asarray(whole_body_pos_max)
        super().__init__(robot)

    def h_2(self, z, *args, **kwargs):
        # Extract values
        q = z[: self.num_joints]
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)

        # Joint Limit Avoidance
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # Singularity Avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        robot_num_pts = robot_collision_positions.shape[0]

        # Whole-body Set Containment
        h_whole_body_upper = (
            jnp.tile(self.whole_body_pos_max, (robot_num_pts, 1))
            - robot_collision_positions
            - robot_collision_radii
        ).ravel()
        h_whole_body_lower = (
            robot_collision_positions
            - jnp.tile(self.whole_body_pos_min, (robot_num_pts, 1))
            - robot_collision_radii
        ).ravel()

        return jnp.concatenate(
            [
                h_joint_limits,
                h_singularity,
                h_whole_body_upper,
                h_whole_body_lower,
            ]
        )

    def h_1(self, z, *args, **kwargs):
        qdot = z[self.num_joints :]
        # Joint velocity limits
        joint_max_vels = jnp.asarray(self.robot.joint_max_velocities)
        qdot_max = joint_max_vels
        qdot_min = -joint_max_vels
        return jnp.concatenate([qdot_max - qdot, qdot - qdot_min])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2

class CollisionsConfig(OSCBFTorqueConfig):
    def __init__(
        self,
        robot: Manipulator,
        whole_body_pos_min: ArrayLike,
        whole_body_pos_max: ArrayLike,
        relax_cbf: bool = False,               # Default to strict
        cbf_relaxation_penalty: float = 1e4    # Default penalty weight
    ):
        base_spheres = np.arange(0, 4)   # First 4 spheres (0, 1, 2, 3)
        ee_spheres = np.arange(9, 21)   # Last 12 spheres (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
        # Create a grid of all combinations between the two groups (4 x 6 = 24 pairs)
        grid_base, grid_ee = np.meshgrid(base_spheres, ee_spheres)
        # Flatten them into 1D arrays for JAX slicing
        self.self_col_row_idx = grid_base.ravel()
        self.self_col_col_idx = grid_ee.ravel()
        len_h_self = len(self.self_col_row_idx)

        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-2
        self.whole_body_pos_min = np.asarray(whole_body_pos_min)
        self.whole_body_pos_max = np.asarray(whole_body_pos_max)
        super().__init__(robot)

    def h_1(self, z, *args, **kwargs):
        """Relative Degree 1: Velocity Limits & Dynamic Capsule Collision Avoidance"""
        q = z[: self.num_joints]
        qdot = z[self.num_joints :]

        joint_max_vels = jnp.asarray(self.robot.joint_max_velocities)
        h_qdot = jnp.concatenate([joint_max_vels - qdot, qdot - (-joint_max_vels)])

        # Extract dynamic obstacle capsules (Starts and Ends)
        if len(args) >= 5:
            obs_start = args[0]      # Shape: (N_obs, 3)
            obs_end = args[1]        # Shape: (N_obs, 3)
            obs_radii = args[2]      # Shape: (N_obs,)
            obs_vel_start = args[3]  # Shape: (N_obs, 3)
            obs_vel_end = args[4]    # Shape: (N_obs, 3)
        else:
            obs_start = jnp.full((6, 3), 100.0)
            obs_end = jnp.full((6, 3), 100.0)
            obs_radii = jnp.zeros(6)
            obs_vel_start = jnp.zeros((6, 3))
            obs_vel_end = jnp.zeros((6, 3))

        rob_pos_rad, rob_vel_rad = jax.jvp(
            self.robot.link_collision_data, (q,), (qdot,)
        )
        rob_pos = rob_pos_rad[:, :3]
        rob_radii = rob_pos_rad[:, 3]
        rob_vel = rob_vel_rad[:, :3]

        # Find closest point on each capsule centerline to each robot sphere center
        # Utillize broadcasting to compute for all robot spheres and all capsules in one go
        A = obs_start[None, :, :] # Start of segment (1, N_obs, 3)
        B = obs_end[None, :, :]   # End of segment (1, N_obs, 3)
        P = rob_pos[:, None, :]   # Robot spheres (N_rob, 1, 3)

        AB = B - A
        AP = P - A # Shape (N_rob, N_obs, 3)

        AB_dot_AB = jnp.sum(AB * AB, axis=-1) + 1e-6 # Avoid div by zero
        AP_dot_AB = jnp.sum(AP * AB, axis=-1)
        t = AP_dot_AB / AB_dot_AB

        # Clip t to [0, 1] to keep the closest point strictly on the segment
        t = jnp.clip(t, 0.0, 1.0)

        # Compute the closest point C on the capsule centerline to each robot sphere center
        C = A + t[..., None] * AB

        # Interpolate the velocity of the obstacle at point C
        vel_A = obs_vel_start[None, :, :]
        vel_B = obs_vel_end[None, :, :]
        vel_C = vel_A + t[..., None] * (vel_B - vel_A)

        # Standard CBF calculations using C
        delta_p = P - C
        dist = jnp.linalg.norm(delta_p, axis=2)
        normals = delta_p / (dist[..., None] + 1e-6)

        delta_v = rob_vel[:, None, :] - vel_C
        h0_dot = jnp.sum(normals * delta_v, axis=2)

        radii_sum = rob_radii[:, None] + obs_radii[None, :]
        safety_padding = 0.0 # use for testing, but better way is to adapt capsule radii
        h0 = dist - radii_sum - safety_padding

        # Lower gamma = earlier responding, higher gamma = later responding. Value of 2.0 proved good in real life testing
        # gamma = float(os.getenv("CBF_GAMMA", 1.0))
        gamma = 1.75
        h_collision_dynamic = (h0_dot + gamma * h0).ravel() # constraint: h_dot + gamma * h >= 0 -> h_dot >= -gamma * h

        return jnp.concatenate([h_qdot, h_collision_dynamic])

    def h_2(self, z, *args, **kwargs):
        """Relative Degree 2: Static Position Limits & Workspace Containment"""
        # Extract values
        q = z[: self.num_joints]
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)

        # Joint Limit Avoidance
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # Singularity Avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

        # Get robot sphere positions
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        robot_num_pts = robot_collision_positions.shape[0]

        # Self-Collision Avoidance
        pos_A = robot_collision_positions[self.self_col_row_idx]
        pos_B = robot_collision_positions[self.self_col_col_idx]
        rad_A = robot_collision_radii[self.self_col_row_idx]
        rad_B = robot_collision_radii[self.self_col_col_idx]
        diffs = pos_A - pos_B
        dists = jnp.sqrt(jnp.sum(diffs**2, axis=-1) + 1e-8)
        h_self_collision = dists - (rad_A.ravel() + rad_B.ravel())

        # Static Table Avoidance (Z-axis floor)
        h_table = (
            robot_collision_positions[:, 2] - robot_collision_radii.ravel()
        )

        # Whole-body Set Containment (Workspace bounding box)
        h_whole_body_upper = (
            jnp.tile(self.whole_body_pos_max, (robot_num_pts, 1))
            - robot_collision_positions
            - robot_collision_radii
        ).ravel()

        h_whole_body_lower = (
            robot_collision_positions
            - jnp.tile(self.whole_body_pos_min, (robot_num_pts, 1))
            - robot_collision_radii
        ).ravel()

        return jnp.concatenate([
            h_joint_limits,
            # h_whole_body_upper,
            # h_whole_body_lower,
            h_table,
            h_singularity,
            h_self_collision,
        ])

    def alpha(self, h):
        # Lower = more conservative, higher = more aggressive
        # Formula: h_ddot + alpha h_dot + alpha*alpha_2 h >= 0
        # alpha: constant, alpha_2 damping
        # Value of 4.0 proved good in real life testing, 25 in simulation
        return 4.0 * h

    def alpha_2(self, h_2):
        # Value of 2.0 proved good in real life testing, 15 in simulation
        return 2.0 * h_2

@partial(jax.jit, static_argnums=(0, 1, 2))
def compute_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
    obs_start: ArrayLike,
    obs_end: ArrayLike,
    obs_radii: ArrayLike,
    obs_vel_start: ArrayLike,
    obs_vel_end: ArrayLike,
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)

    nullspace_posture_goal = jnp.array(
        [0.0, -jnp.pi / 6, 0.0, -3 * jnp.pi / 4, 0.0, 5 * jnp.pi / 9, 0.0]
    )

    # Compute nominal control
    u_nom = osc_controller(
        q, qdot, pos=ee_tmat[:3, 3], rot=ee_tmat[:3, :3],
        des_pos=z_ee_des[:3], des_rot=jnp.reshape(z_ee_des[3:12], (3, 3)),
        des_vel=z_ee_des[12:15], des_omega=z_ee_des[15:18],
        des_accel=jnp.zeros(3), des_alpha=jnp.zeros(3),
        des_q=nullspace_posture_goal, des_qdot=jnp.zeros(robot.num_joints),
        J=J, M=M, M_inv=M_inv, g=g, c=c,
    )

    # Apply the CBF safety filter, passing the dynamic obstacles as kwargs
    tau = cbf.safety_filter(
        z,
        u_nom,
        obs_start,
        obs_end,
        obs_radii,
        obs_vel_start,
        obs_vel_end,
    )

    # Infeasibility check
    is_infeasible = jnp.any(jnp.isnan(tau))
    is_feasible = jnp.logical_not(is_infeasible)

    tau_limits = jnp.asarray(robot.joint_max_forces)

    # If infeasible, fall back to nominal control, but keep it bounded.
    safe_tau = jnp.where(is_feasible, tau, u_nom)
    safe_tau = jnp.clip(safe_tau, -tau_limits, tau_limits)

    tau_cmd = safe_tau - g
    tau_cmd = jnp.clip(tau_cmd, -tau_limits, tau_limits)

    tau_nom_cmd = u_nom - g
    tau_nom_cmd = jnp.clip(tau_nom_cmd, -tau_limits, tau_limits)

    return tau_cmd, tau_nom_cmd, ee_tmat[:3, 3], is_feasible


@partial(jax.jit, static_argnums=(0,))
def compute_robot_collision_data(robot: Manipulator, q: ArrayLike, qdot: ArrayLike):
    """Computes both positions/radii and velocities of collision spheres in one JIT-compiled pass."""
    pos_rad, vel_rad = jax.jvp(robot.link_collision_data, (q,), (qdot,))
    return pos_rad, vel_rad

def main(args=None):
    rclpy.init(args=args)
    node = OSCBFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
    finally:
        node.publish_zero_torques()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
