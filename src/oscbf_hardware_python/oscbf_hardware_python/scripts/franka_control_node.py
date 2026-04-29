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
from std_msgs.msg import Float64MultiArray, Float32MultiArray
from sensor_msgs.msg import JointState
from oscbf_msgs.msg import EEState
from visualization_msgs.msg import MarkerArray

from oscbf_hardware_python.utils.rotations_and_transforms import xyzw_to_rotation_numpy
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.core.controllers import PoseTaskTorqueController

jax.config.update("jax_enable_x64", True)

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
        safety_padding = 0.1 # use for testing, but better way is to adapt capsule radii
        h0 = dist - radii_sum - safety_padding

        # Lower gamma = earlier responding, higher gamma = later responding. Value of 2.0 proved good in real life testing
        gamma = 50
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

        # Static Table Avoidance (Z-axis floor)
        # h_table = (
        #     robot_collision_positions[:, 2] - robot_collision_radii.ravel()
        # )

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
            h_whole_body_upper, 
            h_whole_body_lower, 
            h_singularity,
        ])

    def alpha(self, h):
        # Lower = more conservative, higher = more aggressive
        # Value of 4.0 proved good in real life testing
        return 20.0 * h

    def alpha_2(self, h_2):
        # Value of 2.0 proved good in real life testing
        return 20.0 * h_2

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

    return tau - g

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
        Returns True ONLY on the first frame of a collision.
        """
        robot_positions, robot_radii = robot_spheres
        current_pos = self.get_current_position()
        
        distances = np.linalg.norm(robot_positions - current_pos, axis=1)
        touching_now = np.any(distances < (self.radius + robot_radii.flatten()))
        
        collision_triggered = False
        if touching_now and not self.is_currently_colliding:
            collision_triggered = True  # First hit
        
        self.is_currently_colliding = touching_now
        return collision_triggered

class OSCBFNode(Node):
    MAX_TRACKED_OBS = 3 # Pad to this many obstacles to keep matrix sizes constant

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

        self.tracker_sub = self.create_subscription(
            MarkerArray, "/tracker_centroids", self.tracker_callback, qos_profile
        )

        self.sphere_pub = self.create_publisher(
            Float32MultiArray, "franka/robot_spheres", qos_profile
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

        self.tracked_joints = {}  # Format: {'left_arm': {'start': pos, 'end': pos}}
        self.tracked_vels = {}  # Format: {'left_arm': {'start': vel, 'end': vel}}
        self.last_tracker_pos = {}

        self.tracker_centroids = np.zeros((0, 3)) # Initialize as empty array
        self.tracker_velocities = np.zeros((0, 3)) # Initialize as empty array
        
        self.get_logger().info("Loading Franka model...")
        self.robot = load_panda()

        self.counter = 0
        self.counter2 = 0

        # Initialize tracking metrics
        self.num_collisions = 0

        # Create and manage multiple flying obstacles
        self.num_flying_obstacles = int(os.getenv("OBSTACLE_COUNT", 2))
        self.radius = float(os.getenv("OBSTACLE_RADIUS", 0.05))
        self.speed = float(os.getenv("OBSTACLE_SPEED", 1.0))
        start_point_min = np.array([0.1+self.radius, -0.8+self.radius, 0+self.radius])
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

        kp_pos = 50.0 # Initial value = 50
        kp_rot = 20.0 # Initial value = 20
        kd_pos = 20.0
        kd_rot = 10.0
        kp_joint = 10.0 # Initial value = 10
        kd_joint = 5.0
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

    def _get_padded_obstacles(self):
        start_list, end_list, rad_list, v_start_list, v_end_list = [], [], [], [], []
        self.counter += 1
        
        def add_capsule(j_start, j_end, radius):
            """Helper to add a capsule if both joints are currently tracked"""
            if j_start in self.tracked_joints and j_end in self.tracked_joints:
                start_list.append(self.tracked_joints[j_start])
                end_list.append(self.tracked_joints[j_end])
                rad_list.append(radius)
                v_start_list.append(self.tracked_vels[j_start])
                v_end_list.append(self.tracked_vels[j_end])

        def add_sphere(joint, radius):
            """A sphere is just a capsule where start and end are the same point"""
            add_capsule(joint, joint, radius)

        # Wrists (15cm Spheres)
        add_sphere("fused_left wrist", 0.15)
        add_sphere("fused_right wrist", 0.15)
        
        # Lower Arms (8cm Capsules)
        add_capsule("fused_left elbow", "fused_left wrist", 0.08)
        add_capsule("fused_right elbow", "fused_right wrist", 0.08)
        
        # Upper Arms (10cm Capsules)
        add_capsule("fused_left shoulder", "fused_left elbow", 0.10)
        add_capsule("fused_right shoulder", "fused_right elbow", 0.10)
        
        # Head (10cm Sphere)
        add_sphere("fused_nose", 0.10) 
        
        # Torso Sides (15cm Capsules)
        add_capsule("fused_left shoulder", "fused_left hip", 0.15)
        add_capsule("fused_right shoulder", "fused_right hip", 0.15)

        # How many obstacles did we successfully build?
        num_tracked = len(start_list)
        
        # We must pad up to exactly MAX_TRACKED_OBS (10) for JAX to stay compiled
        num_padding = self.MAX_TRACKED_OBS - num_tracked

        # Add static obstacles
        # start_list.append(np.array([0.45, 0.0, 0.0]))
        # end_list.append(np.array([0.45, 0.0, 0.1]))
        # rad_list.append(0.15)
        # v_start_list.append(np.zeros(3))
        # v_end_list.append(np.zeros(3))

        # Add random flying obstacle
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

        # Publish them such that they can be visualized in PyBullet
        if self.counter % 20 == 0: # Publish at 50Hz to avoid flooding
            obstacle_pos_msg = Float32MultiArray()
            for i, obs in enumerate(self.flying_obstacles):
                obs.update()
                pos = obs.get_current_position()
                # Data: [ID, x, y, z, radius]
                msg_data = [float(i)] + pos.tolist() + [float(obs.radius)]
                
                obstacle_pos_msg.data = msg_data
                self.obstacle_pub.publish(obstacle_pos_msg)
        
            # Count number of collisions with robot spheres
            # Get robot sphere positions
            q = self.last_joint_state[: self.robot.num_joints] if self.last_joint_state is not None else np.zeros(self.robot.num_joints)
            robot_collision_pos_rad = self.robot.link_collision_data(q)
            robot_collision_positions = robot_collision_pos_rad[:, :3]
            robot_collision_radii = robot_collision_pos_rad[:, 3, None]
            for obs in self.flying_obstacles:
                # This only returns True on the very first frame of contact
                if obs.check_collision_event(robot_spheres=[robot_collision_positions, robot_collision_radii]):
                    self.num_collisions += 1
                    self.get_logger().warn(
                        f"NEW Collision! Total: {self.num_collisions}"
                    )
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

    def tracker_callback(self, msg: MarkerArray):
        t_now = self.get_clock().now()
        dt = (t_now - getattr(self, 'last_tracker_time', t_now)).nanoseconds / 1e9
        self.last_tracker_time = t_now
        
        for marker in msg.markers:
            name = marker.ns  # e.g., "fused_left wrist", "fused_nose", etc.
            pos_base = np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
            self.tracked_joints[name] = pos_base
            
            # Simple EMA Velocity to avoid noisy differentiation and a too sharp velocity increase (which happens when dynamic obstacles are received at low frequency)
            if name in self.last_tracker_pos and dt > 0.005:
                raw_v = (pos_base - self.last_tracker_pos[name]) / dt
                old_v = self.tracked_vels.get(name, np.zeros(3))
                smooth_v = 1.0 * raw_v + 0 * old_v
                # Cap speed
                speed = np.linalg.norm(smooth_v)
                self.tracked_vels[name] = (smooth_v / speed * 1.5) if speed > 1.5 else smooth_v
            else:
                self.tracked_vels[name] = np.zeros(3)
                
            self.last_tracker_pos[name] = pos_base

        # After updating all joints, build the capsules

    def _jit_compile(self):
        """Dummy data to trigger JIT compilation of the control function during initialization."""
        z = np.zeros(self.robot.num_joints * 2)
        z_ee_des = np.concatenate([np.ones(3), np.eye(3).ravel(), np.zeros(3), np.zeros(3)])
        # Compile passing the padded dynamic arrays
        _ = np.asarray(
            # Call _get_padded_obstacles to get initial dummy data for compilation
            # This ensures the JAX function is compiled with the correct array shapes.
            compute_control(
                self.robot, self.osc_controller, self.cbf, z, z_ee_des, 
                *self._get_padded_obstacles()
            )
        )
    
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
        
    def publish_control(self):
        if self.last_joint_state is None or self.last_received_target_state is None:
            return

        self.counter2 += 1

        # Update and publish obstacle data at the control frequency
        self.curr_obs_start, self.curr_obs_end, self.curr_obs_radii, self.curr_obs_v_start, self.curr_obs_v_end = self._get_padded_obstacles()

        msg = Float64MultiArray()
        # Feed the dynamic obstacle arrays into the compiled loop
        tau = compute_control(
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
        msg.data = tau.tolist()
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
        """Final save before the node is killed by the bash script."""
        file_path = "grid_search_results.csv"
        file_exists = os.path.isfile(file_path)
        
        # We focus only on the parameters and the final collision count
        with open(file_path, "a") as f:
            writer = csv.writer(f)
            # Write header only if the file is brand new
            if not file_exists:
                writer.writerow(["N", "Radius", "Speed", "Total_Collisions"])
            
            writer.writerow([
                self.num_flying_obstacles, 
                self.radius, 
                self.speed, 
                self.num_collisions
            ])
        self.get_logger().info(f"DATA LOGGED: {self.num_collisions} collisions detected.")

    def destroy_node(self):
        self.log_results()
        super().destroy_node()

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
