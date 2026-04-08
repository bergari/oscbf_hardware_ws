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
    ):
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-2
        self.whole_body_pos_min = np.asarray(whole_body_pos_min)
        self.whole_body_pos_max = np.asarray(whole_body_pos_max)
        super().__init__(robot)

    # def h_1(self, z, *args, **kwargs):
    #     """Relative Degree 1: Joint Velocity Limits"""
    #     qdot = z[self.num_joints :]      
    #     # Joint velocity limits
    #     joint_max_vels = jnp.asarray(self.robot.joint_max_velocities)
    #     qdot_max = joint_max_vels
    #     qdot_min = -joint_max_vels
    #     return jnp.concatenate([qdot_max - qdot, qdot - qdot_min])

    def h_1(self, z, *args, **kwargs):
        """Relative Degree 1: Velocity Limits & Dynamic Collision Avoidance"""
        q = z[: self.num_joints]
        qdot = z[self.num_joints :]      

        # Joint velocity limits
        joint_max_vels = jnp.asarray(self.robot.joint_max_velocities)
        h_qdot = jnp.concatenate([joint_max_vels - qdot, qdot - (-joint_max_vels)])

        # Extract dynamic obstacles
        if len(args) >= 3:
            obs_pos = args[0]
            obs_radii = args[1]
            obs_vel = args[2] # Obstacle velocity
        else:
            obs_pos = jnp.full((6, 3), 100.0)
            obs_radii = jnp.zeros(6)
            obs_vel = jnp.zeros((6, 3))

        # Calculate Robot Sphere Positions and Velocities
        # JVP calculates v_robot = J(q) * qdot instantly for all spheres
        rob_pos_rad, rob_vel_rad = jax.jvp(
            self.robot.link_collision_data, (q,), (qdot,)
        )
        rob_pos = rob_pos_rad[:, :3]
        rob_radii = rob_pos_rad[:, 3]
        rob_vel = rob_vel_rad[:, :3]

        # Calculate Distance and Normal Vectors
        # Broadcasting to compare every robot sphere against every obstacle
        delta_p = rob_pos[:, None, :] - obs_pos[None, :, :]  # Shape: (N_rob, N_obs, 3)
        dist = jnp.linalg.norm(delta_p, axis=2)              # Shape: (N_rob, N_obs)
        normals = delta_p / (dist[..., None] + 1e-6)         # Shape: (N_rob, N_obs, 3)

        # Calculate Relative Approach Speed (h0_dot)
        delta_v = rob_vel[:, None, :] - obs_vel[None, :, :]  # Shape: (N_rob, N_obs, 3)
        h0_dot = jnp.sum(normals * delta_v, axis=2)          # Dot product along 3rd axis

        # Calculate Base Distance (h0)
        radii_sum = rob_radii[:, None] + obs_radii[None, :]
        safety_padding = 0.05 # Add a safety margin to the combined radii (just for debugging)
        h0 = dist - radii_sum - safety_padding

        # Formulate the Dynamic CBF
        # Higher gamma softer response, lower gamma more aggressive
        # CBF requires h0_dot + gamma * h0 >= 0 (note: h0_dot is negative when approach robot)
        # With gamma = 1, if h0_dot is -0.5 m/s when h0 is 0.5m, we get -0.5 + 1*0.5 = 0, right at the boundary. 
        # If gamma = 2, we get -0.5 + 2*0.5 = 0.5, giving us a safety margin. Meaning the robot would need to be approached faster or be closer for the CBF to activate.
        gamma = 2
        h_collision_dynamic = (h0_dot + gamma * h0).ravel()

        return jnp.concatenate([h_qdot, h_collision_dynamic])

    def h_2(self, z, *args, **kwargs):
        """Relative Degree 2: Static Position Limits & Workspace Containment"""
        # Extract values
        q = z[: self.num_joints]
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)

        if len(args) >= 2:
            collision_positions = args[0]
            collision_radii = args[1]
        else:
            collision_positions = jnp.full((6, 3), 100.0)
            collision_radii = jnp.zeros(6)

        # Joint Limit Avoidance
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # Singularity Avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

        # Get robot sphere positions (WE NEED THIS UNCOMMENTED)
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        robot_num_pts = robot_collision_positions.shape[0]

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

        # Collision Avoidance with obstacles (Static, based on position only)
        # center_deltas = (
        #     robot_collision_positions[:, None, :] - collision_positions[None, :, :]
        # ).reshape(-1, 3)
        # radii_sums = (
        #     robot_collision_radii[:, None] + collision_radii[None, :]
        # ).reshape(-1)
        # safety_padding = 0.0 # Add a safety margin to the combined radii
        # h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums - safety_padding

        return jnp.concatenate([
            h_joint_limits, 
            h_whole_body_upper, 
            h_whole_body_lower, 
            h_singularity, 
            h_table,
            # h_collision
        ])

    def alpha(self, h):
        # Lower = more conservative, higher = more aggressive
        # For some reason increasing this from 2 to 4 resullts in a steady state offset
        return 4.0 * h

    def alpha_2(self, h_2):
        return 2.0 * h_2

@partial(jax.jit, static_argnums=(0, 1, 2))
def compute_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
    obs_positions: ArrayLike,
    obs_radii: ArrayLike,
    obs_velocities: ArrayLike,
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
        obs_positions, 
        obs_radii,
        obs_velocities
    )

    return tau - g

class OSCBFNode(Node):
    MAX_TRACKED_OBS = 5 # Pad to this many obstacles to keep matrix sizes constant

    def __init__(
        self,
        whole_body_pos_min: ArrayLike = (-0.5, -0.75, -0.05), # Define workspace limits
        whole_body_pos_max: ArrayLike = (1.0, 0.75, 1.0),
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

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.control_freq = 1000
        self.timer = self.create_timer(1 / self.control_freq, self.publish_control)

        self.last_torque_cmd = None
        self.last_joint_state = None
        self.last_received_target_state = None
        self.tracker_centroids = np.zeros((0, 3)) # Initialize as empty array
        self.tracker_velocities = np.zeros((0, 3)) # Initialize as empty array
        
        self.get_logger().info("Loading Franka model...")
        self.robot = load_panda()

        # Transform form camera frame to robot base frame (if desired to do manually else set to identity)
        self.T_C_to_B = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Initialize CBF 
        self.get_logger().info("Creating CBF...")
        self.cbf_config = CollisionsConfig(self.robot, whole_body_pos_min, whole_body_pos_max)
        self.cbf = CBF.from_config(self.cbf_config)
        
        # Initialize the padded arrays for dynamic obstacles
        self.current_obs_pos, self.current_obs_radii, self.current_obs_vel = self._get_padded_obstacles()

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
        """Creates fixed-size obstacle arrays for JAX."""
        # Use for testing/demo purposes, currently set far away from robot 
        virtual_obstacle_pos = np.array([[100, 0.0, 0.45]])
        virtual_obstacle_radius = np.array([0.20])
        virtual_obstacle_vel = np.array([[0.0, 0.0, 0.0]])
        
        pos_list = [virtual_obstacle_pos]
        rad_list = [virtual_obstacle_radius]
        vel_list = [virtual_obstacle_vel]
        
        num_tracked = min(self.tracker_centroids.shape[0], self.MAX_TRACKED_OBS)
        
        # Add real tracked markers
        if num_tracked > 0:
            ones = np.ones((num_tracked, 1))
            centroids_homogeneous = np.hstack([self.tracker_centroids[:num_tracked], ones])
            centroids_base_frame = (self.T_C_to_B @ centroids_homogeneous.T).T[:, :3]
            self.get_logger().info(f"Tracker centroids (base frame): {centroids_base_frame}")
            pos_list.append(centroids_base_frame)
            rad_list.append(np.full(num_tracked, 0.15)) # Change this if you want different radii for the tracked obstacles
            vel_list.append(self.tracker_velocities[:num_tracked])
            
        # Pad the rest with dummy obstacles far away
        num_padding = self.MAX_TRACKED_OBS - num_tracked
        if num_padding > 0:
            pos_list.append(np.full((num_padding, 3), 100.0)) # 100 meters away
            rad_list.append(np.zeros(num_padding))            # 0 radius
            vel_list.append(np.zeros((num_padding, 3)))     # 0 velocity
            
        return np.vstack(pos_list), np.concatenate(rad_list), np.vstack(vel_list)

    def tracker_callback(self, msg: MarkerArray):
        marker_positions = []
        marker_velocities = []
        t_now = self.get_clock().now()
        min_dt = 0.005 
        
        if not hasattr(self, 'last_tracker_pos'):
            self.last_tracker_pos = None
            self.last_tracker_time = t_now
            self.filtered_vel = np.zeros(3) # Initialize the filter memory

        dt = (t_now - self.last_tracker_time).nanoseconds / 1e9
        self.last_tracker_time = t_now

        for marker in msg.markers:
            if marker.ns == "fused_left wrist":
                pos = np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
                marker_positions.append(pos)
                
                if self.last_tracker_pos is not None and dt > min_dt:
                    # Raw velocity
                    raw_vel_vector = (pos - self.last_tracker_pos) / dt
                    raw_vel_base = self.T_C_to_B[:3, :3] @ raw_vel_vector 
                    
                    # Apply EMA Filter to smooth velocity reading to avoid spikes in CBF response
                    # Can increase filter weight once we get higher hz of obstacle positions
                    # 0.4 means trust the new reading 40%, trust the old history 60%
                    filter_weight = 0.4 
                    self.filtered_vel = (filter_weight * raw_vel_base) + ((1.0 - filter_weight) * self.filtered_vel)

                    # Cap smoothed velocity
                    speed = np.linalg.norm(self.filtered_vel)
                    if speed > 1.5: # Prevent errors
                        self.filtered_vel = (self.filtered_vel / speed) * 1.5
                        
                    marker_velocities.append(self.filtered_vel)
                else:
                    marker_velocities.append(np.zeros(3))
                
                self.last_tracker_pos = pos

        self.tracker_centroids = np.array(marker_positions) if marker_positions else np.zeros((0, 3))
        self.tracker_velocities = np.array(marker_velocities) if marker_velocities else np.zeros((0, 3))
        self.current_obs_pos, self.current_obs_radii, self.current_obs_vel = self._get_padded_obstacles()

    def _jit_compile(self):
        """Dummy data to trigger JIT compilation of the control function during initialization."""
        z = np.zeros(self.robot.num_joints * 2)
        z_ee_des = np.concatenate([np.ones(3), np.eye(3).ravel(), np.zeros(3), np.zeros(3)])
        # Compile passing the padded dynamic arrays
        _ = np.asarray(
            compute_control(
                self.robot, self.osc_controller, self.cbf, z, z_ee_des,
                self.current_obs_pos, self.current_obs_radii, self.current_obs_vel
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
        msg = Float64MultiArray()
        # Feed the dynamic obstacle arrays into the compiled loop
        tau = compute_control(
            self.robot,
            self.osc_controller,
            self.cbf,
            self.last_joint_state,
            self.last_received_target_state,
            self.current_obs_pos,
            self.current_obs_radii,
            self.current_obs_vel
        )
        msg.data = tau.tolist()
        self.torque_pub.publish(msg)

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
