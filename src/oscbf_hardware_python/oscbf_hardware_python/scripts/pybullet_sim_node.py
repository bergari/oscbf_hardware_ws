#!/usr/bin/env python3
"""
Test the OSCBF ROS2 communication with pybullet

This will mimic the franka c++ communication node which listens to joint torques
and publishes joint states
"""

import numpy as np
import pybullet
import csv
import os
from pybullet_utils.bullet_client import BulletClient

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

from oscbf_hardware_python.utils.rotations_and_transforms import xyzw_to_rotation_numpy
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.core.controllers import PoseTaskTorqueController
from oscbf.core.manipulation_env import FrankaTorqueControlEnv
from oscbf_hardware_python.utils.visualization import visualize_3D_box


def create_clipped_reference_limit_visual(
    client,
    radius=0.7,
    lower=None,
    upper=None,
    rgba=(0, 1, 0, 0.4),
    resolution=96,
):
    """Visualize the spherical reference region clipped by the target sampling box."""
    lower = np.array([0.25, -0.7, 0.0]) if lower is None else np.asarray(lower)
    upper = np.array([0.7, 0.7, 0.7]) if upper is None else np.asarray(upper)
    x_min = max(float(lower[0]), -radius)
    x_max = min(float(upper[0]), radius)
    if x_min >= x_max:
        raise ValueError("Reference limit x bounds do not intersect the sphere.")

    theta_min = np.arccos(x_max / radius)
    theta_max = np.arccos(x_min / radius)
    theta_values = np.linspace(theta_min, theta_max, resolution)
    phi_values = np.linspace(0.0, np.pi, resolution)

    vertices = []
    indices = []
    vertex_cache = {}

    def add_vertex(vertex):
        vertex = [float(value) for value in vertex]
        key = tuple(round(value, 10) for value in vertex)
        if key in vertex_cache:
            return vertex_cache[key]
        vertex_cache[key] = len(vertices)
        vertices.append(vertex)
        return len(vertices) - 1

    def add_triangles(triangle_indices):
        indices.extend(int(index) for index in triangle_indices)

    def add_parametric_surface(u_values, v_values, point_fn, reverse=False):
        index_grid = np.empty((len(u_values), len(v_values)), dtype=int)
        for i, u in enumerate(u_values):
            for j, v in enumerate(v_values):
                index_grid[i, j] = add_vertex(point_fn(u, v))

        for i in range(len(u_values) - 1):
            for j in range(len(v_values) - 1):
                quad = [
                    index_grid[i, j],
                    index_grid[i + 1, j],
                    index_grid[i + 1, j + 1],
                    index_grid[i, j + 1],
                ]
                if reverse:
                    add_triangles(
                        [quad[0], quad[2], quad[1], quad[0], quad[3], quad[2]]
                    )
                else:
                    add_triangles(
                        [quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]]
                    )

    # Curved spherical shell. The phi endpoints are exactly the z=0 boundary.
    add_parametric_surface(
        theta_values,
        phi_values,
        lambda theta, phi: [
            radius * np.cos(theta),
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
        ],
    )

    # Flat x-min cut plane, stitched to the spherical shell at rho=cap_radius.
    cap_radius = np.sqrt(max(radius * radius - x_min * x_min, 0.0))
    rho_values = np.linspace(0.0, cap_radius, resolution)
    add_parametric_surface(
        rho_values,
        phi_values,
        lambda rho, phi: [x_min, rho * np.cos(phi), rho * np.sin(phi)],
        reverse=True,
    )

    # Flat z=0 cut plane. Build positive/negative y halves from the same
    # theta samples as the spherical shell, so the round-to-flat edge is shared.
    radial_fracs = np.linspace(0.0, 1.0, resolution)

    def bottom_point(frac, theta, y_sign):
        rho_min = x_min / max(np.cos(theta), 1e-9)
        rho = rho_min + frac * (radius - rho_min)
        return [rho * np.cos(theta), y_sign * rho * np.sin(theta), 0.0]

    add_parametric_surface(
        radial_fracs,
        theta_values,
        lambda frac, theta: bottom_point(frac, theta, 1.0),
        reverse=True,
    )
    add_parametric_surface(
        radial_fracs,
        theta_values,
        lambda frac, theta: bottom_point(frac, theta, -1.0),
    )

    visual_id = client.createVisualShape(
        pybullet.GEOM_MESH,
        vertices=vertices,
        indices=indices,
        rgbaColor=rgba,
    )
    return client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_id,
        basePosition=[0, 0, 0],
    )


class PybulletNode(Node):
    def __init__(self):
        super().__init__("pybullet_node")

        self.freq = 1000
        self.env = FrankaTorqueControlEnv(timestep=1 / self.freq, target_pos=(0.5, 0, 0.2), load_floor=True)
        self.obstacle_ids = {}

        # Visualize workspace limits
        start_point_min = np.array([0.1, -0.8, 0])
        start_point_max = np.array([0.85, 0.8, 1.0])
        visualize_3D_box(
            box=(start_point_min, start_point_max), rgba=(1, 0, 0, 0.2), client=self.env.client
        )

        # Visualize reachable reference targets from traj_node.generate_safe_waypoint().
        create_clipped_reference_limit_visual(client=self.env.client)

        self.csv_filename = "torque_data.csv"
        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        header = ["timestamp"] + [f"joint_{i}_torque" for i in range(self.env.num_joints)]
        self.csv_writer.writerow(header)
        self.counter = 0

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.torque_sub = self.create_subscription(
            Float64MultiArray,
            "franka/torque_command",
            self.torque_callback,
            qos_profile,
        )

        self.obstacle_sub = self.create_subscription(
            Float32MultiArray,
            "franka/obstacle_data",
            self.obstacle_callback,
            qos_profile,
        )
        self._last_z = None

        self.ee_state_sub = self.create_subscription(
            EEState, "ee_state", self.ee_state_callback, qos_profile
        )

        self.last_received_target_state = None

        self.joint_state_pub = self.create_publisher(
            JointState, "franka/joint_states", qos_profile
        )

        self.timer = self.create_timer(1 / self.freq, self.publish_joint_states)

    def torque_callback(self, msg):
        if self._last_z is None:
            self.get_logger().warn("No joint state received yet.")
            return
        # Convert the Float64MultiArray message to a numpy array
        joint_torques = np.array(msg.data)

        # Need to add back gravity compensation
        zero_velocities = [0.0] * self.env.num_joints
        zero_accelerations = [0.0] * self.env.num_joints
        gravity_torques = self.env.client.calculateInverseDynamics(
            self.env.robot,
            self._last_z[: self.env.num_joints].tolist(),
            zero_velocities,
            zero_accelerations,
        )
        joint_torques += gravity_torques

        # Apply the joint torques to the robot in the simulation
        self.env.apply_control(joint_torques)
        #print("applied torques")

        # Log the torques to a CSV file
        self.counter += 1
        if self.counter % 100 == 0: # Log every 100th message (10Hz) to avoid excessive file writing
            timestamp = self.get_clock().now().nanoseconds / 1e9
            row = [timestamp] + joint_torques.tolist()
            self.csv_writer.writerow(row)
        # Step sim
        self.env.step()

    def obstacle_callback(self, msg):
        data = np.array(msg.data, dtype=float)
        if len(data) == 5:
            obs_id_idx = int(data[0])
            start = data[1:4]
            end = start
            radius = data[4]
        elif len(data) == 8:
            obs_id_idx = int(data[0])
            start = data[1:4]
            end = data[4:7]
            radius = data[7]
        else:
            return

        if radius <= 0.0:
            self._remove_obstacle_visual(obs_id_idx)
            return

        pos, orientation, shape_key = self._obstacle_visual_pose(start, end, radius)
        obstacle_record = self.obstacle_ids.get(obs_id_idx)

        if obstacle_record is not None and obstacle_record["shape_key"] == shape_key:
            self.env.client.resetBasePositionAndOrientation(
                obstacle_record["body_id"], pos, orientation
            )
            return

        self._remove_obstacle_visual(obs_id_idx)
        visual_id = self._create_obstacle_visual_shape(shape_key)
        body_id = self.env.client.createMultiBody(
            baseVisualShapeIndex=visual_id,
            basePosition=pos,
            baseOrientation=orientation,
        )
        self.obstacle_ids[obs_id_idx] = {
            "body_id": body_id,
            "shape_key": shape_key,
        }

    def _remove_obstacle_visual(self, obs_id_idx):
        obstacle_record = self.obstacle_ids.pop(obs_id_idx, None)
        if obstacle_record is not None:
            self.env.client.removeBody(obstacle_record["body_id"])

    def _obstacle_visual_pose(self, start, end, radius):
        segment = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length = float(np.linalg.norm(segment))

        if length < 1e-6:
            shape_key = ("sphere", round(float(radius), 5))
            return start.tolist(), [0, 0, 0, 1], shape_key

        direction = segment / length
        z_axis = np.array([0.0, 0.0, 1.0])
        rotation_axis = np.cross(z_axis, direction)
        axis_norm = np.linalg.norm(rotation_axis)
        dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))

        if axis_norm < 1e-8:
            if dot > 0:
                orientation = [0, 0, 0, 1]
            else:
                orientation = pybullet.getQuaternionFromAxisAngle([1, 0, 0], np.pi)
        else:
            rotation_axis = (rotation_axis / axis_norm).tolist()
            angle = float(np.arccos(dot))
            orientation = pybullet.getQuaternionFromAxisAngle(rotation_axis, angle)

        pos = ((np.asarray(start, dtype=float) + np.asarray(end, dtype=float)) / 2.0).tolist()
        shape_key = ("capsule", round(float(radius), 5), round(length, 5))
        return pos, orientation, shape_key

    def _create_obstacle_visual_shape(self, shape_key):
        if shape_key[0] == "sphere":
            return self.env.client.createVisualShape(
                pybullet.GEOM_SPHERE,
                radius=shape_key[1],
                rgbaColor=[1, 0, 0, 1],
            )

        return self.env.client.createVisualShape(
            pybullet.GEOM_CAPSULE,
            radius=shape_key[1],
            length=shape_key[2],
            rgbaColor=[1, 0, 0, 1],
        )

    def ee_state_callback(self, msg: EEState):
        pos: Point = msg.pose.position
        self.last_received_target_state = (pos.x, pos.y, pos.z)
        self.env.target_pos = self.last_received_target_state
        quat: Quaternion = msg.pose.orientation
        vel: Vector3 = msg.twist.linear
        omega: Vector3 = msg.twist.angular

        # Update the position and orientation of the target object in PyBullet
        self.env.client.resetBasePositionAndOrientation(
            self.env.target,
            [pos.x, pos.y, pos.z],
            [quat.x, quat.y, quat.z, quat.w]
        )
        self.env.client.resetBaseVelocity(
            self.env.target,
            [vel.x, vel.y, vel.z],
            [omega.x, omega.y, omega.z]
        )

    def publish_joint_states(self):
        z = self.env.get_joint_state()
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        # names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        # joint_state_msg.name = names
        joint_state_msg.position = z[: self.env.num_joints].tolist()
        joint_state_msg.velocity = z[self.env.num_joints :].tolist()
        self.joint_state_pub.publish(joint_state_msg)

        self._last_z = z


def main(args=None):
    rclpy.init(args=args)
    node = PybulletNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
