#!/usr/bin/env python3
"""
Test the OSCBF ROS2 communication with pybullet

This will mimic the franka c++ communication node which listens to joint torques
and publishes joint states
"""

import numpy as np
import pybullet
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
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from oscbf_msgs.msg import EEState

from oscbf_hardware_python.utils.rotations_and_transforms import xyzw_to_rotation_numpy
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.core.controllers import PoseTaskTorqueController
from oscbf.core.manipulation_env import FrankaTorqueControlEnv


class PybulletNode(Node):
    def __init__(self):
        super().__init__("pybullet_node")

        self.freq = 1000
        self.env = FrankaTorqueControlEnv(timestep=1 / self.freq)

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

        self._last_z = None

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

        # Step sim
        self.env.step()

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
