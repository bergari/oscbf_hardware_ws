"""ROS2 node for publishing desired end-effector states from an example trajectory"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from oscbf_msgs.msg import EEState
from geometry_msgs.msg import Point, Quaternion, Vector3

from oscbf_hardware_python.utils.trajectory import SinusoidalTaskTrajectory


class EETrajNode(Node):
    def __init__(self):
        super().__init__("ee_traj_node")

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Based on the traj from the dynamic motion demo
        # TODO: make these values inputs
        self.traj = SinusoidalTaskTrajectory(
            init_pos=(0.45, 0, 0.45),
            init_rot=np.array(
                [
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ]
            ),
            amplitude=(0.2, 0, 0),
            angular_freq=(1, 0, 0),
            phase=(0, 0, 0),
        )

        self.freq = 100
        self.timer = self.create_timer(1 / self.freq, self.publish_ee_state)

        self.ee_state_pub = self.create_publisher(EEState, "ee_state", qos_profile)

    def publish_ee_state(self):
        """Publish the desired end-effector state"""
        time = self.get_clock().now()
        secs, nanosecs = time.seconds_nanoseconds()
        t = secs + nanosecs / 1e9

        pos = self.traj.position(t)
        rot = self.traj.rotation(t)
        vel = self.traj.velocity(t)
        omega = self.traj.omega(t)
        xyzw = rmat_to_quat(rot)

        msg = EEState()
        msg.header.stamp = time.to_msg()
        msg.pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
        msg.pose.orientation = Quaternion(x=xyzw[0], y=xyzw[1], z=xyzw[2], w=xyzw[3])
        msg.twist.linear = Vector3(x=vel[0], y=vel[1], z=vel[2])
        msg.twist.angular = Vector3(x=omega[0], y=omega[1], z=omega[2])

        self.ee_state_pub.publish(msg)


def rmat_to_quat(rmat: np.ndarray) -> np.ndarray:
    """Converts a rotation matrix into XYZW quaternions

    (computer graphics solution by Shoemake 1994, same as NASA's code)

    Args:
        rmat (np.ndarray): (3,3) rotation matrix

    Returns:
        np.ndarray: XYZW quaternions
    """

    tr = rmat[0, 0] + rmat[1, 1] + rmat[2, 2]
    if tr >= 0:
        s4 = 2.0 * np.sqrt(tr + 1.0)
        x = (rmat[2, 1] - rmat[1, 2]) / s4
        y = (rmat[0, 2] - rmat[2, 0]) / s4
        z = (rmat[1, 0] - rmat[0, 1]) / s4
        w = s4 / 4.0
    elif rmat[0, 0] > rmat[1, 1] and rmat[0, 0] > rmat[2, 2]:
        s4 = 2.0 * np.sqrt(1.0 + rmat[0, 0] - rmat[1, 1] - rmat[2, 2])
        x = s4 / 4.0
        y = (rmat[0, 1] + rmat[1, 0]) / s4
        z = (rmat[2, 0] + rmat[0, 2]) / s4
        w = (rmat[2, 1] - rmat[1, 2]) / s4
    elif rmat[1, 1] > rmat[2, 2]:
        s4 = 2.0 * np.sqrt(1.0 + rmat[1, 1] - rmat[0, 0] - rmat[2, 2])
        x = (rmat[0, 1] + rmat[1, 0]) / s4
        y = s4 / 4.0
        z = (rmat[1, 2] + rmat[2, 1]) / s4
        w = (rmat[0, 2] - rmat[2, 0]) / s4
    else:
        s4 = 2.0 * np.sqrt(1.0 + rmat[2, 2] - rmat[0, 0] - rmat[1, 1])
        x = (rmat[2, 0] + rmat[0, 2]) / s4
        y = (rmat[1, 2] + rmat[2, 1]) / s4
        z = s4 / 4.0
        w = (rmat[1, 0] - rmat[0, 1]) / s4

    return np.array([x, y, z, w])


def main(args=None):
    rclpy.init(args=args)
    node = EETrajNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
