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
from std_msgs.msg import Float32MultiArray

# Import your newly added trajectory class
from oscbf_hardware_python.utils.trajectory import LinearTaskTrajectory, SmoothLinearTrajectory


class EETrajNode(Node):
    def __init__(self):
        super().__init__("ee_traj_node")

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Time tracking
        self.switch_interval = 10.0  # Switch trajectory every 10 seconds
        self.last_switch_time = None 
        
        # Define a list of target waypoints to cycle through
        self.target_waypoints = [
            self.generate_safe_waypoint(),
        ]
        self.current_target_idx = 0
        self.traj = None   
        self.freq = 1000

        self.ros_timer = self.create_timer(1 / self.freq, self.publish_ee_state)
        self.ee_state_pub = self.create_publisher(EEState, "ee_state", qos_profile)
        self.ee_pos_array_pub = self.create_publisher(Float32MultiArray, "ee_target_pos_array", qos_profile)

    def publish_ee_state(self):
        """Publish the desired end-effector state"""
        time = self.get_clock().now()
        secs, nanosecs = time.seconds_nanoseconds()
        t = secs + nanosecs / 1e9

        # Initialize the first trajectory
        if self.last_switch_time is None:
            self.last_switch_time = t
            start_pos = np.array([0.45, 0.0, 0.45])
            next_pos = np.array([0.45, 0.0, 0.45]) # Keep init pose for the first 10 seconds
            dist = np.linalg.norm(next_pos - start_pos)
            
            # self.traj = QuinticTaskTrajectory(
            #     start_pos=start_pos,
            #     end_pos=next_pos,
            #     duration=np.linalg.norm(next_pos - start_pos) / 0.1, # 0.1 m/s speed
            #     start_vel=np.zeros(3) # Start from rest
            # )
            self.traj = SmoothLinearTrajectory(
                start_pos=start_pos,
                end_pos=next_pos,
                duration=dist / 0.1
            )
            self.get_logger().info(f"Initialized trajectory! Moving to: {next_pos}")
            return

        # Calculate the elapsed time since the last switch
        elapsed_t = t - self.last_switch_time

        # Check if 10 seconds have elapsed since the last switch
        if elapsed_t >= self.switch_interval:
            # Capture exact state at the time of switching to ensure continuity
            current_pos = self.traj.position(elapsed_t)
            current_vel = self.traj.velocity(elapsed_t) 

            self.last_switch_time = t
            elapsed_t = 0.0 # Reset elapsed time for the newly created trajectory

            # update random target waypoint every switch
            self.target_waypoints[0] = self.generate_safe_waypoint()
            
            # Update to the next target waypoint in the list
            self.current_target_idx = (self.current_target_idx + 1) % len(self.target_waypoints)
            next_pos = np.array(self.target_waypoints[self.current_target_idx])

            # Start the new trajectory from exactly where we are, conserving velocity
            dist = np.linalg.norm(next_pos - current_pos)
            # self.traj = QuinticTaskTrajectory(
            #     start_pos=current_pos,
            #     end_pos=next_pos,
            #     duration=dist / 0.1,  # Adjust duration based on desired speed (0.1 m/s speed)
            #     start_vel=current_vel # CRITICAL: Pass in the current velocity to prevent jitter
            # )
            self.traj = SmoothLinearTrajectory(
                start_pos=current_pos,
                end_pos=next_pos,
                duration=dist / 0.1  # 0.1 m/s average speed
            )
            self.get_logger().info(f"Switched trajectory! Moving to: {next_pos}")

        pos = self.traj.position(elapsed_t)
        rot = self.traj.rotation(elapsed_t)
        vel = self.traj.velocity(elapsed_t)
        omega = self.traj.omega(elapsed_t)
        xyzw = rmat_to_quat(rot)

        msg = EEState()
        msg.header.stamp = time.to_msg()
        msg.pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
        msg.pose.orientation = Quaternion(x=xyzw[0], y=xyzw[1], z=xyzw[2], w=xyzw[3])
        msg.twist.linear = Vector3(x=vel[0], y=vel[1], z=vel[2])
        msg.twist.angular = Vector3(x=omega[0], y=omega[1], z=omega[2])

        self.ee_state_pub.publish(msg)

        array_msg = Float32MultiArray()
        array_msg.data = [float(pos[0]), float(pos[1]), float(pos[2])] 
        self.ee_pos_array_pub.publish(array_msg)

    def generate_safe_waypoint(self):
            """Generates a random waypoint guaranteed to be safely inside the Franka workspace."""
            max_safe_radius = 0.75  # Keep it comfortably under the 0.855m absolute limit
            
            while True:
                target = np.array([
                    np.random.uniform(0.25, 0.65), 
                    np.random.uniform(-0.3, 0.3), 
                    np.random.uniform(0.2, 0.7)
                ])
                
                # Check if the absolute distance from base is within the safe sphere
                if np.linalg.norm(target) <= max_safe_radius:
                    return target

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