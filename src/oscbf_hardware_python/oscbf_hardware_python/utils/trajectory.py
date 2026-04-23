"""Trajectories"""

from abc import ABC, abstractmethod
import numpy as np


class TaskTrajectory(ABC):
    """Base Operational Space Trajectory implementation

    Inherited classes must implement the following methods:
    - position(t: float) -> np.ndarray
    - velocity(t: float) -> np.ndarray
    - acceleration(t: float) -> np.ndarray
    - rotation(t: float) -> np.ndarray
    - omega(t: float) -> np.ndarray
    - alpha(t: float) -> np.ndarray
    """

    def __init__(self):
        pass

    @abstractmethod
    def position(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def velocity(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def acceleration(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def rotation(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def omega(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def alpha(self, t: float) -> np.ndarray:
        pass


class JointTrajectory(ABC):
    """Base Joint Space Trajectory implementation

    Inherited classes must implement the following methods:
    - joint_positions(t: float) -> np.ndarray
    - joint_velocities(t: float) -> np.ndarray
    - joint_accelerations(t: float) -> np.ndarray
    """

    def __init__(self):
        pass

    @abstractmethod
    def joint_positions(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def joint_velocities(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def joint_accelerations(self, t: float) -> np.ndarray:
        pass


class SinusoidalTaskTrajectory(TaskTrajectory):
    """An example sinusoidal task-space position trajectory for the robot to follow

    Args:
        init_pos (np.ndarray): Initial position of the end-effector, shape (3,)
        init_rot (np.ndarray): Initial rotation of the end-effector, shape (3, 3)
        amplitude (np.ndarray): X,Y,Z amplitudes of the sinusoid, shape (3,)
        angular_freq (np.ndarray): X,Y,Z angular frequencies of the sinusoid, shape (3,)
        phase (np.ndarray): X,Y,Z phase offsets of the sinusoid, shape (3,)
    """

    def __init__(
        self,
        init_pos: np.ndarray,
        init_rot: np.ndarray,
        amplitude: np.ndarray,
        angular_freq: np.ndarray,
        phase: np.ndarray,
    ):
        self.init_pos = np.asarray(init_pos)
        self.init_rot = np.asarray(init_rot)
        self.amplitude = np.asarray(amplitude)
        self.angular_freq = np.asarray(angular_freq)
        self.phase = np.asarray(phase)

        assert self.init_pos.shape == (3,)
        assert self.init_rot.shape == (3, 3)
        assert self.amplitude.shape == (3,)
        assert self.angular_freq.shape == (3,)
        assert self.phase.shape == (3,)

    # Simple sinusoidal positional trajectory

    def position(self, t: float) -> np.ndarray:
        return self.init_pos + self.amplitude * np.sin(
            self.angular_freq * t + self.phase
        )

    def velocity(self, t: float) -> np.ndarray:
        return (
            self.amplitude
            * self.angular_freq
            * np.cos(self.angular_freq * t + self.phase)
        )

    def acceleration(self, t: float) -> np.ndarray:
        return (
            -self.amplitude
            * self.angular_freq**2
            * np.sin(self.angular_freq * t + self.phase)
        )

    # Maintain a fixed orientation

    def rotation(self, t: float) -> np.ndarray:
        return self.init_rot

    def omega(self, t: float) -> np.ndarray:
        return np.zeros(3)

    def alpha(self, t: float) -> np.ndarray:
        return np.zeros(3)

class LinearTaskTrajectory(TaskTrajectory):
    """Defines a linear trajectory from a start to an end pose."""

    def __init__(self, start_pos, end_pos, duration, init_rot=None):
        self.start_pos = np.asarray(start_pos)
        self.end_pos = np.asarray(end_pos)
        self.duration = duration
        if init_rot is None:
            self.init_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            self.init_rot = np.asarray(init_rot)

    def position(self, t: float) -> np.ndarray:
        """Linearly interpolates position."""
        s = min(t / self.duration, 1.0)
        return self.start_pos + s * (self.end_pos - self.start_pos)

    def rotation(self, t: float) -> np.ndarray:
        """Keeps rotation constant."""
        return self.init_rot

    def velocity(self, t: float) -> np.ndarray:
        """Returns constant velocity during motion."""
        if t < self.duration:
            return (self.end_pos - self.start_pos) / self.duration
        return np.zeros(3)

    def omega(self, t: float) -> np.ndarray:
        """No angular velocity."""
        return np.zeros(3)

    def acceleration(self, t: float) -> np.ndarray:
        """No linear acceleration."""
        return np.zeros(3)

    def alpha(self, t: float) -> np.ndarray:
        """No angular acceleration."""
        return np.zeros(3)
    
class SmoothLinearTrajectory:
    """Defines a perfectly straight geometric path with a smooth (quintic) velocity/acceleration ramp."""

    def __init__(self, start_pos, end_pos, duration, init_rot=None):
        self.start_pos = np.asarray(start_pos)
        self.end_pos = np.asarray(end_pos)
        
        # Prevent division by zero
        self.duration = max(duration, 0.001)
        
        self.delta_pos = self.end_pos - self.start_pos

        if init_rot is None:
            self.init_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            self.init_rot = np.asarray(init_rot)

    def _get_s_and_derivatives(self, t: float):
        """Calculates a smooth quintic scaling factor s(t) from 0 to 1, and its derivatives."""
        # Clamp time between 0 and duration
        t = min(max(t, 0.0), self.duration)
        
        # Normalized time tau goes from 0.0 to 1.0
        tau = t / self.duration 
        
        # Quintic polynomial for position scaling: 10tau^3 - 15tau^4 + 6tau^5
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        
        # Derivative of s with respect to time (chain rule applied: d(tau)/dt = 1/duration)
        s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.duration
        
        # Second derivative (acceleration scaling)
        s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / (self.duration**2)
        
        return s, s_dot, s_ddot

    def position(self, t: float) -> np.ndarray:
        s, _, _ = self._get_s_and_derivatives(t)
        return self.start_pos + s * self.delta_pos

    def rotation(self, t: float) -> np.ndarray:
        return self.init_rot

    def velocity(self, t: float) -> np.ndarray:
        _, s_dot, _ = self._get_s_and_derivatives(t)
        # Multiplies the smooth scalar velocity profile by the 3D direction vector
        return s_dot * self.delta_pos

    def acceleration(self, t: float) -> np.ndarray:
        _, _, s_ddot = self._get_s_and_derivatives(t)
        return s_ddot * self.delta_pos

    def omega(self, t: float) -> np.ndarray:
        return np.zeros(3)

    def alpha(self, t: float) -> np.ndarray:
        return np.zeros(3)
