import math

from geometry_msgs.msg import Pose
from tf_transformations import euler_from_quaternion


def translate(
    value: float, leftMin: float, leftMax: float, rightMin: float, rightMax: float
) -> float:
    """Proportionally projects given value from one interval onto antoher.

    Args:
        value (float): Value to be projected.
        leftMin (float): Minimal value from the source interval.
        leftMax (float): Maximal value from the source interval.
        rightMin (float): Minimal value from the target interval.
        rightMax (float): Maximal value from the target interbal.

    Returns:
        float: Proportionally projected value in the target interval.
    """
    value = min(max(value, leftMin), leftMax)

    # Figure out how 'wide' each interval is.
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Get the proportional placement of the value in the source interval.
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Cast the proportion on the target interval.
    return rightMin + (valueScaled * rightSpan)


def pose_delta(pose1: Pose, pose2: Pose) -> tuple[float, float, float]:
    """Calculates movement in linear x,y axes and yaw rotation between two odometry poses.

    Args:
        pose1 (Pose): Old pose.
        pose2 (Pose): Current pose.

    Returns:
        tuple[float, float, float]: Movement in x axis, y axis, and yaw rotation.
    """
    dx = pose2.position.x - pose1.position.x
    dy = pose2.position.y - pose1.position.y

    q1 = [
        pose1.orientation.x,
        pose1.orientation.y,
        pose1.orientation.z,
        pose1.orientation.w,
    ]
    q2 = [
        pose2.orientation.x,
        pose2.orientation.y,
        pose2.orientation.z,
        pose2.orientation.w,
    ]

    _, _, yaw1 = euler_from_quaternion(q1)
    _, _, yaw2 = euler_from_quaternion(q2)
    dyaw = yaw2 - yaw1
    dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))

    return dx, dy, dyaw
