#!/usr/bin/env python3

import math
from typing import Optional

from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile
from rclpy.subscription import Subscription
from rclpy.time import Time
from rclpy.timer import Timer

from geometry_msgs.msg import Twist, Point, Pose
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

from aruco_opencv_msgs.msg import ArucoDetection, MarkerPose

from leo_example_follow_aruco_marker.follower_parameters import follower_parameters


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


class ArucoMarkerFollower(Node):
    """A ROS node following specified Aruco Marker."""

    _running: bool = False
    _marker_angle: float = 0.0
    _marker_distance: float = 0.0
    _last_marker_position: Optional[Point] = None
    _twist_cmd: Twist = Twist()
    _last_odom_ts: Optional[Time] = None
    _odom_marker_pose: Pose = Pose()
    _odom_current_pose: Pose = Pose()

    def __init__(self) -> None:
        super().__init__("aruco_follower")

        self._param_listener = follower_parameters.ParamListener(self)
        self._params = self._param_listener.get_params()

        self._param_timer: Timer = self.create_timer(1.0, self.param_timer_callback)

        self._last_marker_ts: Time = self.get_clock().now() - Duration(
            seconds=self._params.marker_timeout + 1.0
        )

        self._cmd_vel_pub: Publisher = self.create_publisher(
            Twist, "cmd_vel", QoSProfile(depth=1)
        )

        self._marker_pose_sub: Subscription = self.create_subscription(
            ArucoDetection,
            "aruco_detections",
            self.marker_callback,
            QoSProfile(depth=1),
        )

        self._merged_odom_sub: Subscription = self.create_subscription(
            Odometry, "merged_odom", self.odom_callback, QoSProfile(depth=1)
        )

        self._run_timer: Timer = self.create_timer(0.1, self.run)

    def param_timer_callback(self) -> None:
        """Callback for the parameters timer - updates the values of parameters."""
        if self._param_listener.is_old(self._params):
            self._param_listener.refresh_dynamic_parameters()
            self._params = self._param_listener.get_params()

    def run(self) -> None:
        """Function controlling the rover. Sends velocity command based on
        marker position and odometry calculations.
        """
        if not self._params.follow_enabled:
            return

        self.update_vel_cmd()
        self._cmd_vel_pub.publish(self._twist_cmd)

    def update_vel_cmd(self) -> None:
        """Updates current velocity command based on recent marker detections and odometry data."""
        # Check for a timeout
        if (
            self._last_marker_ts + Duration(seconds=self._params.marker_timeout)
            < self.get_clock().now()
        ):
            self._twist_cmd.linear.x = 0.0
            self._twist_cmd.angular.z = 0.0
            return

        # Get the absolute angle to the marker
        angle = math.fabs(self._marker_angle)

        # Get the direction multiplier
        dir = -1.0 if self._marker_angle < 0.0 else 1.0

        # Calculate angular command
        if angle < self._params.angle_min:
            ang_cmd = 0.0
        else:
            ang_cmd = translate(
                angle,
                self._params.angle_min,
                self._params.angle_max,
                self._params.min_ang_vel,
                self._params.max_ang_vel,
            )

        # Calculate linear command
        if self._marker_distance >= self._params.distance_min_forward:
            lin_cmd = translate(
                self._marker_distance,
                self._params.distance_min_forward,
                self._params.distance_max_forward,
                self._params.min_lin_vel_forward,
                self._params.max_lin_vel_forward,
            )
        elif self._marker_distance <= self._params.distance_max_reverse:
            lin_cmd = -translate(
                self._marker_distance,
                self._params.distance_min_reverse,
                self._params.distance_max_reverse,
                self._params.max_lin_vel_reverse,
                self._params.min_lin_vel_reverse,
            )
        else:
            lin_cmd = 0.0

        self._twist_cmd.angular.z = dir * ang_cmd
        self._twist_cmd.linear.x = lin_cmd

    def update_marker_angle_and_distance(self) -> None:
        """Updates current distance and angle from the rover to the marker."""
        if self._last_marker_position:
            current_pos_x, current_pos_y, current_yaw = pose_delta(
                self._odom_marker_pose, self._odom_current_pose
            )

            position_x: float = self._last_marker_position.x - current_pos_x
            position_y: float = self._last_marker_position.y - current_pos_y

            self._marker_angle = math.atan(position_y / position_x) - current_yaw
            self._marker_distance = math.sqrt(position_x**2 + position_y**2)

    def marker_callback(self, msg: ArucoDetection) -> None:
        """Callback for marker detection subscriber. Searches for the right marker
        and updates the calculation needed variables.

        Args:
            msg (ArucoDetection): Message received on the topic.
        """
        marker: MarkerPose
        for marker in msg.markers:
            if marker.marker_id != self._params.follow_id:
                continue
            if Time.from_msg(msg.header.stamp) < self._last_marker_ts:
                self.get_logger().warning(
                    "Got marker position with an older timestamp.",
                    throttle_duration_sec=3.0,
                )
                continue

            self._last_marker_ts = Time.from_msg(msg.header.stamp)
            self._last_marker_position = marker.pose.position
            self._odom_marker_pose = self._odom_current_pose

            self.update_marker_angle_and_distance()
            return

    def odom_callback(self, msg: Odometry) -> None:
        """Callback for odometry subscriber. Updates stored poses needed for
        distance and angle calculations.

        Args:
            msg (Odometry): Message received on the topic.
        """
        if self._last_odom_ts:
            start_ts: Time = max(self._last_odom_ts, self._last_marker_ts)

            end_ts: Time = Time.from_msg(msg.header.stamp)
            if end_ts < start_ts:
                self.get_logger().warning(
                    "Reveived odometry has timestamp older than last marker position."
                )

            self._odom_current_pose = msg.pose

            self.update_marker_angle_and_distance()

        self.last_odom_ts = Time.from_msg(msg.header.stamp)

    def cleanup(self) -> None:
        """Cleans ROS entities."""
        self._params.follow_enabled = False
        self.destroy_timer(self._param_timer)
        self.destroy_timer(self._run_timer)
        self.destroy_subscription(self._marker_pose_sub)
        self.destroy_subscription(self._merged_odom_sub)
        self.destroy_publisher(self._cmd_vel_pub)
