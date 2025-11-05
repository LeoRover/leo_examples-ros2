import math
from typing import Optional

from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.time import Time
from rclpy.timer import Timer

from geometry_msgs.msg import Twist, Point, Pose
from nav_msgs.msg import Odometry

from aruco_opencv_msgs.msg import ArucoDetection, MarkerPose

from leo_example_follow_aruco_marker.follower_parameters import follower_parameters
from leo_follow_aruco.utils import pose_delta, translate


class ArucoMarkerFollower(Node):
    """A ROS node following specified Aruco Marker."""

    running: bool = False
    marker_angle: float = 0.0
    marker_distance: float = 0.0
    last_marker_position: Optional[Point] = None
    twist_cmd: Twist = Twist()
    last_odom_ts: Optional[Time] = None
    odom_marker_pose: Pose = Pose()
    odom_current_pose: Pose = Pose()

    def __init__(self) -> None:
        super().__init__("aruco_follower")

        self.param_listener = follower_parameters.ParamListener(self)
        self.params = self.param_listener.get_params()
        self.param_listener.set_user_callback(self.reconfigure_callback)

        self.last_marker_ts: Time = self.get_clock().now() - Duration(
            seconds=self.params.marker_timeout + 1.0
        )

        self.cmd_vel_pub: Publisher = self.create_publisher(Twist, "cmd_vel", 1)

        self.marker_pose_sub: Subscription = self.create_subscription(
            ArucoDetection,
            "aruco_detections",
            self.marker_callback,
            1,
        )

        self.merged_odom_sub: Subscription = self.create_subscription(
            Odometry, "merged_odom", self.odom_callback, 1
        )

        self.run_timer: Timer = self.create_timer(0.1, self.run)

    def reconfigure_callback(self, parameters: follower_parameters.Params) -> None:
        """Callback for the parameters change.
        Updates the values of parameters.

        Args:
            parameters (follower_parameters.Params): The struct with current parameters.
        """
        self.params = parameters

    def run(self) -> None:
        """Function controlling the rover. Sends velocity command based on
        marker position and odometry calculations.
        """
        if not self.params.follow_enabled:
            return

        self.update_vel_cmd()
        self.cmd_vel_pub.publish(self.twist_cmd)

    def update_vel_cmd(self) -> None:
        """Updates current velocity command based on recent marker detections and odometry data."""
        # Check for a timeout
        if (
            self.last_marker_ts + Duration(seconds=self.params.marker_timeout)
            < self.get_clock().now()
        ):
            self.twist_cmd.linear.x = 0.0
            self.twist_cmd.angular.z = 0.0
            return

        # Get the absolute angle to the marker
        angle = math.fabs(self.marker_angle)

        # Get the direction multiplier
        dir = -1.0 if self.marker_angle < 0.0 else 1.0

        # Calculate angular command
        if angle < self.params.angle_min:
            ang_cmd = 0.0
        else:
            ang_cmd = translate(
                angle,
                self.params.angle_min,
                self.params.angle_max,
                self.params.min_ang_vel,
                self.params.max_ang_vel,
            )

        # Calculate linear command
        if self.marker_distance >= self.params.distance_min_forward:
            lin_cmd = translate(
                self.marker_distance,
                self.params.distance_min_forward,
                self.params.distance_max_forward,
                self.params.min_lin_vel_forward,
                self.params.max_lin_vel_forward,
            )
        elif self.marker_distance <= self.params.distance_max_reverse:
            lin_cmd = -translate(
                self.marker_distance,
                self.params.distance_min_reverse,
                self.params.distance_max_reverse,
                self.params.max_lin_vel_reverse,
                self.params.min_lin_vel_reverse,
            )
        else:
            lin_cmd = 0.0

        self.twist_cmd.angular.z = dir * ang_cmd
        self.twist_cmd.linear.x = lin_cmd

    def update_marker_angle_and_distance(self) -> None:
        """Updates current distance and angle from the rover to the marker."""
        if self.last_marker_position:
            current_pos_x, current_pos_y, current_yaw = pose_delta(
                self.odom_marker_pose, self.odom_current_pose
            )

            position_x: float = self.last_marker_position.x - current_pos_x
            position_y: float = self.last_marker_position.y - current_pos_y

            self.marker_angle = math.atan(position_y / position_x) - current_yaw
            self.marker_distance = math.sqrt(position_x**2 + position_y**2)

    def marker_callback(self, msg: ArucoDetection) -> None:
        """Callback for marker detection subscriber. Searches for the right marker
        and updates the calculation needed variables.

        Args:
            msg (ArucoDetection): Message received on the topic.
        """
        marker: MarkerPose
        for marker in msg.markers:
            if marker.marker_id != self.params.follow_id:
                continue
            if Time.from_msg(msg.header.stamp) < self.last_marker_ts:
                self.get_logger().warning(
                    "Got marker position with an older timestamp.",
                    throttle_duration_sec=3.0,
                )
                continue

            self.last_marker_ts = Time.from_msg(msg.header.stamp)
            self.last_marker_position = marker.pose.position
            self.odom_marker_pose = self.odom_current_pose

            self.update_marker_angle_and_distance()
            return

    def odom_callback(self, msg: Odometry) -> None:
        """Callback for odometry subscriber. Updates stored poses needed for
        distance and angle calculations.

        Args:
            msg (Odometry): Message received on the topic.
        """
        if self.last_odom_ts:
            start_ts: Time = max(self.last_odom_ts, self.last_marker_ts)

            end_ts: Time = Time.from_msg(msg.header.stamp)
            if end_ts < start_ts:
                self.get_logger().warning(
                    "Reveived odometry has timestamp older than last marker position."
                )

            self.odom_current_pose = msg.pose

            self.update_marker_angle_and_distance()

        self.last_odom_ts = Time.from_msg(msg.header.stamp)

    def cleanup(self) -> None:
        """Cleans ROS entities."""
        self.params.follow_enabled = False
        self.destroy_timer(self.run_timer)
        self.destroy_subscription(self.marker_pose_sub)
        self.destroy_subscription(self.merged_odom_sub)
        self.destroy_publisher(self.cmd_vel_pub)
