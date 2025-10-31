import datetime
import os
from pathlib import Path
import shutil
import time

from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.subscription import Subscription

from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage

import cv2
import cv_bridge

from numpy.typing import NDArray


class DataSaverNode(Node):
    """A ROS node saving data needed for training line follower NN model."""

    _ready: bool = False
    _end_time: int = 0
    _counter: int = 0
    running: bool = False

    def __init__(
        self, duration: float, camera_topic: str, vel_topic: str, output_dir: str
    ) -> None:
        super().__init__("data_saver")
        self._duration = duration

        self._bridge = cv_bridge.CvBridge()

        self.get_logger().info(
            "Making directory for saved images (if it doesn't exist)."
        )

        if output_dir[0] != "/":
            self._path = os.path.join(Path.home(), output_dir)
        else:
            self._path = output_dir

        Path(self._path).mkdir(parents=True, exist_ok=True)

        date = datetime.datetime.now()
        self._img_base_name = "%s%s%s%s%s-img" % (
            date.day,
            date.month,
            date.year,
            date.hour,
            date.minute,
        )

        self.get_logger().info("Opening label file (creating if doesn't exist).")
        self._label_file = open(os.path.join(self._path, "labels.txt"), "a+")

        self._video_sub: Subscription = self.create_subscription(
            CompressedImage, camera_topic, self.video_callback, QoSProfile(depth=1)
        )
        self._vel_sub: Subscription = self.create_subscription(
            Twist, vel_topic, self.velocity_callback, QoSProfile(depth=1)
        )
        self.running = True

    def video_callback(self, data: CompressedImage) -> None:
        """Saves incoming video messages into specified data directory,
        and labels the file with current velocity command.

        Args:
            data (CompressedImage): Incoming video frame.
        """
        if not self._ready:
            self.get_logger().info("Wating for twist msg.")
            return

        if self._end_time <= time.monotonic():
            self.get_logger().info("Saved enough data. Finishing node.")
            self.get_logger().info(f"Data saved in '{self._path}' directory.")
            self._label_file.close()
            self.running = False

        if self._current_label != (0.0, 0.0):
            cv_img: NDArray = self._bridge.compressed_imgmsg_to_cv2(
                data, desired_encoding="bgr8"
            )
            img_name: str = self._img_base_name + str(self._counter) + ".jpg"

            cv2.imwrite(
                filename=os.path.join(self._path, img_name),
                img=cv_img,
                params=[cv2.IMWRITE_JPEG_QUALITY, 100],
            )
            self._label_file.write("{}:{}\n".format(img_name, self._current_label))
            self._counter += 1

    def velocity_callback(self, data: Twist) -> None:
        """Sets the end time of data recording and updates
        current velocity command.

        Args:
            data (Twist): Current velocity message.
        """
        if not self._ready:
            self._ready = True
            self._end_time = time.monotonic() + self._duration
        self._current_label = (round(data.linear.x, 2), round(data.angular.z, 2))

    def cleanup(self) -> None:
        """Cleans ROS entities, closes and removes created files if node stopped early."""
        if not self._label_file.closed:
            self._label_file.close()

        self.destroy_subscription(self._vel_sub)
        self.destroy_subscription(self._video_sub)

        if self.running:
            self.get_logger().warning(
                "Node stopped during data recording - removing all created files."
            )
            shutil.rmtree(self._path, ignore_errors=True)
