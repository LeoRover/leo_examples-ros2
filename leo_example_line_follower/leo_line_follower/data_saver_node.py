import datetime
import os
from pathlib import Path
import shutil
import time

from rclpy.node import Node
from rclpy.subscription import Subscription

from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage

import cv2
import cv_bridge

from numpy.typing import NDArray


class DataSaverNode(Node):
    """A ROS node saving data needed for training line follower NN model."""

    ready: bool = False
    end_time: int = 0
    counter: int = 0
    running: bool = False

    def __init__(
        self, duration: float, camera_topic: str, vel_topic: str, output_dir: str
    ) -> None:
        super().__init__("data_saver")
        self.duration = duration

        self.bridge = cv_bridge.CvBridge()

        self.get_logger().info(
            "Making directory for saved images (if it doesn't exist)."
        )

        if output_dir[0] != "/":
            self.path = os.path.join(Path.home(), output_dir)
        else:
            self.path = output_dir

        Path(self.path).mkdir(parents=True, exist_ok=True)

        date = datetime.datetime.now()
        self.img_base_name = "%s%s%s%s%s-img" % (
            date.day,
            date.month,
            date.year,
            date.hour,
            date.minute,
        )

        self.get_logger().info("Opening label file (creating if doesn't exist).")
        self.label_file = open(os.path.join(self.path, "labels.txt"), "a+")

        self.video_sub: Subscription = self.create_subscription(
            CompressedImage, camera_topic, self.video_callback, 1
        )
        self.vel_sub: Subscription = self.create_subscription(
            Twist, vel_topic, self.velocity_callback, 1
        )
        self.running = True

    def video_callback(self, data: CompressedImage) -> None:
        """Saves incoming video messages into specified data directory,
        and labels the file with current velocity command.

        Args:
            data (CompressedImage): Incoming video frame.
        """
        if not self.ready:
            self.get_logger().info("Wating for twist msg.")
            return

        if self.end_time <= time.monotonic():
            self.get_logger().info("Saved enough data. Finishing node.")
            self.get_logger().info(f"Data saved in '{self.path}' directory.")
            self.label_file.close()
            self.running = False

        if self.current_label != (0.0, 0.0):
            cv_img: NDArray = self.bridge.compressed_imgmsg_to_cv2(
                data, desired_encoding="bgr8"
            )
            img_name: str = self.img_base_name + str(self.counter) + ".jpg"

            cv2.imwrite(
                filename=os.path.join(self.path, img_name),
                img=cv_img,
                params=[cv2.IMWRITE_JPEG_QUALITY, 100],
            )
            self.label_file.write("{}:{}\n".format(img_name, self.current_label))
            self.counter += 1

    def velocity_callback(self, data: Twist) -> None:
        """Sets the end time of data recording and updates
        current velocity command.

        Args:
            data (Twist): Current velocity message.
        """
        if not self.ready:
            self.ready = True
            self.end_time = time.monotonic() + self.duration
        self.current_label = (round(data.linear.x, 2), round(data.angular.z, 2))

    def cleanup(self) -> None:
        """Cleans ROS entities, closes and removes created files if node stopped early."""
        if not self.label_file.closed:
            self.label_file.close()

        self.destroy_subscription(self.vel_sub)
        self.destroy_subscription(self.video_sub)

        if self.running:
            self.get_logger().warning(
                "Node stopped during data recording - removing all created files."
            )
            shutil.rmtree(self.path, ignore_errors=True)
