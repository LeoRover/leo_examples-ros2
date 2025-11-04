import copy

from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from sensor_msgs.msg import CompressedImage, Image

from leo_example_line_follower.follower_parameters import follower_parameters

from leo_line_follower.utils import simple_mask, double_range_mask, get_colors_from_mask

import cv2
import cv_bridge

from numpy.typing import NDArray


class ColorMaskNode(Node):
    """ROS node responsible for detecting and publishing color mask images."""

    def __init__(self):
        super().__init__("color_mask_finder")
        self._param_listener = follower_parameters.ParamListener(self)
        self._params = self._param_listener.get_params()
        self._param_listener.set_user_callback(self.reconfigure_callback)

        self._bridge = cv_bridge.CvBridge()

        self._mask_function = simple_mask

        self._mask_pub: Publisher = self.create_publisher(
            Image, "color_mask", 1
        )

        self._colors_caught_pub: Publisher = self.create_publisher(
            CompressedImage, "colors_caught/compressed", 1
        )

        self._video_sub: Subscription = self.create_subscription(
            CompressedImage,
            "camera/image_color/compressed",
            self.video_callback,
            1
        )

        self.get_logger().info("Starting color_mask_finder node")

    def reconfigure_callback(self, parameters: follower_parameters.Params) -> None:
        """Callback for the parameters change.
        Updates the values of parameters and switches color filtering function.

        Args:
            parameters (follower_parameters.Params): The struct with current parameters.
        """
        self._mask_function = (
            simple_mask
            if parameters.hue_min < parameters.hue_max
            else double_range_mask
        )

        self._params = parameters

    def video_callback(self, data: Image) -> None:
        """Callback for the camera image topic.
        Processes incoming camera frames, applies color masking, and publishes results.

        Args:
            data (Image): The received message.
        """
        cv_img: NDArray = self._bridge.compressed_imgmsg_to_cv2(
            data, desired_encoding="passthrough"
        )
        cv_img = cv_img[200 : cv_img.shape[0], :]
        cv_img = cv2.resize(cv_img, (160, 120))

        color_mask = self.get_mask(cv_img)
        colors_caught = get_colors_from_mask(color_mask, cv_img)

        self.publish_imgs(color_mask, colors_caught)

    def get_mask(self, img: NDArray) -> NDArray:
        """Generates a color mask from the provided image using HSV thresholding.

        Args:
            img (NDArray): The target image.

        Returns:
            NDArray: Grayscale image representing the obtained color mask.
        """
        copy_img = copy.deepcopy(img)
        hsv_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2HSV)
        mask = self._mask_function(hsv_img, self._params)

        return mask

    def publish_imgs(self, mask: NDArray, colors: NDArray) -> None:
        """Converts processed OpenCV images to ROS messages and publishes them.

        Args:
            mask (NDArray): Grayscale image representing the detected color mask.
            colors (NDArray): Image showing regions matching the color mask.
        """
        mask_to_ros = self._bridge.cv2_to_imgmsg(mask, encoding="8UC1")
        colors_to_ros = self._bridge.cv2_to_compressed_imgmsg(colors)

        self._mask_pub.publish(mask_to_ros)
        self._colors_caught_pub.publish(colors_to_ros)

    def print_vals(self) -> None:
        """Displays current HSV bounds of the color mask on the terminal."""
        print("Your chosen hsv bounds - copy them to correct yaml file", flush=True)
        print(f"hue_min: {self._params.hue_min}", flush=True)
        print(f"hue_max: {self._params.hue_max}", flush=True)
        print(f"sat_min: {self._params.sat_min}", flush=True)
        print(f"sat_max: {self._params.sat_max}", flush=True)
        print(f"val_min: {self._params.val_min}", flush=True)
        print(f"val_max: {self._params.val_max}", flush=True)

    def cleanup(self) -> None:
        """Cleans ROS entities."""
        self.destroy_subscription(self._video_sub)
        self.destroy_publisher(self._colors_caught_pub)
        self.destroy_publisher(self._mask_pub)
        self.print_vals()
