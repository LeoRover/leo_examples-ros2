from typing import Optional

from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Twist

from leo_example_line_follower.follower_parameters import follower_parameters

from leo_line_follower.utils import simple_mask, double_range_mask

import cv2
import cv_bridge

from numpy.typing import NDArray
import numpy as np

from ai_edge_litert.interpreter import Interpreter


class LineFollowerNode(Node):
    """A ROS node performing as line follower."""

    def __init__(self, model_path=None, velocity_topic=None, video_topic=None) -> None:
        super().__init__("line_follower")
        self._param_listener = follower_parameters.ParamListener(self)
        self._params = self._param_listener.get_params()
        self._param_listener.set_user_callback(self.reconfigure_callback)

        self._bridge = cv_bridge.CvBridge()

        try:
            self._interpreter = Interpreter(model_path=model_path)
            self._interpreter.allocate_tensors()
        except ValueError as e:
            self.get_logger().error(f"Couldnt load tflite model: {model_path}")
            return

        self._mask_func = simple_mask

        self._vel_pub: Publisher = self.create_publisher(
            Twist, velocity_topic, 1
        )

        self._mask_pub: Optional[Publisher] = None
        if self._params.publish_mask:
            self._mask_pub = self.create_publisher(
                Image, "color_mask", 1
            )

        self._video_sub: Subscription = self.create_subscription(
            CompressedImage, video_topic, self.video_callback, 1
        )

    def reconfigure_callback(self, parameters: follower_parameters.Params) -> None:
        """Callback for the parameters change.
        Creates or destroys mask publisher and updates the values
        of parameters and switches color filtering function.

        Args:
            parameters (follower_parameters.Params): The struct with current parameters.
        """
        self._mask_function = (
            simple_mask
            if parameters.hue_min < parameters.hue_max
            else double_range_mask
        )

        if not self._mask_pub and parameters.publish_mask:
            self._mask_pub = self.create_publisher(
                Image, "color_mask", 1
            )
        elif self._mask_pub and not parameters.publish_mask:
            self.destroy_publisher(self._mask_pub)
            self._mask_pub = None

        self._params = parameters

    def video_callback(self, data: CompressedImage) -> None:
        """Processes incoming video frames.
        Parses the frames into NN model input, obtains velocity
        commands from the model and publishes them.

        Args:
            data (CompressedImage): Incoming video frame.
        """
        cv_img = self._bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        processed_img = self.preprocess_img(cv_img)

        steering = self.get_steering(processed_img)

        if self._params.publish_mask:
            self.publish_mask(processed_img)

        self.get_logger().debug(f"steering: {steering[0]}, {steering[1]}")

        if self._params.follow_enabled:
            self.publish_vel(steering)

    def publish_vel(self, steering: tuple[float, float]) -> None:
        """Sends velocity commands for the robot.

        Args:
            steering (tuple[float, float]): Velocity commands obtained from NN model.
        """
        vel_msg = Twist()
        vel_msg.linear.x = float(steering[0])
        vel_msg.angular.z = float(steering[1])
        self._vel_pub.publish(vel_msg)

    def publish_mask(self, mask: NDArray) -> None:
        """Prepares color mask (NN model input) and publishes it.

        Args:
            mask (NDArray): Color mask parsed as NN model input.
        """
        mask *= 255.0
        img_msg = self._bridge.cv2_to_imgmsg(mask, encoding="32FC1")
        self._mask_pub.publish(img_msg)

    def get_steering(self, img: NDArray) -> tuple[float, float]:
        """Runs NN model interference - obtains robots velocity
        commands based on the camera image.

        Args:
            img (NDArray): An input-ready image for the NN model.

        Returns:
            tuple[float, float]: NN model output - linear and angular velocity commands.
        """
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        self._interpreter.allocate_tensors()

        # providing input
        self._interpreter.set_tensor(input_details[0]["index"], [img])
        # running interferance
        self._interpreter.invoke()

        # getting answer
        linear_x = self._interpreter.get_tensor(output_details[0]["index"])[0][0]
        angular_z = self._interpreter.get_tensor(output_details[1]["index"])[0][0]
        self.get_logger().debug(f"prediction = ({linear_x}, {angular_z})")

        return linear_x, angular_z

    def preprocess_img(self, img: NDArray) -> NDArray:
        """Preprocesses camera image into NN model input format.

        Args:
            img (NDArray): Current image frame.

        Returns:
            NDArray: Processed input image ready for model interference.
        """
        # changing BGR to HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # cropping img
        crop_img = hsv_img[200 : hsv_img.shape[0], :]
        # getting color mask
        color_mask = self._mask_func(crop_img, self._params)
        # converting int balues to float
        float_img = color_mask.astype(np.float32)
        # resizing
        resized_img = cv2.resize(float_img, (160, 120))
        # normalize
        final_img = resized_img / 255.0

        return final_img[:, :, np.newaxis]

    def cleanup(self) -> None:
        """Cleans ROS entities."""
        self.destroy_subscription(self._video_sub)

        self.destroy_publisher(self._vel_pub)

        if self._mask_pub:
            self.destroy_publisher(self._mask_pub)
