import copy
from typing import TypeAlias, Sequence
from pathlib import Path
import yaml

from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.parameter import Parameter
from rclpy.subscription import Subscription

from rcl_interfaces.msg import ParameterDescriptor, IntegerRange, SetParametersResult
from sensor_msgs.msg import Image, CompressedImage

import cv2
from numpy.typing import NDArray

import cv_bridge

from ai_edge_litert.interpreter import Interpreter

BoxInfo: TypeAlias = tuple[tuple[int, int], tuple[int, int], str, tuple[int, int, int]]


class ObjectDetectorNode(Node):
    """A ROS node demonstrating usage of object detection NN."""

    def __init__(
        self, model_path: str = None, video_topic: str = None, labels_file: str = None
    ) -> None:
        super().__init__("object_detector")

        self.bridge: cv_bridge.CvBridge = cv_bridge.CvBridge()

        try:
            self.interpreter: Interpreter = Interpreter(model_path=model_path)
        except ValueError as e:
            self.get_logger().error(f"Could not load tflite model: '{model_path}'.")
            raise

        input_details = self.interpreter.get_input_details()
        self.input_shape: tuple[int, int] = tuple(
            input_details[0]["shape"][1:3].tolist()
        )
        self.interpreter.allocate_tensors()

        self.read_labels(labels_file)
        self.build_color_dict()
        self.scales: bool = False

        self.declare_parameter(
            "~confidence",
            65,
            ParameterDescriptor(
                description="Detection confidence threshold percentage (int, range 0-100).",
                integer_range=[IntegerRange(from_value=0, to_value=100, step=1)],
            ),
        )
        self.confidence_threshold: int = self.get_parameter("~confidence").value
        self.add_on_set_parameters_callback(self.param_callback)

        self.detection_pub: Publisher = self.create_publisher(
            CompressedImage, "detections/compressed", 1
        )

        self.video_sub: Subscription = self.create_subscription(
            Image, video_topic, self.video_callback, 1
        )

        self.get_logger().info("Starting node.")

    def param_callback(self, params: Sequence[Parameter]) -> SetParametersResult:
        """Callback triggered on parameter change.

        Args:
            params (Sequence[Parameter]): Node parameters.

        Returns:
            SetParametersResult: Result of parameter change action.
        """
        for param in params:
            if param.name == "~confidence" and param.type_ == param.Type.INTEGER:
                self.confidence_threshold = param.value
                self.get_logger().info(
                    f"Updated detection confidence threshold to {param.value}."
                )
                return SetParametersResult(successful=True)
        return SetParametersResult(successful=False, reason="No such parameter")

    def get_scales(self, img: NDArray) -> None:
        """Calculates the scaling factors for the images from ros topic and NN input.

        Args:
            img (NDArray): Image received from ros topic converted to opencv object.
        """
        self.final_height: int = img.shape[0]
        self.final_width: int = img.shape[1]

        self.scale_x: float = self.final_width / self.input_shape[0]
        self.scale_y: float = self.final_height / self.input_shape[1]

        self.scales = True

    def translate_point(self, point: tuple[int, int]) -> tuple[int, int]:
        """Translates point from NN image into ros topic image.

        Args:
            point (tuple[int, int]): A point from NN detection bounding box.

        Returns:
            tuple[int, int]: Point with translated coordinates.
        """
        x_translated = point[0] * self.scale_x
        y_translated = point[1] * self.scale_y

        return (int(x_translated), int(y_translated))

    def read_labels(self, file_path: str) -> None:
        """Reads labels for the detection model from provided file.

        Args:
            file_path (str): Absolute path to file with detection labels.
        """
        self.labels: Sequence[str] = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.labels.append(line.strip())

    def build_color_dict(self) -> None:
        """Parses the label and color parameters to produce a dictionary mapping labels
        to their respective colors.
        """
        self.label_colors: dict = dict()
        self.declare_parameter(
            "~config_path",
            str(
                Path(get_package_share_directory("leo_example_object_detection"))
                / "config"
                / "labels_config.yaml"
            ),
            ParameterDescriptor(
                description="Path to the yaml file with label-color configuration."
            ),
        )

        config_path = self.get_parameter("~config_path").value

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        for label, color in config_data.get("labels", {}).items():
            self.label_colors[label.replace("_", " ")] = color

    def video_callback(self, data: Image) -> None:
        """Processes incoming video messages by applying preprocessing and object detection,
        and publishes the resulting compressed image with bounding boxes over detected objects.

        Args:
            data (Image): Incoming video frame.
        """
        cv_img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        if not self.scales:
            self.get_scales(cv_img)

        processed_img = self.preprocess(rgb_img)
        boxes, labels, confidences = self.detect_objects(processed_img)
        final_boxes = self.get_boxes(boxes, labels, confidences)
        final_img = self.draw_detections(rgb_img, final_boxes)

        try:
            msg = self.bridge.cv2_to_compressed_imgmsg(final_img, "jpeg")
            self.detection_pub.publish(msg)
        except cv_bridge.CvBridgeError() as e:
            self.get_logger().error(e)

    def preprocess(self, img: NDArray) -> NDArray:
        """Resizes the provided opencv image to match the NN requirements.

        Args:
            img (NDArray): Input image to run the object detection on.

        Returns:
            NDArray: Preprocessed opencv image ready for NN input.
        """
        cpy = copy.deepcopy(img)
        resized: NDArray = cv2.resize(cpy, self.input_shape)

        return resized

    def detect_objects(self, img: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Performs object detection on the provided image.

        Args:
            img (NDArray): Input image for the neural network.

        Returns:
            tuple[NDArray, NDArray, NDArray]:
                A tuple containing respectively bounding box coordinates (ranging from 0.0 to 1.0),
                class IDs of the detected objects and confidence scores for each detection.
        """
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()

        # providing input
        self.interpreter.set_tensor(input_details[0]["index"], [img])
        # running interferance
        self.interpreter.invoke()

        # getting answer
        boxes = self.interpreter.get_tensor(output_details[0]["index"])[0]
        labels = self.interpreter.get_tensor(output_details[1]["index"])[0]
        confidence = self.interpreter.get_tensor(output_details[2]["index"])[0]

        return boxes, labels, confidence

    def get_boxes(
        self,
        boxes: Sequence[tuple[int, int, int, int]],
        labels: Sequence[int],
        confidence: Sequence[float],
    ) -> Sequence[BoxInfo]:
        """Generates formatted bounding box data for detected objects.

        Args:
            boxes (Sequence[tuple[int, int, int, int]]): Bounding box coordinates in normalized form (ymin, xmin, ymax, xmax).
            labels (Sequence[int]): Class label indices corresponding to detected objects.
            confidence (Sequence[float]): Confidence scores for each detection, ranging from 0.0 to 1.0.

        Returns:
            Sequence[BoxInfo]: Bounding box information, where each element includes:
                * start_point (tuple[int, int]): Top-left pixel coordinates.
                * end_point (tuple[int, int]): Bottom-right pixel coordinates.
                * text (str): Label string with confidence.
                * color (tuple[int, int, int]): RGB color for visualization.
        """
        final_boxes = []
        for box, label, conf in zip(boxes, labels, confidence):
            if int(conf * 100) > self.confidence_threshold:
                top = int(box[0] * self.input_shape[1])
                left = int(box[1] * self.input_shape[0])
                bottom = int(box[2] * self.input_shape[1])
                right = int(box[3] * self.input_shape[0])

                color = self.label_colors.get(self.labels[int(label)], (0, 0, 102))

                text = self.labels[int(label)] + " " + str(round(conf * 100, 2)) + "%"

                final_start_point = self.translate_point((left, top))
                final_end_point = self.translate_point((right, bottom))

                final_boxes.append((final_start_point, final_end_point, text, color))

        return final_boxes

    def draw_detections(self, img: NDArray, final_boxes: Sequence[BoxInfo]) -> NDArray:
        """Draws bounding boxes with text labels onto target image.

        Args:
            img (NDArray): Target image to draw the detections on.
            final_boxes (Sequence[BoxInfo]): Boxes with detections info used for drawing.

        Returns:
           NDArray: Image with bounding boxes drawn.
        """
        for start, end, text, color in final_boxes:
            # main border
            img = cv2.rectangle(img, start, end, color, 2)

            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            text_w, text_h = text_size

            # text background
            cv2.rectangle(
                img,
                (start[0], start[1] + 2),
                (start[0] + text_w, start[1] + 2 + text_h),
                color,
                -1,
            )

            img = cv2.putText(
                img,
                text,
                (start[0], start[1] + text_h - 1),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                1,
            )

        return img

    def cleanup(self) -> None:
        """Cleans ROS entities."""
        self.destroy_subscription(self.video_sub)
        self.destroy_publisher(self.detection_pub)
