import copy
import cv2
from numpy.typing import NDArray

from leo_example_line_follower.follower_parameters import follower_parameters


def simple_mask(img: NDArray, params: follower_parameters.Params) -> NDArray:
    """Obtains color mask from image in a simple threshold manner:
    each of HSV components needs to be inbetween specidied bounds.

    Args:
        img (NDArray): Image to get the mask from.
        params (follower_parameters.Params): Parameter structure with bounds for HSV components.

    Returns:
        NDArray: Grayscale image representing the detected color mask.
    """
    color_mask = cv2.inRange(
        img,
        (
            params.hue_min,
            params.sat_min,
            params.val_min,
        ),
        (
            params.hue_max,
            params.sat_max,
            params.val_max,
        ),
    )
    return color_mask


def double_range_mask(img: NDArray, params: follower_parameters.Params) -> NDArray:
    """Obtains color mask from image in a dual range manner.
    Used when in specified bounds for hue component, maximum has smaller
    value than minimum. Then the hue values have to be in one of the ranges:
    - [0 - hue_max]
    - [hue_min - 180]

    Args:
        img (NDArray): Image to get the mask from.
        params (follower_parameters.Params): Parameter structure with bounds for HSV components.

    Returns:
        NDArray: Grayscale image representing the detected color mask.
    """
    lower_mask = cv2.inRange(
        img,
        (
            0,
            params.sat_min,
            params.val_min,
        ),
        (
            params.hue_max,
            params.sat_max,
            params.val_max,
        ),
    )

    upper_mask = cv2.inRange(
        img,
        (
            params.hue_min,
            params.sat_min,
            params.val_min,
        ),
        (
            179,
            params.sat_max,
            params.val_max,
        ),
    )

    final_mask = lower_mask + upper_mask

    return final_mask


def get_colors_from_mask(mask: NDArray, img: NDArray) -> NDArray:
    """Applies the given mask to an RGB image.

    Args:
        mask (NDArray): Binary mask used to extract color regions.
        img (NDArray): The target image.

    Returns:
        NDArray: Image containing only the masked color regions.
    """
    copy_img = copy.deepcopy(img)
    colors_caught = cv2.bitwise_and(copy_img, copy_img, mask=mask)

    return colors_caught
