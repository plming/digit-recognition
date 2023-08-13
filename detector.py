from typing import Optional, Tuple

import cv2
import numpy as np

from bounding_box import BoundingBox


def detect_digit(image: np.ndarray) -> Optional[Tuple[BoundingBox, np.ndarray]]:
    """
    주어진 이미지에서 숫자를 검출합니다.
    :param image: 검출할 이미지
    :return: 숫자가 검출된 경우 (경계 상자, 이진화된 이미지), 검출되지 않은 경우 None
    """
    blur = cv2.GaussianBlur(image, (9, 9), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(white_contours) == 0:
        return None
    white_contour = max(white_contours, key=cv2.contourArea)
    white_box = BoundingBox(*cv2.boundingRect(white_contour))

    red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours = sorted(red_contours, key=cv2.contourArea, reverse=True)
    for red_contour in red_contours:
        red_box = BoundingBox(*cv2.boundingRect(red_contour))
        if red_box.is_inside(white_box):
            region_of_interest = red_box.crop(red_mask)
            return red_box, region_of_interest

    return None
