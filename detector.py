from typing import Optional

import cv2
import numpy as np

from bounding_box import BoundingBox


def detect_digit(bgr_image: np.ndarray) -> Optional[BoundingBox]:
    """
    주어진 이미지에서 숫자가 있는 영역을 찾아냅니다.
    :param bgr_image: 숫자를 찾을 BGR 이미지
    :return: 숫자가 발견될 경우 숫자가 있는 영역, 발견되지 않을 경우 None
    """
    blur = cv2.GaussianBlur(bgr_image, (9, 9), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 흰색(번호판 배경 색상) 검출하기
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(white_contours) == 0:
        return None
    white_contour = max(white_contours, key=cv2.contourArea)
    white_box = BoundingBox(*cv2.boundingRect(white_contour))

    # 빨간색(번호판 숫자 색상) 검출하기
    red_mask1 = cv2.inRange(hsv, (0, 128, 128), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 128, 128), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours = sorted(red_contours, key=cv2.contourArea, reverse=True)
    for red_contour in red_contours:
        red_box = BoundingBox(*cv2.boundingRect(red_contour))
        if red_box.is_inside(white_box):
            return red_box

    return None
