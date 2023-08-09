from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class Region:
    """
    숫자 영역을 나타내는 클래스입니다.
    """
    x: int
    y: int
    width: int
    height: int
    mask: np.ndarray


def detect_digit(image: np.ndarray) -> Optional[Region]:
    """
    주어진 이미지에서 숫자 영역을 탐지합니다. 이미지에 숫자는 최대 1개만 존재한다 가정합니다.
    숫자는 흰 배경에 빨간색으로 쓰여져 있어야 합니다.
    :param image: 숫자 영역을 찾을 BGR 이미지
    :return: 숫자 영역을 찾았다면 Region 객체, 아니라면 None
    """
    blur = cv2.GaussianBlur(image, (9, 9), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    cv2.imshow('mask', mask)
    morphology = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(morphology, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    region = Region(x, y, w, h, mask[y:y + h, x:x + w])
    return region
