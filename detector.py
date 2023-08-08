from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int
    mask: np.ndarray


def detect_digit(image: np.ndarray) -> Optional[Region]:
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
