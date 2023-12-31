from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """
    경계 상자를 나타내는 클래스입니다.
    """
    x: int
    y: int
    width: int
    height: int

    @property
    def left_top(self) -> np.ndarray:
        """
        경계 상자의 좌상단 좌표를 반환합니다.
        :return: 경계 상자의 좌상단 좌표
        """
        return np.array([self.x, self.y])

    @property
    def right_bottom(self) -> np.ndarray:
        """
        경계 상자의 우하단 좌표를 반환합니다.
        :return: 경계 상자의 우하단 좌표
        """
        return np.array([self.x + self.width, self.y + self.height])

    def crop(self, image: np.ndarray) -> np.ndarray:
        """
        주어진 이미지에서 이 영역을 잘라냅니다.
        :param image: 영역을 잘라낼 이미지
        :return: 잘라낸 영역
        """
        return image[self.y:self.y + self.height, self.x:self.x + self.width].copy()

    def is_inside(self, other) -> bool:
        """
        이 영역이 다른 영역 내부에 있는지 확인합니다.
        :param other: 다른 영역
        :return: 이 영역이 다른 영역 내부에 있다면 True, 아니라면 False
        """
        return (other.x <= self.x and other.y <= self.y and
                self.x + self.width <= other.x + other.width and
                self.y + self.height <= other.y + other.height)
