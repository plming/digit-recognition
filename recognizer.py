import cv2
import numpy as np
from keras.models import Model


class DigitRecognizer:
    """
    숫자 인식기 클래스입니다.
    """

    def __init__(self, model: Model):
        """
        숫자 인식기 생성자입니다.
        :param model: MNIST로 학습된 Keras 모델
        """
        self.model = model

    def run(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        주어진 이미지에서 숫자를 인식합니다.
        :param bgr_image: 숫자가 쓰여진 BGR 이미지
        :return: 각 숫자에 대한 예측 확률을 담은 배열
        """
        assert bgr_image.ndim == 3

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 0xFF, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # MNIST 데이터셋의 전처리 과정
        # 참고: http://yann.lecun.com/exdb/mnist/

        # normalize size to 20x20 with preserving aspect ratio
        aspect_ratio = binary.shape[1] / binary.shape[0]
        if aspect_ratio > 1:
            new_width = 20
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = 20
            new_width = int(new_height * aspect_ratio)

        binary = cv2.resize(binary, (new_width, new_height), interpolation=cv2.INTER_AREA)
        x = cv2.copyMakeBorder(binary,
                               (28 - new_height) // 2, (28 - new_height) // 2,
                               (28 - new_width) // 2, (28 - new_width) // 2,
                               cv2.BORDER_CONSTANT, value=0)

        x = cv2.resize(x, (28, 28), interpolation=cv2.INTER_AREA)

        # x값이 [0, 1] 범위에 있도록 scaling
        x_min = np.min(x)
        x_max = np.max(x)
        x = (x - x_min) / (x_max - x_min)

        if __debug__:
            cv2.imshow('digit', cv2.resize(x, (112, 112)))

        x = x.reshape((1, 28, 28, 1))
        probabilities = self.model.predict(x, verbose=0)
        assert probabilities.shape == (1, 10)

        return probabilities[0]
