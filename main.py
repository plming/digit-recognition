import cv2

# 실행 시 장치 번호 확인하기
DEVICE_INDEX = 0
capture = cv2.VideoCapture(DEVICE_INDEX)

_ret, frame = capture.read()

# 비디오 입력 테스트
# Expected: (480, 640, 3)
print(f"비디오 크기: {frame.shape}")

while True:
    ret, frame = capture.read()

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break