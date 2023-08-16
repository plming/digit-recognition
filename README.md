# digit-recognition

## 소개

[부산대학교 총장배 창의비행체 경진대회](https://cafe.naver.com/pnucreative/2072)의 회전익 번호판 인식 임무에 사용할 프로그램입니다.

## 설치 및 실행

1. 파이썬 가상환경 설정

```bash
python3 -m venv venv
```

2. 의존성 설치

```bash
pip install -r requirements.txt
```

3. 실행

```bash
python3 main.py
```

## 이미지 세부사항

- 야외에서 하나의 숫자가 적힌 번호판을 촬영한 사진(scene text image)
- 글자 크기: 25cm
- 배경색: 흰색
- 글꼴색: 빨간색

## 임무 수행 방법

1. Digit Detection
    - 숫자가 적힌 이미지 내 영역을 추출
    - 번호판에 대한 색상 정보에 기반하여, 색상 필터로 영역 추출
2. Digit Recognition
    - MNIST 데이터셋으로 학습한 CNN 모델 사용
    - Detection 단계에서 추출한 숫자 이미지를 입력으로 사용
    - 0~9 숫자 중 하나로 분류

## 참고자료

- [MNIST 데이터셋](http://yann.lecun.com/exdb/mnist)
- [Introduction to Scene Text Detection and Recognition](http://dmqm.korea.ac.kr/activity/seminar/320)