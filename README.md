# Digit Recognizer

MNIST 데이터셋으로 학습한 CNN 모델을 사용해 손글씨 숫자(0~9)를 인식하는 프로젝트입니다. 브라우저에서 숫자를 그려 예측하는 Flask 웹 버전과, 데스크톱에서 실행하는 Tkinter 버전을 함께 제공합니다.

## 주요 기능

- 손글씨 숫자 0~9 인식
- MNIST 기반 CNN 모델 사용
- 캔버스 입력 이미지를 MNIST 형식으로 전처리
- 예측 숫자와 confidence/probability 출력
- Flask 웹 앱 버전
- Tkinter 데스크톱 앱 버전
- PyInstaller 패키징 스크립트 포함

## 프로젝트 구조

```text
.
├── web_version/
│   ├── app.py              # Flask 웹 앱, 모델 학습/예측
│   ├── app_standalone.py   # 자동 브라우저 실행용 standalone 버전
│   ├── mnist_model.keras   # 학습된 모델
│   └── launchers/
├── desktop_version/
│   ├── app.py              # Tkinter 데스크톱 앱
│   ├── requirements.txt
│   ├── build_macos.sh
│   ├── build_windows.bat
│   └── mnist_model.keras
└── CLAUDE.md
```

## 웹 버전 실행

```bash
git clone https://github.com/perust/digit-recognizer.git
cd digit-recognizer/web_version

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install tensorflow flask pillow scipy numpy

python app.py
```

브라우저에서 `http://127.0.0.1:8080`에 접속합니다.

## 데스크톱 버전 실행

```bash
cd digit-recognizer/desktop_version
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## 모델/전처리 개요

- 입력 이미지를 grayscale로 변환
- 숫자 영역 bounding box 검출
- 정사각형 패딩 후 20x20 리사이즈
- 28x28 캔버스 중앙 배치
- center-of-mass 보정
- CNN softmax 출력으로 숫자별 확률 계산

## 다시 학습하기

웹 버전에서 `mnist_model.keras`를 삭제하고 앱을 실행하면 새 모델을 학습하도록 구성되어 있습니다.
