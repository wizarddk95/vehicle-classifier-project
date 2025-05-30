# Vehicle Classifier Project

이 프로젝트는 차량 이미지를 분류하는 딥러닝 모델을 학습하고 평가하는 프로젝트입니다.

## 프로젝트 구조

```
vehicle_classifier_project/
├── analysis/              # 모델 분석 및 시각화 관련 코드
├── checkpoints/           # 학습된 모델 체크포인트 저장
├── data/                  # 학습 데이터셋
├── dataloaders/           # 데이터 로딩 관련 코드
│   └── loaders.py        # 데이터로더 구현
├── models/               # 모델 관련 코드
│   └── train_utils.py    # 학습 유틸리티 함수
├── notebook/             # Jupyter 노트북
│   ├── train_validation.ipynb  # 모델 학습 및 검증
│   ├── visualization.ipynb     # 데이터 및 결과 시각화
│   └── test_submission.ipynb   # 테스트 데이터 예측 및 제출
├── preprocessing/        # 데이터 전처리 관련 코드
│   └── crop_roof.ipynb   # 차량 지붕 영역 추출
├── output_analysis/      # 분석 결과 저장
├── runs/                 # TensorBoard 로그
├── submissions/          # 제출 파일
└── requirements.txt      # 프로젝트 의존성
```

## 환경 설정

### 요구사항
- Python 3.11+
- CUDA 지원 GPU (권장)

### 설치 방법

1. 가상환경 생성 및 활성화:
```bash
python -m venv pytorch_env
source pytorch_env/bin/activate  # Linux/Mac
# 또는
pytorch_env\Scripts\activate  # Windows
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

## 주요 기능

- 차량 이미지 분류를 위한 딥러닝 모델 학습
- Swin Transformer 기반 모델 사용
- TensorBoard를 통한 학습 모니터링
- 조기 종료(Early Stopping) 구현
- 학습/검증 데이터 분할
- 모델 성능 분석 및 시각화
- 차량 지붕 영역 추출을 통한 전처리
- 데이터 및 결과 시각화 도구

## 데이터 전처리

데이터 전처리는 `preprocessing/crop_roof.ipynb`에서 수행됩니다:
- 차량 이미지에서 지붕 영역을 자동으로 추출
- 이미지 크기 정규화
- 데이터 증강(augmentation) 적용

## 모델 학습

모델 학습은 `notebook/train_validation.ipynb`에서 수행됩니다. 주요 특징:

- Cosine Annealing with Warm Restarts 스케줄러 사용
- AdamW 옵티마이저
- Cross Entropy Loss
- Early Stopping (patience=5)
- 배치 크기: 32
- 학습 에포크: 50

## 결과 분석 및 시각화

학습된 모델의 성능은 다음 지표로 평가됩니다:
- 정확도 (Accuracy)
- 로그 손실 (Log Loss)

분석 및 시각화는 `notebook/visualization.ipynb`에서 수행됩니다:
- 학습/검증 손실 및 정확도 그래프
- 혼동 행렬(Confusion Matrix)
- 클래스별 성능 분석
- 예측 결과 시각화

분석 결과는 `output_analysis/` 디렉토리에 저장됩니다.

## 테스트 및 제출

테스트 데이터에 대한 예측 및 제출 파일 생성은 `notebook/test_submission.ipynb`에서 수행됩니다:
- 테스트 데이터 로드 및 전처리
- 학습된 모델을 사용한 예측
- 제출 파일 생성

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 