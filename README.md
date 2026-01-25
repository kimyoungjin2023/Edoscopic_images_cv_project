# 👁️ Multi-task Vision: Object Detection & Segmentation

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+cu121%2B-EE4C2C)
![Docker](https://img.shields.io/badge/Docker-Available-2496ED)

> **Object Detection과 Segmentation을 동시에 수행하는 멀티태스크 비전 모델 개발**
> 의료 영상(내시경 데이터)을 활용하여 실전 문제 해결을 목표로 한 **팀 기반 포트폴리오 프로젝트**입니다.

---

## 🚀 Key Features

본 프로젝트는 단일 모델 내에서 두 가지 비전 태스크를 효율적으로 처리하기 위해 다음과 같은 기능을 구현했습니다.

* **Multi-task Learning:** Object Detection + Segmentation 동시 학습 구조 설계
* **Data Pipeline:** COCO-style annotation 파싱 및 전처리 자동화
* **Loss Function:**
    * **Detection:** IoU 기반 Bounding Box Loss
    * **Segmentation:** BCE(Binary Cross Entropy) + Dice Loss 결합
* **Experiment Mgmt:** YAML Config 기반의 유연한 실험 관리
* **Modular Code:** 유지보수와 확장이 용이한 모듈형 구조 (`src/` 분리)
* **Reproducibility:** Docker를 이용한 동일한 실행 환경 보장

---

## 🛠 Tech Stack

프로젝트에 사용된 주요 기술 스택입니다.

| Category | Technology |
| :--- | :--- |
| **Framework** | ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) |
| **Vision Libs** | ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) ![Albumentations](https://img.shields.io/badge/-Albumentations-F05032?style=flat) ![Ultralytics](https://img.shields.io/badge/-Ultralytics-0070FF?style=flat&logo=ultralytics&logoColor=white)|
| **Environment** | ![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat&logo=docker&logoColor=white) |

---

👥 Team
|Member |	Role |
| :--- | :--- |
|이정결 |	Model design & training|
|박소윤 |	Dataset preprocessing|
|한지수 |	Evaluation & visualization|

---

## 📁 Project Structure

```text
project/
├── src/
│   ├── train.py           # 학습 실행 스크립트
│   ├── eval.py            # 모델 평가 및 검증
│   ├── infer.py           # 추론(Inference) 실행
│   ├── models/            # 모델 아키텍처 정의
│   ├── datasets/          # 데이터 로더 및 전처리
│   ├── losses/            # Custom Loss 함수 정의
│   └── utils/             # 유틸리티 함수 모음
├── configs/               # 실험 설정 파일 (.yaml)
├── data/
│   ├── images/            # 원본 이미지
│   ├── annotations/       # COCO format json
│   └── README.md
├── scripts/               # 쉘 스크립트 모음
├── requirements.txt       # 의존성 패키지 목록
├── pyproject.toml         # 프로젝트 설정
├── Dockerfile             # 도커 빌드 파일
└── README.md              # 프로젝트 문서
```

---

## 🧱 Hardware Environment

실험 및 학습은 아래 환경에서 진행되었습니다.

| Component | Specification |
| :--- | :--- |
| **GPU** | NVIDIA RTX 4060 Ti (8GB) |
| **CUDA** | Version 11.7 |
| **CPU** | AMD Ryzen 5 5600 6-core |
| **RAM** | 16GB |

---

## 🐳 How to Run (Docker)

Docker를 사용하여 복잡한 환경 설정 없이 바로 프로젝트를 실행할 수 있습니다.

### 1. Build Image
```bash
docker build -t multitask-vision .
docker run --gpus all -it multitask-vision
```

---

# 객체 탐지 모델 아키텍처 비교 (Object Detection Architecture Comparison)

R-CNN 계열(2-Stage)의 발전 과정과 YOLO(1-Stage)의 구조적 차이를 보여주는 비교 다이어그램입니다.

# Object Detection Models Comparison

## R-CNN
- Region Proposal 기반 객체 검출
- Selective Search로 영역 생성
- CNN을 각 영역마다 적용 → 매우 느림

## Fast R-CNN
- 전체 이미지를 한 번만 CNN에 통과
- RoI Pooling 도입
- End-to-End 학습 가능

## Faster R-CNN
- Region Proposal Network (RPN) 도입
- Selective Search 제거
- 높은 정확도의 Two-stage Detector

## YOLO
- 객체 검출을 하나의 회귀 문제로 해결
- 매우 빠른 속도
- 실시간 객체 검출 가능

### 모델별 핵심 요약

* **R-CNN**: Selective Search로 영역을 제안하고, 각 영역마다 CNN을 돌려 속도가 매우 느림.
* **Fast R-CNN**: 이미지 전체를 한 번만 CNN에 통과시키고(Feature Map 공유), RoI Pooling을 도입하여 속도 개선.
* **Faster R-CNN**: 병목이었던 영역 제안(Region Proposal) 과정을 RPN(Region Proposal Network)으로 대체하여 완전한 딥러닝 구조(End-to-End) 완성.
* **YOLO**: 별도의 영역 제안 과정 없이 그리드(Grid) 방식을 사용하여 물체의 위치와 종류를 한 번에 예측(One-Stage)하여 실시간 처리 가능.

* # Object Detection Algorithms: R-CNN vs Fast R-CNN vs Faster R-CNN vs YOLO

## 📌 알고리즘 비교

| 알고리즘 | 구조 | 장점 | 단점 |
|----------|------|------|------|
| **R-CNN (2014)** | Selective Search → CNN → SVM + BBox | 정확도 높음 | 속도 매우 느림 |
| **Fast R-CNN (2015)** | CNN → Feature Map → RoI Pooling → Softmax + BBox | 속도 개선 | Selective Search 필요 |
| **Faster R-CNN (2016)** | CNN → RPN → RoI Pooling → Softmax + BBox | Selective Search 제거, 속도 향상 | 실시간 부족 |
| **YOLO (2016~)** | Grid 분할 → 각 셀에서 BBox + Class 예측 | 매우 빠름, 실시간 가능 | 작은 객체 탐지 약함 |

## 🖼️ 구조 비교 다이어그램
![비교 다이어그램](https://copilot.microsoft.com/th/id/BCO.0a7487ed-1e7d-40d6-9908-76796146df7f.png)

