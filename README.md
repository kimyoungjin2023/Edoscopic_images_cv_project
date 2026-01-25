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

![R-CNN Comparison](https://copilot.microsoft.com/th/id/BCO.590fd018-a8c4-446e-8602-d4d0243660eb.png)

![R-CNN Comparison](images/rcnn_comparison.png)

