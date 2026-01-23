# Multi-task Vision: Object Detection & Segmentation

본 프로젝트는 **Object Detection과 Segmentation을 동시에 수행하는 멀티태스크 비전 모델**을 개발하는  
**팀 기반 포트폴리오 프로젝트**입니다.

PyTorch 기반으로 학습 파이프라인을 직접 설계하였으며,  
의료 영상(내시경 이미지) 데이터를 활용하여 실전 문제에 가까운 환경을 목표로 했습니다.

---

## 🚀 Key Features

- Object Detection + Segmentation 멀티태스크 학습
- COCO-style annotation 기반 데이터 처리
- IoU 기반 Bounding Box Loss
- BCE + Dice Segmentation Loss
- Config 기반 실험 관리
- 확장 가능한 모듈형 코드 구조
- Docker 기반 재현 가능한 실행 환경

---

## 📁 Project Structure

```text
project/
├── src/
│   ├── train.py
│   ├── eval.py
│   ├── infer.py
│   ├── models/
│   ├── datasets/
│   ├── losses/
│   └── utils/
├── configs/
├── data/
│   ├── images/
│   ├── annotations/
│   └── README.md
├── scripts/
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── README.md
🛠 Tech Stack
PyTorch

OpenCV

Albumentations

Docker

🧱 HW
GPU: NVIDIA RTX 4060 Ti (8GB)

CUDA: 11.7

CPU: x86_64

RAM: 32GB

🐳 Docker
본 프로젝트는 Docker 기반으로 실행 환경을 재현할 수 있습니다.


docker build -t multitask-vision .
docker run --gpus all -it multitask-vision
