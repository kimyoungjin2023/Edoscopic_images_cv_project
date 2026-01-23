# Multi-task Vision: Object Detection & Segmentation

PyTorch 기반 **Object Detection + Segmentation 멀티태스크 학습 프로젝트**입니다.  
단일 모델로 bounding box 예측과 pixel-wise segmentation을 동시에 수행합니다.

---

## 🚀 Features

- Detection + Segmentation 멀티태스크 학습
- IoU-based bounding box loss
- BCE + Dice segmentation loss
- Config 기반 실험 관리
- Modular & clean codebase (research / production friendly)

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
│   ├── README.md
│   └── sample/
├── scripts/
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── README.md
'''

---

### Hardware
- Tested on NVIDIA RTX 4060 Ti (8GB)
- CUDA 11.7
