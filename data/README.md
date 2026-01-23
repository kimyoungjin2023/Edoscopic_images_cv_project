# Dataset Guide

본 프로젝트는 **Object Detection + Segmentation 멀티태스크 학습**을 위한  
데이터 구조를 사용합니다.

Detection과 Segmentation 데이터는 **동일한 이미지 기준**으로 구성됩니다.

---

## 📁 Directory Structure

```text
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── masks/
│   ├── train/
│   ├── val/
│   └── test/
└── README.md
