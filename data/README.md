# Dataset Guide — Endoscopy Synthetic Dataset (AI Hub)

이 프로젝트는 **AI Hub 내시경 이미지 합성데이터** 기반으로  
Object Detection + Segmentation 모델 학습을 목표로 합니다.  
(데이터셋: 위/대장 내시경 병변 합성 이미지) :contentReference[oaicite:2]{index=2}

---

## 📁 Directory Structure

```text
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── README.md

