# Dataset Guide — Endoscopy Synthetic Dataset (AI Hub)

이 프로젝트는 **AI Hub 내시경 이미지 합성데이터** 기반으로  
Object Detection + Segmentation 모델 학습을 목표로 합니다.  
(데이터셋: 위/대장 내시경 병변 합성 이미지)

---

## 📁 Directory Structure

```text
data/
├── train/
│   ├── 원천데이터(IMG)
│   └── 라벨링 데이터(JSON)
├── val/
│   ├── 원천데이터(IMG)
│   └── 라벨링 데이터(JSON)
├── test/
│   ├── 원천데이터(IMG)
│   └── 라벨링 데이터(JSON)
└── README.md

```

---

## 데이터셋

|데이터|양|
|:---|:---|
|Train|6000장(위, 대장:궤양, 암, 용종)각각 1000장|
|val|900장(위, 대장:궤양, 암, 용종)각각 150장|
|test|2000장(위, 대장: 궤양(250장), 암(500장), 용종(250장))|
