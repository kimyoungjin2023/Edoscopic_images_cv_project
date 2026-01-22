# Edoscopic_images_cv_project
Edoscopic_images_cv_project(cv_project)

# Project Name

## Overview
Object Detection & Segmentation using XXX

## Installation
pip install -r requirements.txt

## Dataset Structure

### 실제의 위, 대장 내시경의 궤양, 용종, 암 이미지를 기반으로 위 20,000장(궤양 5,000장, 용종 5,000장, 암 10,000장), 대장 20,000장(궤양 5,000장, 용종 5,000장, 암 10,000장) 총 40,000장의 내시경 이미지 합성이미지를 생성

## Training
python src/train.py --config configs/config.yaml

## Inference
python src/infer.py --weights ...

## Results
(mAP, IoU, 이미지)
