#!/usr/bin/env python3
"""
의료 영상 Segmentation Inference 스크립트
Mask R-CNN 모델로 대장/위 내시경 이미지의 병변(궤양, 암, 용종)을 검출
"""

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import unicodedata

# ⭐ Import 경로 수정
try:
    from datasets.medical_folder_dataset import MedicalFolderDataset
except ImportError:
    from medical_folder_dataset import MedicalFolderDataset

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("⚠️ 한글 폰트 설정 실패 (시각화는 정상 작동)")


def get_model(num_classes, checkpoint_path):
    """학습된 Mask R-CNN 모델 로드"""
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    
    model = maskrcnn_resnet50_fpn(weights=None)
    
    # Box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    
    # Checkpoint 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"📦 Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return model


def normalize_class_name(name):
    name = str(name).strip()

    # ⭐ 유니코드 정규화 (이게 핵심)
    name = unicodedata.normalize("NFC", name)

    # 괄호 제거
    if '(' in name:
        name = name.split('(')[0].strip()

    # 종양 → 용종 통일
    if name == '종양':
        name = '용종'

    return name


def visualize_predictions(image, outputs, ground_truth, class_names, threshold=0.5, save_path=None):
    """예측 결과 시각화"""
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    masks = outputs['masks'].cpu().numpy()
    
    # Threshold 적용
    keep = scores > threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    masks = masks[keep]
    
    # ⭐ Ground truth 정규화
    gt_normalized = normalize_class_name(ground_truth)
    
    print(f"\n{'='*60}")
    print(f"📊 Predictions (threshold={threshold})")
    print(f"{'='*60}")
    print(f"Ground Truth: '{gt_normalized}'")
    print(f"Total detections: {len(scores)}")
    
    correct_count = 0
    wrong_count = 0
    
    if len(scores) > 0:
        print(f"Score range: {scores.min():.3f} - {scores.max():.3f}\n")
        print("Predicted classes:")
        
        for i, (label, score) in enumerate(zip(labels, scores)):
            # ⭐ 클래스 이름 가져오고 정규화
            class_name = class_names.get(int(label), f"class_{label}")
            pred_normalized = normalize_class_name(class_name)
            
            # ⭐ 디버그 출력 (첫 예측만)
            if i == 0:
                print(f"  [DEBUG] Raw class_name: '{class_name}'")
                print(f"  [DEBUG] Normalized: '{pred_normalized}'")
                print(f"  [DEBUG] GT normalized: '{gt_normalized}'")
                print(f"  [DEBUG] Match: {pred_normalized == gt_normalized}\n")
            
            # ⭐ 정규화된 값으로 비교
            is_correct = (pred_normalized == gt_normalized)
            
            if is_correct:
                correct_count += 1
                status = "CORRECT"
                emoji = "✓"
            else:
                wrong_count += 1
                status = "WRONG"
                emoji = "✗"
            
            print(f"  [{i}] {emoji} '{pred_normalized}': {score:.3f} [{status}]")
        
        print(f"\nSummary: {correct_count} correct, {wrong_count} wrong")
    else:
        print("No detections above threshold")
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 원본 이미지
    axes[0].imshow(image_np)
    axes[0].set_title(f"Original\nGround Truth: {gt_normalized}", fontsize=12, weight='bold')
    axes[0].axis('off')
    
    # 2. Bounding Boxes
    axes[1].imshow(image_np)
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        
        class_name = class_names.get(int(label), f"class_{label}")
        pred_normalized = normalize_class_name(class_name)
        is_correct = (pred_normalized == gt_normalized)
        
        # 색상: 초록=정답, 빨강=오답
        color = 'green' if is_correct else 'red'
        
        rect = plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False, edgecolor=color, linewidth=3
        )
        axes[1].add_patch(rect)
        
        # 라벨 텍스트
        axes[1].text(
            x1, y1-5, f"{pred_normalized}: {score:.2f}",
            color=color, fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )
    
    title = f"Detections (n={len(boxes)})"
    if len(boxes) > 0:
        title += f"\nCorrect: {correct_count}, Wrong: {wrong_count}"
    axes[1].set_title(title, fontsize=12, weight='bold')
    axes[1].axis('off')
    
    # 3. Segmentation Masks
    axes[2].imshow(image_np)
    if len(masks) > 0:
        combined_mask = masks[:, 0, :, :].sum(axis=0)
        axes[2].imshow(combined_mask, alpha=0.6, cmap='jet')
    axes[2].set_title("Segmentation Masks", fontsize=12, weight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.close()


def batch_test(model, dataset, device, class_names, num_samples=10, threshold=0.5):
    """배치 테스트 - 여러 샘플 평가"""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    results = {
        'correct': 0,
        'wrong': 0,
        'no_detection': 0
    }
    
    os.makedirs("outputs/batch_inference", exist_ok=True)
    
    # 디버그: 클래스 매핑 확인
    print("\n" + "="*60)
    print("DEBUG: Class Mapping Check")
    print("="*60)
    print("\nCLASS_NAMES dictionary:")
    for label, name in class_names.items():
        normalized = normalize_class_name(name)
        print(f"  {label}: '{name}' → '{normalized}'")
    
    print("\nSample ground truths:")
    for i in range(min(3, len(dataset))):
        orig = dataset.samples[i]['class_name']
        norm = normalize_class_name(orig)
        print(f"  [{i}] '{orig}' → '{norm}'")
    print("="*60 + "\n")
    
    # 각 샘플 테스트
    for i in range(min(num_samples, len(dataset))):
        sample = dataset.samples[i]
        
        # 이미지 로드
        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize((384, 384))
        image_tensor = transform(image).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model([image_tensor])[0]
        
        # 가장 높은 점수의 예측 찾기
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        
        keep = scores > threshold
        
        # ⭐ Ground truth 정규화
        gt_norm = normalize_class_name(sample['class_name'])
        
        if keep.sum() > 0:
            # Threshold 이상인 것 중 최고 점수
            filtered_scores = scores[keep]
            filtered_labels = labels[keep]
            
            best_idx = filtered_scores.argmax()
            predicted_label = int(filtered_labels[best_idx])
            predicted_class = class_names.get(predicted_label, 'unknown')
            predicted_score = filtered_scores[best_idx]
            
            # ⭐ 예측 정규화
            pred_norm = normalize_class_name(predicted_class)
            
            # ⭐ 결과 판정
            if pred_norm == gt_norm:
                results['correct'] += 1
                print(f"✓ [{i}] CORRECT: '{gt_norm}' → '{pred_norm}' ({predicted_score:.3f})")
            else:
                results['wrong'] += 1
                print(f"✗ [{i}] WRONG: '{gt_norm}' → '{pred_norm}' ({predicted_score:.3f})")
        else:
            results['no_detection'] += 1
            print(f"? [{i}] NO DETECTION: '{gt_norm}'")
        
        # 시각화 저장
        save_path = f"outputs/batch_inference/sample_{i:03d}.png"
        visualize_predictions(
            image_tensor, outputs, sample['class_name'],
            class_names, threshold, save_path
        )
    
    # 최종 통계
    total = sum(results.values())
    detected = total - results['no_detection']
    
    print("\n" + "="*60)
    print("BATCH TEST RESULTS")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"Correct predictions: {results['correct']} ({results['correct']/total*100:.1f}%)")
    print(f"Wrong predictions: {results['wrong']} ({results['wrong']/total*100:.1f}%)")
    print(f"No detections: {results['no_detection']} ({results['no_detection']/total*100:.1f}%)")
    
    if detected > 0:
        accuracy = results['correct'] / detected * 100
        print(f"\nAccuracy (excluding no detections): {accuracy:.1f}%")
    else:
        print(f"\nAccuracy: N/A (no detections)")
    
    print("="*60)


def main():
    device = torch.device("cpu")
    print(f"🖥️ Using device: {device}\n")
    
    # 클래스 정의 (Dataset의 IDX_TO_CLASS와 동일하게)
    CLASS_NAMES = {
        0: 'background',
        1: '궤양 (ulcer)',
        2: '암 (cancer)',
        3: '종양 (tumor)'
    }
    
    # 모델 로드
    print("="*60)
    print("Loading Model")
    print("="*60)
    
    checkpoint_path = 'outputs/medical_seg/best_model.pth'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'outputs/medical_seg/model_epoch5.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model not found at {checkpoint_path}")
        print("Please train the model first!")
        return
    
    model = get_model(num_classes=4, checkpoint_path=checkpoint_path)
    model.to(device)
    model.eval()
    
    # Detection threshold 설정
    model.roi_heads.score_thresh = 0.05
    model.roi_heads.nms_thresh = 0.5
    
    print("✅ Model loaded successfully!\n")
    
    # 테스트 모드 선택
    print("="*60)
    print("TEST OPTIONS")
    print("="*60)
    print("1. Single image test (with multiple thresholds)")
    print("2. Batch test (10 samples, accuracy measurement)")
    print("3. Custom image path")
    print()
    
    mode = input("Enter choice (1/2/3): ").strip()
    
    if mode in ["1", "2"]:
        # Dataset 로드
        print("\nLoading dataset...")
        try:
            dataset = MedicalFolderDataset(
                image_root='/Users/admin/Downloads/datasets/1.Training/1.원천데이터',
                label_root='/Users/admin/Downloads/datasets/1.Training/2.라벨링데이터',
                organ_type='대장',
                transforms=None,
                resize=(384, 384),
                max_samples=50
            )
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("Please check the data paths!")
            return
        
        if mode == "1":
            # Single image test
            print(f"\nDataset has {len(dataset)} samples")
            idx = int(input(f"Enter sample index (0-{len(dataset)-1}): "))
            
            if idx < 0 or idx >= len(dataset):
                print(f"❌ Invalid index! Must be between 0 and {len(dataset)-1}")
                return
            
            sample = dataset.samples[idx]
            
            # 이미지 로드
            image = Image.open(sample['image_path']).convert('RGB')
            image = image.resize((384, 384))
            
            transform = transforms.ToTensor()
            image_tensor = transform(image).to(device)
            
            print(f"\nTesting: {os.path.basename(sample['image_path'])}")
            print(f"Ground truth: {sample['class_name']}")
            
            # Inference
            with torch.no_grad():
                outputs = model([image_tensor])[0]
            
            # 여러 threshold로 테스트
            os.makedirs("outputs/inference", exist_ok=True)
            
            for threshold in [0.3, 0.5, 0.7]:
                print(f"\n{'='*60}")
                print(f"Testing with threshold = {threshold}")
                print(f"{'='*60}")
                
                save_path = f"outputs/inference/result_thresh{threshold:.1f}.png"
                visualize_predictions(
                    image_tensor, outputs, sample['class_name'],
                    CLASS_NAMES, threshold, save_path
                )
        
        elif mode == "2":
            # Batch test
            print("\nRunning batch test on 10 random samples...\n")
            batch_test(
                model, dataset, device, CLASS_NAMES,
                num_samples=10, threshold=0.5
            )
    
    elif mode == "3":
        # Custom image
        img_path = input("Enter image path: ").strip()
        
        if not os.path.exists(img_path):
            print(f"❌ File not found: {img_path}")
            return
        
        image = Image.open(img_path).convert('RGB')
        print(f"Original size: {image.size}")
        image = image.resize((384, 384))
        
        transform = transforms.ToTensor()
        image_tensor = transform(image).to(device)
        
        print("\nRunning inference...")
        with torch.no_grad():
            outputs = model([image_tensor])[0]
        
        os.makedirs("outputs/custom", exist_ok=True)
        save_path = "outputs/custom/result.png"
        
        visualize_predictions(
            image_tensor, outputs, "Unknown",
            CLASS_NAMES, threshold=0.5, save_path=save_path
        )
    
    else:
        print("❌ Invalid choice!")
    
    print("\n✅ Inference completed!")


if __name__ == "__main__":
    main()
