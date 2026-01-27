# train_medical_segmentation.py (완전 수정 버전)
import torch
import os
import argparse
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from datasets.medical_folder_dataset import MedicalFolderDataset, collate_fn_filter_empty
from tqdm import tqdm
import time

def get_device(force_mps=False):
    """
    최적의 device 선택
    ⚠️ Mask R-CNN은 MPS에서 불안정 → 기본적으로 CPU 사용
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🖥️ Using device: CUDA - {torch.cuda.get_device_name(0)}")
        return device, True  # (device, can_use_workers)
    
    if torch.backends.mps.is_available():
        if force_mps:
            device = torch.device("mps")
            print(f"🖥️ Using device: MPS (Apple Silicon) - EXPERIMENTAL!")
            print("   ⚠️ May crash with Mask R-CNN. Use --cpu if unstable.")
            return device, False  # MPS는 num_workers=0 필요
        else:
            print("⚠️ MPS available but disabled (Mask R-CNN compatibility)")
            print("   Use --force-mps to override (may crash)")
    
    device = torch.device("cpu")
    print(f"🖥️ Using device: CPU")
    return device, True  # CPU는 workers 사용 가능


def get_model(num_classes):
    """Mask R-CNN 모델 생성"""
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    
    return model


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """1 epoch 학습 (AMP 제거 - 안정성 우선)"""
    model.train()
    epoch_loss = 0
    num_batches = 0
    skipped_batches = 0
    
    loss_components = {
        'loss_classifier': 0,
        'loss_box_reg': 0,
        'loss_mask': 0,
        'loss_objectness': 0,
        'loss_rpn_box_reg': 0
    }

    start_time = time.time()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            skipped_batches += 1
            continue
        
        images, targets = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        try:
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            if torch.isnan(loss):
                print(f"\n⚠️ NaN at epoch {epoch}, batch {batch_idx}")
                skipped_batches += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            
            for k, v in loss_dict.items():
                if k in loss_components:
                    loss_components[k] += v.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{epoch_loss/num_batches:.4f}'
            })
        
        except RuntimeError as e:
            if "MPS" in str(e) or "Metal" in str(e):
                print(f"\n❌ MPS Error at batch {batch_idx}")
                print("   Try running with --cpu flag")
                raise  # MPS 에러는 즉시 중단
            else:
                print(f"\n❌ Error at batch {batch_idx}: {e}")
                skipped_batches += 1
                continue

    elapsed = time.time() - start_time
    
    if num_batches == 0:
        print("⚠️ No valid batches!")
        return 0.0
    
    if skipped_batches > len(dataloader) * 0.3:
        print(f"⚠️ Warning: {skipped_batches} batches skipped")
    
    avg_loss = epoch_loss / num_batches
    
    print(f"\n[Epoch {epoch}] Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s | Skipped: {skipped_batches}")
    if epoch % 5 == 0 or epoch == 1:
        print("  Loss components:")
        for k, v in loss_components.items():
            print(f"    {k}: {v/num_batches:.4f}")
    
    return avg_loss


def validate(model, dataloader, device):
    """Validation"""
    model.eval()
    val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None:
                continue
            
            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            try:
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                val_loss += loss.item()
                num_batches += 1
            except:
                continue
    
    if num_batches == 0:
        return 0.0
    
    return val_loss / num_batches


def main(args):
    # ⭐ Device 선택 (안정성 우선)
    if args.cpu:
        device = torch.device("cpu")
        can_use_workers = True
        print(f"🖥️ Using device: CPU (forced)")
    else:
        device, can_use_workers = get_device(args.force_mps)
    
    # ⭐ num_workers 조정
    if not can_use_workers:
        print(f"⚠️ Setting num_workers=0 for device compatibility")
        args.num_workers = 0
    
    transform = transforms.ToTensor()
    
    # 데이터셋 로드
    datasets = []
    
    if args.organ == 'colon' or args.organ == 'both':
        print("\n" + "="*60)
        print("Loading COLON dataset...")
        print("="*60)
        colon_dataset = MedicalFolderDataset(
            image_root=args.image_root,
            label_root=args.label_root,
            organ_type='대장',
            transforms=transform,
            min_area=args.min_area,
            resize=(args.img_size, args.img_size),
            max_samples=args.max_samples
        )
        datasets.append(colon_dataset)
    
    if args.organ == 'stomach' or args.organ == 'both':
        print("\n" + "="*60)
        print("Loading STOMACH dataset...")
        print("="*60)
        stomach_dataset = MedicalFolderDataset(
            image_root=args.image_root,
            label_root=args.label_root,
            organ_type='위',
            transforms=transform,
            min_area=args.min_area,
            resize=(args.img_size, args.img_size),
            max_samples=args.max_samples
        )
        datasets.append(stomach_dataset)
    
    # 데이터셋 합치기
    if len(datasets) > 1:
        print(f"\n🔗 Combining datasets...")
        full_dataset = ConcatDataset(datasets)
    else:
        full_dataset = datasets[0]
    
    # Train/Val split
    if args.val_split > 0:
        train_size = int(len(full_dataset) * (1 - args.val_split))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        print(f"\n📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f"\n📊 Total training samples: {len(train_dataset)}")
    
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🔄 Batches per epoch: ~{len(train_dataset) // args.batch_size}")
    
    # ⭐ DataLoader (안전 설정)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_filter_empty,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_filter_empty,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0)
        )
    
    # num_classes 계산
    if isinstance(full_dataset, ConcatDataset):
        base_dataset = full_dataset.datasets[0]
    else:
        base_dataset = full_dataset
    
    num_classes = len(base_dataset.IDX_TO_CLASS) + 1
    print(f"🎯 Num classes (with background): {num_classes}")
    
    # 모델
    model = get_model(num_classes=num_classes)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=0.1
    )
    
    # 학습
    print("\n" + "="*60)
    print("🚀 Starting training...")
    print("="*60 + "\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_one_epoch(
            model, dataloader, optimizer, device, epoch
        )
        lr_scheduler.step()
        
        # Validation
        if val_loader and epoch % 5 == 0:
            val_loss = validate(model, val_loader, device)
            print(f"  📉 Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
                print(f"  💾 Best model saved!")
        
        # 정기 저장
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"model_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, save_path)
            print(f"  💾 Saved: {save_path}\n")
    
    print("✅ Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 데이터
    parser.add_argument('--image-root', type=str,
                    default='/Users/admin/Downloads/datasets/1.Training/1.원천데이터')
    parser.add_argument('--label-root', type=str,
                    default='/Users/admin/Downloads/datasets/1.Training/2.라벨링데이터')
    parser.add_argument('--organ', type=str, choices=['colon', 'stomach', 'both'],
                    default='colon')
    
    # 속도/안정성
    parser.add_argument('--img-size', type=int, default=384)
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=2)
    
    # 학습
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-step', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-interval', type=int, default=5)
    parser.add_argument('--min-area', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='outputs/medical_seg')
    parser.add_argument('--val-split', type=float, default=0.1)
    
    # ⭐ Device 옵션
    parser.add_argument('--cpu', action='store_true',
                    help='Force use CPU (most stable)')
    parser.add_argument('--force-mps', action='store_true',
                    help='Force use MPS (may crash with Mask R-CNN)')
    
    args = parser.parse_args()
    main(args)
