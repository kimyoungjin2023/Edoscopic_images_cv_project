import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics import YOLO

from datasets.endoscopy_dataset import EndoscopySegDataset
from models.unet import UNet


def train_unet(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
):
    model.train()
    total_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # 1. YOLOv8 Detection
    # =========================
    yolo_model = YOLO("yolov8n.pt")  # or yolov8s.pt

    yolo_model.train(
        data="configs/yolo_data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="outputs/detection",
        name="yolov8"
    )

    # =========================
    # 2. U-Net Segmentation
    # =========================
    seg_dataset = EndoscopySegDataset(
        image_dir="data/images/train",
        mask_dir="data/masks/train",
        image_size=640
    )

    seg_loader = DataLoader(
        seg_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    unet = UNet(
        in_channels=3,
        out_channels=1
    ).to(device)

    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 50

    for epoch in range(num_epochs):
        loss = train_unet(
            unet,
            seg_loader,
            optimizer,
            criterion,
            device
        )

        print(f"[Epoch {epoch+1}/{num_epochs}] Seg Loss: {loss:.4f}")

        os.makedirs("outputs/segmentation", exist_ok=True)
        torch.save(
            unet.state_dict(),
            f"outputs/segmentation/unet_epoch{epoch+1}.pth"
        )


if __name__ == "__main__":
    main()
