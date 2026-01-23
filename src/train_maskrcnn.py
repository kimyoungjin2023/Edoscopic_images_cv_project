import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from datasets.coco_dataset import COCODataset


def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(
        in_features, num_classes
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = \
        torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask,
            256,
            num_classes
        )

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = COCODataset(
        img_dir="data/images/train",
        ann_file="data/annotations/train.json",
        transforms=None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    model = get_model(num_classes=2)  # background + lesion
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4
    )

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}")

        torch.save(
            model.state_dict(),
            f"outputs/maskrcnn_epoch{epoch+1}.pth"
        )


if __name__ == "__main__":
    main()
