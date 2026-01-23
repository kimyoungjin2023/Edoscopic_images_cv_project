import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask


class COCODataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        """
        img_dir: 이미지 폴더 경로
        ann_file: COCO annotation json 경로
        """
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # ---------- 이미지 로드 ----------
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # ---------- annotation 로드 ----------
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            # bbox: [x, y, w, h] → [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

            # polygon → binary mask
            rles = coco_mask.frPyObjects(
                ann["segmentation"],
                img_info["height"],
                img_info["width"]
            )
            mask = coco_mask.decode(rles)
            mask = np.any(mask, axis=2) if mask.ndim == 3 else mask
            masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
