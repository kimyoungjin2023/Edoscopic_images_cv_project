import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# Detection Losses
# =====================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (N, C)
        targets: (N, C) one-hot
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class IoULoss(nn.Module):
    """
    IoU Loss (can be extended to GIoU/DIoU/CIoU)
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        pred, target: (N, 4) -> [x1, y1, x2, y2]
        """
        iou = self.iou(pred, target)
        return 1.0 - iou.mean()

    @staticmethod
    def iou(box1, box2):
        x1 = torch.max(box1[:, 0], box2[:, 0])
        y1 = torch.max(box1[:, 1], box2[:, 1])
        x2 = torch.min(box1[:, 2], box2[:, 2])
        y2 = torch.min(box1[:, 3], box2[:, 3])

        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        union = area1 + area2 - inter + 1e-6
        return inter / union


# =====================================================
# Segmentation Losses
# =====================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for binary or multi-class segmentation
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        preds: (N, C, H, W) logits
        targets: (N, C, H, W) one-hot
        """
        preds = torch.sigmoid(preds)

        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            preds.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()


class SegLoss(nn.Module):
    """
    BCE + Dice (most commonly used)
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        dice = self.dice(preds, targets)
        return self.bce_weight * bce + self.dice_weight * dice


# =====================================================
# Multi-task Total Loss
# =====================================================

class MultiTaskLoss(nn.Module):
    """
    Detection + Segmentation Loss
    """
    def __init__(self,
                 cls_weight=1.0,
                 box_weight=5.0,
                 seg_weight=1.0):
        super().__init__()

        self.cls_loss = FocalLoss()
        self.box_loss = IoULoss()
        self.seg_loss = SegLoss()

        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.seg_weight = seg_weight

    def forward(self, outputs, targets):
        """
        outputs:
            {
                "cls_logits": (N, C),
                "boxes": (N, 4),
                "masks": (N, C, H, W)
            }

        targets:
            {
                "labels": (N, C),
                "boxes": (N, 4),
                "masks": (N, C, H, W)
            }
        """
        loss_dict = {}

        cls_loss = self.cls_loss(
            outputs["cls_logits"], targets["labels"]
        )
        box_loss = self.box_loss(
            outputs["boxes"], targets["boxes"]
        )
        seg_loss = self.seg_loss(
            outputs["masks"], targets["masks"]
        )

        total_loss = (
            self.cls_weight * cls_loss +
            self.box_weight * box_loss +
            self.seg_weight * seg_loss
        )

        loss_dict["cls_loss"] = cls_loss
        loss_dict["box_loss"] = box_loss
        loss_dict["seg_loss"] = seg_loss
        loss_dict["total_loss"] = total_loss

        return total_loss, loss_dict
