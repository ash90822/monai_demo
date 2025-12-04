import torch
from monai.metrics import DiceMetric
from monai.metrics import MeanIoU

dice_metric = DiceMetric(include_background=True, reduction="mean")
iou_metric = MeanIoU(include_background=True, reduction="mean")

def compute_metrics(pred, mask):
    """
    pred: (B, 1, H, W) or (B, C, H, W)
    mask: (B, 1, H, W) or (B, C, H, W)
    """
    pred_bin = (pred > 0.5).float()

    dice_tensor = dice_metric(pred_bin, mask)  # 可能 shape = (B,) 或 (B, C)
    iou_tensor  = iou_metric(pred_bin, mask)

    dice = dice_tensor.mean().item()
    iou  = iou_tensor.mean().item()

    return dice, iou


def compute_per_class_dice(logits, masks, num_classes, threshold=0.5):
    """
    logits: (B, C, H, W) 來自 model(x)
    masks:  (B, 1, H, W) 或 (B, H, W)，整數 label (0 ~ C-1)
    num_classes: 總類別數（含背景）
    回傳：
        mean_dice: float
        dice_per_class: tensor, shape (num_classes,)
    """

    # ---- 1. 轉成預測類別 pred_label (B, H, W) ----
    if logits.shape[1] == 1 and num_classes == 2:
        # 二類：用 sigmoid + threshold
        probs_fg = torch.sigmoid(logits)
        pred_label = (probs_fg > threshold).long().squeeze(1)  # (B,H,W), 0/1
    else:
        # 多類：softmax + argmax
        probs = torch.softmax(logits, dim=1)
        pred_label = probs.argmax(dim=1)  # (B,H,W)

    # ---- 2. 把 GT 壓成 (B, H, W) ----
    if masks.ndim == 4:
        gt = masks.squeeze(1).long()
    else:
        gt = masks.long()

    # ---- 3. per-class dice 計算 ----
    eps = 1e-6
    dice_numer = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    dice_denom = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)

    for c in range(num_classes):
        pred_c = (pred_label == c)
        gt_c   = (gt == c)

        intersection = (pred_c & gt_c).sum()
        union        = pred_c.sum() + gt_c.sum()

        dice_numer[c] += 2.0 * intersection
        dice_denom[c] += union + eps

    dice_per_class = dice_numer / dice_denom
    mean_dice = dice_per_class.mean().item()

    return mean_dice, dice_per_class