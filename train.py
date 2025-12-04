# train.py
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from monai.losses import DiceLoss

from datasets.basic_dataset import ChestCTDataset
from transforms.basic_transforms import get_transforms
from configs.config import CFG


NUM_CLASSES = CFG.classes.num_classes
LABEL_NAMES = CFG.classes.label_names


# ---------- metrics: per-class Dice ----------

def compute_per_class_dice(logits, masks, num_classes=NUM_CLASSES):
    """
    logits: (B,C,H,W)
    masks:  (B,1,H,W) 或 (B,H,W)，整數 label (0..C-1)
    回傳:
        mean_dice: float
        dice_per_class: (C,) tensor
    """
    if logits.shape[1] == 1 and num_classes == 2:
        probs_fg = torch.sigmoid(logits)
        pred_label = (probs_fg > 0.5).long().squeeze(1)
    else:
        probs = torch.softmax(logits, dim=1)
        pred_label = probs.argmax(dim=1)  # (B,H,W)

    if masks.ndim == 4:
        gt = masks.squeeze(1).long()
    else:
        gt = masks.long()

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


def evaluate(model, val_loader, device, num_classes=NUM_CLASSES):
    model.eval()
    dice_sum = torch.zeros(num_classes, dtype=torch.float32, device=device)
    dice_cnt = torch.zeros(num_classes, dtype=torch.float32, device=device)
    eps = 1e-6

    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)

            logits = model(img)
            _, dice_per_class = compute_per_class_dice(logits, mask, num_classes)

            dice_sum += dice_per_class
            dice_cnt += torch.ones_like(dice_per_class)

    dice_mean = dice_sum / (dice_cnt + eps)
    mean_dice = dice_mean.mean().item()
    return mean_dice, dice_mean


# ---------- 訓練主程式 ----------

def train(csv_path: str | None = None, epochs = CFG.train.epochs):
    # 1. 讀資料 & 切 train/val
    if csv_path is None:
        csv_path = CFG.paths.train_csv
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 2. 建立 transforms
    train_trans, val_trans = get_transforms()

    # 3. 建立 Dataset / DataLoader
    train_ds = ChestCTDataset(train_df, transforms=train_trans)
    val_ds   = ChestCTDataset(val_df,   transforms=val_trans)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.train.batch_size,
        shuffle=True,
        num_workers=CFG.train.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.train.batch_size,
        shuffle=False,
        num_workers=CFG.train.num_workers,
    )

    # 4. 設定 device & model & loss & optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device =", device)

    from models.basic_unet import get_basic_unet
    model = get_basic_unet().to(device)

    loss_fn = DiceLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.train.lr)
    ckpt_path = CFG.paths.checkpoint
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    best_val_dice = -1.0
    best_epoch = -1
    no_improve_epochs = 0

    # 5. training loop + early stopping
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        model.train()
        running_loss = 0.0

        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)

            logits = model(img)
            loss = loss_fn(logits, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_loader))
        print(f"Train loss: {avg_train_loss:.4f}")

        # ----- Validation -----
        val_mean_dice, val_dice_per_class = evaluate(model, val_loader, device, NUM_CLASSES)
        print(f"Val mean Dice: {val_mean_dice:.4f}")
        for c, d in enumerate(val_dice_per_class.tolist()):
            print(f"  - {LABEL_NAMES[c]}: Dice = {d:.4f}")

        # ----- Early Stopping -----
        if val_mean_dice > best_val_dice + CFG.train.min_delta:
            best_val_dice = val_mean_dice
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → New best model saved (epoch {epoch}, Dice={best_val_dice:.4f})")
        else:
            no_improve_epochs += 1
            print(f"  No improvement for {no_improve_epochs} epoch(s).")

        if no_improve_epochs >= CFG.train.patience:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch = {best_epoch}, Dice={best_val_dice:.4f}")
            break

    # 6. 訓練結束後，用 best model 再算一次完整 per-class Dice
    print("\nLoading best model and computing final per-class Dice on validation set...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    final_mean_dice, final_dice_per_class = evaluate(model, val_loader, device, NUM_CLASSES)
    print(f"Final Val mean Dice: {final_mean_dice:.4f}")
    for c, d in enumerate(final_dice_per_class.tolist()):
        print(f"  - {LABEL_NAMES[c]}: Dice = {d:.4f}")


if __name__ == "__main__":
    train()
