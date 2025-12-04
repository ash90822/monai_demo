# infer.py
import os
import json
import base64

import cv2
import torch
import pandas as pd
import numpy as np
import pydicom

from models.basic_unet import get_basic_unet
from transforms.basic_transforms import HETransformd
from monai.transforms import Compose, EnsureTyped, Resized
from configs.config import CFG

# === 類別設定：index 0 是背景，只用來做 label_map，不會輸出到 JSON shapes ===
LABEL_NAMES = CFG.classes.label_names
IMG_SIZE = CFG.data.img_size

# ---------------------------------------------------
# 1. 影像編碼成 Base64 PNG（寫入 LabelMe imageData）
# ---------------------------------------------------
def dicom_to_base64_png(dcm_path, 
                        window_center=CFG.infer.window_center, 
                        window_width=CFG.infer.window_width
                        ):
    """
    將「原始 DICOM」轉成 PNG Base64，用於 LabelMe imageData。
    如果有提供 window_center / window_width，就套用 windowing。
    """
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)

    # MONOCHROME1 要反轉
    if hasattr(ds, "PhotometricInterpretation"):
        if ds.PhotometricInterpretation == "MONOCHROME1":
            img = np.max(img) - img

    # Windowing（例如 CT 常用 40/400，可在呼叫時指定）
    if window_center is not None and window_width is not None:
        c = float(window_center)
        w = float(window_width)
        img = np.clip(img, c - w / 2, c + w / 2)

    # normalize 到 0~255
    img = img - img.min()
    img = img / (img.max() + 1e-6)
    img_u8 = (img * 255).astype(np.uint8)

    success, encoded = cv2.imencode(".png", img_u8)
    if not success:
        raise RuntimeError("PNG encode failed.")
    b64 = base64.b64encode(encoded).decode("utf-8")
    return b64



# ---------------------------------------------------
# 2. DICOM → HU
# ---------------------------------------------------
def dicom_to_hu(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    slope = float(ds.get("RescaleSlope", 1))
    intercept = float(ds.get("RescaleIntercept", 0))
    hu = img * slope + intercept

    # MONOCHROME1 需要反轉
    if hasattr(ds, "PhotometricInterpretation"):
        if ds.PhotometricInterpretation == "MONOCHROME1":
            hu = np.max(hu) - hu

    return hu.astype(np.float32)  # (H,W)


# ---------------------------------------------------
# 3. 讀 DICOM + HE + resize → (1,IMG_SIZE,IMG_SIZE) Tensor
# ---------------------------------------------------
def load_dicom_with_he(path, img_size=IMG_SIZE):
    """
    讀取 DICOM，做：
      HU → HETransformd (body mask + percentile + HE) → resize
    回傳: torch.Tensor, shape = (1, H, W), 值 0~1
    """
    hu = dicom_to_hu(path)  # (H,W)

    he_params = dict(
        pmin=1.0,
        pmax=99.0,
        use_body_mask=True,
        body_hu_thresh=-300,
    )

    infer_trans = Compose([
        HETransformd(keys=("image",), **he_params),
        Resized(
            keys=("image",),
            spatial_size=(img_size, img_size),
            mode=("bilinear",),
        ),
        EnsureTyped(keys=("image",)),
    ])

    data = {"image": hu[np.newaxis, ...]}  # (1,H,W)
    data = infer_trans(data)

    # data["image"]: torch.Tensor or MetaTensor, shape (1,H,W)
    img_t = data["image"]  # (1,H,W)
    return img_t


# ---------------------------------------------------
# 4. 載入模型
# ---------------------------------------------------
def load_model(ckpt_path, device="cpu"):
    """載入訓練好的 UNet 模型權重（multi-class）"""
    model = get_basic_unet()  # out_channels 應該已設定為 3
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------
# 5. 模型推論 → label_map (H,W) & 前處理後影像 (H,W) 0~1
# ---------------------------------------------------
def model_inference_to_label_map(model, dcm_path, device="cpu"):
    """
    輸出:
      label_map: (H,W) 內含 0/1/2
      image_np:  (H,W) 0~1，用來寫入 imageData
    """
    # 讀 DICOM + HE + resize，得到 (1,H,W) Tensor
    img_t = load_dicom_with_he(dcm_path, img_size=IMG_SIZE)  # (1,H,W)
    image_np = img_t.cpu().numpy()[0]                        # (H,W), 0~1

    # 增加 batch 維度 (B,C,H,W)
    img_t = img_t.unsqueeze(0).to(device)  # (1,1,H,W)

    with torch.no_grad():
        logits = model(img_t)              # (1,3,H,W)
        probs = torch.softmax(logits, dim=1)

    probs_np = probs.cpu().numpy()[0]      # (3,H,W)
    label_map = probs_np.argmax(axis=0).astype(np.uint8)  # (H,W)，0/1/2

    return label_map, image_np


# ---------------------------------------------------
# 6. label_map → LabelMe JSON（不包含背景）
# ---------------------------------------------------
def label_map_to_labelme_json(label_map, image_np, img_path, label_names):
    """
    label_map: (H,W)，0=背景, 1/2=器官
    image_np:  (H,W) 0~1，拿來編碼成 imageData
    產生 LabelMe JSON dict，**只包含器官類別的 shapes（不含背景）**
    """
    h, w = label_map.shape
    shapes = []

    # 這裡刻意從 class_id=1 開始，背景 0 完全不建 shape
    for class_id in range(1, len(label_names)):
        class_name = label_names[class_id]

        binary = (label_map == class_id).astype(np.uint8) * 255
        if binary.max() == 0:
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cnt.shape[0] < 3:
                continue

            # 若要濾掉小區域，可打開以下兩行：
            # area = cv2.contourArea(cnt)
            # if area < 50:
            #     continue

            pts = cnt.squeeze(1).astype(float)  # (N,2)
            points = pts.tolist()               # [[x1,y1], [x2,y2], ...]

            shapes.append({
                "label": class_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            })

    imageData = dicom_to_base64_png(img_path)

    json_dict = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,                         # 只會有 liver / spleen，沒有 bg
        "imagePath": os.path.basename(img_path),  # 這裡保留原 DICOM 名稱
        "imageData": imageData,
        "imageHeight": int(h),
        "imageWidth": int(w),
    }

    return json_dict


# ---------------------------------------------------
# 7. main：讀 test.csv 的 dcm_path 逐張做推論
# ---------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device =", device)

    model = load_model(CFG.paths.checkpoint, device=device)

    df = pd.read_csv(CFG.paths.test_csv)
    if "dcm_path" not in df.columns:
        raise ValueError("test.csv 必須包含 dcm_path 欄位")

    os.makedirs(CFG.paths.output_dir, exist_ok=True)

    for idx, row in df.iterrows():
        dcm_path = row["dcm_path"]
        print(f"[{idx+1}/{len(df)}] Infer: {dcm_path}")

        label_map, image_np = model_inference_to_label_map(model, dcm_path, device=device)

        json_dict = label_map_to_labelme_json(label_map, image_np, dcm_path, LABEL_NAMES)

        base_name = os.path.splitext(os.path.basename(dcm_path))[0]
        json_path = os.path.join(CFG.paths.output_dir, f"{base_name}.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=2)

        print(f"  → LabelMe JSON saved to: {json_path}")

    print("Inference completed.")
