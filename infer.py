# infer.py
import os
import json
import base64

import cv2
import torch
import pandas as pd
import numpy as np
import pydicom

from models.basic_unet import get_basic_nnUnet
from transforms.basic_transforms import HETransformd
from monai.transforms import Compose, EnsureTyped, Resized
from configs.config import CFG

LABEL_NAMES = CFG.classes.label_names
IMG_SIZE = CFG.data.img_size

# ---------------------------------------------------
# 1. 影像編碼成 Base64 PNG（寫入 LabelMe imageData）
# ---------------------------------------------------
def dicom_to_base64_png(
    dcm_path,
    window_center=None,
    window_width=None,
):
    """
    將「原始 DICOM（轉 HU）」依 window_center / window_width 做一次 windowing，
    再線性映射到 0~255，輸出 PNG base64，給 LabelMe 的 imageData 用。

    若沒有手動給 window_center / window_width：
      1) 優先使用 DICOM header 的 WindowCenter / WindowWidth
      2) 若 header 沒有，就用 HU 的分位數做個穩健 window（例如 1% ~ 99%）
    """
    ds = pydicom.dcmread(dcm_path)

    # ---- 1) 轉 HU ----
    img = ds.pixel_array.astype(np.float32)
    slope = float(ds.get("RescaleSlope", 1.0))
    intercept = float(ds.get("RescaleIntercept", 0.0))
    hu = img * slope + intercept

    # MONOCHROME1 要反轉
    if hasattr(ds, "PhotometricInterpretation"):
        if ds.PhotometricInterpretation == "MONOCHROME1":
            hu = np.max(hu) - hu

    # ---- 2) 取得 window ----
    wc = window_center
    ww = window_width

    # 若沒手動指定，就試著從 DICOM header 拿
    if wc is None or ww is None:
        # DICOM 的 WindowCenter / WindowWidth 可能是 Multi-valued
        dicom_wc = ds.get("WindowCenter", None)
        dicom_ww = ds.get("WindowWidth", None)

        if dicom_wc is not None and dicom_ww is not None:
            # 可能是 pydicom 的 MultiValue，取第一個
            try:
                wc = float(dicom_wc[0]) if hasattr(dicom_wc, "__len__") else float(dicom_wc)
                ww = float(dicom_ww[0]) if hasattr(dicom_ww, "__len__") else float(dicom_ww)
            except Exception:
                wc = float(dicom_wc)
                ww = float(dicom_ww)
        else:
            # 如果 header 也沒給，就用分位數做一個「自動 window」
            # 例如 1% ~ 99% 或 5% ~ 95%（可調）
            pmin, pmax = 1.0, 99.0
            lo = np.percentile(hu, pmin)
            hi = np.percentile(hu, pmax)
            wc = (lo + hi) / 2.0
            ww = (hi - lo)

    # ---- 3) 根據 window 做一次 clip + 線性映射 ----
    if ww <= 0:
        ww = 1.0

    low = wc - ww / 2.0
    high = wc + ww / 2.0

    hu_clipped = np.clip(hu, low, high)

    # 線性映射到 [0,1]
    img01 = (hu_clipped - low) / (high - low + 1e-6)
    img01 = np.clip(img01, 0.0, 1.0)

    # ---- 4) 轉成 0~255 uint8 再 encode PNG ----
    img_u8 = (img01 * 255).astype(np.uint8)

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
    model = get_basic_nnUnet()  # out_channels 應該已設定為 3
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
      label_map: (H_resize, W_resize) 內含 0/1/2
      image_np:  (H_resize, W_resize) 0~1（HE+resize 後，可拿來 debug）
      orig_h, orig_w: 原始 DICOM 的高寬
    """
    # 先讀一次原始 HU，記錄原本大小
    hu = dicom_to_hu(dcm_path)        # (H_orig, W_orig)
    orig_h, orig_w = hu.shape

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
            spatial_size=(IMG_SIZE, IMG_SIZE),
            mode=("bilinear",),
        ),
        EnsureTyped(keys=("image",)),
    ])

    data = {"image": hu[np.newaxis, ...]}   # (1,H_orig,W_orig)
    data = infer_trans(data)
    img_t = data["image"]                   # (1,H_resize,W_resize)
    image_np = img_t.cpu().numpy()[0]       # (H_resize,W_resize), 0~1

    # 增加 batch 維度 (B,C,H,W)
    img_t = img_t.unsqueeze(0).to(device)   # (1,1,H_resize,W_resize)

    with torch.no_grad():
        logits = model(img_t)               # (1,3,H_resize,W_resize)
        probs = torch.softmax(logits, dim=1)

    probs_np = probs.cpu().numpy()[0]       # (3,H_resize,W_resize)
    label_map = probs_np.argmax(axis=0).astype(np.uint8)

    return label_map, image_np, orig_h, orig_w


# ---------------------------------------------------
# 6. label_map → LabelMe JSON（不包含背景）
# ---------------------------------------------------
def label_map_to_labelme_json(label_map, img_path, label_names, orig_h, orig_w):
    """
    label_map:   (H_resize, W_resize)，0=背景, 1/2=器官
    img_path:    DICOM 路徑（用來產出原圖 imageData）
    label_names: ["bg", "liver", "spleen", ...]
    orig_h, orig_w: 原始 DICOM 影像大小
    """
    h_resized, w_resized = label_map.shape
    shapes = []

    # 從 resize 空間 → 原始空間 的縮放係數
    scale_y = orig_h / h_resized
    scale_x = orig_w / w_resized

    for class_id in range(1, len(label_names)):  # 跳過背景 0
        class_name = label_names[class_id]

        binary = (label_map == class_id).astype(np.uint8) * 255
        if binary.max() == 0:
            continue

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if cnt.shape[0] < 3:
                continue

            # (N,1,2) → (N,2)
            pts = cnt.squeeze(1).astype(float)  # (N,2) in resized space

            # 座標放大回原始空間
            pts[:, 0] *= scale_x   # x 對應 width
            pts[:, 1] *= scale_y   # y 對應 height

            points = pts.tolist()

            shapes.append({
                "label": class_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            })

    # imageData：這裡用原始 DICOM→PNG 的影像
    imageData = dicom_to_base64_png(img_path,
    window_center=CFG.infer.window_center,
    window_width=CFG.infer.window_width
    )

    json_dict = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(img_path),
        "imageData": imageData,
        # 這裡一定要填「原始影像大小」，不是 label_map 的大小
        "imageHeight": int(orig_h),
        "imageWidth": int(orig_w),
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

        # 原本是：label_map, image_np = ...
        label_map, image_np, orig_h, orig_w = model_inference_to_label_map(
            model, dcm_path, device=device
        )

        json_dict = label_map_to_labelme_json(
            label_map,
            dcm_path,
            LABEL_NAMES,
            orig_h,
            orig_w,
        )

        base_name = os.path.splitext(os.path.basename(dcm_path))[0]
        json_path = os.path.join(CFG.paths.output_dir, f"{base_name}.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=2)

        print(f"  → LabelMe JSON saved to: {json_path}")


    print("Inference completed.")
