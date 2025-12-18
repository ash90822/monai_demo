import os
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import torch
import pydicom

from models.basic_unet import get_basic_nnUnet
from transforms.basic_transforms import HETransformd
from monai.transforms import Compose, EnsureTyped, Resized
from configs.config import CFG

LABEL_NAMES = CFG.classes.label_names
IMG_SIZE = CFG.data.img_size


def dicom_to_base64_png(dcm_path, window_center=None, window_width=None):
    ds = pydicom.dcmread(dcm_path)

    img = ds.pixel_array.astype(np.float32)
    slope = float(ds.get("RescaleSlope", 1.0))
    intercept = float(ds.get("RescaleIntercept", 0.0))
    hu = img * slope + intercept

    if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
        hu = np.max(hu) - hu

    wc = window_center
    ww = window_width

    if wc is None or ww is None:
        dicom_wc = ds.get("WindowCenter", None)
        dicom_ww = ds.get("WindowWidth", None)
        if dicom_wc is not None and dicom_ww is not None:
            try:
                wc = float(dicom_wc[0]) if hasattr(dicom_wc, "__len__") else float(dicom_wc)
                ww = float(dicom_ww[0]) if hasattr(dicom_ww, "__len__") else float(dicom_ww)
            except Exception:
                wc = float(dicom_wc)
                ww = float(dicom_ww)
        else:
            lo = np.percentile(hu, 1.0)
            hi = np.percentile(hu, 99.0)
            wc = (lo + hi) / 2.0
            ww = (hi - lo)

    if ww <= 0:
        ww = 1.0

    low = wc - ww / 2.0
    high = wc + ww / 2.0

    hu_clipped = np.clip(hu, low, high)
    img01 = (hu_clipped - low) / (high - low + 1e-6)
    img_u8 = (np.clip(img01, 0.0, 1.0) * 255).astype(np.uint8)

    ok, encoded = cv2.imencode(".png", img_u8)
    if not ok:
        raise RuntimeError("PNG encode failed.")
    return base64.b64encode(encoded).decode("utf-8")


def dicom_to_hu(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    slope = float(ds.get("RescaleSlope", 1))
    intercept = float(ds.get("RescaleIntercept", 0))
    hu = img * slope + intercept

    if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
        hu = np.max(hu) - hu

    return hu.astype(np.float32)


def model_inference_to_label_map(model, dcm_path: str, device: str = "cpu"):
    hu = dicom_to_hu(dcm_path)
    orig_h, orig_w = hu.shape

    he_params = dict(
        pmin=CFG.data.pmin,
        pmax=CFG.data.pmax,
        use_body_mask=CFG.data.use_body_mask,
        body_hu_thresh=CFG.data.body_hu_thresh,
    )

    infer_trans = Compose([
        HETransformd(keys=("image",), **he_params),
        Resized(keys=("image",), spatial_size=(IMG_SIZE, IMG_SIZE), mode=("bilinear",)),
        EnsureTyped(keys=("image",)),
    ])

    data = {"image": hu[np.newaxis, ...]}
    data = infer_trans(data)
    img_t = data["image"]                 # (1,H,W)
    image_np = img_t.cpu().numpy()[0]     # (H,W)

    img_t = img_t.unsqueeze(0).to(device) # (1,1,H,W)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)

    label_map = probs.cpu().numpy()[0].argmax(axis=0).astype(np.uint8)
    return label_map, image_np, orig_h, orig_w


def label_map_to_labelme_json(label_map, img_path, label_names, orig_h, orig_w):
    h_resized, w_resized = label_map.shape
    shapes = []
    scale_y = orig_h / h_resized
    scale_x = orig_w / w_resized

    for class_id in range(1, len(label_names)):
        class_name = label_names[class_id]
        binary = (label_map == class_id).astype(np.uint8) * 255
        if binary.max() == 0:
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cnt.shape[0] < 3:
                continue
            pts = cnt.squeeze(1).astype(float)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            shapes.append({
                "label": class_name,
                "points": pts.tolist(),
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            })

    imageData = dicom_to_base64_png(
        img_path,
        window_center=CFG.infer.window_center,
        window_width=CFG.infer.window_width
    )

    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(img_path),
        "imageData": imageData,
        "imageHeight": int(orig_h),
        "imageWidth": int(orig_w),
    }


class ModelService:
    """
    GUI/CLI 共用：load_model() + predict_to_labelme()
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_loaded = False

        self.ckpt_path = str(CFG.paths.checkpoint)
        self.output_dir = Path(CFG.paths.output_dir)

    def load_model(self, ckpt_path: Optional[str] = None):
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        model = get_basic_nnUnet()
        state = torch.load(self.ckpt_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()

        self.model = model
        self.is_loaded = True

    def predict_to_labelme(self, dcm_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded.")

        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        label_map, image_np, orig_h, orig_w = model_inference_to_label_map(
            self.model, dcm_path, device=self.device
        )

        json_dict = label_map_to_labelme_json(
            label_map, dcm_path, LABEL_NAMES, orig_h, orig_w
        )

        base_name = os.path.splitext(os.path.basename(dcm_path))[0]
        json_path = out_dir / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=2)

        return {
            "dcm_path": dcm_path,
            "json_path": str(json_path),
            "num_shapes": len(json_dict.get("shapes", [])),
            "device": self.device,
        }
