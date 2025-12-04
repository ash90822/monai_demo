# datasets/basic_dataset.py
import os
import json
import numpy as np
import cv2
import pydicom
from torch.utils.data import Dataset
from configs.config import CFG


class ChestCTDataset(Dataset):
    def __init__(
        self,
        df,
        img_col="dcm_path",
        json_col="json_path",
        transforms=None,
    ):
        """
        df: DataFrame，至少包含：
            - img_col: DICOM 路徑（預設 dcm_path）
            - json_col: LabelMe JSON 路徑（多器官標註）
        transforms: MONAI dict transforms（會吃 {"image": ..., "mask": ...}）
        """
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.json_col = json_col
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    # ---------- 1. DICOM → HU ----------

    @staticmethod
    def _dicom_to_hu(ds):
        """pixel_array -> HU"""
        img = ds.pixel_array.astype(np.float32)
        slope = float(ds.get("RescaleSlope", 1))
        intercept = float(ds.get("RescaleIntercept", 0))
        return img * slope + intercept

    def load_dicom_hu(self, path):
        ds = pydicom.dcmread(path)
        hu = self._dicom_to_hu(ds)

        # MONOCHROME1 反轉
        if hasattr(ds, "PhotometricInterpretation"):
            if ds.PhotometricInterpretation == "MONOCHROME1":
                hu = np.max(hu) - hu

        return hu.astype(np.float32)  # (H,W) in HU

    # ---------- 2. JSON → multi-class mask ----------

    def json_to_mask(self, json_path, image_shape):
        """
        將 LabelMe JSON 轉成 mask (H,W)，整數類別：
        0=背景, 1=器官1, 2=器官2, ...
        """
        mask = np.zeros(image_shape, dtype=np.uint8)

        if (json_path is None) or (not os.path.exists(json_path)):
            return mask

        with open(json_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        # TODO: 這裡替換成你實際的 label 名稱
        LABEL_MAP = CFG.classes.json_label_map

        for shape in ann.get("shapes", []):
            label_name = shape.get("label", None)
            if label_name not in LABEL_MAP:
                continue
            class_id = LABEL_MAP[label_name]

            pts = shape.get("points", [])
            if not pts or len(pts) < 3:
                continue

            pts = np.array(pts, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2:
                continue

            pts = np.round(pts).astype(np.int32)
            if pts.shape[0] < 3:
                continue

            cv2.fillPoly(mask, [pts], class_id)

        return mask

    # ---------- 3. __getitem__ ----------

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dcm_path = row[self.img_col]
        json_path = row.get(self.json_col, None)

        # HU image
        hu = self.load_dicom_hu(dcm_path)                # (H,W) float32
        mask = self.json_to_mask(json_path, hu.shape)    # (H,W) uint8

        data = {
            "image": hu[np.newaxis, ...],   # (1,H,W)
            "mask":  mask[np.newaxis, ...], # (1,H,W)
        }

        if self.transforms is not None:
            data = self.transforms(data)

        # transforms 後通常變成 tensor
        return data["image"], data["mask"]
