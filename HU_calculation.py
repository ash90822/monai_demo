# HU_calculation.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd

# 直接重用你現有的 dicom_to_hu（建議你也可把它集中到 infer_service.py 再 import）
from infer_service import dicom_to_hu


@dataclass
class HUStatsResult:
    csv_path: Path
    df: pd.DataFrame
    matched: int
    missing_json: List[str]
    missing_dcm: List[str]


def _base_key(p: Path) -> str:
    """以檔名（去副檔名）作為對齊 key：xxx.dcm <-> xxx.json"""
    return p.stem


def poly_to_mask(h: int, w: int, points: List[List[float]]) -> np.ndarray:
    """LabelMe polygon points -> bool mask (H,W)"""
    pts = np.array(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return np.zeros((h, w), dtype=bool)

    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def compute_hu_stats_from_labelme(
    dicom_dir: str | Path,
    mask_dir: str | Path,
    out_csv_name: str = "hu_stats.csv",
    stop_flag: Optional[callable] = None,     # stop_flag(): bool
    progress_cb: Optional[callable] = None,   # progress_cb(i, total)
    log_cb: Optional[callable] = None,        # log_cb(msg)
) -> HUStatsResult:
    """
    - dicom_dir: 你 Select Folder 的 DICOM 根目錄（會遞迴找 *.dcm）
    - mask_dir:  output_dir（放 LabelMe json 的資料夾）
    - 輸出 csv 到 mask_dir / out_csv_name
    - stop_flag: GUI Stop 用（回傳 True 就中斷）
    """
    dicom_dir = Path(dicom_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    dcm_list = sorted(dicom_dir.rglob("*.dcm"))
    json_list = sorted(mask_dir.glob("*.json"))

    dcm_map: Dict[str, Path] = {_base_key(p): p for p in dcm_list}
    json_map: Dict[str, Path] = {_base_key(p): p for p in json_list}

    common = sorted(set(dcm_map.keys()) & set(json_map.keys()))
    missing_json = sorted(set(dcm_map.keys()) - set(json_map.keys()))
    missing_dcm = sorted(set(json_map.keys()) - set(dcm_map.keys()))

    if log_cb:
        log_cb(f"[HU] DICOM={len(dcm_list)}, JSON(mask)={len(json_list)}, matched={len(common)}")
        if missing_json[:5]:
            log_cb("[HU] Missing JSON examples: " + ", ".join(missing_json[:5]))
        if missing_dcm[:5]:
            log_cb("[HU] Missing DICOM examples: " + ", ".join(missing_dcm[:5]))

    if len(common) == 0:
        raise RuntimeError("No matched DICOM/JSON by filename. Check naming and folders.")

    rows: List[Dict[str, Any]] = []
    total = len(common)

    for i, k in enumerate(common, start=1):
        if stop_flag and stop_flag():
            if log_cb:
                log_cb("[HU] Stopped by user.")
            break

        dcm_path = dcm_map[k]
        json_path = json_map[k]

        hu = dicom_to_hu(str(dcm_path))  # (H,W)
        H, W = hu.shape

        with open(json_path, "r", encoding="utf-8") as f:
            jd = json.load(f)

        shapes = jd.get("shapes", [])
        row: Dict[str, Any] = {
            "case": k,
            "dcm_path": str(dcm_path),
            "json_path": str(json_path),
        }

        # 沒有 shape 也寫一列
        if not shapes:
            rows.append(row)
            if progress_cb:
                progress_cb(i, total)
            continue

        label_masks: Dict[str, np.ndarray] = {}
        for sh in shapes:
            label = sh.get("label", "unknown")
            pts = sh.get("points", [])
            m = poly_to_mask(H, W, pts)
            if label in label_masks:
                label_masks[label] = (label_masks[label] | m)
            else:
                label_masks[label] = m

        for label, m in label_masks.items():
            vals = hu[m]
            if vals.size == 0:
                continue
            row[f"{label}_n"] = int(vals.size)
            row[f"{label}_mean"] = float(np.mean(vals))
            row[f"{label}_std"] = float(np.std(vals))
            row[f"{label}_median"] = float(np.median(vals))
            row[f"{label}_p05"] = float(np.percentile(vals, 5))
            row[f"{label}_p95"] = float(np.percentile(vals, 95))

        rows.append(row)

        if progress_cb:
            progress_cb(i, total)

    df = pd.DataFrame(rows)
    csv_path = mask_dir / out_csv_name
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    if log_cb:
        log_cb(f"[HU] Saved CSV: {csv_path}")

    return HUStatsResult(
        csv_path=csv_path,
        df=df,
        matched=len(common),
        missing_json=missing_json,
        missing_dcm=missing_dcm,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--out_csv", default="hu_stats.csv")
    args = parser.parse_args()

    def _log(x): print(x, flush=True)

    compute_hu_stats_from_labelme(
        args.dicom_dir,
        args.mask_dir,
        out_csv_name=args.out_csv,
        log_cb=_log
    )
