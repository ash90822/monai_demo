# transforms/basic_transforms.py
import numpy as np
import cv2
from configs.config import CFG

from monai.transforms import (
    MapTransform,
    Compose,
    EnsureTyped,
    Resized,
    RandFlipd,
    RandAffined,
)


class HETransformd(MapTransform):
    """
    對 "image" 做：
      HU → (可選 body mask) → 分位數縮放 → HE → 0~1 float
    假設輸入 image shape: (1,H,W) 或 (H,W)，值為 HU。
    """

    def __init__(
        self,
        keys=("image",),
        pmin=1.0,
        pmax=99.0,
        use_body_mask=True,
        body_hu_thresh=-300,
    ):
        super().__init__(keys)
        self.pmin = pmin
        self.pmax = pmax
        self.use_body_mask = use_body_mask
        self.body_hu_thresh = body_hu_thresh

    def _make_body_mask(self, hu):
        m = (hu > self.body_hu_thresh).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(m)
        if num_labels <= 1:
            return m.astype(bool)
        areas = [(labels == lbl).sum() for lbl in range(1, num_labels)]
        lbl_max = int(np.argmax(areas)) + 1
        return labels == lbl_max

    def _hu_to_unit(self, hu, mask=None):
        if mask is not None and mask.any():
            roi = hu[mask]
        else:
            roi = hu

        lo = np.percentile(roi, self.pmin)
        hi = np.percentile(roi, self.pmax)
        if hi <= lo:
            hi = lo + 1.0
        x = (hu - lo) / (hi - lo)
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _apply_he(unit01):
        img_u8 = (np.clip(unit01, 0, 1) * 255).astype(np.uint8)
        he = cv2.equalizeHist(img_u8)
        return he.astype(np.float32) / 255.0

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]  # (1,H,W) 或 (H,W)

            if img.ndim == 3 and img.shape[0] == 1:
                hu = img[0]
            elif img.ndim == 2:
                hu = img
            else:
                raise RuntimeError(
                    f"HETransformd expects (1,H,W) or (H,W), got {img.shape}"
                )

            body_mask = None
            if self.use_body_mask:
                body_mask = self._make_body_mask(hu)

            unit01 = self._hu_to_unit(hu, mask=body_mask)
            he = self._apply_he(unit01)

            if body_mask is not None:
                he = he * body_mask.astype(np.float32)

            d[key] = he[np.newaxis, ...].astype(np.float32)  # (1,H,W)

        return d


def get_transforms(img_size=None):
    """
    回傳 train_trans, val_trans：

    train_trans:
      1) HETransformd: HU → [0,1] + HE
      2) Resized:      resize 到 (img_size, img_size)
      3) RandFlipd:    垂直翻轉 (prob=0.5)  ≈ imgaug.Flipud(0.5)
      4) RandFlipd:    水平翻轉 (prob=0.5)  ≈ imgaug.Fliplr(0.5)
      5) RandAffined:  rotate/shear/scale   ≈ imgaug.Affine(...)
      6) EnsureTyped

    val_trans:
      1) HETransformd
      2) Resized
      3) EnsureTyped
    """
    if img_size is None:
        img_size = CFG.data.img_size

    he_params = dict(
        pmin=CFG.data.pmin,
        pmax=CFG.data.pmax,
        use_body_mask=CFG.data.use_body_mask,
        body_hu_thresh=CFG.data.body_hu_thresh,
    )

    # 將角度轉成 rad：
    # imgaug: rotate=(-45,45)  degrees → (-π/4, π/4)
    rotate_min = -45.0 * np.pi / 180.0
    rotate_max = 45.0 * np.pi / 180.0

    # imgaug: shear=(-16,16) degrees → 約 (-0.28, 0.28) rad
    shear_min = -16.0 * np.pi / 180.0
    shear_max = 16.0 * np.pi / 180.0

    train_trans = Compose([
        # 1. HU -> HE (0~1)
        HETransformd(keys=("image",), **he_params),

        # 2. resize 到固定大小（和你原本 imgaug 的作法一致：先 resize 再增強）
        Resized(
            keys=("image", "mask"),
            spatial_size=(img_size, img_size),
            mode=("bilinear", "nearest"),
        ),

        # 3. 垂直翻轉（對應 imgaug.Flipud(0.5)）
        RandFlipd(
            keys=("image", "mask"),
            spatial_axis=0,   # 0 = up-down
            prob=0.5,
        ),

        # 4. 水平翻轉（對應 imgaug.Fliplr(0.5)）
        RandFlipd(
            keys=("image", "mask"),
            spatial_axis=1,   # 1 = left-right
            prob=0.5,
        ),

        # 5. Affine：rotate + shear + scale（不做平移）
        RandAffined(
            keys=("image", "mask"),
            prob=1.0,  # 原本 imgaug.Affine 是一定會做，只是參數隨機
            rotate_range=(rotate_min, rotate_max),  # 約 (-45°, 45°)
            shear_range=(shear_min, shear_max),     # 約 (-16°, 16°)
            scale_range=(0.8, 1.2),                 # x,y 皆在 0.8~1.2
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),

        # 6. 轉成 Torch Tensor / MetaTensor
        EnsureTyped(keys=("image", "mask")),
    ])

    val_trans = Compose([
        HETransformd(keys=("image",), **he_params),
        Resized(
            keys=("image", "mask"),
            spatial_size=(img_size, img_size),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=("image", "mask")),
    ])

    return train_trans, val_trans
