# utils/io.py
import numpy as np
import pydicom
import cv2

def load_image_single(path):
    """
    回傳 numpy array: (1, H, W)
    自動處理：
    - DICOM pixel_array
    - OpenCV (PNG/JPEG)
    - normalize to 0~1
    """
    if path.lower().endswith(".dcm"):
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)

        # normalize to 0~1
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = img / 255.0

    return img[np.newaxis, ...]
