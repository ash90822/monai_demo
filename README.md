# ChestCT Segmentation (MONAI + PyTorch)

本專案使用 **MONAI + PyTorch** 建立一個可對 Chest CT DICOM 影像進行多類別器官分割的流程，並且提供完整的  
**訓練（train.py）**、**推論（infer.py）**、**Dataset 與 Transform**、**LabelMe JSON 匯出** 的模組化架構。

---
```
## 📁 專案結構

monai_demo/
│── README.md
│── config.py # 超參數 + 類別設定集中管理
│── train.py # 訓練主程式（含 Early Stopping）
│── infer.py # 推論 + 匯出 LabelMe JSON
│
├── datasets/
│ └── basic_dataset.py # DICOM → HU → mask（LabelMe JSON）Dataset
│
├── models/
│ └── basic_unet.py # MONAI UNet 模型定義
│
├── transforms/
│ └── basic_transforms.py # HE 前處理 + MONAI 版增強（Flip / Rotate / Affine）
│
├── utils/
│ ├── io.py # 自訂 I/O 函式（如 load_image_single）
│ └── metrics.py # Dice / IoU 計算
│
└── .gitignore # 忽略 data / outputs / checkpoints 等大型資料
```
---

## 🚀 功能說明

### ✔ 1. DICOM → HU → HE → Resize（前處理）
- 自訂 HETransformd
- 支援 body mask（避免背景參與分位數）
- 支援多器官 LabelMe JSON mask

### ✔ 2. Data Augmentation（完整 MONAI 版本）
- RandFlipd（左右 / 上下）
- RandAffined（rotate / shear / scale）
- Resized

### ✔ 3. 多器官 UNet 分割
- MONAI UNet backbone
- out_channels 與類別數自動對應 config.py
- 訓練採用 Dice Loss（softmax 版）

### ✔ 4. Early Stopping
- patience / min_delta 於 config.py 中設定
- 最佳模型會自動存成 `checkpoints/best.pth`

### ✔ 5. 推論（infer.py）
- 讀 DICOM → HE + resize 做推論
- 依 argmax(prob) 生成 label_map
- 匯出 **LabelMe JSON polygon**
- JSON 的 imageData 存原始 DICOM （非 HE 影像）

---

## ⚙ 環境安裝

```bash
conda create -n monai_env python=3.10
conda activate monai_env

pip install monai torch torchvision
pip install pydicom opencv-python imgaug pandas
🏋️‍♂️ 訓練模型
bash
複製程式碼
python train.py
所有訓練超參均於 config.py 管理，包括：

batch size

learning rate

epochs

patience

img_size

HE 設定

類別名稱

UNet 結構設定

🔍 推論 DICOM
bash
複製程式碼
python infer.py
輸出路徑：./outputs/*.json

每個 JSON 包含：

原始影像（PNG base64）

各器官 polygon（LabelMe shapes）

🧩 config.py（集中管理設定）
你可以在這裡調整：

類別名稱：["bg", "liver", "spleen"]

模型 out_channels

HE 參數（pmin/pmax/body_mask）

訓練參數（lr/epochs/batch）

推論影像的 window 值（40/400）

路徑設定（train.csv / test.csv / outputs）

不用再每次打開 train.py 或 infer.py 修改。
