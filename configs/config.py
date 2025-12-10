# config.py
from dataclasses import dataclass, field
from typing import Tuple, List, Dict


# ------------ 路徑相關 ------------
@dataclass
class PathsConfig:
    train_csv: str = "./data/train.csv"
    test_csv: str = "./data/test.csv"
    checkpoint: str = "./checkpoints/best.pth"
    output_dir: str = "./outputs"


# ------------ 資料 / HE / 影像大小 ------------
@dataclass
class DataConfig:
    img_size: int = 512
    # HE / body mask
    pmin: float = 1.0
    pmax: float = 99.0
    use_body_mask: bool = True
    body_hu_thresh: float = -300


# ------------ 類別設定（最重要） ------------
@dataclass
class ClassesConfig:
    # index 0 = 背景，其餘依序是器官
    label_names: List[str] = field(default_factory=lambda: ["bg", "liver", "spleen"])

    # LabelMe JSON 裡的 label 字串 → 整數 id（不含背景）
    json_label_map: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        # 若沒自己指定 json_label_map，就自動用 label_names[1:] 建立
        if not self.json_label_map:
            self.json_label_map = {
                name: idx
                for idx, name in enumerate(self.label_names)
                if idx > 0  # 跳過背景
            }

    @property
    def num_classes(self) -> int:
        return len(self.label_names)


# ------------ 模型結構超參 ------------
@dataclass
class ModelConfig:
    in_channels: int = 1
    out_channels: int = 3          # 要跟 ClassesConfig.num_classes 一致
    channels: Tuple[int, ...] = (32, 64, 128, 256)
    strides: Tuple[int, ...] = (2, 2, 2)
    num_res_units: int = 2


# ------------ 訓練超參 ------------
@dataclass
class TrainConfig:
    batch_size: int = 4
    lr: float = 1e-4
    epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-4
    num_workers: int = 0  # DataLoader workers
    weight_decay: float = 1e-2


# ------------ 推論相關設定 ------------
@dataclass
class InferConfig:
    # DICOM 顯示用 window（寫進 LabelMe imageData 的那張圖）
    window_center: float = 40.0
    window_width: float = 400.0


# ------------ 全域設定物件 ------------
class CFG:
    paths = PathsConfig()
    data = DataConfig()
    classes = ClassesConfig()
    model = ModelConfig()
    train = TrainConfig()
    infer = InferConfig()
