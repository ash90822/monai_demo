import sys, subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QTextEdit, QGroupBox, QAbstractItemView,
    QDoubleSpinBox, QMessageBox, QProgressBar, QCheckBox
)

from infer_service import ModelService, dicom_to_hu
from configs.config import CFG


@dataclass
class Sample:
    image_path: Path
    label: Optional[int] = None
    meta: Dict[str, Any] = None


class InferenceWorker(QThread):
    progress = pyqtSignal(int, int)
    message = pyqtSignal(str)
    result = pyqtSignal(dict)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, model_service: ModelService, samples: List[Sample], output_dir: str):
        super().__init__()
        self.model_service = model_service
        self.samples = samples
        self.output_dir = output_dir
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            total = len(self.samples)
            if total == 0:
                self.message.emit("No samples to run.")
                self.finished_ok.emit()
                return

            self.message.emit(f"Start inference: {total} DICOM files")
            for i, s in enumerate(self.samples, start=1):
                if self._stop:
                    self.message.emit("Inference stopped by user.")
                    break
                out = self.model_service.predict_to_labelme(str(s.image_path), output_dir=self.output_dir)
                self.result.emit(out)
                self.progress.emit(i, total)

            self.message.emit("Inference done.")
            self.finished_ok.emit()

        except Exception as e:
            self.failed.emit(str(e))


class PreviewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._img = None  # HU float32 (H,W)
        self._wl_enabled = True
        self._window = 400.0
        self._level = 40.0

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(480, 480)
        self.image_label.setStyleSheet("QLabel { background: #111; color: #ddd; }")

        controls = QGroupBox("Preview Controls")
        grid = QGridLayout()

        self.chk_wl = QCheckBox("Enable Window/Level")
        self.chk_wl.setChecked(True)
        self.chk_wl.stateChanged.connect(self._on_wl_toggle)

        self.win_spin = QDoubleSpinBox()
        self.win_spin.setRange(1.0, 10000.0)
        self.win_spin.setValue(self._window)
        self.win_spin.valueChanged.connect(self._on_wl_change)

        self.lev_spin = QDoubleSpinBox()
        self.lev_spin.setRange(-5000.0, 5000.0)
        self.lev_spin.setValue(self._level)
        self.lev_spin.valueChanged.connect(self._on_wl_change)

        grid.addWidget(self.chk_wl, 0, 0, 1, 2)
        grid.addWidget(QLabel("Window"), 1, 0)
        grid.addWidget(self.win_spin, 1, 1)
        grid.addWidget(QLabel("Level"), 2, 0)
        grid.addWidget(self.lev_spin, 2, 1)
        controls.setLayout(grid)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(controls)
        self.setLayout(layout)

    def load_dicom(self, path: Path):
        hu = dicom_to_hu(str(path))
        self._img = hu

        if CFG.infer.window_center is not None and CFG.infer.window_width is not None:
            self._level = float(CFG.infer.window_center)
            self._window = float(CFG.infer.window_width)
        else:
            v = hu[np.isfinite(hu)]
            lo = float(np.percentile(v, 1))
            hi = float(np.percentile(v, 99))
            self._window = max(1.0, hi - lo)
            self._level = (hi + lo) / 2.0

        self.win_spin.blockSignals(True)
        self.lev_spin.blockSignals(True)
        self.win_spin.setValue(self._window)
        self.lev_spin.setValue(self._level)
        self.win_spin.blockSignals(False)
        self.lev_spin.blockSignals(False)

        self._refresh_view()

    @staticmethod
    def _apply_window_level(arr: np.ndarray, window: float, level: float) -> np.ndarray:
        low = level - window / 2.0
        high = level + window / 2.0
        x = np.clip(arr, low, high)
        x = (x - low) / max(1e-6, (high - low))
        return (x * 255.0).astype(np.uint8)

    def _on_wl_toggle(self):
        self._wl_enabled = self.chk_wl.isChecked()
        self._refresh_view()

    def _on_wl_change(self):
        self._window = float(self.win_spin.value())
        self._level = float(self.lev_spin.value())
        self._refresh_view()

    def _refresh_view(self):
        if self._img is None:
            self.image_label.setText("No image loaded")
            return

        if self._wl_enabled:
            img8 = self._apply_window_level(self._img, self._window, self._level)
        else:
            vmin, vmax = float(np.nanmin(self._img)), float(np.nanmax(self._img))
            denom = max(1e-6, vmax - vmin)
            img8 = np.clip((self._img - vmin) / denom * 255.0, 0, 255).astype(np.uint8)

        h, w = img8.shape
        qimg = QImage(img8.data, w, h, w, QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_view()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Inference GUI (PyQt6)")
        self.resize(1200, 720)

        self.model_service = ModelService()
        self.samples: List[Sample] = []
        self.current_sample: Optional[Sample] = None
        self.worker: Optional[InferenceWorker] = None

        self.output_dir = str(CFG.paths.output_dir)

        # Left
        left = QWidget()
        left_layout = QVBoxLayout()

        btn_row = QHBoxLayout()
        self.btn_select_folder = QPushButton("Select Folder (DICOM)")
        self.btn_select_folder.clicked.connect(self.on_select_folder)

        self.btn_select_output = QPushButton("Select Output Dir")
        self.btn_select_output.clicked.connect(self.on_select_output_dir)

        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.clicked.connect(self.on_load_model)

        # ---- Training controls ----
        self.train_img_dir = ""
        self.train_label_dir = ""

        train_row = QHBoxLayout()
        self.btn_pick_train_img = QPushButton("Train Images Dir")
        self.btn_pick_train_lbl = QPushButton("Train Labels Dir")
        self.btn_start_train = QPushButton("Start Train")
        self.btn_start_train.clicked.connect(self.on_start_train)

        self.btn_pick_train_img.clicked.connect(self.on_pick_train_img_dir)
        self.btn_pick_train_lbl.clicked.connect(self.on_pick_train_label_dir)

        train_row.addWidget(self.btn_pick_train_img)
        train_row.addWidget(self.btn_pick_train_lbl)
        train_row.addWidget(self.btn_start_train)

        btn_row.addWidget(self.btn_select_folder)
        btn_row.addWidget(self.btn_select_output)
        btn_row.addWidget(self.btn_load_model)

        left_layout.addLayout(train_row)

        self.case_list = QListWidget()
        self.case_list.itemSelectionChanged.connect(self.on_case_selected)
        self.case_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        infer_row = QHBoxLayout()
        self.btn_run_selected = QPushButton("Run Selected")
        self.btn_run_selected.clicked.connect(self.on_run_selected)

        self.btn_run_all = QPushButton("Run All")
        self.btn_run_all.clicked.connect(self.on_run_all)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)

        infer_row.addWidget(self.btn_run_selected)
        infer_row.addWidget(self.btn_run_all)
        infer_row.addWidget(self.btn_stop)

        self.progress = QProgressBar()
        self.progress.setValue(0)

        left_layout.addLayout(btn_row)
        left_layout.addWidget(QLabel("Case List"))
        left_layout.addWidget(self.case_list, stretch=1)
        left_layout.addLayout(infer_row)
        left_layout.addWidget(self.progress)
        left.setLayout(left_layout)

        # Right
        right = QWidget()
        right_layout = QVBoxLayout()
        self.preview = PreviewPanel()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(160)

        right_layout.addWidget(self.preview, stretch=1)
        right_layout.addWidget(QLabel("Log"))
        right_layout.addWidget(self.log)
        right.setLayout(right_layout)

        central = QWidget()
        main_layout = QHBoxLayout()
        main_layout.addWidget(left, stretch=1)
        main_layout.addWidget(right, stretch=2)
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self._log(f"Ready. Device={self.model_service.device}")

    def _log(self, msg: str):
        t = time.strftime("%H:%M:%S")
        self.log.append(f"[{t}] {msg}")

    def _scan_folder(self, folder: Path) -> List[Sample]:
        samples = []
        for p in folder.rglob("*.dcm"):
            samples.append(Sample(image_path=p, meta={"name": p.name}))
        samples.sort(key=lambda s: str(s.image_path))
        return samples

    def _refresh_case_list(self):
        self.case_list.clear()
        for s in self.samples:
            item = QListWidgetItem(s.image_path.name)
            item.setData(Qt.ItemDataRole.UserRole, s)
            self.case_list.addItem(item)

    def on_select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select a folder containing DICOM (*.dcm)")
        if not folder:
            return
        folder = Path(folder)

        self.samples = self._scan_folder(folder)
        self._log(f"Selected: {folder}")
        self._log(f"Found {len(self.samples)} DICOM files.")
        self._refresh_case_list()

        if self.samples:
            self.case_list.setCurrentRow(0)

    def on_select_output_dir(self):
        out = QFileDialog.getExistingDirectory(self, "Select output directory for LabelMe JSON")
        if not out:
            return
        self.output_dir = out
        self._log(f"[OUTPUT] Set output directory: {self.output_dir}")

    def on_case_selected(self):
        items = self.case_list.selectedItems()
        if not items:
            return
        s = items[0].data(Qt.ItemDataRole.UserRole)
        self.current_sample = s
        self._log(f"Selected: {s.image_path}")
        self.preview.load_dicom(s.image_path)

    def on_load_model(self):
        try:
            default_dir = str(Path("checkpoints").resolve())
            # 若你 CFG.paths.checkpoint 是完整路徑，也可用它的 parent 當 default_dir
            if hasattr(CFG.paths, "checkpoint"):
                try:
                    default_dir = str(Path(CFG.paths.checkpoint).resolve().parent)
                except Exception:
                    pass

            ckpt, _ = QFileDialog.getOpenFileName(
                self,
                "Select checkpoint file",
                default_dir,
                "PyTorch Weights (*.pt *.pth *.bin *.ckpt);;All Files (*)"
            )

            # 允許取消：取消就用 CFG 預設
            if not ckpt:
                ckpt = str(CFG.paths.checkpoint)

            self._log(f"Loading model: {ckpt}")
            self.model_service.load_model(ckpt_path=ckpt)
            self._log("Model loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Load Model Failed", str(e))

    def _start_worker(self, samples: List[Sample]):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Inference is already running.")
            return
        if not self.model_service.is_loaded:
            QMessageBox.warning(self, "Model not loaded", "Please click 'Load Model' first.")
            return

        self.progress.setValue(0)
        self.btn_stop.setEnabled(True)
        self.btn_run_all.setEnabled(False)
        self.btn_run_selected.setEnabled(False)

        self.worker = InferenceWorker(self.model_service, samples, output_dir=self.output_dir)
        self.worker.message.connect(self._log)
        self.worker.progress.connect(self.on_progress)
        self.worker.result.connect(self.on_result)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()
    
    


    def on_run_selected(self):
        items = self.case_list.selectedItems()
        if not items:
            QMessageBox.information(self, "No selection", "Please select DICOM files (Ctrl/Shift supported).")
            return
        samples = [it.data(Qt.ItemDataRole.UserRole) for it in items]
        self._start_worker(samples)

    def on_run_all(self):
        if not self.samples:
            QMessageBox.information(self, "Empty", "No DICOM found. Please select a folder first.")
            return
        self._start_worker(self.samples)

    def on_pick_train_img_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select training images directory")
        if not d:
            return
        self.train_img_dir = d
        self._log(f"[TRAIN] images_dir = {d}")

    def on_pick_train_label_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select training labels directory")
        if not d:
            return
        self.train_label_dir = d
        self._log(f"[TRAIN] labels_dir = {d}")

    def on_start_train(self):
        if not self.train_img_dir or not self.train_label_dir:
            QMessageBox.information(self, "Missing", "Please select both training images dir and labels dir.")
            return

        pairs, missing_labels, missing_images = validate_pairs(self.train_img_dir, self.train_label_dir)

        self._log(f"Pair check: matched={len(pairs)}, missing_labels={len(missing_labels)}, missing_images={len(missing_images)}")
        if missing_labels[:5]:
            self._log("Missing labels examples: " + ", ".join(missing_labels[:5]))
        if missing_images[:5]:
            self._log("Missing images examples: " + ", ".join(missing_images[:5]))

        if len(pairs) == 0:
            QMessageBox.warning(self, "Invalid", "No matched image/label pairs. Check filenames and extensions.")
            return
        if missing_labels or missing_images:
            # 你要嚴格就直接 return；或允許部分對齊訓練
            QMessageBox.warning(self, "Not aligned", "Some files are not aligned. Check log for details.")
            return

        # 背景訓練
        self._log("Start training worker...")
        self.train_worker = TrainWorker(self.train_img_dir, self.train_label_dir)
        self.train_worker.message.connect(self._log)
        self.train_worker.finished_ok.connect(lambda rc: self._log(f"Train finished. return_code={rc}"))
        self.train_worker.failed.connect(lambda e: self._log(f"Train failed: {e}"))
        self.train_worker.start()


    def on_stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self._log("Stopping...")

    def on_progress(self, done: int, total: int):
        self.progress.setValue(int(done / max(1, total) * 100))

    def on_result(self, out: dict):
        self._log(f"Saved: {out['json_path']} | shapes={out['num_shapes']} | {out['dcm_path']}")

    def on_finished(self):
        self.btn_stop.setEnabled(False)
        self.btn_run_all.setEnabled(True)
        self.btn_run_selected.setEnabled(True)
        self.progress.setValue(100)
        self._log("Worker finished.")

    def on_failed(self, err: str):
        self.btn_stop.setEnabled(False)
        self.btn_run_all.setEnabled(True)
        self.btn_run_selected.setEnabled(True)
        QMessageBox.critical(self, "Inference Failed", err)
        self._log(f"Worker failed: {err}")

class TrainWorker(QThread):
    message = pyqtSignal(str)
    finished_ok = pyqtSignal(int)   # return code
    failed = pyqtSignal(str)

    def __init__(self, img_dir: str, label_dir: str, extra_args: Optional[List[str]] = None):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.extra_args = extra_args or []

    def run(self):
        try:
            cmd = [
                sys.executable, "-u", "train.py",   # <- 關鍵：-u
                "--img_dir", self.img_dir,
                "--label_dir", self.label_dir,
                *self.extra_args
            ]
            self.message.emit("[TRAIN] " + " ".join(cmd))

            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,            # <- 行緩衝（搭配 -u 最有效）
                universal_newlines=True
            )

            assert p.stdout is not None
            for line in iter(p.stdout.readline, ""):
                if line == "" and p.poll() is not None:
                    break
                self.message.emit(line.rstrip("\n"))

            rc = p.wait()
            self.finished_ok.emit(rc)

        except Exception as e:
            self.failed.emit(str(e))

def validate_pairs(img_dir: str, label_dir: str,
                   img_exts=(".png", ".jpg", ".jpeg", ".dcm"),
                   label_exts=(".json")):
        img_dir = Path(img_dir)
        label_dir = Path(label_dir)

        def stem_key(p: Path) -> str:
            # 處理 .nii.gz
            name = p.name.lower()
            if name.endswith(".nii.gz"):
                return p.name[:-7]  # remove .nii.gz
            return p.stem

        imgs = [p for p in img_dir.iterdir() if p.is_file() and (p.suffix.lower() in img_exts or p.name.lower().endswith(".nii.gz"))]
        lbls = [p for p in label_dir.iterdir() if p.is_file() and (p.suffix.lower() in label_exts or p.name.lower().endswith(".nii.gz"))]

        img_map = {stem_key(p): p for p in imgs}
        lbl_map = {stem_key(p): p for p in lbls}

        img_keys = set(img_map.keys())
        lbl_keys = set(lbl_map.keys())

        missing_labels = sorted(list(img_keys - lbl_keys))
        missing_images = sorted(list(lbl_keys - img_keys))
        matched = sorted(list(img_keys & lbl_keys))

        pairs = [(img_map[k], lbl_map[k]) for k in matched]
        return pairs, missing_labels, missing_images

def main():
    app = QApplication([])
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
