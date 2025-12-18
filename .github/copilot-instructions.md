# Copilot instructions for monai_demo

This file gives compact, actionable guidance for AI coding agents working in this repository.

Purpose
- Provide the repo "big picture" and concrete conventions so an agent can make small-to-moderate changes safely.

Big picture
- This project implements multi-class Chest CT organ segmentation using MONAI + PyTorch.
- Pipelines: `data (DICOM + LabelMe JSON)` → `transforms (HE + resize + augment)` → `models (DynUNet)` → `train/infer`.
- Key files: `configs/config.py` (single source of truth for hyperparams and paths), `train.py`, `infer.py`, `datasets/basic_dataset.py`, `transforms/basic_transforms.py`, `models/basic_unet.py`, `utils/io.py`.

Important conventions & invariants (follow these exactly)
- `CFG` in `configs/config.py` centralizes all runtime settings. Always prefer changing behavior via `CFG` instead of ad-hoc edits in `train.py`/`infer.py`.
- `CFG.classes.label_names` has index-0 = background. `CFG.model.out_channels` must equal `CFG.classes.num_classes` (used throughout training/inference).
- Label mapping for training/inference: `CFG.classes.json_label_map` maps LabelMe `label` strings → integer class ids (1..N). `datasets.basic_dataset.ChestCTDataset.json_to_mask` uses this map.
- Input images are single-channel HU values; transforms expect `image` shaped `(1,H,W)` or `(H,W)` and produce `(1,H,W)` in [0,1]. See `transforms/HETransformd`.
- Augmentation order: HETransformd (HU→unit→HE) → `Resized` → flips → affine → `EnsureTyped`. Keep this ordering if modifying augment pipeline.

Developer workflows (commands)
- Create conda env and install minimal deps (from README):
  - `conda create -n monai_env python=3.10`
  - `conda activate monai_env`
  - `pip install monai torch torchvision pydicom opencv-python imgaug pandas`
- Train: `python train.py` (train hyperparameters, file paths, and model sizes controlled by `configs/config.py`).
- Inference on CSV: `python infer.py` — expects `CFG.paths.test_csv` to contain a `dcm_path` column; outputs LabelMe JSON files to `CFG.paths.output_dir`.

Data expectations / examples
- CSV layout: `train.csv` and `test.csv` rows contain at least `dcm_path`; `train.csv` should also include `json_path` for annotated samples (LabelMe JSON). See `datasets/basic_dataset.ChestCTDataset.__init__`.
- LabelMe JSON produced by `infer.py`:
  - `imageData` contains original DICOM encoded as PNG base64 (use `infer.dicom_to_base64_png`).
  - `shapes` contains polygons in original DICOM coordinate space (the code rescales from resized space back to original using `orig_h, orig_w`).

Model & checkpoints
- UNet: `models/basic_unet.get_basic_nnUnet()` creates a MONAI `DynUNet`. Keep `CFG.model.channels/strides` and `CFG.model.out_channels` consistent.
- Checkpoint path: `CFG.paths.checkpoint` (default `./checkpoints/best.pth`). Training uses early stopping and saves the best model there.

Common pitfalls and quick fixes
- Mismatched channels/classes: when you see shape errors during `model.load_state_dict` or inference outputs with wrong number of classes, verify `CFG.model.out_channels == CFG.classes.num_classes`.
- Transforms errors: `HETransformd` expects HU input (not normalized image). If a test shows unexpected contrast/zeros, ensure DICOM -> HU conversion is used (`datasets.basic_dataset.load_dicom_hu` or `infer.dicom_to_hu`).
- Label mismatch: `datasets.basic_dataset.json_to_mask` silently skips unknown labels. If annotations disappear, check `CFG.classes.json_label_map`.

Where to make changes
- Add new classes: update `CFG.classes.label_names` and (optionally) `json_label_map` in `configs/config.py`; ensure `CFG.model.out_channels` matches.
- Change preprocessing: edit `transforms/basic_transforms.py` (preserve HE → resize → augment order).
- Change model backbone: update `models/basic_unet.py` and keep `get_basic_nnUnet()` signature and return type compatible with training/inference code (i.e., model(img) returns logits `(B,C,H,W)`).

Files to inspect for examples
- `configs/config.py` — config dataclasses and `CFG` object
- `train.py` — training loop, early stopping, metrics (`compute_per_class_dice`) and optimizer setup
- `infer.py` — full inference flow, `dicom_to_base64_png`, `model_inference_to_label_map`, `label_map_to_labelme_json`
- `datasets/basic_dataset.py` — DICOM→HU, LabelMe JSON → multi-class mask
- `transforms/basic_transforms.py` — `HETransformd` behavior and augmentation ordering

If uncertain, prefer minimal, isolated edits and add a sanity check run:
- Example quick sanity run after a change: `python -c "from configs.config import CFG; print(CFG.model.out_channels, CFG.classes.num_classes)"`

Questions for the maintainer (to improve these instructions)
- Are there CI tests or environment files (requirements.txt / environment.yml) you prefer agents to use?
- Any labeling conventions (naming or ignored labels) that are not encoded in `CFG.classes.json_label_map`?

End of instructions — ask me to iterate or to add examples/PR checklist.
