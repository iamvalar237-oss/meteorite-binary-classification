import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG")


def _cfg(config: Dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    """Read nested config values from a plain dict loaded from YAML."""
    return config.get(section, {}).get(key, default)


def _require_file(path: os.PathLike) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def _require_columns(df: pd.DataFrame, path: os.PathLike, columns) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}; found columns: {list(df.columns)}")


def resolve_image_dir(root_dir: os.PathLike, image_dir: str) -> Path:
    """
    Resolve image directory and tolerate Kaggle's common nested layout:
    train_images/train_images/*.jpg or test_images/test_images/*.jpg.
    """
    root_dir = Path(root_dir)
    base_dir = root_dir / image_dir
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {base_dir}")

    if any(p.is_file() and p.suffix in SUPPORTED_EXTENSIONS for p in base_dir.iterdir()):
        return base_dir

    nested_dir = base_dir / base_dir.name
    if nested_dir.is_dir() and any(p.is_file() and p.suffix in SUPPORTED_EXTENSIONS for p in nested_dir.iterdir()):
        return nested_dir

    return base_dir


def resolve_image_path(image_dir: os.PathLike, image_id: Any) -> Path:
    """Resolve one image path from id, trying common extensions only when id has no suffix."""
    image_dir = Path(image_dir)
    image_id = str(image_id)
    direct_path = image_dir / image_id

    if Path(image_id).suffix:
        if direct_path.exists():
            return direct_path
        raise FileNotFoundError(f"Image id '{image_id}' was not found in directory: {image_dir}")

    for ext in SUPPORTED_EXTENSIONS:
        candidate = image_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Image id '{image_id}' was not found in directory: {image_dir}. "
        f"Tried extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


def _resolve_sample_weight_path(config: Dict[str, Any]) -> Optional[Path]:
    """Resolve sample weight CSV path from config, returning None if not enabled."""
    sw_cfg = config.get("sample_weight", {})
    if not sw_cfg.get("enabled", False):
        return None
    raw_path = sw_cfg.get("path", "")
    if not raw_path:
        return None
    return Path(raw_path)


def _load_sample_weights(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Load sample weight CSV if enabled and file exists, otherwise return None."""
    path = _resolve_sample_weight_path(config)
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"sample_weight.path does not exist: {path}")
    sw_cfg = config.get("sample_weight", {})
    id_col = sw_cfg.get("id_col", "id")
    weight_col = sw_cfg.get("weight_col", "sample_weight")
    sw_df = pd.read_csv(path)
    _require_columns(sw_df, path, [id_col, weight_col])
    sw_df = sw_df[[id_col, weight_col]].copy()
    sw_df[id_col] = sw_df[id_col].astype(str)
    sw_df[weight_col] = sw_df[weight_col].astype(float)
    return sw_df



def _load_external_train_rows(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    ext_csv = _cfg(config, "data", "external_train_csv", "")
    if not ext_csv:
        return None
    root = Path(_cfg(config, "data", "root_dir", "."))
    ext_path = Path(ext_csv)
    if not ext_path.is_absolute():
        ext_path = root / ext_path
    if not ext_path.exists():
        raise FileNotFoundError(f"external_train_csv not found: {ext_path}")
    ext_df = pd.read_csv(ext_path)
    required = {"id", "path", "label"}
    missing = required - set(ext_df.columns)
    if missing:
        raise ValueError(f"external_train_csv missing columns {missing}; found: {list(ext_df.columns)}")
    ext_df = ext_df.copy()
    ext_df["_external_train"] = 1
    ext_df["_external_neg"] = 1
    ext_df["_external_neg_path"] = ext_df["path"]
    return ext_df


def _load_external_negatives(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Load external hard negative CSV (label=0 images) if configured, else None."""
    ext_neg_csv = _cfg(config, "data", "external_neg_csv", "")
    if not ext_neg_csv:
        return None
    root_dir = Path(_cfg(config, "data", "root_dir", "."))
    ext_path = Path(ext_neg_csv)
    if not ext_path.is_absolute():
        ext_path = root_dir / ext_path
    if not ext_path.is_file():
        raise FileNotFoundError(f"external_neg_csv not found: {ext_path}")
    id_col = _cfg(config, "data", "id_col", "id")
    ext_df = pd.read_csv(ext_path)
    if "path" not in ext_df.columns:
        raise ValueError(f"external_neg_csv must contain 'path' column; found: {list(ext_df.columns)}")
    # Use the absolute path column; assign label=0
    ext_df = ext_df.copy()
    ext_df[id_col] = ext_df["path"].astype(str)
    ext_df["_external_neg"] = 1
    return ext_df


def load_train_dataframe(config: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, int]]]:
    """Load training labels and map non-0/1 labels deterministically if needed."""
    root_dir = Path(_cfg(config, "data", "root_dir", "."))
    train_csv = root_dir / _cfg(config, "data", "train_csv", "train_labels.csv")
    id_col = _cfg(config, "data", "id_col", "id")
    label_col = _cfg(config, "data", "label_col", "label")

    df = pd.read_csv(_require_file(train_csv))
    _require_columns(df, train_csv, [id_col, label_col])
    df = df.copy()
    df[id_col] = df[id_col].astype(str)

    # Append external hard negatives (label=0, train folds only)
    ext_df = _load_external_negatives(config)
    if ext_df is not None:
        ext_rows = []
        for _, ext_row in ext_df.iterrows():
            ext_rows.append({id_col: ext_row[id_col], label_col: 0, "_external_neg": 1, "_external_neg_path": ext_row["path"]})
        ext_part = pd.DataFrame(ext_rows)
        df = pd.concat([df, ext_part], axis=0, ignore_index=True)

    # Append generic external train rows (positive or negative, train folds only)
    ext_train_df = _load_external_train_rows(config)
    if ext_train_df is not None:
        ext_rows = []
        for _, ext_row in ext_train_df.iterrows():
            ext_rows.append({
                id_col: str(ext_row["id"]),
                label_col: int(ext_row["label"]),
                "_external_train": 1,
                "_external_neg": 1,
                "_external_neg_path": ext_row["path"],
            })
        ext_part = pd.DataFrame(ext_rows)
        df = pd.concat([df, ext_part], axis=0, ignore_index=True)

    unique_labels = sorted(df[label_col].dropna().unique().tolist())
    if len(unique_labels) != 2:
        raise ValueError(f"Expected exactly 2 classes for binary classification, got {len(unique_labels)}: {unique_labels}")
    label_mapping = None
    if set(unique_labels) != {0, 1}:
        label_mapping = {str(label): idx for idx, label in enumerate(unique_labels)}
        df[f"{label_col}_original"] = df[label_col]
        df[label_col] = df[label_col].map(lambda value: label_mapping[str(value)])

    df[label_col] = df[label_col].astype(int)

    # Merge sample weights if enabled
    sw_df = _load_sample_weights(config)
    if sw_df is not None:
        weight_col = config.get("sample_weight", {}).get("weight_col", "sample_weight")
        sw_id_col = config.get("sample_weight", {}).get("id_col", id_col)
        if sw_id_col == "path":
            # External negatives use abs path as id; merge on the _external_neg_path column
            sw_df = sw_df.rename(columns={sw_id_col: "_merge_path"})
            df = df.merge(sw_df, left_on="_external_neg_path", right_on="_merge_path", how="left")
            df = df.drop(columns=["_merge_path"])
        else:
            df = df.merge(sw_df, on=sw_id_col, how="left")
        df[weight_col] = df[weight_col].fillna(1.0)

    df.attrs["label_mapping"] = label_mapping
    return df, label_mapping


def load_test_dataframe(config: Dict[str, Any]) -> pd.DataFrame:
    """Load sample submission while preserving the original row order."""
    root_dir = Path(_cfg(config, "data", "root_dir", "."))
    sample_submission = root_dir / _cfg(config, "data", "sample_submission", "sample_submission.csv")
    id_col = _cfg(config, "data", "id_col", "id")

    df = pd.read_csv(_require_file(sample_submission))
    _require_columns(df, sample_submission, [id_col])
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    return df


def build_transforms(config: Dict[str, Any], mode: str):
    """Build torchvision transforms for train/valid/test."""
    if mode not in {"train", "valid", "test"}:
        raise ValueError(f"Invalid transform mode: {mode}. Expected train, valid or test.")

    image_size = int(_cfg(config, "model", "image_size", 384))
    if mode == "train":
        scale_min = float(_cfg(config, "augment", "random_resized_crop_scale_min", 0.75))
        scale_max = float(_cfg(config, "augment", "random_resized_crop_scale_max", 1.0))
        rotation_degrees = float(_cfg(config, "augment", "rotation_degrees", 15))
        random_grayscale_p = float(_cfg(config, "augment", "random_grayscale_p", 0.05))
        random_erasing_p = float(_cfg(config, "augment", "random_erasing_p", 0.5))
        erase_scale_min = float(_cfg(config, "augment", "random_erasing_scale_min", 0.02))
        erase_scale_max = float(_cfg(config, "augment", "random_erasing_scale_max", 0.20))

        transform_list = [
            transforms.RandomResizedCrop(image_size, scale=(scale_min, scale_max)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(rotation_degrees),
        ]
        if bool(_cfg(config, "augment", "color_jitter", True)):
            transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03))
        transform_list.extend(
            [
                transforms.RandomGrayscale(p=random_grayscale_p),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=random_erasing_p, scale=(erase_scale_min, erase_scale_max)),
            ]
        )
        return transforms.Compose(transform_list)

    resize_size = int(image_size * 1.15)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class MeteoriteDataset(Dataset):
    """Dataset for meteorite binary classification and Kaggle test inference."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: os.PathLike,
        id_col: str = "id",
        label_col: str = "label",
        transforms=None,
        mode: str = "train",
    ):
        if mode not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid dataset mode: {mode}. Expected train, valid or test.")
        if id_col not in df.columns:
            raise ValueError(f"DataFrame is missing id column: {id_col}")
        if mode != "test" and label_col not in df.columns:
            raise ValueError(f"DataFrame is missing label column: {label_col}")

        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.id_col = id_col
        self.label_col = label_col
        self.transforms = transforms
        self.mode = mode
        self._has_sample_weight = "sample_weight" in df.columns and mode != "test"
        self._external_neg_mode = "_external_neg" in df.columns and mode == "train"

        self.image_ids = self.df[id_col].astype(str).tolist()
        # Build image paths: external negatives use stored abs path, others resolve from image_dir
        self.image_paths = []
        for idx in range(len(self.df)):
            ext_path = self.df.iloc[idx].get("_external_neg_path")
            if pd.notna(ext_path) and ext_path:
                self.image_paths.append(Path(str(ext_path)))
            else:
                self.image_paths.append(resolve_image_path(self.image_dir, self.image_ids[idx]))
        self.labels = None if mode == "test" else self.df[label_col].astype(float).tolist()
        self.sample_weights = None
        if self._has_sample_weight:
            self.sample_weights = self.df["sample_weight"].astype(float).tolist()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)

        image_id = self.image_ids[index]
        if self.mode == "test":
            return image, image_id

        label = torch.tensor(self.labels[index], dtype=torch.float32)
        if self._has_sample_weight:
            weight = torch.tensor(self.sample_weights[index], dtype=torch.float32)
            return image, label, weight, image_id
        return image, label, image_id


# Backward-compatible alias for older notebooks/scripts in this project.
StoneDataset = MeteoriteDataset