import argparse
import copy
import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MeteoriteDataset, build_transforms, load_test_dataframe, resolve_image_dir
from models import build_model
from utils import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-fold checkpoint inference and Kaggle submission generation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing fold*_best.pth, fold*_best_f1.pth, or fold*_best_loss.pth checkpoints")
    parser.add_argument("--output", default=None, help="Submission output path; default: checkpoint-dir/submission.csv")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    parser.add_argument("--tta", action="store_true", help="Use deterministic flip TTA")
    parser.add_argument("--device", default=None, help="Default: cuda if available else cpu")
    return parser.parse_args()


def find_checkpoints(checkpoint_dir: Path) -> List[Path]:
    patterns = ["fold*_best.pth", "fold*_best_f1.pth", "fold*_best_loss.pth"]
    for pattern in patterns:
        checkpoints = sorted(checkpoint_dir.glob(pattern))
        if checkpoints:
            return checkpoints
    raise FileNotFoundError(
        f"No fold checkpoints found under {checkpoint_dir}; "
        f"expected one of: {', '.join(patterns)}"
    )


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load full training checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_threshold(args: argparse.Namespace, checkpoint_dir: Path) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    metrics_path = checkpoint_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        if "oof_best_threshold" in metrics:
            return float(metrics["oof_best_threshold"])
    warnings.warn("Could not find oof_best_threshold in metrics.json; fallback to threshold=0.5")
    return 0.5


def apply_tta(images: torch.Tensor, tta_index: int) -> torch.Tensor:
    if tta_index == 0:
        return images
    if tta_index == 1:
        return torch.flip(images, dims=[3])
    if tta_index == 2:
        return torch.flip(images, dims=[2])
    if tta_index == 3:
        return torch.flip(images, dims=[2, 3])
    raise ValueError(f"Invalid TTA index: {tta_index}")


def _stable_sigmoid(logit: np.ndarray) -> np.ndarray:
    """Stable sigmoid: clip extremes before exp to avoid overflow."""
    logit = np.asarray(logit, dtype=np.float64)
    logit = np.clip(logit, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logit))


@torch.no_grad()
def predict_checkpoint(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_tta: bool, amp_enabled: bool) -> Dict[str, float]:
    """Run one checkpoint and return id -> averaged TTA logit (not probability)."""
    model.eval()
    tta_count = 4 if use_tta else 1
    id_to_logit: Dict[str, float] = {}

    for images, image_ids in tqdm(loader, desc="Predict", leave=False):
        images = images.to(device, non_blocking=True)
        batch_logits = []
        for tta_index in range(tta_count):
            augmented = apply_tta(images, tta_index)
            with autocast(enabled=amp_enabled):
                logits = model(augmented)
            batch_logits.append(logits.float().detach().cpu().numpy())
        avg_logits = np.mean(batch_logits, axis=0)
        for image_id, logit in zip(image_ids, avg_logits):
            id_to_logit[str(image_id)] = float(logit)
    return id_to_logit


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output) if args.output else checkpoint_dir / "submission.csv"
    prediction_path = output_path.parent / "test_predictions.csv"
    ensure_dir(output_path.parent)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    id_col = data_cfg.get("id_col", "id")
    label_col = data_cfg.get("label_col", "label")

    sample_df = load_test_dataframe(config)
    image_dir = resolve_image_dir(data_cfg.get("root_dir", "."), data_cfg.get("test_img_dir", "test_images"))
    test_dataset = MeteoriteDataset(
        sample_df,
        image_dir=image_dir,
        id_col=id_col,
        label_col=label_col,
        transforms=build_transforms(config, "test"),
        mode="test",
    )
    loader = DataLoader(
        test_dataset,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    checkpoints = find_checkpoints(checkpoint_dir)
    fold_logit_maps = []
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    for checkpoint_path in checkpoints:
        checkpoint = load_checkpoint(checkpoint_path, device)
        model_config = copy.deepcopy(checkpoint.get("config", config))
        # Inference only needs the saved fine-tuned weights; avoid downloading pretrained weights again.
        model_config.setdefault("model", {})["pretrained"] = False
        model_config["model"].pop("pretrained_checkpoint_path", None)
        model = build_model(model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        fold_logit_maps.append(predict_checkpoint(model, loader, device, args.tta, amp_enabled))

    image_ids = sample_df[id_col].astype(str).tolist()
    final_logits = []
    for image_id in image_ids:
        missing_folds = [idx for idx, logit_map in enumerate(fold_logit_maps) if image_id not in logit_map]
        if missing_folds:
            raise RuntimeError(f"Missing prediction for image id '{image_id}' from fold indexes: {missing_folds}")
        fold_logits = [logit_map[image_id] for logit_map in fold_logit_maps]
        final_logits.append(float(np.mean(fold_logits)))

    final_logits = np.array(final_logits, dtype=np.float64)
    probabilities = _stable_sigmoid(final_logits)

    threshold = load_threshold(args, checkpoint_dir)
    predictions = (probabilities >= threshold).astype(int)

    pred_df = pd.DataFrame({
        "id": image_ids,
        "logit": final_logits,
        "probability": probabilities,
        "prediction": predictions,
    })
    pred_df.to_csv(prediction_path, index=False)

    submission_df = sample_df.copy()
    if label_col not in submission_df.columns:
        raise ValueError(f"sample_submission must contain label column '{label_col}' to preserve column format")
    submission_df[label_col] = predictions
    submission_df.to_csv(output_path, index=False)
    print(f"Saved test predictions to {prediction_path}")
    print(f"Saved submission to {output_path} with threshold={threshold:.4f}")


if __name__ == "__main__":
    main()