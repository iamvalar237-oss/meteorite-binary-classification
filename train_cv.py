import argparse
import copy
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MeteoriteDataset, build_transforms, load_train_dataframe, resolve_image_dir
from models import build_model
from utils import (
    AverageMeter,
    compute_binary_metrics,
    count_parameters,
    ensure_dir,
    load_yaml,
    logits_to_probs,
    save_config_snapshot,
    save_json,
    search_best_threshold,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-fold ConvNeXt fine-tuning for meteorite binary classification")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--fold", default="all", help="Fold id: 0/1/2/3/4/all")
    parser.add_argument("--epochs", type=int, default=None, help="Override config train.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config train.batch_size")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Use only first N training samples for smoke test")
    parser.add_argument("--max-valid-samples", type=int, default=None, help="Use only first N validation samples for smoke test")
    parser.add_argument("--device", default=None, help="Default: cuda if available else cpu")
    parser.add_argument("--no-progress", action="store_true", help="Disable batch-level tqdm progress bars")
    parser.add_argument("--quiet-warnings", action="store_true", help="Hide torch.cuda.amp FutureWarning messages")
    return parser.parse_args()


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("train", {})["batch_size"] = args.batch_size
    return config


def make_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config.get("train", {})
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": float(train_cfg.get("lr_backbone", 1e-5))},
            {"params": model.head.parameters(), "lr": float(train_cfg.get("lr_head", 1e-4))},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )


def make_criterion(
    train_df: pd.DataFrame,
    config: Dict[str, Any],
    device: torch.device,
    use_reduction_none: bool = False,
) -> nn.Module:
    label_col = config.get("data", {}).get("label_col", "label")
    pos_weight_cfg = config.get("loss", {}).get("pos_weight", "auto")
    pos_weight = None
    if pos_weight_cfg == "auto":
        num_positive = int((train_df[label_col] == 1).sum())
        num_negative = int((train_df[label_col] == 0).sum())
        if num_positive > 0:
            pos_weight = torch.tensor([num_negative / max(num_positive, 1)], dtype=torch.float32, device=device)
    elif pos_weight_cfg is not None:
        pos_weight = torch.tensor([float(pos_weight_cfg)], dtype=torch.float32, device=device)
    reduction = "none" if use_reduction_none else "mean"
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)


def class_counts(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    """Return JSON-safe class counts."""
    return {str(key): int(value) for key, value in df[label_col].value_counts().sort_index().to_dict().items()}


def format_seconds(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS for readable training logs."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _iter_with_optional_tqdm(loader: DataLoader, show_progress: bool, desc: str):
    if show_progress:
        return tqdm(loader, desc=desc, leave=False)
    return loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
    grad_clip_norm: float,
    show_progress: bool = True,
    progress_desc: str = "Train",
    use_sample_weight: bool = False,
) -> float:
    model.train()
    meter = AverageMeter()
    for batch in _iter_with_optional_tqdm(loader, show_progress, progress_desc):
        if use_sample_weight:
            images, labels, weights, _ = batch
            weights = weights.to(device, non_blocking=True)
        else:
            images, labels, _ = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
            if use_sample_weight:
                # criterion uses reduction="none", so loss is a vector
                loss = (loss * weights).mean()

        scaler.scale(loss).backward()
        if grad_clip_norm and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        meter.update(loss.item(), n=images.size(0))
    return meter.avg


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    show_progress: bool = True,
    progress_desc: str = "Valid",
) -> Tuple[float, List[str], np.ndarray, np.ndarray]:
    model.eval()
    meter = AverageMeter()
    image_ids: List[str] = []
    y_true: List[float] = []
    y_prob: List[float] = []

    for batch in _iter_with_optional_tqdm(loader, show_progress, progress_desc):
        if len(batch) == 4:
            images, labels, _, ids = batch
        elif len(batch) == 3:
            images, labels, ids = batch
        else:
            raise ValueError(f"Unexpected validation batch length: {len(batch)}")
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        probs = logits_to_probs(logits)

        meter.update(loss.item(), n=images.size(0))
        image_ids.extend(list(ids))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_prob.extend(probs.tolist())

    return meter.avg, image_ids, np.asarray(y_true), np.asarray(y_prob)


def run_fold(
    fold: int,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    df: pd.DataFrame,
    config: Dict[str, Any],
    label_mapping,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    threshold_cfg = config.get("threshold", {})
    output_dir = ensure_dir(config.get("output", {}).get("dir", "outputs/convnext_tiny_384"))
    id_col = data_cfg.get("id_col", "id")
    label_col = data_cfg.get("label_col", "label")

    train_df = df.iloc[train_indices].reset_index(drop=True)
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    # Remove external negatives from validation folds (train only)
    if "_external_neg" in valid_df.columns:
        ext_count = (valid_df["_external_neg"] == 1).sum()
        if ext_count > 0:
            valid_df = valid_df[valid_df["_external_neg"] != 1].reset_index(drop=True)
            print(f"  Removed {ext_count} external negatives from valid fold (train-only)")
    if "_external_neg" in train_df.columns:
        ext_count = (train_df["_external_neg"] == 1).sum()
        if ext_count > 0:
            print(f"  Fold {fold} train: {ext_count} external negatives added")
    if args.max_train_samples is not None:
        train_df = train_df.head(args.max_train_samples).reset_index(drop=True)
    if args.max_valid_samples is not None:
        valid_df = valid_df.head(args.max_valid_samples).reset_index(drop=True)

    image_dir = resolve_image_dir(data_cfg.get("root_dir", "."), data_cfg.get("train_img_dir", "train_images"))
    train_dataset = MeteoriteDataset(
        train_df, image_dir=image_dir, id_col=id_col, label_col=label_col, transforms=build_transforms(config, "train"), mode="train"
    )
    valid_dataset = MeteoriteDataset(
        valid_df, image_dir=image_dir, id_col=id_col, label_col=label_col, transforms=build_transforms(config, "valid"), mode="valid"
    )
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 4))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = build_model(config).to(device)
    total_params, trainable_params = count_parameters(model)
    print(
        f"Model: {config.get('model', {}).get('model_name', 'convnext_tiny')} | "
        f"image_size: {config.get('model', {}).get('image_size', 384)} | "
        f"total parameters: {total_params:,} | trainable parameters: {trainable_params:,} | batch_size: {batch_size}"
    )

    use_sample_weight = bool(config.get("sample_weight", {}).get("enabled", False))
    optimizer = make_optimizer(model, config)
    # Training criterion with reduction="none" when sample_weight is enabled
    train_criterion = make_criterion(train_df, config, device, use_reduction_none=use_sample_weight)
    # Validation criterion always uses default reduction (mean)
    valid_criterion = make_criterion(train_df, config, device, use_reduction_none=False)
    epochs = int(train_cfg.get("epochs", 15))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 5))

    best_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5
    best_loss_saved = float("inf")
    best_predictions = None
    no_improve_epochs = 0
    per_epoch_metrics = []
    fold_start_time = time.time()
    show_progress = not getattr(args, "no_progress", False)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_start_time = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            train_criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            grad_clip_norm,
            show_progress=show_progress,
            progress_desc=f"fold{fold} epoch{epoch} train",
            use_sample_weight=use_sample_weight,
        )
        train_time_sec = time.time() - train_start_time

        valid_start_time = time.time()
        valid_loss, valid_ids, y_true, y_prob = validate_one_epoch(
            model,
            valid_loader,
            valid_criterion,
            device,
            amp_enabled,
            show_progress=show_progress,
            progress_desc=f"fold{fold} epoch{epoch} valid",
        )
        valid_time_sec = time.time() - valid_start_time
        threshold, valid_f1 = search_best_threshold(
            y_true,
            y_prob,
            float(threshold_cfg.get("min", 0.05)),
            float(threshold_cfg.get("max", 0.95)),
            float(threshold_cfg.get("step", 0.005)),
        )
        metrics = compute_binary_metrics(y_true, y_prob, threshold)
        scheduler.step()
        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]

        epoch_time_sec = time.time() - epoch_start_time
        elapsed_fold_time_sec = time.time() - fold_start_time
        avg_epoch_time_sec = elapsed_fold_time_sec / max(epoch, 1)
        remaining_epochs = max(epochs - epoch, 0)
        eta_fold_time_sec = avg_epoch_time_sec * remaining_epochs

        projected_best_f1 = max(best_f1, valid_f1)
        projected_best_epoch = epoch if valid_f1 > best_f1 else best_epoch
        print(
            f"\n[fold={fold} epoch={epoch}/{epochs}]\n"
            f"train_loss={train_loss:.5f} valid_loss={valid_loss:.5f} "
            f"f1={valid_f1:.5f} best_f1={projected_best_f1:.5f} best_epoch={projected_best_epoch}\n"
            f"threshold={threshold:.3f} precision={metrics['precision']:.5f} "
            f"recall={metrics['recall']:.5f} accuracy={metrics['accuracy']:.5f}\n"
            f"lr_backbone={current_lr_backbone:.2e} lr_head={current_lr_head:.2e}\n"
            f"time: train={format_seconds(train_time_sec)} valid={format_seconds(valid_time_sec)} "
            f"epoch={format_seconds(epoch_time_sec)} elapsed={format_seconds(elapsed_fold_time_sec)} "
            f"eta_fold={format_seconds(eta_fold_time_sec)}",
            flush=True,
        )

        per_epoch_metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_f1": valid_f1,
                "threshold": threshold,
                "metrics": metrics,
                "lr_backbone": current_lr_backbone,
                "lr_head": current_lr_head,
                "train_time_sec": train_time_sec,
                "valid_time_sec": valid_time_sec,
                "epoch_time_sec": epoch_time_sec,
                "elapsed_fold_time_sec": elapsed_fold_time_sec,
            }
        )

        # Track best-f1 checkpoint (always saved when new best f1 found)
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            best_epoch = epoch
            best_threshold = threshold
            best_predictions = pd.DataFrame(
                {
                    "id": valid_ids,
                    "label": y_true.astype(int),
                    "probability": y_prob,
                    "prediction": (y_prob >= threshold).astype(int),
                    "threshold": threshold,
                    "fold": fold,
                }
            )
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "fold": fold,
                "epoch": epoch,
                "best_f1": best_f1,
                "best_threshold": best_threshold,
                "model_name": config.get("model", {}).get("model_name", "convnext_tiny"),
                "image_size": config.get("model", {}).get("image_size", 384),
                "config": config,
                "label_mapping": label_mapping,
            }
            torch.save(checkpoint, output_dir / f"fold{fold}_best_f1.pth")
            torch.save(checkpoint, output_dir / f"fold{fold}_best.pth")
            no_improve_epochs = 0

        # Track best-loss checkpoint (lowest valid_loss so far)
        if epoch == 1 or valid_loss < best_loss_saved:
            best_loss_saved = valid_loss
            torch.save(checkpoint, output_dir / f"fold{fold}_best_loss.pth")

    # Save last checkpoint
    torch.save(checkpoint, output_dir / f"fold{fold}_last.pth")

    if best_predictions is None:
        raise RuntimeError(f"Fold {fold} finished without validation predictions.")

    best_predictions.to_csv(output_dir / f"fold{fold}_valid_predictions.csv", index=False)
    fold_metrics = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "best_loss": best_loss_saved,
        "class_counts": class_counts(train_df, label_col),
        "label_mapping": label_mapping,
        "epochs": per_epoch_metrics,
    }
    save_json(fold_metrics, output_dir / f"fold{fold}_metrics.json")
    return best_predictions, fold_metrics


def main() -> None:
    args = parse_args()
    if getattr(args, "quiet_warnings", False):
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")
    config = apply_cli_overrides(load_yaml(args.config), args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(int(config.get("seed", 42)))

    output_dir = ensure_dir(config.get("output", {}).get("dir", "outputs/convnext_tiny_384"))
    save_config_snapshot(config, output_dir)

    df, label_mapping = load_train_dataframe(config)
    label_col = config.get("data", {}).get("label_col", "label")

    # Ensure external negatives are excluded from StratifiedKFold split
    has_external = "_external_neg" in df.columns
    if has_external:
        split_df = df[df["_external_neg"] != 1].reset_index(drop=True)
    else:
        split_df = df

    n_splits = int(config.get("data", {}).get("n_splits", 5))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(config.get("seed", 42)))
    folds = list(splitter.split(split_df, split_df[label_col]))

    # Remap fold indices to align with full df (external negs get appended at the end)
    if has_external:
        # split_df indices are 0..N-1 (no externals). Map to original df positions.
        base_indices = split_df.index.values  # positions in the full df
        ext_indices = df[df["_external_neg"] == 1].index.values  # positions of external negs
        remapped_folds = []
        for train_idx, valid_idx in folds:
            remapped_train = np.concatenate([base_indices[train_idx], ext_indices])
            remapped_valid = base_indices[valid_idx]
            remapped_folds.append((remapped_train, remapped_valid))
        folds = remapped_folds

    if args.fold == "all":
        selected_folds = list(range(n_splits))
    else:
        selected_folds = [int(args.fold)]
        if selected_folds[0] < 0 or selected_folds[0] >= n_splits:
            raise ValueError(f"--fold must be one of 0..{n_splits - 1} or all; got {args.fold}")

    all_predictions = []
    per_fold_metrics = []
    all_start_time = time.time()
    total_selected_folds = len(selected_folds)
    for fold_position, fold in enumerate(selected_folds, start=1):
        fold_wall_start = time.time()
        preds, metrics = run_fold(fold, folds[fold][0], folds[fold][1], df, config, label_mapping, args, device)
        all_predictions.append(preds)
        per_fold_metrics.append(metrics)

        elapsed_all_time_sec = time.time() - all_start_time
        avg_fold_time_sec = elapsed_all_time_sec / max(fold_position, 1)
        remaining_folds = max(total_selected_folds - fold_position, 0)
        eta_all_time_sec = avg_fold_time_sec * remaining_folds
        fold_time_sec = time.time() - fold_wall_start
        if total_selected_folds > 1:
            print(
                f"\n[all-folds progress] finished={fold_position}/{total_selected_folds} "
                f"last_fold_time={format_seconds(fold_time_sec)} "
                f"elapsed={format_seconds(elapsed_all_time_sec)} eta_all={format_seconds(eta_all_time_sec)}",
                flush=True,
            )

    if args.fold == "all":
        oof = pd.concat(all_predictions, axis=0, ignore_index=True)
        oof.to_csv(output_dir / "oof_predictions.csv", index=False)
        threshold, oof_f1 = search_best_threshold(
            oof["label"].values,
            oof["probability"].values,
            float(config.get("threshold", {}).get("min", 0.05)),
            float(config.get("threshold", {}).get("max", 0.95)),
            float(config.get("threshold", {}).get("step", 0.005)),
        )
        oof_metrics = compute_binary_metrics(oof["label"].values, oof["probability"].values, threshold)
        metrics_json = {
            "per_fold": [
                {"fold": m["fold"], "best_f1": m["best_f1"], "best_threshold": m["best_threshold"], "best_epoch": m["best_epoch"]}
                for m in per_fold_metrics
            ],
            "oof_f1": oof_f1,
            "oof_best_threshold": threshold,
            "oof_metrics": oof_metrics,
            "class_counts": class_counts(df, label_col),
            "label_mapping": label_mapping,
            "config_snapshot": config,
        }
        save_json(metrics_json, output_dir / "metrics.json")


if __name__ == "__main__":
    main()