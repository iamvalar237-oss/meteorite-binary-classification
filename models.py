from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn


def _resolve_pretrained_checkpoint_path(checkpoint_path: str, data_root_dir: str = ".") -> Path:
    """Resolve a local pretrained checkpoint relative to data.root_dir or project root."""
    raw_path = Path(checkpoint_path).expanduser()
    project_root = Path(__file__).resolve().parent

    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        data_root = Path(data_root_dir).expanduser()
        candidates = []
        if data_root.is_absolute():
            candidates.append(data_root / raw_path)
        else:
            candidates.append(project_root / data_root / raw_path)
        candidates.append(project_root / raw_path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked_paths = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Configured model.pretrained_checkpoint_path does not exist: '{checkpoint_path}'. "
        f"Checked paths: {checked_paths}"
    )


class BinaryTimmModel(nn.Module):
    """Generic timm backbone with a lightweight binary classification head."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        dropout: float = 0.3,
        pretrained_checkpoint_path: Optional[Path] = None,
    ):
        super().__init__()
        try:
            import timm

            if pretrained_checkpoint_path is not None:
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=0,
                    pretrained_cfg_overlay={"file": str(pretrained_checkpoint_path)},
                )
            else:
                self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception as exc:
            if pretrained_checkpoint_path is not None:
                raise RuntimeError(
                    f"Failed to create timm model '{model_name}' from local pretrained checkpoint "
                    f"'{pretrained_checkpoint_path}'. Please check that model.model_name matches the checkpoint weights."
                ) from exc
            raise RuntimeError(
                f"Failed to create timm model '{model_name}'. "
                "Please check the model_name in config and your installed timm version."
            ) from exc

        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            raise RuntimeError(f"Could not infer feature dimension for timm model '{model_name}'.")

        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits.squeeze(1)


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Build a pretrained ConvNeXt/timm backbone plus binary head from config."""
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    model_name = model_cfg.get("model_name", "convnext_tiny")
    pretrained = bool(model_cfg.get("pretrained", True))
    dropout = float(model_cfg.get("dropout", 0.3))
    num_classes = int(model_cfg.get("num_classes", 1))
    if num_classes != 1:
        raise ValueError("This binary pipeline expects model.num_classes: 1")
    checkpoint_path = str(model_cfg.get("pretrained_checkpoint_path", "")).strip()
    resolved_checkpoint_path = None
    if checkpoint_path:
        resolved_checkpoint_path = _resolve_pretrained_checkpoint_path(
            checkpoint_path=checkpoint_path,
            data_root_dir=str(data_cfg.get("root_dir", ".")),
        )
    return BinaryTimmModel(
        model_name=model_name,
        pretrained=pretrained,
        dropout=dropout,
        pretrained_checkpoint_path=resolved_checkpoint_path,
    )