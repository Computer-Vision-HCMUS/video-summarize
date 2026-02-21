"""CNN feature extractor (ResNet-based placeholder)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# Optional: use timm for more backbones
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class CNNFeatureExtractor(nn.Module):
    """
    ResNet-based feature extractor. Outputs (T, feature_dim) for a sequence of frames.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        feature_dim: Optional[int] = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._build_backbone(backbone, pretrained)
        self.feature_dim = feature_dim or self._infer_feature_dim()

    def _build_backbone(self, backbone: str, pretrained: bool) -> None:
        if backbone == "resnet50":
            m = models.resnet50(weights="DEFAULT" if pretrained else None)
            self._features = nn.Sequential(*list(m.children())[:-1])
            self._out_dim = 2048
        elif backbone == "resnet18":
            m = models.resnet18(weights="DEFAULT" if pretrained else None)
            self._features = nn.Sequential(*list(m.children())[:-1])
            self._out_dim = 512
        elif backbone == "googlenet":
            m = models.googlenet(weights="DEFAULT" if pretrained else None, transform_input=True)
            m.fc = nn.Identity()
            self._features = m
            self._out_dim = 1024
        elif HAS_TIMM and backbone:
            self._features = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            self._out_dim = self._features.num_features
        else:
            m = models.resnet50(weights="DEFAULT" if pretrained else None)
            self._features = nn.Sequential(*list(m.children())[:-1])
            self._out_dim = 2048
        self._features.eval()

    def _infer_feature_dim(self) -> int:
        return getattr(self, "_out_dim", 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W) or (B, C, H, W).
        Returns (B, T, D) or (B, D).
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            out = self._features(x)
            if isinstance(out, tuple):
                out = out[0]
            out = out.view(B, T, -1)
        else:
            out = self._features(x)
            if isinstance(out, tuple):
                out = out[0]
            out = out.view(out.size(0), -1)
        return out

    def extract_from_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract features from list of RGB numpy frames (H, W, 3).
        Returns (T, feature_dim) numpy array.
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.eval()
        features_list = []
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch = frames[i : i + batch_size]
                tensors = torch.stack([transform(f) for f in batch]).to(self.device)
                out = self.forward(tensors)
                features_list.append(out.cpu().numpy())
        return np.vstack(features_list).astype(np.float32)
