# models/tuning_modules/side_tuning.py
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class _SideConvStem(nn.Module):
    """
    Lightweight side network: few 3x3 convs + BN + ReLU, then a 1x1 to C channels.
    """
    def __init__(self, in_ch: int, mid: int, out_ch: int, depth: int = 3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
                  nn.BatchNorm2d(mid), nn.ReLU(inplace=True)]
        for _ in range(max(0, depth - 2)):
            layers += [nn.Conv2d(mid, mid, 3, padding=1, bias=False),
                       nn.BatchNorm2d(mid), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(mid, out_ch, 1, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):              # [B,3,H,W] -> [B,C,H,W]
        return self.net(x)


class SideTuningClassifier(nn.Module):
    """
    Side-tuning wrapper.

    - Expects a backbone that returns features from .forward_features(x) or plain .forward(x)
      as either [B,C] or [B,C,H,W].
    - Keeps the backbone frozen; trains a small side net + classifier head + blend alpha.

    Memory optimization:
      * Blend AFTER global pooling (vector space [B,C]) to avoid broadcasting over [H,W].
      * Optional activation checkpointing on side_net to reduce peak VRAM
        (see torch.utils.checkpoint docs).  # official docs: https://pytorch.org/docs/stable/checkpoint.html
    """
    def __init__(self,
                 base_model: nn.Module,
                 num_classes: int,
                 side_width: int = 64,
                 side_depth: int = 3,
                 learn_alpha: bool = True,
                 alpha_init: float = 0.5,
                 use_checkpoint: bool = True):
        super().__init__()
        self.base = base_model
        self.use_checkpoint = use_checkpoint

        # Freeze the base
        for p in self.base.parameters():
            p.requires_grad = False

        # Infer base feature dim C
        C = getattr(self.base, 'num_features', None)
        if C is None:
            raise RuntimeError("Backbone must expose .num_features (int)")

        # Small side-net that maps image -> [B, C, H, W]
        self.side_net = _SideConvStem(3, side_width, C, depth=side_depth)

        # Alpha in (0,1) via logit parameterization
        a0 = float(alpha_init)
        a0 = min(max(a0, 0.0), 1.0)
        eps = 1e-4
        a0 = max(min(a0, 1.0 - eps), eps)
        a0_t = torch.tensor(a0)
        self.alpha_logit = nn.Parameter(torch.log(a0_t / (1.0 - a0_t)),
                                        requires_grad=learn_alpha)

        # Pool + classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(C, num_classes)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def _base_feats(self, x: torch.Tensor) -> torch.Tensor:
        # No grad on the frozen backbone
        with torch.no_grad():
            if hasattr(self.base, 'forward_features'):
                b = self.base.forward_features(x)
            else:
                b = self.base(x)
        if b.dim() == 4:
            b = self.pool(b).flatten(1)
        return b  # [B, C]

    def _side_feats(self, x: torch.Tensor) -> torch.Tensor:
        # Optionally activation checkpoint the side path
        if self.training and self.use_checkpoint:
            s = checkpoint(self.side_net, x)
        else:
            s = self.side_net(x)
        s = self.pool(s).flatten(1)    # [B, C]
        return s

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b_vec = self._base_feats(x)    # [B, C], no grad
        s_vec = self._side_feats(x)    # [B, C], trainable
        a = self.alpha
        fused = (1.0 - a) * b_vec + a * s_vec
        return fused                   # [B, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)   # [B, C]
        return self.head(z)            # [B, num_classes]
