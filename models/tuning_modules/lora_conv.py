# models/tuning_modules/lora_conv.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, r: int = 4, alpha: float = 1.0):
        super().__init__()
        assert isinstance(base, nn.Conv2d)
        self.base = base

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)

        # freeze base conv
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # only support groups==1 in this simple baseline
        if self.base.groups != 1:
            self.lora_down = None
            self.lora_up = None
            return

        in_ch = self.base.in_channels
        out_ch = self.base.out_channels
        kH, kW = self.base.kernel_size

        # down: 1x1 (in -> r)
        self.lora_down = nn.Conv2d(in_ch, self.r, kernel_size=1, bias=False)
        # up: same kernel/stride/pad/dilation as base (r -> out)
        self.lora_up = nn.Conv2d(
            self.r, out_ch,
            kernel_size=(kH, kW),
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=1,
            bias=False,
        )

        # init: start near-zero so it's identity-like
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        y = self.base(x)
        if self.lora_down is None:
            return y
        return y + self.scaling * self.lora_up(self.lora_down(x))


def apply_lora_conv2d(model: nn.Module, r: int = 4, alpha: float = 1.0):
    """
    Replace Conv2d layers (groups==1) by LoRAConv2d wrappers, in-place.
    """
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d):
                wrapped = LoRAConv2d(child, r=r, alpha=alpha)
                setattr(parent, name, wrapped)
