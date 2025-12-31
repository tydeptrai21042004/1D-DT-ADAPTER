# models/__init__.py

import torch
import torch.nn as nn

from .backbones import *           # resnet50, ...
from .heads import *               # LinearHead
from .tuning_modules import set_tuning_config
from .layers.ws_conv import WSConv2d

# Side-tuning wrapper
from .tuning_modules.side_tuning import SideTuningClassifier

__all__ = ['build_model']


def replace_conv2d_with_my_conv2d(net, ws_eps=None):
    if ws_eps is None:
        return
    for m in net.modules():
        to_update = {}
        for name, sub in m.named_children():
            if isinstance(sub, nn.Conv2d) and sub.bias is None:
                to_update[name] = sub
        for name, sub in to_update.items():
            m._modules[name] = WSConv2d(
                sub.in_channels, sub.out_channels, sub.kernel_size, sub.stride,
                sub.padding, sub.dilation, sub.groups, sub.bias is not None,
            )
            m._modules[name].load_state_dict(sub.state_dict())
            m._modules[name].weight.requires_grad = sub.weight.requires_grad
            if sub.bias is not None:
                m._modules[name].bias.requires_grad = sub.bias.requires_grad
    for m in net.modules():
        if isinstance(m, WSConv2d):
            m.ws_eps = ws_eps


def _safe_tuning_config(tuning_method, args):
    """
    Some repos don't define a config for 'sidetune'; in that case,
    just return a neutral config so backbones build cleanly.
    """
    try:
        return set_tuning_config(tuning_method, args)
    except NotImplementedError:
        if str(tuning_method) == 'sidetune':
            return {"method": "full"}  # base acts like vanilla backbone
        raise


def build_model(model_name, pretrained=True, num_classes=1000, input_size=224,
                tuning_method='full', args=None, **kwargs):
    """
    Build a backbone and apply the requested parameter-efficient tuning method.
    'sidetune' wraps a frozen backbone with a lightweight side network + alpha blending.
    """
    # 1) Build the base backbone
    tuning_config = _safe_tuning_config(tuning_method, args)
    base = eval(model_name)(
        pretrained=pretrained,
        tuning_config=tuning_config,
        input_resolution=input_size,
        **kwargs
    )

    # 2) Wrap (sidetune) or attach standard head
    if str(tuning_method) == 'sidetune':
        model = SideTuningClassifier(
            base_model=base,                                      # ✅ correct kw
            num_classes=num_classes,
            side_width=int(getattr(args, 'sidetune_width', 64)),
            side_depth=int(getattr(args, 'sidetune_depth', 3)),
            learn_alpha=bool(getattr(args, 'sidetune_learn_alpha', True)),
            alpha_init=float(getattr(args, 'sidetune_alpha', 0.5)),
            use_checkpoint=True,                                  # saves VRAM (see docs)
        )
    else:
        model = base
        model.head = LinearHead(model.num_features, num_classes, drop=0.2)

    # 3) Freeze/unfreeze according to tuning method
    if tuning_method == 'full':
        pass
    elif tuning_method == 'prompt':
        for name, p in model.named_parameters():
            if name.startswith('head'):         # train head
                continue
            if name.startswith('norm'):         # keep norms trainable
                continue
            if 'tuning_module' in name:         # prompt modules
                continue
            p.requires_grad = False
    elif tuning_method == 'adapter':
        raise NotImplementedError
    elif tuning_method == 'sidetune':
        # Train only side network + alpha + head; freeze the base
        for name, p in model.named_parameters():
            train_ok = (
                name.startswith('side_net.') or                   # ✅ correct prefix
                name.startswith('head.') or
                name == 'alpha_logit'
            )
            p.requires_grad = train_ok
    elif tuning_method == 'linear':
        for name, p in model.named_parameters():
            if name.startswith('head') or name.startswith('norm'):
                continue
            p.requires_grad = False
    elif tuning_method == 'norm':
        for name, p in model.named_parameters():
            if name.startswith('head'):
                continue
            if ('bn' in name) or ('gn' in name) or ('norm' in name):
                continue
            if 'before_head' in name:
                continue
            p.requires_grad = False
    elif tuning_method == 'bias':
        for name, p in model.named_parameters():
            if name.startswith('head') or name.startswith('norm') or ('bias' in name):
                continue
            p.requires_grad = False
    elif tuning_method in ('conv_adapt', 'repnet'):
        for name, p in model.named_parameters():
            if name.startswith('head'):
                continue
            if 'tuning_module' in name:
                continue
            if 'norm' in name:
                continue
            p.requires_grad = False
    elif tuning_method == 'conv_adapt_norm':
        for name, p in model.named_parameters():
            if name.startswith('head'):
                continue
            if 'tuning_module' in name:
                continue
            if ('bn' in name) or ('gn' in name) or ('norm' in name):
                continue
            if 'before_head' in name:
                continue
            p.requires_grad = False
    elif tuning_method in ('conv_adapt_bias', 'repnet_bias'):
        for name, p in model.named_parameters():
            if name.startswith('head'):
                continue
            if 'tuning_module' in name:
                continue
            if 'bias' in name:
                continue
            if name.startswith('norm'):
                continue
            p.requires_grad = False

    if 'repnet' in str(tuning_method):
        replace_conv2d_with_my_conv2d(model, 1e-5)

    # 4) Debug: list trainable params (optional)
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"{n} is trainable")

    return model
