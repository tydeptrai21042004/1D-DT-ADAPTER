# models/tuning_modules/__init__.py

from .prompter import PadPrompter
from .conv_adapter import ConvAdapter, LinearAdapter
from .program_module import ProgramModule

def set_tuning_config(tuning_method, args):
    """
    Return a small config dict describing the chosen tuning method.
    Also normalizes a few legacy/alias names so old strings still work.

    Supported families:
      conv_adapt | conv_adapt_norm | conv_adapt_bias | conv | conv-adapter | conv_adapter
      prompt
      full | linear | norm | repnet | repnet_bias | bias
      hcc | hcc_adapter
      residual | residual_adapter | residual_adapters | ra
      sidetune | side-tuning | sidetuning | side_tune
    """
    # ---- Normalize aliases ----------------------------------------------------
    alias = {
        "conv": "conv_adapt",
        "conv-adapter": "conv_adapt",
        "conv_adapter": "conv_adapt",

        "hcc_adapter": "hcc",

        "residual_adapter": "residual",
        "residual_adapters": "residual",
        "ra": "residual",

        # new: side-tuning aliases
        "side-tuning": "sidetune",
        "sidetuning": "sidetune",
        "side_tune": "sidetune",
    }
    tm = alias.get(str(tuning_method), str(tuning_method))

    # ---- Conv-Adapter family --------------------------------------------------
    if tm in ("conv_adapt", "conv_adapt_norm", "conv_adapt_bias"):
        return {
            "method": tm,
            "kernel_size": getattr(args, "kernel_size", 3),
            "adapt_size": getattr(args, "adapt_size", 8),
            "adapt_scale": getattr(args, "adapt_scale", 1.0),
        }

    # ---- Prompt ---------------------------------------------------------------
    if tm == "prompt":
        return {
            "method": tm,
            "prompt_size": getattr(args, "prompt_size", 10),
        }

    # ---- Simple switches ------------------------------------------------------
    if tm in ("full", "linear", "norm", "repnet", "repnet_bias", "bias"):
        return {"method": tm}

    # ---- Hartley–Cosine Adapter (HCC) ----------------------------------------
    if tm == "hcc":
        return {
            "method": "hcc",
            "M":              getattr(args, "hcc_M", 1),
            "h":              getattr(args, "hcc_h", 1),
            "axis":           getattr(args, "hcc_axis", "hw"),
            "per_channel":    getattr(args, "hcc_per_channel", True),
            "tie_sym":        getattr(args, "hcc_tie_sym", True),
            "use_pw":         getattr(args, "hcc_use_pw", True),
            "pw_ratio":       getattr(args, "hcc_pw_ratio", 8),
            # optional flags in your repo; ignored by HCCAdapter if unused
            "residual_scale": getattr(args, "hcc_residual_scale", 1.0),
            "gate_init":      getattr(args, "hcc_gate_init", 0.1),
            "padding_mode":   getattr(args, "hcc_padding", "reflect"),
        }

    # ---- Residual Adapters ----------------------------------------------------
    if tm == "residual":
        return {
            "method": "residual",
            "mode":        getattr(args, "ra_mode", "parallel"),
            "reduction":   getattr(args, "ra_reduction", 16),
            "norm":        getattr(args, "ra_norm", "bn"),
            "act":         getattr(args, "ra_act", "relu"),
            "gate_init":   getattr(args, "ra_gate_init", 0.0),
            "stages":      getattr(args, "ra_stages", "1,2,3,4"),
        }

    # ---- Side-Tuning (wrapper does the work; config kept minimal) -------------
    if tm == "sidetune":
        return {
            "method": "sidetune",
            "alpha": getattr(args, "sidetune_alpha", 0.5),
            "learn_alpha": getattr(args, "sidetune_learn_alpha", True),
            "side_width": getattr(args, "sidetune_width", 64),
            "side_depth": getattr(args, "sidetune_depth", 3),
        }

    # ---- Unknown --------------------------------------------------------------
    raise NotImplementedError(f"Unknown tuning_method: {tuning_method}")
