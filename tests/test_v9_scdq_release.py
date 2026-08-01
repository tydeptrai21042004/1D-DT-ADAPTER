from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from torchvision.models import resnet18

import main as training

ROOT = Path(__file__).resolve().parents[1]


def test_main_manifest_uses_final_scdq_configuration():
    manifest = yaml.safe_load((ROOT / "configs/paper/cnn_three_seed_manifest.yaml").read_text())
    preset = manifest["method_presets"]["dt1d"]
    args = preset["args"]
    assert preset["label"] == "SCDQ-DT1D-Adapter"
    assert args["dt_minimal_quotient_realization"] is True
    assert args["dt_quotient_support_cap"] == 4
    assert args["dt_no_pw"] is True
    assert args["dt_padding"] == "reflect"
    assert args["dt_exact_cost_realization"] is False


def test_scdq_ablation_manifest_has_thirteen_three_seed_variants():
    manifest = yaml.safe_load((ROOT / "configs/experiments/scdq_three_seed_manifest.yaml").read_text())
    target = manifest["targets"]["table_14_15_scdq_ablation"]
    assert manifest["default_seeds"] == [0, 1, 2]
    assert len(target["variants"]) == 13
    assert "scdq4_reflect_core_final" in target["variants"]
    assert "legacy_v8_reflect_pointwise" in target["variants"]
    assert "mlq8_replicate_pointwise" in target["variants"]


def test_strict_bitfit_does_not_collapse_to_linear_probe():
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 101)
    args = training.get_args_parser().parse_args([])
    args.tuning_method = "bitfit"
    args.bitfit_train_head = True
    training.set_trainability_policy(model, args)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    head_count = sum(p.numel() for name, p in model.named_parameters() if training._is_head_param(name))
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_trainable > head_count
    assert "bn1.bias" in trainable
    assert "bn1.weight" not in trainable
    assert getattr(model.bn1, "_dt1d_force_eval", False) is True


def test_run_from_config_dry_run_writes_metadata(tmp_path: Path):
    config = {
        "schema_version": 2,
        "experiment_id": "metadata_test",
        "target": "unit_test",
        "kind": "ablation",
        "method_preset": "dt1d",
        "method_label": "SCDQ-DT1D-Adapter",
        "variant": "final",
        "independent_seed": 0,
        "args": {
            "dataset": "fake", "backbone": "resnet18", "weights": "none",
            "device": "cpu", "epochs": 1, "batch_size": 2,
            "tuning_method": "dt", "freeze_backbone": True,
        },
    }
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    subprocess.check_call([
        sys.executable, str(ROOT / "tools/run_from_config.py"), str(config_path),
        "--output-dir", str(output_dir), "--dry-run",
    ], cwd=ROOT)
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["target"] == "unit_test"
    assert metadata["variant"] == "final"
    assert "args" not in metadata
