from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_cnn_paper", ROOT / "tools/run_cnn_paper.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_runner_materializes_three_independent_seed_configs(tmp_path):
    manifest = MODULE.load_manifest(ROOT / "configs/paper/cnn_three_seed_manifest.yaml")
    runs = list(MODULE.iter_runs(
        manifest=manifest,
        names=["table_14_15"],
        seeds=[0, 1, 2],
        requested_methods="full,linear",
        output_root=tmp_path / "outputs",
        data_path=tmp_path / "data",
        device="cpu",
        smoke=True,
    ))
    assert len(runs) == 6
    seen = set()
    for config_path, output_dir, config in runs:
        parsed = yaml.safe_load(config_path.read_text())
        seen.add((parsed["method_preset"], parsed["independent_seed"]))
        assert parsed["args"]["dataset"] == "fake"
        assert parsed["args"]["backbone"] == "resnet18"
        assert output_dir.is_dir()
    assert seen == {(m, s) for m in ("full", "linear") for s in (0, 1, 2)}


def test_non_cnn_backbone_is_rejected():
    try:
        MODULE.ensure_cnn_backbone("vit_b_16")
    except SystemExit:
        pass
    else:
        raise AssertionError("ViT backbone was not rejected by CNN-only runner")
