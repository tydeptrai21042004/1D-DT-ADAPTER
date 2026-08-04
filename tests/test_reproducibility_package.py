from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    return yaml.safe_load((ROOT / "configs/paper/cnn_three_seed_manifest.yaml").read_text())


def test_required_publication_files_exist():
    required = [
        "requirements.txt", "environment.yml", "CITATION.cff",
        "REPRODUCIBILITY.md", "MANUSCRIPT_TO_CODE.md", "MANUSCRIPT_ALIGNMENT_NOTES.md",
        "configs/paper/cnn_three_seed_manifest.yaml",
        "tools/run_cnn_paper.py", "tools/aggregate_cnn_paper.py",
        "KAGGLE_CNN_THREE_SEED_RUN.sh",
    ]
    for relative in required:
        assert (ROOT / relative).is_file(), relative


def test_manifest_is_cnn_only_and_three_seed():
    cfg = manifest()
    assert cfg["scope"] == "cnn_classification_only"
    assert cfg["default_seeds"] == [0, 1, 2]
    forbidden = ("vit", "swin", "transformer", "deit", "beit", "clip")
    for target in cfg["targets"].values():
        assert not any(token in target["backbone"].lower() for token in forbidden)


def test_comparison_targets_include_dt_full_and_linear():
    cfg = manifest()
    for name, target in cfg["targets"].items():
        if target["kind"] in {"comparison", "tradeoff_figure"}:
            assert {"dt1d", "full", "linear"}.issubset(target["methods"]), name


def test_dt_preset_uses_canonical_names():
    args = manifest()["method_presets"]["dt1d"]["args"]
    assert args["tuning_method"] == "dt"
    assert not any(k.startswith("hcc_") for k in args)


def test_caltech_splits_are_exact_disjoint_partitions():
    for seed in (0, 1, 2):
        split = json.loads((ROOT / f"splits/caltech101/seed{seed}_holdout20.json").read_text())
        train = split["train_indices"]
        val = split["val_indices"]
        test = split["test_indices"]
        assert len(train) == 6942
        assert len(val) == 868
        assert len(test) == 867
        assert set(train).isdisjoint(val)
        assert set(train).isdisjoint(test)
        assert set(val).isdisjoint(test)
        assert len(set(train + val + test)) == split["dataset_length"] == 8677


def test_legacy_module_is_only_a_shim():
    text = (ROOT / "models/hcc_adapter.py").read_text()
    assert "from .dt1d_adapter import DT1DAdapter" in text
    assert "class DT1DAdapter" not in text


def test_all_publication_configs_are_committed_portable_and_non_smoke():
    cfg = manifest()
    expected = 0
    for target_name, target in cfg["targets"].items():
        if target["kind"] == "ablation":
            names = [f"dt1d__{variant}_seed{seed}.yaml" for variant in target["variants"] for seed in cfg["default_seeds"]]
        else:
            names = [f"{method}_seed{seed}.yaml" for method in target["methods"] for seed in cfg["default_seeds"]]
        expected += len(names)
        for name in names:
            path = ROOT / "configs/paper/generated" / target_name / name
            assert path.is_file(), path
            generated = yaml.safe_load(path.read_text())
            args = generated["args"]
            assert args["data_path"] == "data"
            assert args["device"] == "cuda"
            assert args["deterministic"] is True
            assert args["dataset"] != "fake"
            assert str(args["weights"]).lower() != "none"
    assert len(list((ROOT / "configs/paper/generated").rglob("*.yaml"))) == expected
