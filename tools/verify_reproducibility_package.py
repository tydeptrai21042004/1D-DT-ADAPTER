#!/usr/bin/env python3
"""Fail fast when CNN three-seed reproducibility assets are missing or inconsistent."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "paper" / "cnn_three_seed_manifest.yaml"
REQUIRED = [
    "README.md", "REPRODUCIBILITY.md", "MANUSCRIPT_TO_CODE.md", "MANUSCRIPT_ALIGNMENT_NOTES.md", "VERSION", "CHANGELOG.md",
    "requirements.txt", "requirements-kaggle.txt", "environment.yml",
    "CITATION.cff", "codemeta.json", ".zenodo.json", "LICENSE",
    "models/dt1d_adapter.py", "models/hcc_adapter.py",
    "configs/paper/cnn_three_seed_manifest.yaml",
    "splits/caltech101/seed0_holdout20.json", "splits/caltech101/seed1_holdout20.json", "splits/caltech101/seed2_holdout20.json",
    "reproducibility/seeds.json", "reproducibility/table_to_command.csv",
    "tools/run_from_config.py", "tools/run_cnn_paper.py", "tools/aggregate_cnn_paper.py", "tools/preflight_cnn_matrix.py",
    "tools/plot_figure_01_three_seed.py", "tools/plot_figure_02_spectral.py",
    "tools/plot_figure_03_architecture.py", "tools/plot_figure_04_tradeoff.py",
    "scripts/run_all_cnn_tables_three_seed.sh",
    "KAGGLE_CNN_THREE_SEED_RUN.sh", "CREATE_AND_PUSH_CNN_THREE_SEED_BRANCH.sh",
    "logs/validation/cnn_model_preflight.json", "logs/validation/cnn_method_smoke_summary.json",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Missing reproducibility files:\n" + "\n".join(missing))

    for table in [f"{i:02d}" for i in range(2, 14)] + ["14_15", "18_19"]:
        path = ROOT / "scripts" / "tables" / f"table_{table}_three_seed.sh"
        if not path.is_file():
            raise SystemExit(f"Missing table runner: {path.relative_to(ROOT)}")
    for figure in ("01_three_seed", "02_deterministic", "03_deterministic", "04_three_seed"):
        path = ROOT / "scripts" / "figures" / f"figure_{figure}.sh"
        if not path.is_file():
            raise SystemExit(f"Missing figure runner: {path.relative_to(ROOT)}")

    # Parse active Python and YAML files.
    python_files = [ROOT / "main.py", *sorted((ROOT / "models").rglob("*.py")), *sorted((ROOT / "datasets").rglob("*.py")), *sorted((ROOT / "tools").rglob("*.py")), *sorted((ROOT / "tests").rglob("*.py"))]
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    if manifest.get("scope") != "cnn_classification_only":
        raise SystemExit("Manifest scope must be cnn_classification_only.")
    if manifest.get("default_seeds") != [0, 1, 2]:
        raise SystemExit("The official independent seed set must be [0, 1, 2].")

    forbidden_backbone_tokens = ("vit", "swin", "transformer", "deit", "beit", "clip")
    comparison_targets = 0
    for target_name, target in manifest["targets"].items():
        backbone = str(target["backbone"]).lower()
        if any(token in backbone for token in forbidden_backbone_tokens):
            raise SystemExit(f"Non-CNN backbone in {target_name}: {backbone}")
        if target.get("kind") in {"comparison", "tradeoff_figure"}:
            comparison_targets += 1
            methods = set(target.get("methods", []))
            if not {"full", "linear", "dt1d"}.issubset(methods):
                raise SystemExit(f"{target_name} must contain dt1d, full, and linear.")

    generated_expected = 0
    for target_name, target in manifest["targets"].items():
        if target.get("kind") == "ablation":
            expected_names = [
                f"dt1d__{variant}_seed{seed}.yaml"
                for variant in target.get("variants", {})
                for seed in manifest["default_seeds"]
            ]
        else:
            expected_names = [
                f"{method}_seed{seed}.yaml"
                for method in target.get("methods", [])
                for seed in manifest["default_seeds"]
            ]
        generated_dir = ROOT / "configs" / "paper" / "generated" / target_name
        for filename in expected_names:
            path = generated_dir / filename
            if not path.is_file():
                raise SystemExit(f"Missing generated config: {path.relative_to(ROOT)}")
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            args = cfg.get("args", {})
            if args.get("data_path") != "data" or args.get("device") != "cuda":
                raise SystemExit(f"Non-portable generated config: {path.relative_to(ROOT)}")
            if args.get("deterministic") is not True:
                raise SystemExit(f"Generated config is not deterministic: {path.relative_to(ROOT)}")
            if args.get("dataset") == "fake" or str(args.get("weights", "")).lower() == "none":
                raise SystemExit(f"Smoke configuration leaked into publication configs: {path.relative_to(ROOT)}")
        generated_expected += len(expected_names)
    actual_generated = len(list((ROOT / "configs" / "paper" / "generated").rglob("*.yaml")))
    if actual_generated != generated_expected:
        raise SystemExit(f"Generated config count mismatch: expected={generated_expected}, actual={actual_generated}")

    method_presets = manifest["method_presets"]
    if method_presets["full"]["args"].get("tuning_method") != "full":
        raise SystemExit("Full fine-tuning preset is invalid.")
    if method_presets["linear"]["args"].get("tuning_method") != "linear":
        raise SystemExit("Linear probing preset is invalid.")
    if method_presets["dt1d"]["args"].get("tuning_method") != "dt":
        raise SystemExit("Canonical DT1D preset must use tuning_method=dt.")
    if any(str(k).startswith("hcc_") for k in method_presets["dt1d"]["args"]):
        raise SystemExit("Canonical DT1D preset contains deprecated hcc_* keys.")

    split_report = {}
    for seed in (0, 1, 2):
        path = ROOT / "splits" / "caltech101" / f"seed{seed}_holdout20.json"
        split = json.loads(path.read_text(encoding="utf-8"))
        train, val, test = split["train_indices"], split["val_indices"], split["test_indices"]
        if len(train) != 6942 or len(val) != 868 or len(test) != 867:
            raise SystemExit(f"Caltech101 seed {seed} split sizes are incorrect.")
        combined = train + val + test
        if len(set(combined)) != split["dataset_length"] or len(combined) != split["dataset_length"]:
            raise SystemExit(f"Caltech101 seed {seed} split is not a disjoint full partition.")
        split_report[str(seed)] = {"train": len(train), "validation": len(val), "test": len(test)}

    canonical = (ROOT / "models" / "dt1d_adapter.py").read_text(encoding="utf-8")
    shim = (ROOT / "models" / "hcc_adapter.py").read_text(encoding="utf-8")
    if "class DT1DAdapter" not in canonical:
        raise SystemExit("Canonical DT1DAdapter class is missing.")
    if "from .dt1d_adapter import DT1DAdapter" not in shim:
        raise SystemExit("Legacy module must remain a thin compatibility shim.")
    if ").to(device=fused.device, dtype=fused.dtype)" not in canonical:
        raise SystemExit("AMP-safe fused-dtype accumulation is missing.")

    kaggle = (ROOT / "KAGGLE_CNN_THREE_SEED_RUN.sh").read_text(encoding="utf-8")
    if "dt1d-v8-cnn-three-seed" not in kaggle:
        raise SystemExit("Kaggle runner does not name the v0.8 branch.")
    if 'SEEDS="${SEEDS:-0,1,2}"' not in kaggle:
        raise SystemExit("Kaggle runner does not default to seeds 0,1,2.")

    for shell_path in sorted(ROOT.rglob("*.sh")):
        shell_text = shell_path.read_text(encoding="utf-8")
        for forbidden in ("git apply", "PY_AMP_PATCH", "text.replace(original, corrected"):
            if forbidden in shell_text:
                raise SystemExit(f"Runner {shell_path.relative_to(ROOT)} patches source at runtime: {forbidden}")

    report = {
        "status": "ok",
        "version": (ROOT / "VERSION").read_text(encoding="utf-8").strip(),
        "branch": (ROOT / "BRANCH_NAME.txt").read_text(encoding="utf-8").strip(),
        "scope": manifest["scope"],
        "official_seeds": manifest["default_seeds"],
        "target_count": len(manifest["targets"]),
        "comparison_targets_with_full_and_linear": comparison_targets,
        "committed_per_run_configs": generated_expected,
        "caltech101_splits": split_report,
        "required_file_sha256": {name: digest(ROOT / name) for name in REQUIRED},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
