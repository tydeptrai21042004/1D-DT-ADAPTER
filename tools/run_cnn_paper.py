#!/usr/bin/env python3
"""Run CNN-only manuscript tables/figures with independent seeds.

The default seed set is 0, 1, 2. Each method receives the same dataset split for a
fixed seed and a different deterministic split/data order for a different seed when
the dataset has no official validation partition.
"""
from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "paper" / "cnn_three_seed_manifest.yaml"


def parse_csv_ints(text: str) -> list[int]:
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Seeds must be unique independent runs.")
    return values


def parse_csv_strings(text: str) -> list[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def deep_merge(*maps: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mapping in maps:
        for key, value in mapping.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid manifest: {path}")
    for required in ("default_seeds", "common_args", "method_presets", "targets"):
        if required not in data:
            raise SystemExit(f"Manifest is missing {required!r}.")
    return data


def ensure_cnn_backbone(backbone: str) -> None:
    lowered = backbone.lower()
    forbidden = ("vit", "swin", "transformer", "deit", "beit", "clip")
    if any(token in lowered for token in forbidden):
        raise SystemExit(f"CNN-only package rejected non-CNN backbone: {backbone}")


def target_names(requested: str, manifest: dict[str, Any]) -> list[str]:
    all_names = list(manifest["targets"])
    if requested == "all":
        return all_names
    names = parse_csv_strings(requested)
    unknown = [name for name in names if name not in manifest["targets"]]
    if unknown:
        raise SystemExit(f"Unknown target(s): {unknown}. Available: {all_names}")
    return names


def selected_methods(
    target: dict[str, Any],
    manifest: dict[str, Any],
    requested: str | None,
) -> list[str]:
    if requested is None or requested == "target":
        methods = list(target.get("methods", []))
    elif requested == "all-cnn":
        methods = list(manifest["all_cnn_methods"])
    else:
        methods = parse_csv_strings(requested)
    unknown = [method for method in methods if method not in manifest["method_presets"]]
    if unknown:
        raise SystemExit(f"Unknown method preset(s): {unknown}")
    return methods


def write_run_config(
    *,
    manifest: dict[str, Any],
    target_name: str,
    target: dict[str, Any],
    method_name: str,
    seed: int,
    output_root: Path,
    data_path: Path,
    device: str,
    variant_name: str | None = None,
    variant_args: dict[str, Any] | None = None,
    smoke: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    preset = manifest["method_presets"][method_name]
    common = copy.deepcopy(manifest["common_args"])
    target_args = {
        "dataset": target["dataset"],
        "backbone": target["backbone"],
        "epochs": int(target["epochs"]),
        "batch_size": int(target["batch_size"]),
    }
    run_args = deep_merge(common, target_args, target.get("args", {}), preset.get("args", {}), variant_args or {})
    # Keep committed YAML portable. The launcher passes the actual data path and
    # device to run_from_config at execution time.
    run_args.update({"seed": int(seed), "data_path": "data", "device": device})
    split_candidate = ROOT / "splits" / str(target["dataset"]) / f"seed{seed}_holdout20.json"
    if split_candidate.is_file():
        run_args["split_file"] = str(split_candidate.relative_to(ROOT))

    if smoke:
        run_args.pop("split_file", None)
        run_args.update({
            "dataset": "fake",
            "weights": "none",
            "device": "cpu",
            "input_size": 64,
            "epochs": 1,
            "batch_size": 4,
            "fake_train_size": 12,
            "fake_val_size": 8,
            "fake_test_size": 8,
            "fake_num_classes": 5,
            "num_workers": 0,
            "use_amp": False,
            "pin_mem": False,
            "profile_efficiency": False,
            "measure_eval_latency": False,
            "save_ckpt": False,
            "deterministic": True,
        })

    ensure_cnn_backbone(str(run_args["backbone"]))
    run_key = method_name if variant_name is None else f"{method_name}__{variant_name}"
    output_dir = output_root / target_name / str(run_args["dataset"]) / str(run_args["backbone"]) / run_key / f"seed_{seed}"
    # Smoke tests must never overwrite the publication configurations committed
    # under configs/paper/generated.
    generated_base = (output_root / "_generated_configs") if smoke else (ROOT / "configs" / "paper" / "generated")
    generated_dir = generated_base / target_name
    generated_dir.mkdir(parents=True, exist_ok=True)
    config_path = generated_dir / f"{run_key}_seed{seed}.yaml"

    config = {
        "schema_version": 2,
        "experiment_id": f"{target_name}_{run_key}_seed{seed}",
        "target": target_name,
        "kind": target.get("kind"),
        "manuscript_tables": target.get("manuscript_tables", []),
        "manuscript_figures": target.get("manuscript_figures", []),
        "method_preset": method_name,
        "method_label": preset.get("label", method_name),
        "variant": variant_name,
        "independent_seed": int(seed),
        "description": manifest.get("description"),
        "args": run_args,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(
        json.dumps({k: v for k, v in config.items() if k != "args"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path, output_dir, config


def iter_runs(
    *,
    manifest: dict[str, Any],
    names: Iterable[str],
    seeds: list[int],
    requested_methods: str | None,
    output_root: Path,
    data_path: Path,
    device: str,
    smoke: bool,
):
    for target_name in names:
        target = manifest["targets"][target_name]
        ensure_cnn_backbone(str(target["backbone"]))
        methods = selected_methods(target, manifest, requested_methods)
        if target.get("kind") == "ablation":
            if requested_methods not in (None, "target", "dt1d"):
                raise SystemExit("Ablation target table_02 only supports DT1D variants.")
            for variant_name, variant_args in target.get("variants", {}).items():
                for seed in seeds:
                    yield write_run_config(
                        manifest=manifest,
                        target_name=target_name,
                        target=target,
                        method_name="dt1d",
                        seed=seed,
                        output_root=output_root,
                        data_path=data_path,
                        device=device,
                        variant_name=variant_name,
                        variant_args=variant_args,
                        smoke=smoke,
                    )
        else:
            for method in methods:
                for seed in seeds:
                    yield write_run_config(
                        manifest=manifest,
                        target_name=target_name,
                        target=target,
                        method_name=method,
                        seed=seed,
                        output_root=output_root,
                        data_path=data_path,
                        device=device,
                        smoke=smoke,
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="all", help="Target name, comma-separated targets, or all.")
    parser.add_argument("--seeds", type=parse_csv_ints, default=None, help="Independent seeds, e.g. 0,1,2.")
    parser.add_argument("--methods", default=None, help="target, all-cnn, or comma-separated presets.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "cnn_paper_three_seed")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-if-complete", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use a tiny local FakeData run on CPU.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional debugging cap.")
    ns = parser.parse_args()

    manifest_path = ns.manifest if ns.manifest.is_absolute() else ROOT / ns.manifest
    manifest = load_manifest(manifest_path)
    seeds = ns.seeds if ns.seeds is not None else [int(v) for v in manifest["default_seeds"]]
    names = target_names(ns.target, manifest)
    output_root = ns.output_root if ns.output_root.is_absolute() else ROOT / ns.output_root
    data_path = ns.data_path if ns.data_path.is_absolute() else ROOT / ns.data_path

    runs = list(iter_runs(
        manifest=manifest,
        names=names,
        seeds=seeds,
        requested_methods=ns.methods,
        output_root=output_root,
        data_path=data_path.resolve(),
        device=ns.device,
        smoke=ns.smoke,
    ))
    if ns.max_runs is not None:
        runs = runs[: max(0, ns.max_runs)]
    if not runs:
        raise SystemExit("No runs selected.")

    plan = {
        "manifest": str(manifest_path),
        "targets": names,
        "seeds": seeds,
        "methods_request": ns.methods or "target",
        "run_count": len(runs),
        "smoke": ns.smoke,
        "runs": [
            {"config": str(config), "output_dir": str(out), "experiment_id": meta["experiment_id"]}
            for config, out, meta in runs
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "execution_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))

    for index, (config_path, output_dir, _meta) in enumerate(runs, start=1):
        command = [
            sys.executable,
            str(ROOT / "tools" / "run_from_config.py"),
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--data-path",
            str(data_path.resolve()),
            "--device",
            "cpu" if ns.smoke else ns.device,
        ]
        if ns.dry_run:
            command.append("--dry-run")
        if ns.skip_if_complete:
            command.append("--skip-if-complete")
        print(f"[{index}/{len(runs)}] {shlex.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            return completed.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
