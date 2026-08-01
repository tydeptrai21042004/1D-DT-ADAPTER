#!/usr/bin/env python3
"""Instantiate the CNN experiment matrix and verify trainability policies.

This is a no-dataset, no-pretrained-download preflight. It validates that every
selected target/method can construct its CNN model and that Full fine-tuning,
Linear probing, and PEFT trainability policies are internally consistent.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASS_COUNTS = {
    "dtd": 47,
    "flowers102": 102,
    "svhn": 10,
    "oxford_iiit_pet": 37,
    "food101": 101,
    "caltech101": 101,
    "eurosat": 10,
    "fgvc_aircraft": 100,
}

# Historical manuscript counts used only for transparent alignment diagnostics.
HISTORICAL_DT1D_COUNTS = {
    "table_03": 422_111,
    "table_04": 534_806,
    "table_05": 64_830,
    "table_06": 64_830,
    "table_07": 64_830,
    "table_08": 346_298,
    "table_09": 66_349,
    "table_10": 64_317,
    "table_11": 141_465,
    "table_12": 59_481,
    "table_13": 59_481,
    "table_14_15": 64_317,
    "table_18_19": 602_767,
}


def deep_merge(*maps: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mapping in maps:
        for key, value in mapping.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
    return result


def target_names(requested: str, targets: dict[str, Any]) -> list[str]:
    if requested == "all":
        return list(targets)
    names = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = sorted(set(names) - set(targets))
    if unknown:
        raise SystemExit(f"Unknown target(s): {unknown}")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/paper/cnn_three_seed_manifest.yaml")
    parser.add_argument("--target", default="all", help="Target name, comma-separated targets, or all.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict-historical-counts", action="store_true")
    ns = parser.parse_args()

    import main as training

    manifest_path = ns.manifest if ns.manifest.is_absolute() else ROOT / ns.manifest
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    names = target_names(ns.target, manifest["targets"])
    base_parser = training.get_args_parser()
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    seen: set[str] = set()

    for target_name in names:
        target = manifest["targets"][target_name]
        variants = target.get("variants") if target.get("kind") == "ablation" else None
        units = []
        if variants:
            units = [("dt1d", variant_name, variant_args) for variant_name, variant_args in variants.items()]
        else:
            units = [(method, None, {}) for method in target.get("methods", [])]

        for method_name, variant_name, variant_args in units:
            preset = manifest["method_presets"][method_name]
            merged = deep_merge(
                manifest["common_args"],
                {
                    "dataset": target["dataset"],
                    "backbone": target["backbone"],
                    "epochs": target["epochs"],
                    "batch_size": target["batch_size"],
                },
                target.get("args", {}),
                preset.get("args", {}),
                variant_args,
            )
            signature = json.dumps(
                {
                    "target": target_name,
                    "method": method_name,
                    "variant": variant_name,
                    "backbone": merged["backbone"],
                    "dataset": merged["dataset"],
                    "args": preset.get("args", {}),
                    "variant_args": variant_args,
                },
                sort_keys=True,
            )
            if signature in seen:
                continue
            seen.add(signature)

            args = base_parser.parse_args([])
            for key, value in merged.items():
                if hasattr(args, key):
                    setattr(args, key, value)
            args.weights = "none"
            args.clip_model = None
            args.nb_classes = CLASS_COUNTS[str(target["dataset"])]
            args = training.canonicalize_args(args)

            try:
                model, adapter_ids = training.build_model_for_experiment(args)
                model = training.set_trainability_policy(model, args, adapter_ids)
                total = sum(parameter.numel() for parameter in model.parameters())
                trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

                if method_name == "full" and trainable != total:
                    raise AssertionError(f"Full fine-tuning has {trainable}/{total} trainable parameters")
                if method_name == "linear":
                    if trainable <= 0 or trainable >= total:
                        raise AssertionError(f"Linear probing has invalid count {trainable}/{total}")
                    if any(not training._is_head_param(name) for name in trainable_names):
                        raise AssertionError("Linear probing exposed non-head parameters")
                if method_name == "bitfit":
                    non_head_biases = [
                        name for name in trainable_names
                        if name.endswith(".bias") and not training._is_head_param(name)
                    ]
                    if not non_head_biases:
                        raise AssertionError(
                            "BitFit exposed no backbone bias parameters and collapsed to Linear probing"
                        )
                if method_name not in {"full", "linear"} and not (0 < trainable < total):
                    raise AssertionError(f"PEFT method has invalid count {trainable}/{total}")

                historical = HISTORICAL_DT1D_COUNTS.get(target_name) if method_name == "dt1d" and variant_name is None else None
                count_match = None if historical is None else trainable == historical
                record = {
                    "target": target_name,
                    "dataset": target["dataset"],
                    "backbone": target["backbone"],
                    "method": method_name,
                    "variant": variant_name,
                    "total_parameters": total,
                    "trainable_parameters": trainable,
                    "historical_manuscript_trainable_parameters": historical,
                    "historical_count_match": count_match,
                    "status": "ok",
                }
                records.append(record)
                if ns.strict_historical_counts and count_match is False:
                    failures.append(
                        f"{target_name}/{method_name}: source={trainable}, manuscript={historical}"
                    )
            except Exception as exc:
                records.append({
                    "target": target_name,
                    "dataset": target["dataset"],
                    "backbone": target["backbone"],
                    "method": method_name,
                    "variant": variant_name,
                    "status": "failed",
                    "error": repr(exc),
                })
                failures.append(f"{target_name}/{method_name}/{variant_name}: {exc!r}")
            finally:
                if "model" in locals():
                    del model
                gc.collect()

    report = {
        "status": "ok" if not failures else "failed",
        "scope": "cnn_classification_only",
        "targets": names,
        "checked_model_configurations": len(records),
        "failures": failures,
        "records": records,
        "note": (
            "Historical count mismatches are diagnostic unless --strict-historical-counts is used. "
            "Historical counts describe the prior manuscript; the v0.9.1 SCDQ source is authoritative for new reruns."
        ),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if ns.output is not None:
        output = ns.output if ns.output.is_absolute() else ROOT / ns.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
