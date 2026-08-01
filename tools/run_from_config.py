#!/usr/bin/env python3
"""Run a paper experiment from a versioned YAML configuration.

The runner writes the exact resolved configuration, executable command,
environment snapshot, source revision, pretrained-weight metadata, and stdout
log before/while launching ``main.py``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def resolve_weight_metadata(backbone: str, requested: str) -> dict[str, Any]:
    result: dict[str, Any] = {"backbone": backbone, "requested": requested}
    try:
        from torchvision.models import get_model_weights
        enum = get_model_weights(backbone)
        weight = enum.DEFAULT if requested.upper() == "DEFAULT" else getattr(enum, requested.upper())
        result.update({
            "resolved_enum": f"{enum.__name__}.{weight.name}",
            "url": getattr(weight, "url", None),
            "meta": {k: v for k, v in getattr(weight, "meta", {}).items() if k in {"num_params", "min_size", "categories"}},
        })
        if "categories" in result["meta"]:
            result["meta"]["categories"] = f"{len(result['meta']['categories'])} categories"
    except Exception as exc:
        result["resolution_error"] = repr(exc)
    return result


def capture_environment() -> dict[str, Any]:
    packages = [
        "torch", "torchvision", "timm", "numpy", "pandas", "scipy",
        "scikit-learn", "Pillow", "PyYAML", "thop", "fvcore",
        "torchmetrics", "pycocotools", "opencv-python-headless", "pytest",
    ]
    env: dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: package_version(name) for name in packages},
    }
    try:
        import torch
        env["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        env["torch_runtime_error"] = repr(exc)
    try:
        env["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip().splitlines()
    except Exception:
        env["nvidia_smi"] = None
    return env


def normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def resolve_paths(args_map: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(args_map)
    for key in ("split_file",):
        value = resolved.get(key)
        if value and not Path(str(value)).is_absolute():
            resolved[key] = str((ROOT / str(value)).resolve())
    return resolved


def build_command(args_map: dict[str, Any], output_dir: Path) -> list[str]:
    command = [sys.executable, str(ROOT / "main.py")]
    merged = dict(args_map)
    merged["output_dir"] = str(output_dir)
    for key, value in merged.items():
        if value is None:
            continue
        if isinstance(value, list):
            command.append(f"--{key}")
            command.extend(normalize_value(v) for v in value)
        else:
            command.extend([f"--{key}", normalize_value(value)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-if-complete", action="store_true")
    ns = parser.parse_args()

    config_path = ns.config if ns.config.is_absolute() else ROOT / ns.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("args"), dict):
        raise SystemExit("Configuration must contain an 'args' mapping.")

    output_dir = ns.output_dir if ns.output_dir.is_absolute() else ROOT / ns.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if ns.skip_if_complete and (output_dir / "test_summary.json").is_file():
        print(f"[reuse] Complete run already exists: {output_dir}")
        return 0

    args_map = resolve_paths(config["args"])
    if ns.data_path is not None:
        args_map["data_path"] = str(ns.data_path.resolve())
    elif not Path(str(args_map.get("data_path", "data"))).is_absolute():
        args_map["data_path"] = str((ROOT / str(args_map.get("data_path", "data"))).resolve())
    if ns.device is not None:
        args_map["device"] = ns.device

    command = build_command(args_map, output_dir)
    revision = {
        "branch": git_value("branch", "--show-current"),
        "commit": git_value("rev-parse", "HEAD"),
        "dirty": bool(git_value("status", "--porcelain")),
    }
    try:
        source_config_label = str(config_path.relative_to(ROOT))
    except ValueError:
        source_config_label = str(config_path)

    run_metadata = {key: value for key, value in config.items() if key != "args"}
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )

    resolved = {
        "schema_version": 1,
        "source_config": source_config_label,
        "source_config_sha256": sha256(config_path),
        "experiment_id": config.get("experiment_id"),
        "manuscript_tables": config.get("manuscript_tables", []),
        "manuscript_figures": config.get("manuscript_figures", []),
        "independent_seed": config.get("independent_seed"),
        "description": config.get("description"),
        "args": args_map,
        "output_dir": str(output_dir),
        "source_revision": revision,
        "pretrained_weights": resolve_weight_metadata(str(args_map["backbone"]), str(args_map["weights"])),
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "environment.json").write_text(json.dumps(capture_environment(), indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\n" + shlex.join(command) + "\n", encoding="utf-8")
    os.chmod(output_dir / "command.sh", 0o755)

    print(shlex.join(command))
    if ns.dry_run:
        return 0

    started = dt.datetime.now(dt.timezone.utc)
    with (output_dir / "stdout.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    finished = dt.datetime.now(dt.timezone.utc)
    status = {
        "return_code": code,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
    }
    (output_dir / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
