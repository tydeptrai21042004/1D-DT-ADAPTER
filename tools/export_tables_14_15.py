#!/usr/bin/env python3
"""Export manuscript Tables 14 and 15 from one completed paper run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing required run artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    ns = parser.parse_args()
    run_dir = ns.run_dir.resolve()
    out = (ns.output_dir or run_dir / "manuscript_tables").resolve()
    out.mkdir(parents=True, exist_ok=True)

    test = load(run_dir / "test_summary.json")
    eff = load(run_dir / "efficiency_profile.json")
    conv = load(run_dir / "convergence_summary.json")
    ref = load(ROOT / "reproducibility" / "manuscript_reference.json")

    table14 = [
        {"metric": "Best validation top-1 (%)", "observed": conv.get("best_val_acc1"), "manuscript": ref["table_14"]["best_validation_accuracy_percent"]},
        {"metric": "Best epoch (zero-based)", "observed": conv.get("best_epoch"), "manuscript": ref["table_14"]["best_epoch_zero_based"]},
        {"metric": "Test top-1 (%)", "observed": test.get("acc1"), "manuscript": ref["table_14"]["test_top1_percent"]},
        {"metric": "Test top-5 (%)", "observed": test.get("acc5"), "manuscript": ref["table_14"]["test_top5_percent"]},
        {"metric": "Test loss", "observed": test.get("loss"), "manuscript": ref["table_14"]["test_loss"]},
    ]
    table15 = [
        {"metric": "Trainable parameters", "observed": eff.get("trainable_params", conv.get("n_trainable_parameters")), "manuscript": ref["table_15"]["trainable_parameters"]},
        {"metric": "Total parameters", "observed": eff.get("total_params", conv.get("n_total_parameters")), "manuscript": ref["table_15"]["total_parameters"]},
        {"metric": "FLOPs (GMACs)", "observed": eff.get("flops_g"), "manuscript": ref["table_15"]["flops_gmacs"]},
        {"metric": "Latency (ms/image)", "observed": eff.get("latency_ms_per_image"), "manuscript": ref["table_15"]["latency_ms_per_image"]},
        {"metric": "FPS", "observed": eff.get("fps"), "manuscript": ref["table_15"]["fps"]},
        {"metric": "Mean epoch time (s)", "observed": conv.get("mean_epoch_time_sec"), "manuscript": ref["table_15"]["mean_epoch_time_seconds"]},
        {"metric": "Total training time (s)", "observed": conv.get("total_train_time_sec"), "manuscript": ref["table_15"]["total_training_time_seconds"]},
    ]

    for number, rows in ((14, table14), (15, table15)):
        for row in rows:
            observed, manuscript = row["observed"], row["manuscript"]
            row["delta_observed_minus_manuscript"] = None if observed is None else float(observed) - float(manuscript)
        (out / f"table_{number}.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        with (out / f"table_{number}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        lines = [f"# Manuscript Table {number}", "", "| Metric | Observed | Manuscript | Difference |", "|---|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['metric']} | {row['observed']} | {row['manuscript']} | {row['delta_observed_minus_manuscript']} |")
        if number == 15:
            lines += ["", "> Timing values depend on the exact GPU, driver, CUDA, PyTorch, warm-up, and batch size. Structural metrics must match exactly; timing is reported with the captured environment."]
        (out / f"table_{number}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Exported Tables 14 and 15 to {out}")


if __name__ == "__main__":
    main()
