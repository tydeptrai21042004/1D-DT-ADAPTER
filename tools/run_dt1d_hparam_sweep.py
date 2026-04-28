# tools/run_dt1d_hparam_sweep.py
"""Run a validation-only DT1D hyperparameter sweep for reviewer-requested sensitivity analysis."""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess


def parse_list(value: str, cast=str):
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="flowers102")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--output_root", type=str, default="outputs_hparam_sweep")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--M", type=str, default="1,2,3")
    parser.add_argument("--h", type=str, default="1,2,3")
    parser.add_argument("--axis", type=str, default="h,w,hw")
    parser.add_argument("--alpha_group", type=str, default="1,8,16,32")
    parser.add_argument("--pw_ratio", type=str, default="4,8,16")
    parser.add_argument("--no_pw", type=str, default="False,True")
    parser.add_argument("--gate_init", type=str, default="0.0,0.01,0.1")
    args = parser.parse_args()

    M_list = parse_list(args.M, int)
    h_list = parse_list(args.h, int)
    axis_list = parse_list(args.axis, str)
    ag_list = parse_list(args.alpha_group, int)
    pw_ratio_list = parse_list(args.pw_ratio, int)
    no_pw_list = [v.lower() in ("1", "true", "yes", "y") for v in parse_list(args.no_pw, str)]
    gate_list = parse_list(args.gate_init, float)

    os.makedirs(args.output_root, exist_ok=True)

    for M, h, axis, ag, pw_ratio, no_pw, gate in itertools.product(
        M_list, h_list, axis_list, ag_list, pw_ratio_list, no_pw_list, gate_list
    ):
        name = f"M{M}_h{h}_axis{axis}_ag{ag}_pw{pw_ratio}_nopw{int(no_pw)}_gate{gate}"
        out = os.path.join(args.output_root, args.dataset, args.backbone, name, f"seed_{args.seed}")
        cmd = [
            "python", "main.py",
            "--dataset", args.dataset,
            "--data_path", args.data_path,
            "--backbone", args.backbone,
            "--tuning_method", "hcc",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
            "--seed", str(args.seed),
            "--output_dir", out,
            "--profile_efficiency", "True",
            "--final_test", "False",
            "--dt_M", str(M),
            "--dt_h", str(h),
            "--dt_axis", axis,
            "--dt_alpha_group", str(ag),
            "--dt_pw_ratio", str(pw_ratio),
            "--dt_no_pw", str(no_pw),
            "--dt_gate_init", str(gate),
        ]
        print("[RUN]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
