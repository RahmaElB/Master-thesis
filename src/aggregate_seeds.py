"""
Phase 4: report test results across multiple random seeds as mean +/- std. 
Reads the *_test.csv files saved by src/train.py and aggregates the numeric 
test metrics such as test_acc, test_f1, and test_auc across runs. Each run uses 
the same architecture, hyperparameters, and official train/val/test split; 
only the random seed changes. 

This lets me check how stable the results 
are across independent training runs while keeping the split fixed for 
comparison with Ouyang et al. Usage (one model at a time):
    python -m src.aggregate_seeds \
        --results_dir results \
        --run_names baseline_64f_112px_clipp1_pretrained \
                     baseline_64f_112px_clipp1_pretrained_seed123 \
                     baseline_64f_112px_clipp1_pretrained_seed2024 \
        --group_name baseline \
        --out_dir results/robustness
"""

import argparse
import os

import numpy as np
import pandas as pd


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results",
                    help="directory containing <run_name>_test.csv files from src/train.py")
    p.add_argument("--run_names", type=str, nargs="+", required=True,
                    help="the run_name values for each seed of the SAME architecture/config "
                         "(without the _test.csv suffix), e.g. the original seed=42 run plus "
                         "the new seed=123/seed=2024 runs")
    p.add_argument("--group_name", type=str, required=True,
                    help="label for this architecture in the combined output, e.g. 'baseline'")
    p.add_argument("--out_dir", type=str, default="results/robustness")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    missing = []
    for run_name in args.run_names:
        path = os.path.join(args.results_dir, f"{run_name}_test.csv")
        if not os.path.exists(path):
            missing.append(path)
            continue
        row = pd.read_csv(path).iloc[0].to_dict()
        row["run_name"] = run_name
        rows.append(row)

    if missing:
        print(f"[warn] {len(missing)} run(s) not found yet (still training, or wrong "
              f"run_name?): {missing}")

    if len(rows) < 2:
        print(f"Only {len(rows)}/{len(args.run_names)} runs found for '{args.group_name}' - "
              "need at least 2 to report mean +/- std. Re-run once the remaining seed(s) "
              "finish training.")
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))
        return

    df = pd.DataFrame(rows)
    per_seed_path = os.path.join(args.out_dir, f"{args.group_name}_per_seed.csv")
    df.to_csv(per_seed_path, index=False)
    print(f"Per-seed results for '{args.group_name}' ({len(df)} seeds):")
    print(df[["run_name", "test_acc", "test_f1", "test_auc"]].to_string(index=False))
    print(f"\nSaved per-seed table to: {per_seed_path}")

    numeric_cols = [c for c in df.columns if c not in ("run_name", "model") and
                     pd.api.types.is_numeric_dtype(df[c])]
    summary = {"group": args.group_name, "n_seeds": len(df)}
    for col in numeric_cols:
        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_std"] = df[col].std(ddof=1) if len(df) > 1 else 0.0

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(args.out_dir, f"{args.group_name}_seed_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{args.group_name}: "
          f"Acc = {summary['test_acc_mean']:.4f} +/- {summary['test_acc_std']:.4f}, "
          f"F1 = {summary['test_f1_mean']:.4f} +/- {summary['test_f1_std']:.4f}, "
          f"AUC = {summary['test_auc_mean']:.4f} +/- {summary['test_auc_std']:.4f}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
