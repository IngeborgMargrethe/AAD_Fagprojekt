from __future__ import annotations

"""
Plot reconstruction accuracy and classification results from the outputs of
compute_eelbrain_backward.py / eelbrain_backward.py.

Expected input directory contents
---------------------------------
- backward_cv_single_talker.json
- backward_cv_two_talker_attended.json
- backward_cv_two_talker_ignored.json
- attention_classification.json   (optional, only if --classify was used)

This script creates:
- reconstruction_accuracy.png
- classification_results.png      (if attention_classification.json exists)
- summary_metrics.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def plot_reconstruction(results_dir: Path) -> Path:
    st = load_json(results_dir / "backward_cv_single_talker.json")
    att = load_json(results_dir / "backward_cv_two_talker_attended.json")
    itt = load_json(results_dir / "backward_cv_two_talker_ignored.json")

    labels = ["Single-talker", "Two-talker\n(attended)", "Two-talker\n(ignored)"]
    fold_data = [
        _to_float_array(st["fold_corrs"]),
        _to_float_array(att["fold_corrs"]),
        _to_float_array(itt["fold_corrs"]),
    ]
    means = [float(np.nanmean(x)) for x in fold_data]
    sems = [
        float(np.nanstd(x, ddof=1) / np.sqrt(np.sum(np.isfinite(x))))
        if np.sum(np.isfinite(x)) > 1 else 0.0
        for x in fold_data
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))

    ax.bar(x, means, yerr=sems, capsize=5)
    for i, vals in enumerate(fold_data):
        xi = np.full(len(vals), i, dtype=float)
        ax.scatter(xi, vals, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reconstruction correlation (r)")
    ax.set_title("Backward model reconstruction accuracy")
    ax.axhline(0, linewidth=1)
    fig.tight_layout()

    out = results_dir / "reconstruction_accuracy.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_classification(results_dir: Path) -> Path | None:
    cls_path = results_dir / "attention_classification.json"
    if not cls_path.exists():
        return None

    cls = load_json(cls_path)
    corr_att = _to_float_array(cls["corr_att"])
    corr_itt = _to_float_array(cls["corr_itt"])
    accuracy = float(cls["accuracy"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(corr_att))
    width = 0.38

    ax.bar(x - width / 2, corr_att, width=width, label="Correlation with attended")
    ax.bar(x + width / 2, corr_itt, width=width, label="Correlation with ignored")

    ax.set_xlabel("Two-talker trial")
    ax.set_ylabel("Correlation (r)")
    ax.set_title(f"Attention classification results (accuracy = {accuracy:.3f})")
    ax.legend()
    ax.axhline(0, linewidth=1)
    fig.tight_layout()

    out = results_dir / "classification_results.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def save_summary(results_dir: Path) -> Path:
    st = load_json(results_dir / "backward_cv_single_talker.json")
    att = load_json(results_dir / "backward_cv_two_talker_attended.json")
    itt = load_json(results_dir / "backward_cv_two_talker_ignored.json")

    summary = {
        "single_talker_mean_corr": float(st["mean_corr"]),
        "two_talker_attended_mean_corr": float(att["mean_corr"]),
        "two_talker_ignored_mean_corr": float(itt["mean_corr"]),
    }

    cls_path = results_dir / "attention_classification.json"
    if cls_path.exists():
        cls = load_json(cls_path)
        summary["classification_accuracy"] = float(cls["accuracy"])
        summary["mean_corr_attended_classification"] = float(np.nanmean(_to_float_array(cls["corr_att"])))
        summary["mean_corr_ignored_classification"] = float(np.nanmean(_to_float_array(cls["corr_itt"])))

    out = results_dir / "summary_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Plot reconstruction and classification results from backward-model outputs.")
    parser.add_argument("--results-dir", required=True, help="Directory containing backward-model JSON outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    required = [
        results_dir / "backward_cv_single_talker.json",
        results_dir / "backward_cv_two_talker_attended.json",
        results_dir / "backward_cv_two_talker_ignored.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required result files: {missing}")

    recon_path = plot_reconstruction(results_dir)
    cls_path = plot_classification(results_dir)
    summary_path = save_summary(results_dir)

    print(f"Saved: {recon_path}")
    if cls_path is not None:
        print(f"Saved: {cls_path}")
    else:
        print("No attention_classification.json found; skipped classification plot.")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
