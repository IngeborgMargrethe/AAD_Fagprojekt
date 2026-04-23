from __future__ import annotations

"""
Plot and summarize Fuglsang WAV-based TRF results.

Expected input
--------------
A folder containing one or more of:
- single_env.pickle
- single_onset.pickle
- single_env_onset.pickle
- attended_env.pickle
- ignored_env.pickle
- attended_ignored_env.pickle
- attended_env_onset.pickle
- ignored_env_onset.pickle

Outputs
-------
- summary_metrics.json
- model_comparison_mean_r.png
- model_comparison_mean_pe.png
- optional per-model kernel/predictive-power images if supported by the local
  Eelbrain version
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import eelbrain as eb  # type: ignore
except Exception as e:
    raise ImportError("This script requires eelbrain in the active environment.") from e


MODEL_FILES = {
    "single_env": "single_env.pickle",
    "single_onset": "single_onset.pickle",
    "single_env_onset": "single_env_onset.pickle",
    "attended_env": "attended_env.pickle",
    "ignored_env": "ignored_env.pickle",
    "attended_ignored_env": "attended_ignored_env.pickle",
    "attended_env_onset": "attended_env_onset.pickle",
    "ignored_env_onset": "ignored_env_onset.pickle",
}


def parse_args():
    p = argparse.ArgumentParser(description="Plot Fuglsang WAV TRF results")
    p.add_argument("--results-dir", required=True, help="Directory containing WAV TRF .pickle files")
    return p.parse_args()


def safe_get_attr(obj: Any, names: list[str], default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def ndvar_to_array(x):
    try:
        return np.asarray(x.x)
    except Exception:
        return np.asarray(x)


def summarize_trf(trf, name: str) -> dict[str, Any]:
    out: dict[str, Any] = {"model": name}

    r = safe_get_attr(trf, ["r", "r_test", "r_fit"])
    proportion_explained = safe_get_attr(trf, ["proportion_explained", "proportion_explained_test"])
    h = safe_get_attr(trf, ["h", "h_scaled", "kernel"])

    if r is not None:
        arr = ndvar_to_array(r)
        out["mean_r"] = float(np.nanmean(arr))
        out["max_r"] = float(np.nanmax(arr))
        out["min_r"] = float(np.nanmin(arr))

    if proportion_explained is not None:
        arr = ndvar_to_array(proportion_explained)
        out["mean_proportion_explained"] = float(np.nanmean(arr))
        out["max_proportion_explained"] = float(np.nanmax(arr))
        out["min_proportion_explained"] = float(np.nanmin(arr))

    if h is not None:
        arr = ndvar_to_array(h)
        out["kernel_shape"] = list(arr.shape)

    return out


def save_json(obj: dict[str, Any], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def try_save_matplotlib_figure(fig_obj, outpath: Path):
    try:
        if hasattr(fig_obj, "figure"):
            fig_obj.figure.savefig(outpath, dpi=150, bbox_inches="tight")
            return True
        if hasattr(fig_obj, "_figure"):
            fig_obj._figure.savefig(outpath, dpi=150, bbox_inches="tight")
            return True
        if hasattr(fig_obj, "savefig"):
            fig_obj.savefig(outpath, dpi=150, bbox_inches="tight")
            return True
    except Exception:
        return False
    return False


def plot_predictive_power(trf, title: str, outpath: Path):
    proportion_explained = safe_get_attr(trf, ["proportion_explained", "proportion_explained_test"])
    r = safe_get_attr(trf, ["r", "r_test", "r_fit"])

    plotted = False

    if proportion_explained is not None:
        try:
            p = eb.plot.Topomap(proportion_explained, title=title)
            plotted = try_save_matplotlib_figure(p, outpath)
            try:
                p.close()
            except Exception:
                pass
            if plotted:
                return
        except Exception:
            pass

    if r is not None:
        try:
            p = eb.plot.Topomap(r, title=title)
            plotted = try_save_matplotlib_figure(p, outpath)
            try:
                p.close()
            except Exception:
                pass
            if plotted:
                return
        except Exception:
            pass

    if not plotted:
        print(f"Could not save predictive-power plot for {title}")


def plot_kernel(trf, title: str, outpath: Path):
    h = safe_get_attr(trf, ["h", "h_scaled", "kernel"])
    if h is None:
        print(f"No kernel found for {title}")
        return

    plotted = False

    try:
        p = eb.plot.Array(h, title=title)
        plotted = try_save_matplotlib_figure(p, outpath)
        try:
            p.close()
        except Exception:
            pass
        if plotted:
            return
    except Exception:
        pass

    try:
        p = eb.plot.UTS(h, title=title)
        plotted = try_save_matplotlib_figure(p, outpath)
        try:
            p.close()
        except Exception:
            pass
        if plotted:
            return
    except Exception:
        pass

    if not plotted:
        print(f"Could not save kernel plot for {title}")


def plot_metric_bar(summary: dict[str, dict[str, Any]], metric_key: str, ylabel: str, outpath: Path):
    import matplotlib.pyplot as plt

    names = []
    vals = []

    for name, item in summary.items():
        if metric_key in item:
            names.append(name)
            vals.append(item[metric_key])

    if not names:
        print(f"No values found for {metric_key}, skipping {outpath.name}")
        return

    plt.figure(figsize=(10, 4.5))
    plt.bar(names, vals)
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    available = {}
    for model_name, fname in MODEL_FILES.items():
        path = results_dir / fname
        if path.exists():
            available[model_name] = path

    if not available:
        raise FileNotFoundError(f"No known TRF pickle files found in {results_dir}")

    print("Found models:")
    for name in available:
        print(f"  - {name}")

    trfs = {name: eb.load.unpickle(path) for name, path in available.items()}

    summary = {}
    for name, trf in trfs.items():
        info = summarize_trf(trf, name)
        summary[name] = info
        print(f"\n{name}")
        for k, v in info.items():
            if k != "model":
                print(f"  {k}: {v}")

    save_json(summary, results_dir / "summary_metrics.json")
    print(f"\nSaved summary: {results_dir / 'summary_metrics.json'}")

    for name, trf in trfs.items():
        plot_predictive_power(
            trf,
            title=f"{name} predictive power",
            outpath=results_dir / f"{name}_predictive_power.png",
        )

    for name, trf in trfs.items():
        plot_kernel(
            trf,
            title=f"{name} kernel",
            outpath=results_dir / f"{name}_kernel.png",
        )

    plot_metric_bar(
        summary,
        metric_key="mean_r",
        ylabel="Mean correlation (r)",
        outpath=results_dir / "model_comparison_mean_r.png",
    )

    plot_metric_bar(
        summary,
        metric_key="mean_proportion_explained",
        ylabel="Mean proportion explained",
        outpath=results_dir / "model_comparison_mean_pe.png",
    )

    print(f"Saved outputs to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()