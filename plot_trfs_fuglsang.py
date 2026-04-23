from __future__ import annotations

"""
Plot and summarize Fuglsang TRF results saved by estimate_trfs_fuglsang.py

Expected files
--------------
- trf_single_talker.pickle
- trf_attended.pickle
- trf_ignored.pickle
- trf_attended_ignored.pickle

Outputs
-------
- summary_metrics.json
- several PNG figures when plotting succeeds
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


def parse_args():
    p = argparse.ArgumentParser(description="Plot Fuglsang TRF results")
    p.add_argument("--results-dir", required=True, help="Directory containing TRF .pickle files")
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

    # Common Eelbrain result attributes
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
        # Some Eelbrain figure objects expose matplotlib figure through .figure
        if hasattr(fig_obj, "figure"):
            fig_obj.figure.savefig(outpath, dpi=150, bbox_inches="tight")
            return True
        # Sometimes ._figure exists
        if hasattr(fig_obj, "_figure"):
            fig_obj._figure.savefig(outpath, dpi=150, bbox_inches="tight")
            return True
        # Or the object itself is a matplotlib figure
        if hasattr(fig_obj, "savefig"):
            fig_obj.savefig(outpath, dpi=150, bbox_inches="tight")
            return True
    except Exception:
        return False
    return False


def plot_predictive_power(trf, title: str, outpath: Path):
    """
    Try several plotting approaches depending on Eelbrain version/result object.
    """
    proportion_explained = safe_get_attr(trf, ["proportion_explained", "proportion_explained_test"])
    r = safe_get_attr(trf, ["r", "r_test", "r_fit"])

    plotted = False

    # Preferred: topomap of proportion explained
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

    # Fallback: topomap of correlation
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


def plot_trf_kernel(trf, title: str, outpath: Path):
    """
    Try to plot the kernel/TRF itself.
    """
    h = safe_get_attr(trf, ["h", "h_scaled", "kernel"])
    if h is None:
        print(f"No kernel found for {title}")
        return

    plotted = False

    # Try image plot first
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

    # Fallback: try generic plot
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


def plot_comparison_bar(summary: dict[str, dict[str, Any]], outpath: Path):
    """
    Create a simple matplotlib bar plot comparing mean predictive metrics.
    """
    import matplotlib.pyplot as plt

    names = list(summary.keys())
    vals = []
    label = None

    for name in names:
        item = summary[name]
        if "mean_proportion_explained" in item:
            vals.append(item["mean_proportion_explained"])
            label = "Mean proportion explained"
        elif "mean_r" in item:
            vals.append(item["mean_r"])
            label = "Mean correlation"
        else:
            vals.append(np.nan)

    plt.figure(figsize=(8, 4))
    plt.bar(names, vals)
    plt.ylabel(label or "Metric")
    plt.title("TRF model comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    files = {
        "single_talker": results_dir / "trf_single_talker.pickle",
        "attended": results_dir / "trf_attended.pickle",
        "ignored": results_dir / "trf_ignored.pickle",
        "attended_ignored": results_dir / "trf_attended_ignored.pickle",
    }

    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required TRF files: {missing}")

    results_dir.mkdir(parents=True, exist_ok=True)

    trfs = {name: eb.load.unpickle(path) for name, path in files.items()}

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

    # Individual predictive power plots
    for name, trf in trfs.items():
        plot_predictive_power(
            trf,
            title=f"{name} predictive power",
            outpath=results_dir / f"{name}_predictive_power.png",
        )

    # Individual kernel plots
    for name, trf in trfs.items():
        plot_trf_kernel(
            trf,
            title=f"{name} kernel",
            outpath=results_dir / f"{name}_kernel.png",
        )

    # Simple comparison bar plot
    try:
        plot_comparison_bar(summary, results_dir / "model_comparison.png")
    except Exception as e:
        print(f"Could not save model comparison plot: {e}")

    print(f"Saved plots to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()