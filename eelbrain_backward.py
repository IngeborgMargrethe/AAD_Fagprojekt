from __future__ import annotations

"""
Compute Eelbrain-ready trial objects and a backward TRF/decoder on the Fuglsang
preprocessed dataset.

Supports the outputs created by:
- data/example1/preprocessed.npz
- data/example2_mtrf/preprocessed.npz

Main features
-------------
1) Load preprocessed stimulus/EEG trials.
2) Optionally build an Eelbrain Dataset for inspection/export.
3) Fit a backward model (EEG -> stimulus) with ridge regression and
   leave-one-trial-out CV.
4) Optionally perform attention classification by training on single-talker
   trials and testing on two-talker trials.

The backward model is implemented explicitly with NumPy/SciPy ridge regression.
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.io import savemat
from scipy.linalg import solve

try:
    import eelbrain as eb  # type: ignore
    HAVE_EELBRAIN = True
except Exception:
    eb = None
    HAVE_EELBRAIN = False


@dataclass
class DecoderConfig:
    fs: float = 64.0
    tmin_ms: float = 0.0
    tmax_ms: float = 500.0
    alpha_grid: Tuple[float, ...] = tuple(np.logspace(-3, 6, 20))
    zscore_x: bool = True
    zscore_y: bool = True


def _to_list_trials_from_3d(x: np.ndarray, squeeze_last: bool = False) -> List[np.ndarray]:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array (time, features, trials), got shape={x.shape}")
    trials: List[np.ndarray] = []
    for i in range(x.shape[2]):
        xi = np.asarray(x[:, :, i], dtype=float)
        if squeeze_last and xi.shape[1] == 1:
            xi = xi[:, 0]
        trials.append(xi)
    return trials


def load_preprocessed(path: str | Path, mode: str) -> Dict[str, List[np.ndarray]]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    if mode == "example1":
        required = ["target_st", "target_tt", "masker_tt", "eeg_st", "eeg_tt"]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Missing keys in {path}: {missing}")
        return {
            "target_st": _to_list_trials_from_3d(data["target_st"], squeeze_last=True),
            "target_tt": _to_list_trials_from_3d(data["target_tt"], squeeze_last=True),
            "masker_tt": _to_list_trials_from_3d(data["masker_tt"], squeeze_last=True),
            "eeg_st": _to_list_trials_from_3d(data["eeg_st"], squeeze_last=False),
            "eeg_tt": _to_list_trials_from_3d(data["eeg_tt"], squeeze_last=False),
        }

    if mode == "example2":
        required = ["stim_st", "resp_st", "stim_att", "stim_itt", "resp_tt"]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Missing keys in {path}: {missing}")

        def _normalize_item(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            if x.ndim == 2 and x.shape[1] == 1:
                return x[:, 0]
            return x

        def _to_py_list(key: str) -> List[np.ndarray]:
            arr = data[key]
            if arr.dtype == object:
                return [_normalize_item(np.asarray(a)) for a in arr.tolist()]
            return [_normalize_item(np.asarray(a)) for a in arr]

        return {
            "stim_st": _to_py_list("stim_st"),
            "resp_st": _to_py_list("resp_st"),
            "stim_att": _to_py_list("stim_att"),
            "stim_itt": _to_py_list("stim_itt"),
            "resp_tt": _to_py_list("resp_tt"),
        }

    raise ValueError(f"Unknown mode: {mode}")


def build_eelbrain_dataset(trials: Dict[str, List[np.ndarray]], fs: float = 64.0):
    if not HAVE_EELBRAIN:
        raise ImportError("eelbrain is not installed in the current environment.")

    ds = eb.Dataset()

    def _make_ndvar_1d(x: np.ndarray, name: str):
        time = eb.UTS(0.0, 1.0 / fs, len(x))
        return eb.NDVar(np.asarray(x, dtype=float), (time,), name=name)

    def _make_ndvar_2d(x: np.ndarray, name: str):
        x = np.asarray(x, dtype=float)
        time = eb.UTS(0.0, 1.0 / fs, x.shape[0])
        chan = eb.Scalar("channel", np.arange(x.shape[1]))
        return eb.NDVar(x.T, (chan, time), name=name)

    trial_rows = []
    if "target_st" in trials:
        for i, (stim, eeg) in enumerate(zip(trials["target_st"], trials["eeg_st"])):
            trial_rows.append(dict(condition="single_talker", attended=_make_ndvar_1d(stim, "stimulus"), ignored=None, eeg=_make_ndvar_2d(eeg, "eeg"), trial=i))
        for i, (att, itt, eeg) in enumerate(zip(trials["target_tt"], trials["masker_tt"], trials["eeg_tt"])):
            trial_rows.append(dict(condition="two_talker", attended=_make_ndvar_1d(att, "attended"), ignored=_make_ndvar_1d(itt, "ignored"), eeg=_make_ndvar_2d(eeg, "eeg"), trial=i))
    else:
        for i, (stim, eeg) in enumerate(zip(trials["stim_st"], trials["resp_st"])):
            trial_rows.append(dict(condition="single_talker", attended=_make_ndvar_1d(stim, "stimulus"), ignored=None, eeg=_make_ndvar_2d(eeg, "eeg"), trial=i))
        for i, (att, itt, eeg) in enumerate(zip(trials["stim_att"], trials["stim_itt"], trials["resp_tt"])):
            trial_rows.append(dict(condition="two_talker", attended=_make_ndvar_1d(att, "attended"), ignored=_make_ndvar_1d(itt, "ignored"), eeg=_make_ndvar_2d(eeg, "eeg"), trial=i))

        ds = eb.Dataset()
        ds["condition"] = eb.Factor([r["condition"] for r in trial_rows])
        ds["trial"] = eb.Var([r["trial"] for r in trial_rows])
        ds["eeg"] = [r["eeg"] for r in trial_rows]
        ds["attended"] = [r["attended"] for r in trial_rows]
        ds["ignored"] = [r["ignored"] if r["ignored"] is not None else "" for r in trial_rows]
        
    return ds


def ms_to_lag_samples(tmin_ms: float, tmax_ms: float, fs: float) -> np.ndarray:
    lag_min = int(round(tmin_ms * fs / 1000.0))
    lag_max = int(round(tmax_ms * fs / 1000.0))
    if lag_max < lag_min:
        raise ValueError("tmax_ms must be >= tmin_ms")
    return np.arange(lag_min, lag_max + 1, dtype=int)


def make_lag_matrix(eeg: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    eeg = np.asarray(eeg, dtype=float)
    if eeg.ndim != 2:
        raise ValueError(f"Expected eeg with shape (time, channels), got {eeg.shape}")
    n_time, n_chan = eeg.shape
    out = np.zeros((n_time, n_chan * len(lags)), dtype=float)
    for i, lag in enumerate(lags):
        block = np.zeros_like(eeg)
        if lag == 0:
            block[:] = eeg
        elif lag > 0:
            block[lag:, :] = eeg[:-lag, :]
        else:
            block[:lag, :] = eeg[-lag:, :]
        out[:, i * n_chan:(i + 1) * n_chan] = block
    return out


def zscore_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    return mean, std


def zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    XtX = X.T @ X
    Xty = X.T @ y
    reg = alpha * np.eye(X.shape[1])
    return solve(XtX + reg, Xty, assume_a="pos")


def corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError("Arrays must have the same length for correlation")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def concatenate_trials(eeg_trials: Sequence[np.ndarray], stim_trials: Sequence[np.ndarray], lags: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    X_parts = []
    y_parts = []
    for eeg, stim in zip(eeg_trials, stim_trials):
        X_parts.append(make_lag_matrix(eeg, lags))
        y_parts.append(np.asarray(stim, dtype=float).ravel())
    return np.vstack(X_parts), np.concatenate(y_parts)


def backward_cv(eeg_trials: Sequence[np.ndarray], stim_trials: Sequence[np.ndarray], cfg: DecoderConfig) -> Dict[str, np.ndarray | list | float]:
    if len(eeg_trials) != len(stim_trials):
        raise ValueError("eeg_trials and stim_trials must have same length")
    lags = ms_to_lag_samples(cfg.tmin_ms, cfg.tmax_ms, cfg.fs)
    fold_corrs = []
    fold_best_alpha = []
    fold_yhat = []

    for test_idx in range(len(eeg_trials)):
        train_eeg = [e for i, e in enumerate(eeg_trials) if i != test_idx]
        train_stim = [s for i, s in enumerate(stim_trials) if i != test_idx]
        test_eeg = eeg_trials[test_idx]
        test_stim = np.asarray(stim_trials[test_idx], dtype=float).ravel()

        X_train, y_train = concatenate_trials(train_eeg, train_stim, lags)
        X_test = make_lag_matrix(test_eeg, lags)

        if cfg.zscore_x:
            mx, sx = zscore_fit(X_train)
            X_train = zscore_apply(X_train, mx, sx)
            X_test = zscore_apply(X_test, mx, sx)

        if cfg.zscore_y:
            my, sy = zscore_fit(y_train[:, None])
            y_train_z = zscore_apply(y_train[:, None], my, sy).ravel()
        else:
            my = np.array([[0.0]])
            sy = np.array([[1.0]])
            y_train_z = y_train

        best_alpha = None
        best_w = None
        best_score = -np.inf
        for alpha in cfg.alpha_grid:
            w = fit_ridge_closed_form(X_train, y_train_z, alpha)
            yhat_z = X_test @ w
            yhat = (yhat_z * sy.ravel()[0]) + my.ravel()[0]
            score = corr_1d(yhat, test_stim)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_alpha = alpha
                best_w = w

        assert best_w is not None and best_alpha is not None
        yhat_z = X_test @ best_w
        yhat = (yhat_z * sy.ravel()[0]) + my.ravel()[0]
        fold_corrs.append(best_score)
        fold_best_alpha.append(best_alpha)
        fold_yhat.append(yhat)

    return {
        "lags_samples": np.asarray(lags, dtype=int),
        "lags_ms": np.asarray(lags, dtype=float) * (1000.0 / cfg.fs),
        "fold_corrs": np.asarray(fold_corrs, dtype=float),
        "mean_corr": float(np.nanmean(fold_corrs)),
        "best_alpha_per_fold": np.asarray(fold_best_alpha, dtype=float),
        "predictions": fold_yhat,
    }


def attention_classification_from_single_talker(eeg_st: Sequence[np.ndarray], stim_st: Sequence[np.ndarray], eeg_tt: Sequence[np.ndarray], stim_att: Sequence[np.ndarray], stim_itt: Sequence[np.ndarray], cfg: DecoderConfig) -> Dict[str, np.ndarray | float]:
    lags = ms_to_lag_samples(cfg.tmin_ms, cfg.tmax_ms, cfg.fs)
    X_train, y_train = concatenate_trials(eeg_st, stim_st, lags)

    if cfg.zscore_x:
        mx, sx = zscore_fit(X_train)
        X_train = zscore_apply(X_train, mx, sx)
    else:
        mx, sx = None, None

    if cfg.zscore_y:
        my, sy = zscore_fit(y_train[:, None])
        y_train_z = zscore_apply(y_train[:, None], my, sy).ravel()
    else:
        my = np.array([[0.0]])
        sy = np.array([[1.0]])
        y_train_z = y_train

    best_alpha = None
    best_w = None
    best_score = -np.inf
    for alpha in cfg.alpha_grid:
        w = fit_ridge_closed_form(X_train, y_train_z, alpha)
        yhat_train_z = X_train @ w
        yhat_train = (yhat_train_z * sy.ravel()[0]) + my.ravel()[0]
        score = corr_1d(yhat_train, y_train)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_alpha = alpha
            best_w = w

    assert best_w is not None and best_alpha is not None

    corr_att = []
    corr_itt = []
    pred_attended = []

    for eeg, att, itt in zip(eeg_tt, stim_att, stim_itt):
        X = make_lag_matrix(eeg, lags)
        if mx is not None and sx is not None:
            X = zscore_apply(X, mx, sx)
        yhat_z = X @ best_w
        yhat = (yhat_z * sy.ravel()[0]) + my.ravel()[0]
        r_att = corr_1d(yhat, att)
        r_itt = corr_1d(yhat, itt)
        pred_attended.append(int(r_att >= r_itt))
        corr_att.append(r_att)
        corr_itt.append(r_itt)

    return {
        "best_alpha": float(best_alpha),
        "corr_att": np.asarray(corr_att, dtype=float),
        "corr_itt": np.asarray(corr_itt, dtype=float),
        "pred_attended": np.asarray(pred_attended, dtype=int),
        "accuracy": float(np.mean(pred_attended)),
    }


def save_results(outdir: Path, stem: str, results: Dict):
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            summary[k] = v.tolist()
        elif isinstance(v, list):
            summary[k] = [vv.tolist() if isinstance(vv, np.ndarray) else vv for vv in v]
        else:
            summary[k] = v
    with open(outdir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    mat_payload = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            mat_payload[k] = v
        elif isinstance(v, list):
            try:
                mat_payload[k] = np.array(v, dtype=object)
            except Exception:
                pass
        elif isinstance(v, (int, float, np.integer, np.floating)):
            mat_payload[k] = np.array([v])
    savemat(outdir / f"{stem}.mat", mat_payload)

    with open(outdir / f"{stem}.pkl", "wb") as f:
        pickle.dump(results, f)


def parse_args():
    p = argparse.ArgumentParser(description="Compute Eelbrain-ready data objects and a backward model on Fuglsang preprocessed data.")
    p.add_argument("--input", required=True, help="Path to preprocessed.npz")
    p.add_argument("--mode", choices=["example1", "example2"], required=True, help="Which preprocessed format to load")
    p.add_argument("--outdir", default="./results_backward", help="Directory for outputs")
    p.add_argument("--tmin-ms", type=float, default=0.0, help="Minimum lag in ms")
    p.add_argument("--tmax-ms", type=float, default=500.0, help="Maximum lag in ms")
    p.add_argument("--fs", type=float, default=64.0, help="Sampling rate after preprocessing")
    p.add_argument("--classify", action="store_true", help="Also run train-on-single-talker attention classification")
    p.add_argument("--save-eelbrain", action="store_true", help="Save an Eelbrain Dataset pickle if eelbrain is installed")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    cfg = DecoderConfig(fs=args.fs, tmin_ms=args.tmin_ms, tmax_ms=args.tmax_ms)
    trials = load_preprocessed(args.input, args.mode)

    if args.save_eelbrain:
        if not HAVE_EELBRAIN:
            print("eelbrain is not installed; skipping Eelbrain Dataset export.")
        else:
            ds = build_eelbrain_dataset(trials, fs=args.fs)
            outdir.mkdir(parents=True, exist_ok=True)
            with open(outdir / "eelbrain_dataset.pkl", "wb") as f:
                pickle.dump(ds, f)
            print(f"Saved Eelbrain Dataset to: {outdir / 'eelbrain_dataset.pkl'}")

    if args.mode == "example1":
        res_st = backward_cv(trials["eeg_st"], trials["target_st"], cfg)
        save_results(outdir, "backward_cv_single_talker", res_st)
        print(f"Single-talker mean reconstruction correlation: {res_st['mean_corr']:.4f}")

        res_att = backward_cv(trials["eeg_tt"], trials["target_tt"], cfg)
        save_results(outdir, "backward_cv_two_talker_attended", res_att)
        print(f"Two-talker attended mean reconstruction correlation: {res_att['mean_corr']:.4f}")

        res_itt = backward_cv(trials["eeg_tt"], trials["masker_tt"], cfg)
        save_results(outdir, "backward_cv_two_talker_ignored", res_itt)
        print(f"Two-talker ignored mean reconstruction correlation: {res_itt['mean_corr']:.4f}")

        if args.classify:
            cls = attention_classification_from_single_talker(eeg_st=trials["eeg_st"], stim_st=trials["target_st"], eeg_tt=trials["eeg_tt"], stim_att=trials["target_tt"], stim_itt=trials["masker_tt"], cfg=cfg)
            save_results(outdir, "attention_classification", cls)
            print(f"Attention classification accuracy: {cls['accuracy']:.4f}")

    else:
        res_st = backward_cv(trials["resp_st"], trials["stim_st"], cfg)
        save_results(outdir, "backward_cv_single_talker", res_st)
        print(f"Single-talker mean reconstruction correlation: {res_st['mean_corr']:.4f}")

        res_att = backward_cv(trials["resp_tt"], trials["stim_att"], cfg)
        save_results(outdir, "backward_cv_two_talker_attended", res_att)
        print(f"Two-talker attended mean reconstruction correlation: {res_att['mean_corr']:.4f}")

        res_itt = backward_cv(trials["resp_tt"], trials["stim_itt"], cfg)
        save_results(outdir, "backward_cv_two_talker_ignored", res_itt)
        print(f"Two-talker ignored mean reconstruction correlation: {res_itt['mean_corr']:.4f}")

        if args.classify:
            cls = attention_classification_from_single_talker(eeg_st=trials["resp_st"], stim_st=trials["stim_st"], eeg_tt=trials["resp_tt"], stim_att=trials["stim_att"], stim_itt=trials["stim_itt"], cfg=cfg)
            save_results(outdir, "attention_classification", cls)
            print(f"Attention classification accuracy: {cls['accuracy']:.4f}")

    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
