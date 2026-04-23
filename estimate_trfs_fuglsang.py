from __future__ import annotations

"""
Estimate forward TRFs for the Fuglsang dataset from preprocessed .npz files.

Supported inputs
----------------
- data/example1/preprocessed.npz
- data/example2_mtrf/preprocessed.npz

Models
------
- single_talker: EEG <- target/single stimulus
- attended:      EEG <- attended target stream
- ignored:       EEG <- ignored/masker stream
- attended+ignored: EEG <- [attended, ignored]

This script is modeled after the Alice Eelbrain TRF workflow, but adapted to
the Fuglsang preprocessing outputs that already contain trial-wise EEG and
stimulus arrays at a common sampling rate.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import eelbrain as eb  # type: ignore
except Exception as e:
    raise ImportError(
        "This script requires eelbrain in the active environment."
    ) from e


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
            "stim_st": _to_list_trials_from_3d(data["target_st"], squeeze_last=True),
            "stim_att": _to_list_trials_from_3d(data["target_tt"], squeeze_last=True),
            "stim_ign": _to_list_trials_from_3d(data["masker_tt"], squeeze_last=True),
            "resp_st": _to_list_trials_from_3d(data["eeg_st"], squeeze_last=False),
            "resp_tt": _to_list_trials_from_3d(data["eeg_tt"], squeeze_last=False),
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
            "stim_ign": _to_py_list("stim_itt"),
            "resp_tt": _to_py_list("resp_tt"),
        }

    raise ValueError(f"Unknown mode: {mode}")


def make_uts(fs: float, n_samples: int) -> eb.UTS:
    return eb.UTS(0.0, 1.0 / fs, n_samples)


def make_sensor_dim(n_channels: int) -> eb.Scalar:
    return eb.Scalar("channel", np.arange(n_channels))


def make_stim_ndvar(x: np.ndarray, fs: float, name: str) -> eb.NDVar:
    x = np.asarray(x, dtype=float).ravel()
    time = make_uts(fs, len(x))
    return eb.NDVar(x, (time,), name=name)


def make_eeg_ndvar(x: np.ndarray, fs: float, name: str = "eeg") -> eb.NDVar:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected EEG array with shape (time, channels), got {x.shape}")
    time = make_uts(fs, x.shape[0])
    chan = make_sensor_dim(x.shape[1])
    return eb.NDVar(x.T, (chan, time), name=name)


def concatenate_trials_as_ndvars(trials: List[np.ndarray], fs: float, kind: str) -> List[eb.NDVar]:
    out = []
    for i, trial in enumerate(trials):
        if kind == "stim":
            out.append(make_stim_ndvar(trial, fs, name=f"stim_{i:03d}"))
        elif kind == "eeg":
            out.append(make_eeg_ndvar(trial, fs, name=f"eeg_{i:03d}"))
        else:
            raise ValueError(f"Unknown kind: {kind}")
    return out


def fit_trf(
    eeg_trials: List[np.ndarray],
    predictor_trials: List[np.ndarray],
    fs: float,
    tstart: float,
    tstop: float,
    basis: float,
    partitions: int,
    error: str,
    selective_stopping: bool,
):
    if len(eeg_trials) != len(predictor_trials):
        raise ValueError(
            f"Need same number of EEG and predictor trials, got {len(eeg_trials)} and {len(predictor_trials)}"
        )

    eeg_nd = concatenate_trials_as_ndvars(eeg_trials, fs, kind="eeg")
    pred_nd = concatenate_trials_as_ndvars(predictor_trials, fs, kind="stim")

    eeg_concat = eb.concatenate(eeg_nd)
    pred_concat = eb.concatenate(pred_nd)

    trf = eb.boosting(
        eeg_concat,
        pred_concat,
        tstart,
        tstop,
        error=error,
        basis=basis,
        partitions=partitions,
        test=1,
        selective_stopping=selective_stopping,
    )
    return trf


def fit_mtrf_two_predictors(
    eeg_trials: List[np.ndarray],
    pred1_trials: List[np.ndarray],
    pred2_trials: List[np.ndarray],
    fs: float,
    tstart: float,
    tstop: float,
    basis: float,
    partitions: int,
    error: str,
    selective_stopping: bool,
):
    if not (len(eeg_trials) == len(pred1_trials) == len(pred2_trials)):
        raise ValueError(
            "Need same number of EEG, predictor1, predictor2 trials"
        )

    eeg_nd = concatenate_trials_as_ndvars(eeg_trials, fs, kind="eeg")
    pred1_nd = [make_stim_ndvar(x, fs, name="attended") for x in pred1_trials]
    pred2_nd = [make_stim_ndvar(x, fs, name="ignored") for x in pred2_trials]

    eeg_concat = eb.concatenate(eeg_nd)
    pred1_concat = eb.concatenate(pred1_nd)
    pred2_concat = eb.concatenate(pred2_nd)

    trf = eb.boosting(
        eeg_concat,
        [pred1_concat, pred2_concat],
        tstart,
        tstop,
        error=error,
        basis=basis,
        partitions=partitions,
        test=1,
        selective_stopping=selective_stopping,
    )
    return trf


def parse_args():
    p = argparse.ArgumentParser(description="Estimate forward TRFs for Fuglsang preprocessed data with Eelbrain.")
    p.add_argument("--input", required=True, help="Path to preprocessed.npz")
    p.add_argument("--mode", choices=["example1", "example2"], required=True, help="Input format")
    p.add_argument("--outdir", default="./results_trf", help="Output directory")
    p.add_argument("--fs", type=float, default=64.0, help="Sampling rate of preprocessed data")
    p.add_argument("--tstart", type=float, default=-0.100, help="TRF start lag in seconds")
    p.add_argument("--tstop", type=float, default=0.500, help="TRF stop lag in seconds")
    p.add_argument("--basis", type=float, default=0.050, help="Basis window in seconds; 0 for impulse basis")
    p.add_argument("--partitions", type=int, default=5, help="Cross-validation partitions for boosting")
    p.add_argument("--error", choices=["l1", "l2"], default="l1", help="Boosting error metric")
    p.add_argument("--no-selective-stopping", action="store_true", help="Disable selective stopping")
    p.add_argument("--save-dataset", action="store_true", help="Also save an Eelbrain Dataset pickle")
    return p.parse_args()


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def build_dataset(trials: Dict[str, List[np.ndarray]], fs: float):
    rows = []

    for i, (stim, eeg) in enumerate(zip(trials["stim_st"], trials["resp_st"])):
        rows.append(
            dict(
                condition="single_talker",
                trial=i,
                eeg=make_eeg_ndvar(eeg, fs, "eeg"),
                attended=make_stim_ndvar(stim, fs, "stimulus"),
                ignored="",
            )
        )

    for i, (att, ign, eeg) in enumerate(zip(trials["stim_att"], trials["stim_ign"], trials["resp_tt"])):
        rows.append(
            dict(
                condition="two_talker",
                trial=i,
                eeg=make_eeg_ndvar(eeg, fs, "eeg"),
                attended=make_stim_ndvar(att, fs, "attended"),
                ignored=make_stim_ndvar(ign, fs, "ignored"),
            )
        )

    ds = eb.Dataset()
    ds["condition"] = eb.Factor([r["condition"] for r in rows])
    ds["trial"] = eb.Var([r["trial"] for r in rows])
    ds["eeg"] = [r["eeg"] for r in rows]
    ds["attended"] = [r["attended"] for r in rows]
    ds["ignored"] = [r["ignored"] for r in rows]
    return ds


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selective_stopping = not args.no_selective_stopping
    trials = load_preprocessed(args.input, args.mode)

    print(f"Loaded input: {args.input}")
    print(f"Single-talker trials: {len(trials['stim_st'])}")
    print(f"Two-talker trials: {len(trials['stim_att'])}")

    if args.save_dataset:
        ds = build_dataset(trials, args.fs)
        save_pickle(ds, outdir / "eelbrain_dataset.pkl")
        print(f"Saved dataset: {outdir / 'eelbrain_dataset.pkl'}")

    print("Estimating single-talker TRF...")
    trf_st = fit_trf(
        eeg_trials=trials["resp_st"],
        predictor_trials=trials["stim_st"],
        fs=args.fs,
        tstart=args.tstart,
        tstop=args.tstop,
        basis=args.basis,
        partitions=args.partitions,
        error=args.error,
        selective_stopping=selective_stopping,
    )
    eb.save.pickle(trf_st, outdir / "trf_single_talker.pickle")

    print("Estimating attended TRF...")
    trf_att = fit_trf(
        eeg_trials=trials["resp_tt"],
        predictor_trials=trials["stim_att"],
        fs=args.fs,
        tstart=args.tstart,
        tstop=args.tstop,
        basis=args.basis,
        partitions=args.partitions,
        error=args.error,
        selective_stopping=selective_stopping,
    )
    eb.save.pickle(trf_att, outdir / "trf_attended.pickle")

    print("Estimating ignored TRF...")
    trf_ign = fit_trf(
        eeg_trials=trials["resp_tt"],
        predictor_trials=trials["stim_ign"],
        fs=args.fs,
        tstart=args.tstart,
        tstop=args.tstop,
        basis=args.basis,
        partitions=args.partitions,
        error=args.error,
        selective_stopping=selective_stopping,
    )
    eb.save.pickle(trf_ign, outdir / "trf_ignored.pickle")

    print("Estimating combined attended+ignored mTRF...")
    trf_both = fit_mtrf_two_predictors(
        eeg_trials=trials["resp_tt"],
        pred1_trials=trials["stim_att"],
        pred2_trials=trials["stim_ign"],
        fs=args.fs,
        tstart=args.tstart,
        tstop=args.tstop,
        basis=args.basis,
        partitions=args.partitions,
        error=args.error,
        selective_stopping=selective_stopping,
    )
    eb.save.pickle(trf_both, outdir / "trf_attended_ignored.pickle")

    print(f"Saved TRFs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()