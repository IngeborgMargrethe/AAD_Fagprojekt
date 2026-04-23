from __future__ import annotations

"""
Estimate forward TRFs for Fuglsang WAV-based preprocessing outputs.

Expected input
--------------
A .npz created by preprocess_wav_trf.py with keys:
- env_st, env_att, env_ign
- onset_st, onset_att, onset_ign
- eeg_st, eeg_tt
- fs_audio, fs_eeg

Models
------
Single-talker:
- env_st
- onset_st
- env+onset_st

Two-talker:
- env_att
- env_ign
- env_att+env_ign
- env+onset_att
- env+onset_ign

Outputs
-------
Eelbrain BoostingResult pickle files in the output directory.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import eelbrain as eb  # type: ignore
except Exception as e:
    raise ImportError("This script requires eelbrain in the active environment.") from e


def parse_args():
    p = argparse.ArgumentParser(description="Estimate forward TRFs for Fuglsang WAV-based preprocessed data.")
    p.add_argument("--input", required=True, help="Path to WAV-based preprocessed .npz")
    p.add_argument("--outdir", default="./results_trf_wav", help="Output directory")
    p.add_argument("--fs", type=float, default=64.0, help="Sampling rate")
    p.add_argument("--tstart", type=float, default=-0.100, help="TRF start lag in seconds")
    p.add_argument("--tstop", type=float, default=0.250, help="TRF stop lag in seconds")
    p.add_argument("--basis", type=float, default=0.050, help="Basis window in seconds; 0 for impulse basis")
    p.add_argument("--partitions", type=int, default=5, help="Cross-validation partitions")
    p.add_argument("--error", choices=["l1", "l2"], default="l1", help="Boosting error metric")
    p.add_argument("--no-selective-stopping", action="store_true", help="Disable selective stopping")
    return p.parse_args()


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


def load_preprocessed(path: str | Path) -> Dict[str, List[np.ndarray] | float]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    required = [
        "env_st", "env_att", "env_ign",
        "onset_st", "onset_att", "onset_ign",
        "eeg_st", "eeg_tt",
        "fs_audio", "fs_eeg",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    return {
        "env_st": _to_list_trials_from_3d(data["env_st"], squeeze_last=True),
        "env_att": _to_list_trials_from_3d(data["env_att"], squeeze_last=True),
        "env_ign": _to_list_trials_from_3d(data["env_ign"], squeeze_last=True),
        "onset_st": _to_list_trials_from_3d(data["onset_st"], squeeze_last=True),
        "onset_att": _to_list_trials_from_3d(data["onset_att"], squeeze_last=True),
        "onset_ign": _to_list_trials_from_3d(data["onset_ign"], squeeze_last=True),
        "eeg_st": _to_list_trials_from_3d(data["eeg_st"], squeeze_last=False),
        "eeg_tt": _to_list_trials_from_3d(data["eeg_tt"], squeeze_last=False),
        "fs_audio": float(np.asarray(data["fs_audio"]).squeeze()),
        "fs_eeg": float(np.asarray(data["fs_eeg"]).squeeze()),
    }


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


def concatenate_eeg_trials(trials: List[np.ndarray], fs: float) -> eb.NDVar:
    eeg_nd = [make_eeg_ndvar(x, fs, name=f"eeg_{i:03d}") for i, x in enumerate(trials)]
    return eb.concatenate(eeg_nd)


def concatenate_stim_trials(trials: List[np.ndarray], fs: float, name: str) -> eb.NDVar:
    stim_nd = [make_stim_ndvar(x, fs, name=name) for x in trials]
    return eb.concatenate(stim_nd)


def fit_trf_single_predictor(
    eeg_trials: List[np.ndarray],
    predictor_trials: List[np.ndarray],
    predictor_name: str,
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

    eeg_concat = concatenate_eeg_trials(eeg_trials, fs)
    pred_concat = concatenate_stim_trials(predictor_trials, fs, predictor_name)

    return eb.boosting(
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


def fit_trf_multi_predictor(
    eeg_trials: List[np.ndarray],
    predictors: List[List[np.ndarray]],
    predictor_names: List[str],
    fs: float,
    tstart: float,
    tstop: float,
    basis: float,
    partitions: int,
    error: str,
    selective_stopping: bool,
):
    if not predictors:
        raise ValueError("Need at least one predictor set")
    n_trials = len(eeg_trials)
    if any(len(p) != n_trials for p in predictors):
        raise ValueError("All predictor trial lists must match EEG trial count")

    eeg_concat = concatenate_eeg_trials(eeg_trials, fs)
    pred_concats = [
        concatenate_stim_trials(p_trials, fs, name)
        for p_trials, name in zip(predictors, predictor_names)
    ]

    return eb.boosting(
        eeg_concat,
        pred_concats,
        tstart,
        tstop,
        error=error,
        basis=basis,
        partitions=partitions,
        test=1,
        selective_stopping=selective_stopping,
    )


def save_trf(trf, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    eb.save.pickle(trf, outpath)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trials = load_preprocessed(args.input)
    fs = args.fs
    selective_stopping = not args.no_selective_stopping

    print(f"Loaded input: {args.input}")
    print(f"Single-talker trials: {len(trials['env_st'])}")
    print(f"Two-talker trials: {len(trials['env_att'])}")
    print(f"fs_audio: {trials['fs_audio']}, fs_eeg: {trials['fs_eeg']}")

    models = {
        "single_env": lambda: fit_trf_single_predictor(
            trials["eeg_st"], trials["env_st"], "env", fs,
            args.tstart, args.tstop, args.basis, args.partitions, args.error, selective_stopping
        ),
        "single_onset": lambda: fit_trf_single_predictor(
            trials["eeg_st"], trials["onset_st"], "onset", fs,
            args.tstart, args.tstop, args.basis, args.partitions, args.error, selective_stopping
        ),
        "single_env_onset": lambda: fit_trf_multi_predictor(
            trials["eeg_st"], [trials["env_st"], trials["onset_st"]], ["env", "onset"], fs,
            args.tstart, args.tstop, args.basis, args.partitions, args.error, selective_stopping
        ),
        "attended_env": lambda: fit_trf_single_predictor(
            trials["eeg_tt"], trials["env_att"], "attended_env", fs,
            args.tstart, args.tstop, args.basis, args.partitions, args.error, selective_stopping
        ),
        "ignored_env": lambda: fit_trf_single_predictor(
            trials["eeg_tt"], trials["env_ign"], "ignored_env", fs,
            args.tstart, args.tstop, args.basis, args.partitions, args.error, selective_stopping
        ),
        "attended_ignored_env": lambda: fit_trf_multi_predictor(
            trials["eeg_tt"], [trials["env_att"], trials["env_ign"]], ["attended_env", "ignored_env"], fs,
            args.tstart, args.tstop, args.basis, args.partitions, args.error, selective_stopping
        ),
        "attended_env_onset": lambda: fit_trf_multi_predictor(
            trials["eeg_tt"], [trials["env_att"], trials["onset_att"]], ["attended_env", "attended_onset"], fs,
            args.tstart, args.tstop, args.basis, args.partitions, args.error, selective_stopping
        ),
        "ignored_env_onset": lambda: fit_trf_multi_predictor(
            trials["eeg_tt"], [trials["env_ign"], trials["onset_ign"]], ["ignored_env", "ignored_onset"], fs,
            args.tstart, args.tstop, args.basis, args.partitions, args.error, selective_stopping
        ),
    }

    for name, fit_fn in models.items():
        print(f"Estimating: {name}")
        trf = fit_fn()
        save_trf(trf, outdir / f"{name}.pickle")

    print(f"Saved TRFs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()