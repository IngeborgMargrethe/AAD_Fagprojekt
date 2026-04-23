from __future__ import annotations

"""
Preprocess Fuglsang selective-attention data when the stimuli available in the
BIDS subset are MAT files with derived stimulus features rather than raw WAV
files.

This version is intended for subsets like:
    ds-eeg-snhl/
      participants.tsv
      sub-004/eeg/sub-004_task-selectiveattention_eeg.bdf
      sub-004/eeg/sub-004_task-selectiveattention_events.tsv
      stimuli_audio/sub004/target/*.mat
      stimuli_audio/sub004/masker/*.mat

Important limitation
--------------------
The original examplescript1 MATLAB pipeline starts from raw audio and applies a
full auditory-model front-end. If the subset only contains .mat stimulus files,
this Python script instead assumes those MAT files already contain a derived
stimulus representation (typically an envelope-like feature at 512 Hz). It then
applies the later stages of the Fuglsang pipeline:
    lowpass 30 Hz -> resample 64 Hz -> highpass 1 Hz -> lowpass 9 Hz
and aligns the result to preprocessed EEG.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt, resample_poly


def _as_float64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _butter_filter(
    x: np.ndarray,
    fs: float,
    *,
    low: float | None = None,
    high: float | None = None,
    order: int = 4,
    axis: int = 0,
) -> np.ndarray:
    nyq = fs / 2.0
    if low is not None and high is not None:
        wn = [low / nyq, high / nyq]
        btype = "band"
    elif low is not None:
        wn = low / nyq
        btype = "high"
    elif high is not None:
        wn = high / nyq
        btype = "low"
    else:
        raise ValueError("Need low and/or high cutoff")
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, x, axis=axis)


def _resample(x: np.ndarray, fs_in: float, fs_out: float, axis: int = 0) -> np.ndarray:
    if fs_in == fs_out:
        return np.asarray(x)
    from fractions import Fraction
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=axis)


@dataclass
class DerivedStimulusConfig:
    fs_in_default: float = 512.0
    lp_pre: float = 30.0
    fs_out: float = 64.0
    hp_out: float = 1.0
    lp_out: float = 9.0
    filt_order: int = 4


@dataclass
class EEGConfig:
    mastoids: tuple[str, str] = ("TP7", "TP8")
    fs_lowpass: float = 30.0
    fs_out: float = 64.0
    hp_out: float = 1.0
    lp_out: float = 9.0
    crop_tmin: float = 6.0
    crop_tmax: float = 43.0
    scalp_only: bool = True
    eog_regression: bool = True
    filt_order: int = 4


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def load_numeric_feature_from_mat(path: Path):
    """
    Load the most plausible numeric timeseries from a MAT file.

    Returns
    -------
    feat : np.ndarray
        2D array shaped (time, features)
    fs_val : float | None
        Sampling rate if present
    info : dict
        Metadata about how the feature was extracted
    """
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)

    if "dat" in mat:
        dat = mat["dat"]
        if isinstance(dat, np.ndarray) and dat.size > 0:
            obj = dat.flat[0]
            if hasattr(obj, "_fieldnames"):
                feat = getattr(obj, "feat", None)
                fs = getattr(obj, "fs", None)
                t = getattr(obj, "t", None)

                if feat is None:
                    raise ValueError(f"'dat' exists but has no 'feat' field in {path}")

                feat = np.asarray(feat, dtype=float)

                if feat.ndim == 1:
                    feat = feat[:, None]
                elif feat.ndim == 2 and feat.shape[0] < feat.shape[1]:
                    feat = feat.T

                fs_val = None if fs is None else float(np.asarray(fs).squeeze())

                info = {
                    "source_key": "dat.feat",
                    "fs_key": "dat.fs",
                    "t_key": "dat.t" if t is not None else None,
                    "shape": feat.shape,
                }
                return feat, fs_val, info

    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T
            info = {
                "source_key": key,
                "fs_key": None,
                "t_key": None,
                "shape": arr.shape,
            }
            return arr, None, info

    raise ValueError(f"No usable numeric feature found in {path}")


def preprocess_derived_stimulus(
    feature: np.ndarray,
    fs_in: float | None,
    cfg: DerivedStimulusConfig,
) -> np.ndarray:
    if fs_in is None:
        fs_in = cfg.fs_in_default

    x = _as_float64(feature).ravel()
    x = _butter_filter(x, fs_in, high=cfg.lp_pre, order=cfg.filt_order)
    x = _resample(x, fs_in, cfg.fs_out)
    x = _butter_filter(x, cfg.fs_out, low=cfg.hp_out, order=cfg.filt_order)
    x = _butter_filter(x, cfg.fs_out, high=cfg.lp_out, order=cfg.filt_order)
    return x


def crop_toi_1d(x: np.ndarray, fs: float, tmin: float, tmax: float) -> np.ndarray:
    start = max(0, int(round(tmin * fs)))
    stop = min(len(x), int(round(tmax * fs)))
    return x[start:stop]


def preprocess_eeg_bdf(bdf_path: Path, cfg: EEGConfig) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
    ch_names = set(raw.ch_names)

    if all(ch in ch_names for ch in cfg.mastoids):
        raw.set_eeg_reference(ref_channels=list(cfg.mastoids), verbose="ERROR")

    raw.filter(l_freq=None, h_freq=cfg.fs_lowpass, verbose="ERROR")
    raw.resample(cfg.fs_out, verbose="ERROR")

    if cfg.eog_regression:
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        eog_picks = mne.pick_types(raw.info, eog=True, exclude="bads")
        if len(eeg_picks) and len(eog_picks):
            X = raw.get_data(picks=eog_picks).T
            X = np.c_[X, np.ones(X.shape[0])]
            Y = raw.get_data(picks=eeg_picks).T
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            Yhat = X @ beta
            dat = raw.get_data()
            dat[eeg_picks, :] = (Y - Yhat).T
            raw._data = dat

    raw.filter(l_freq=cfg.hp_out, h_freq=None, verbose="ERROR")
    raw.filter(l_freq=None, h_freq=cfg.lp_out, verbose="ERROR")

    if cfg.scalp_only:
        keep = [
            ch for ch in raw.ch_names
            if ch.upper().startswith(
                ("FP", "AF", "F", "FC", "C", "CP", "P", "PO", "O", "FT", "TP", "T", "CZ", "PZ", "OZ", "FZ")
            ) and "EXG" not in ch.upper()
        ]
        if keep:
            raw.pick(keep)

    return raw, {
        "n_channels": len(raw.ch_names),
        "fs": raw.info["sfreq"],
        "channels": raw.ch_names,
    }


def events_target_table(events_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = pd.read_csv(events_path, sep="\t")
    if "trigger_type" not in ev.columns:
        raise ValueError(f"No trigger_type column in {events_path}")
    target = ev.loc[ev["trigger_type"].astype(str) == "targetonset"].copy()
    if "sample" not in target.columns:
        raise ValueError(f"No sample column in {events_path}")
    return ev, target


def extract_eeg_trials(raw: mne.io.BaseRaw, target_events: pd.DataFrame, cfg: EEGConfig) -> list[np.ndarray]:
    fs = raw.info["sfreq"]
    data = raw.get_data().T
    trials: list[np.ndarray] = []

    for _, row in target_events.iterrows():
        sample = int(round(float(row["sample"]) * (fs / 512.0))) if fs != 512.0 else int(row["sample"])
        start = sample - int(round(5 * fs))
        stop = sample + int(round(50 * fs))
        if start < 0 or stop > len(data):
            continue
        trial = data[start:stop]
        trial = crop_toi_1d(trial, fs, cfg.crop_tmin, cfg.crop_tmax)
        trials.append(trial)

    return trials


def _subject_variants(subject: int | str) -> list[str]:
    s = str(subject)
    digits = "".join(ch for ch in s if ch.isdigit())
    n = int(digits)
    return [f"sub-{n:03d}", f"sub{n:03d}", f"sub-{n}", f"sub{n}"]


def resolve_stimulus_path(
    bidsdir: Path,
    subject: int,
    stim_value: str,
    trigger_type: str,
    hearing_status: str | None,
) -> Path:
    stim_value = str(stim_value)
    stem = Path(stim_value).stem
    folder = "target" if trigger_type == "targetonset" else "masker"
    subject_dirs = [bidsdir / "stimuli_audio" / v / folder for v in _subject_variants(subject)]

    prefer_woa = False

    base_names = [stem]
    if stem.endswith("woa"):
        base_names.append(stem[:-3])
    else:
        base_names.append(stem + "woa")

    ordered = []
    for name in base_names:
        if prefer_woa and not name.endswith("woa"):
            ordered.append(name + "woa")
        ordered.append(name)

    seen = set()
    ordered = [x for x in ordered if not (x in seen or seen.add(x))]

    for sdir in subject_dirs:
        for name in ordered:
            for p in [sdir / f"{name}.mat", sdir / f"{name}.wav"]:
                if p.exists():
                    return p
        if sdir.exists():
            for name in ordered:
                hits = list(sdir.rglob(f"{name}.*"))
                hits = [h for h in hits if h.suffix.lower() in {".mat", ".wav"}]
                if hits:
                    return hits[0]

    raise FileNotFoundError(
        f"Could not resolve stimulus file for {stim_value} ({trigger_type}) under stimuli_audio/subXXX/{folder}"
    )


def collect_trial_specs(
    bidsdir: Path,
    subject: int,
    events_path: Path,
    hearing_status: str | None,
) -> list[dict[str, Any]]:
    ev = pd.read_csv(events_path, sep="\t")
    rows = ev.loc[ev["trigger_type"].astype(str).isin(["targetonset", "maskeronset"])].copy()

    specs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for _, row in rows.iterrows():
        trig = str(row["trigger_type"])

        if trig == "targetonset":
            if current is not None:
                specs.append(current)

            current = {
                "target_path": resolve_stimulus_path(bidsdir, subject, row["stim_file"], trig, hearing_status),
                "masker_path": None,
                "masker_delay_sec": None,
                "single_talker_two_talker": row.get("single_talker_two_talker", None),
                "target_onset_sec": float(row.get("onset", np.nan)),
                "target_row": row.to_dict(),
            }

        elif trig == "maskeronset" and current is not None:
            current["masker_path"] = resolve_stimulus_path(bidsdir, subject, row["stim_file"], trig, hearing_status)
            try:
                current["masker_delay_sec"] = float(row["onset"]) - float(current["target_onset_sec"])
            except Exception:
                current["masker_delay_sec"] = 0.0
            current["masker_row"] = row.to_dict()

    if current is not None:
        specs.append(current)

    return specs


def stack_trials_3d(trials: list[np.ndarray]) -> np.ndarray:
    if not trials:
        return np.empty((0, 0, 0))
    min_len = min(t.shape[0] for t in trials)
    arrs = []
    for t in trials:
        x = np.asarray(t)[:min_len]
        if x.ndim == 1:
            x = x[:, None]
        arrs.append(x)
    return np.stack(arrs, axis=2)


def process_subject(
    bidsdir: Path,
    subject: int,
    stim_cfg: DerivedStimulusConfig,
    eeg_cfg: EEGConfig,
) -> dict[str, Any]:
    sub = f"sub-{subject:03d}"
    pfile = bidsdir / "participants.tsv"
    hearing_status = None

    if pfile.exists():
        participants = pd.read_csv(pfile, sep="\t")
        if "participant_id" in participants.columns and "hearing_status" in participants.columns:
            row = participants.loc[participants["participant_id"] == sub]
            if len(row):
                hearing_status = row.iloc[0]["hearing_status"]

    print(f"\nProcessing subject: {sub}")
    print(f"hearing_status = {hearing_status}")

    eeg_dir = bidsdir / sub / "eeg"
    run_specs = [
        (eeg_dir / f"{sub}_task-selectiveattention_eeg.bdf",
         eeg_dir / f"{sub}_task-selectiveattention_events.tsv")
    ]

    if (
        (eeg_dir / f"{sub}_task-selectiveattention_run-2_eeg.bdf").exists()
        and (eeg_dir / f"{sub}_task-selectiveattention_run-2_events.tsv").exists()
    ):
        run_specs.append(
            (
                eeg_dir / f"{sub}_task-selectiveattention_run-2_eeg.bdf",
                eeg_dir / f"{sub}_task-selectiveattention_run-2_events.tsv",
            )
        )

    target_st_trials = []
    target_tt_trials = []
    masker_tt_trials = []
    eeg_st_trials = []
    eeg_tt_trials = []
    trial_meta = []

    for bdf_path, events_path in run_specs:
        raw, eeg_meta = preprocess_eeg_bdf(bdf_path, eeg_cfg)
        _, target_events = events_target_table(events_path)
        eeg_trials = extract_eeg_trials(raw, target_events, eeg_cfg)
        stim_specs = collect_trial_specs(bidsdir, subject, events_path, hearing_status)

        print(f"\nRun: {bdf_path.name}")
        print(f"events file: {events_path.name}")
        print(f"len(eeg_trials) = {len(eeg_trials)}")
        print(f"len(stim_specs) = {len(stim_specs)}")

        if len(eeg_trials) != len(stim_specs):
            raise RuntimeError(
                f"Mismatch between EEG trials and stimulus specs: "
                f"{len(eeg_trials)=}, {len(stim_specs)=} for {events_path}"
            )

        n = len(eeg_trials)
        if n == 0:
            continue

        for trial_idx, (eeg_trial, spec) in enumerate(zip(eeg_trials, stim_specs), start=1):
            print(
                f"Trial {trial_idx:02d}: "
                f"kind={spec['single_talker_two_talker']}, "
                f"target={Path(spec['target_path']).name}, "
                f"masker={None if spec['masker_path'] is None else Path(spec['masker_path']).name}, "
                f"delay={spec['masker_delay_sec']}"
            )
            print(
                f"          event_target={spec['target_row'].get('stim_file')}, "
                f"event_masker={None if 'masker_row' not in spec else spec['masker_row'].get('stim_file')}"
            )

            tgt_raw, fs_t, tgt_info = load_numeric_feature_from_mat(spec["target_path"])
            tgt = preprocess_derived_stimulus(tgt_raw, fs_t, stim_cfg)
            tgt = crop_toi_1d(tgt, stim_cfg.fs_out, eeg_cfg.crop_tmin, eeg_cfg.crop_tmax)

            msk = None
            msk_info = None
            if spec["masker_path"] is not None:
                msk_raw, fs_m, msk_info = load_numeric_feature_from_mat(spec["masker_path"])
                msk = preprocess_derived_stimulus(msk_raw, fs_m, stim_cfg)
                if spec["masker_delay_sec"]:
                    pad = int(round(float(spec["masker_delay_sec"]) * stim_cfg.fs_out))
                    if pad > 0:
                        msk = np.concatenate([np.zeros(pad), msk])
                msk = crop_toi_1d(msk, stim_cfg.fs_out, eeg_cfg.crop_tmin, eeg_cfg.crop_tmax)

            lengths = [len(tgt), eeg_trial.shape[0]] + ([len(msk)] if msk is not None else [])
            L = min(lengths)

            print(
                f"          len_target={len(tgt)}, "
                f"len_masker={None if msk is None else len(msk)}, "
                f"len_eeg={eeg_trial.shape[0]}, "
                f"aligned={L}"
            )

            tgt = tgt[:L]
            eeg_trial = eeg_trial[:L]
            if msk is not None:
                msk = msk[:L]

            if msk is None:
                target_st_trials.append(tgt)
                eeg_st_trials.append(eeg_trial)
                kind = "singletalker"
            else:
                target_tt_trials.append(tgt)
                masker_tt_trials.append(msk)
                eeg_tt_trials.append(eeg_trial)
                kind = "twotalker"

            trial_meta.append({
                "run_bdf": str(bdf_path),
                "events_path": str(events_path),
                "trial_kind": kind,
                "target_path": str(spec["target_path"]),
                "masker_path": None if spec["masker_path"] is None else str(spec["masker_path"]),
                "masker_delay_sec": spec["masker_delay_sec"],
                "aligned_samples": int(L),
                "target_info": tgt_info,
                "masker_info": msk_info,
                "eeg_meta": eeg_meta,
                "attend_left_right": spec["target_row"].get("attend_left_right"),
                "single_talker_two_talker": spec["target_row"].get("single_talker_two_talker"),
                "stim_file_target_event": spec["target_row"].get("stim_file"),
                "stim_file_masker_event": None if "masker_row" not in spec else spec["masker_row"].get("stim_file"),
            })

    return {
        "meta": {
            "subject": sub,
            "hearing_status": hearing_status,
            "stimulus_config": asdict(stim_cfg),
            "eeg_config": asdict(eeg_cfg),
            "notes": [
                "Designed for MAT stimulus derivatives rather than raw WAV audio",
                "Equivalent output role to examplescript1 srdat structure",
                "audio arrays are (time, features, trials)",
                "eeg arrays are (time, channels, trials)",
            ],
        },
        "target_st": stack_trials_3d(target_st_trials),
        "target_tt": stack_trials_3d(target_tt_trials),
        "masker_tt": stack_trials_3d(masker_tt_trials),
        "eeg_st": stack_trials_3d(eeg_st_trials),
        "eeg_tt": stack_trials_3d(eeg_tt_trials),
        "fs_audio": stim_cfg.fs_out,
        "fs_eeg": eeg_cfg.fs_out,
        "trial_meta": trial_meta,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess Fuglsang data from MAT stimulus derivatives into srdat-like arrays"
    )
    p.add_argument("--bidsdir", type=Path, required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--save-mat", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = process_subject(args.bidsdir, args.subject, DerivedStimulusConfig(), EEGConfig())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.out,
        **{
            k: (np.array(json.dumps(_jsonable(v)), dtype=object) if isinstance(v, (dict, list)) else v)
            for k, v in payload.items()
        },
    )
    print(f"\nSaved {args.out}")

    if args.save_mat:
        savemat(
            args.out.with_suffix(".mat"),
            {k: (json.dumps(_jsonable(v)) if isinstance(v, (dict, list)) else v) for k, v in payload.items()},
            do_compression=True,
        )
        print(f"Saved {args.out.with_suffix('.mat')}")


if __name__ == "__main__":
    main()