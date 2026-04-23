from __future__ import annotations

"""
Preprocess Fuglsang selective-attention data from WAV stimuli.

This version uses the actual audio files from a structure like:

    ds-eeg-snhl/
      participants.tsv
      sub-004/eeg/sub-004_task-selectiveattention_eeg.bdf
      sub-004/eeg/sub-004_task-selectiveattention_events.tsv
      stimuli_audio/sub004/target/*.wav
      stimuli_audio/sub004/masker/*.wav

It computes simple acoustic predictors from WAV:
- broadband envelope
- onset envelope (half-wave rectified derivative of envelope)

Outputs
-------
A .npz file containing:
- single-talker trials
- two-talker attended/ignored trials
- EEG trials
- metadata

All signals are aligned and resampled to a common fs_out.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.io import savemat
from scipy.signal import butter, filtfilt, hilbert, resample_poly


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
class AudioFeatureConfig:
    fs_out: float = 64.0
    envelope_lp: float = 30.0
    hp_out: float = 1.0
    lp_out: float = 9.0
    compression: float = 0.3
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


def load_wav_mono(path: Path) -> tuple[np.ndarray, float]:
    x, fs = sf.read(path)
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, float(fs)


def extract_envelope_and_onset(x: np.ndarray, fs: float, cfg: AudioFeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    x = _as_float64(x).ravel()

    # Hilbert envelope
    env = np.abs(hilbert(x))
    env = np.maximum(env, 0.0)
    env = env ** cfg.compression

    # Low-pass in original fs before resampling
    env = _butter_filter(env, fs, high=cfg.envelope_lp, order=cfg.filt_order)

    # Resample
    env = _resample(env, fs, cfg.fs_out)

    # Final band shaping
    env = _butter_filter(env, cfg.fs_out, low=cfg.hp_out, order=cfg.filt_order)
    env = _butter_filter(env, cfg.fs_out, high=cfg.lp_out, order=cfg.filt_order)

    # Onset envelope: half-wave rectified derivative
    onset = np.diff(env, prepend=env[0])
    onset = np.maximum(onset, 0.0)
    onset = _butter_filter(onset, cfg.fs_out, low=cfg.hp_out, order=cfg.filt_order)
    onset = _butter_filter(onset, cfg.fs_out, high=cfg.lp_out, order=cfg.filt_order)

    return env, onset


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


def resolve_audio_path(
    bidsdir: Path,
    subject: int,
    stim_value: str,
    trigger_type: str,
    variant: str = "plain",
) -> Path:
    stim_value = str(stim_value)
    stem = Path(stim_value).stem  # e.g. t002
    folder = "target" if trigger_type == "targetonset" else "masker"

    if variant == "plain":
        candidate_names = [stem]
    elif variant == "woa":
        candidate_names = [f"{stem}woa"]
    elif variant == "woacontrol":
        candidate_names = [f"{stem}woacontrol"]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    subject_dirs = [bidsdir / "stimuli_audio" / v / folder for v in _subject_variants(subject)]

    for sdir in subject_dirs:
        for name in candidate_names:
            p = sdir / f"{name}.wav"
            if p.exists():
                return p

    raise FileNotFoundError(
        f"Could not resolve WAV for {stim_value} ({trigger_type}) with variant={variant}"
    )


def collect_trial_specs(
    bidsdir: Path,
    subject: int,
    events_path: Path,
    audio_variant: str,
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
                "target_path": resolve_audio_path(bidsdir, subject, row["stim_file"], trig, variant=audio_variant),
                "masker_path": None,
                "masker_delay_sec": None,
                "single_talker_two_talker": row.get("single_talker_two_talker", None),
                "target_onset_sec": float(row.get("onset", np.nan)),
                "target_row": row.to_dict(),
            }

        elif trig == "maskeronset" and current is not None:
            current["masker_path"] = resolve_audio_path(bidsdir, subject, row["stim_file"], trig, variant=audio_variant)
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


def process_subject_wav(
    bidsdir: Path,
    subject: int,
    audio_cfg: AudioFeatureConfig,
    eeg_cfg: EEGConfig,
    audio_variant: str = "plain",
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
    print(f"audio_variant = {audio_variant}")

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

    env_st_trials = []
    env_att_trials = []
    env_ign_trials = []
    onset_st_trials = []
    onset_att_trials = []
    onset_ign_trials = []
    eeg_st_trials = []
    eeg_tt_trials = []
    trial_meta = []

    for bdf_path, events_path in run_specs:
        raw, eeg_meta = preprocess_eeg_bdf(bdf_path, eeg_cfg)
        _, target_events = events_target_table(events_path)
        eeg_trials = extract_eeg_trials(raw, target_events, eeg_cfg)
        stim_specs = collect_trial_specs(bidsdir, subject, events_path, audio_variant)

        print(f"\nRun: {bdf_path.name}")
        print(f"events file: {events_path.name}")
        print(f"len(eeg_trials) = {len(eeg_trials)}")
        print(f"len(stim_specs) = {len(stim_specs)}")

        if len(eeg_trials) != len(stim_specs):
            raise RuntimeError(
                f"Mismatch between EEG trials and stimulus specs: "
                f"{len(eeg_trials)=}, {len(stim_specs)=} for {events_path}"
            )

        for trial_idx, (eeg_trial, spec) in enumerate(zip(eeg_trials, stim_specs), start=1):
            print(
                f"Trial {trial_idx:02d}: "
                f"kind={spec['single_talker_two_talker']}, "
                f"target={Path(spec['target_path']).name}, "
                f"masker={None if spec['masker_path'] is None else Path(spec['masker_path']).name}, "
                f"delay={spec['masker_delay_sec']}"
            )

            x_tgt, fs_tgt = load_wav_mono(spec["target_path"])
            env_tgt, onset_tgt = extract_envelope_and_onset(x_tgt, fs_tgt, audio_cfg)
            env_tgt = crop_toi_1d(env_tgt, audio_cfg.fs_out, eeg_cfg.crop_tmin, eeg_cfg.crop_tmax)
            onset_tgt = crop_toi_1d(onset_tgt, audio_cfg.fs_out, eeg_cfg.crop_tmin, eeg_cfg.crop_tmax)

            env_msk = None
            onset_msk = None
            if spec["masker_path"] is not None:
                x_msk, fs_msk = load_wav_mono(spec["masker_path"])
                env_msk, onset_msk = extract_envelope_and_onset(x_msk, fs_msk, audio_cfg)

                if spec["masker_delay_sec"]:
                    pad = int(round(float(spec["masker_delay_sec"]) * audio_cfg.fs_out))
                    if pad > 0:
                        env_msk = np.concatenate([np.zeros(pad), env_msk])
                        onset_msk = np.concatenate([np.zeros(pad), onset_msk])

                env_msk = crop_toi_1d(env_msk, audio_cfg.fs_out, eeg_cfg.crop_tmin, eeg_cfg.crop_tmax)
                onset_msk = crop_toi_1d(onset_msk, audio_cfg.fs_out, eeg_cfg.crop_tmin, eeg_cfg.crop_tmax)

            lengths = [len(env_tgt), len(onset_tgt), eeg_trial.shape[0]]
            if env_msk is not None:
                lengths.extend([len(env_msk), len(onset_msk)])

            L = min(lengths)

            env_tgt = env_tgt[:L]
            onset_tgt = onset_tgt[:L]
            eeg_trial = eeg_trial[:L]
            if env_msk is not None:
                env_msk = env_msk[:L]
                onset_msk = onset_msk[:L]

            if env_msk is None:
                env_st_trials.append(env_tgt)
                onset_st_trials.append(onset_tgt)
                eeg_st_trials.append(eeg_trial)
                kind = "singletalker"
            else:
                env_att_trials.append(env_tgt)
                env_ign_trials.append(env_msk)
                onset_att_trials.append(onset_tgt)
                onset_ign_trials.append(onset_msk)
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
                "attend_left_right": spec["target_row"].get("attend_left_right"),
                "single_talker_two_talker": spec["target_row"].get("single_talker_two_talker"),
                "stim_file_target_event": spec["target_row"].get("stim_file"),
                "stim_file_masker_event": None if "masker_row" not in spec else spec["masker_row"].get("stim_file"),
                "eeg_meta": eeg_meta,
            })

    return {
        "meta": {
            "subject": sub,
            "hearing_status": hearing_status,
            "audio_config": asdict(audio_cfg),
            "eeg_config": asdict(eeg_cfg),
            "audio_variant": audio_variant,
            "notes": [
                "WAV-based preprocessing",
                "Envelope and onset predictors extracted from raw audio",
                "audio arrays are (time, features, trials)",
                "eeg arrays are (time, channels, trials)",
            ],
        },
        "env_st": stack_trials_3d(env_st_trials),
        "env_att": stack_trials_3d(env_att_trials),
        "env_ign": stack_trials_3d(env_ign_trials),
        "onset_st": stack_trials_3d(onset_st_trials),
        "onset_att": stack_trials_3d(onset_att_trials),
        "onset_ign": stack_trials_3d(onset_ign_trials),
        "eeg_st": stack_trials_3d(eeg_st_trials),
        "eeg_tt": stack_trials_3d(eeg_tt_trials),
        "fs_audio": audio_cfg.fs_out,
        "fs_eeg": eeg_cfg.fs_out,
        "trial_meta": trial_meta,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess Fuglsang WAV stimuli into aligned trial arrays")
    p.add_argument("--bidsdir", type=Path, required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--audio-variant", choices=["plain", "woa", "woacontrol"], default="plain")
    p.add_argument("--save-mat", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = process_subject_wav(
        args.bidsdir,
        args.subject,
        AudioFeatureConfig(),
        EEGConfig(),
        audio_variant=args.audio_variant,
    )

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