from __future__ import annotations

"""
Fuglsang-style preprocessing and mTRF-ready data assembly, modeled after the
user-provided `examplescript2.m`.

What this script is for
-----------------------
This is NOT the same as `fuglsang_preprocess.py`.

- `fuglsang_preprocess.py` mirrors the richer preprocessing / analysis style in
  `examplescript1.m`, including the possibility of building arrays like
  `(time, features, trials)` for later custom ridge-regression analyses.
- This script mirrors `examplescript2.m`, whose purpose is to produce
  preprocessed EEG/audio trials in a form suitable for mTRF-style stimulus
  reconstruction, i.e. lists/cells of trials where each trial is a 1D stimulus
  feature and a 2D EEG response matrix.

The processing logic here follows the simplified pipeline from
`examplescript2.m`:

EEG:
  1) rereference to mastoids/TP7+TP8
  2) segment trials around target onsets
  3) downsample to 64 Hz
  4) high-pass 1 Hz
  5) low-pass 9 Hz
  6) keep scalp EEG channels
  7) truncate to 6-43 s post-trigger

Audio:
  1) average left/right channels
  2) resample to 12 kHz
  3) gammatone/ERB filterbank
  4) full-wave rectification / Hilbert-envelope approximation and power-law 0.3
  5) average across bands
  6) resample to 64 Hz
  7) band-pass 1-9 Hz
  8) truncate to 6-43 s post-trigger

Notes
-----
- This is a practical Python approximation. It is not guaranteed to match the
  original MATLAB + FieldTrip + Auditory Modeling Toolbox output sample by
  sample.
- It supports BIDS-style EEG/event files and external audio paths listed in the
  events TSV.
- The output is saved as an .npz file with object arrays holding trial lists,
  which maps naturally to mTRF-style per-trial inputs.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import mne
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.io import savemat
from scipy.signal import butter, filtfilt, hilbert, resample_poly

try:
    from gammatone.filters import centre_freqs, make_erb_filters, erb_filterbank
    HAVE_GAMMATONE = True
except Exception:
    HAVE_GAMMATONE = False


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _as_float64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)



def _butter_filter(
    x: np.ndarray,
    fs: float,
    *,
    low: float | None = None,
    high: float | None = None,
    order: int = 2,
    axis: int = 0,
) -> np.ndarray:
    nyq = fs / 2.0
    if low is not None and high is not None:
        btype = "bandpass"
        wn = [low / nyq, high / nyq]
    elif low is not None:
        btype = "highpass"
        wn = low / nyq
    elif high is not None:
        btype = "lowpass"
        wn = high / nyq
    else:
        raise ValueError("Specify low and/or high cutoff")
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, x, axis=axis)



def _resample(x: np.ndarray, fs_in: float, fs_out: float, axis: int = 0) -> np.ndarray:
    if fs_in == fs_out:
        return np.asarray(x)
    from fractions import Fraction

    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator, axis=axis)



def _load_raw_eeg(path: Path, preload: bool = True) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(path, preload=preload, verbose="ERROR")
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=preload, verbose="ERROR")
    if suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=preload, verbose="ERROR")
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=preload, verbose="ERROR")
    if suffix == ".set":
        return mne.io.read_raw_eeglab(path, preload=preload, verbose="ERROR")
    raise ValueError(f"Unsupported EEG format: {suffix}")



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


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------


@dataclass
class AudioConfig:
    fs_resample_1: float = 12000.0
    n_bands: int = 28
    f_min: float = 100.0
    f_max: float = 4000.0
    compression: float = 0.3
    fs_out: float = 64.0
    bp_low: float = 1.0
    bp_high: float = 9.0
    crop_tmin: float = 6.0
    crop_tmax: float = 43.0
    filt_order: int = 2


@dataclass
class EEGConfig:
    ref_channels: tuple[str, str] = ("TP8", "TP7")
    segment_tmin: float = -5.0
    segment_tmax: float = 50.0
    fs_out: float = 64.0
    hp: float = 1.0
    lp: float = 9.0
    crop_tmin: float = 6.0
    crop_tmax: float = 43.0
    keep_eeg_only: bool = True


# -----------------------------------------------------------------------------
# Audio preprocessing
# -----------------------------------------------------------------------------


def preprocess_audio_for_mtrf(wav: np.ndarray, fs: float, cfg: AudioConfig) -> np.ndarray:
    """Return a 1D audio feature vector at 64 Hz, cropped to 6-43 s."""
    wav = np.asarray(wav)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    elif wav.ndim != 1:
        raise ValueError(f"Audio must be 1D or 2D, got {wav.shape}")

    wav = _resample(wav, fs, cfg.fs_resample_1)

    if not HAVE_GAMMATONE:
        raise ImportError(
            "The `gammatone` package is required for raw audio preprocessing. "
            "Install `gammatone` before running this script."
        )

    cfs = centre_freqs(cfg.fs_resample_1, cfg.n_bands, cfg.f_min)
    cfs = cfs[(cfs >= cfg.f_min) & (cfs <= cfg.f_max)]
    filters = make_erb_filters(cfg.fs_resample_1, cfs)
    bands = erb_filterbank(wav, filters)  # (bands, time)

    # MATLAB script says full-wave rectification; here Hilbert magnitude is a
    # robust approximation often used for this kind of envelope extraction.
    env = np.abs(hilbert(bands, axis=1)) ** cfg.compression
    feat = env.mean(axis=0)
    feat = _resample(feat, cfg.fs_resample_1, cfg.fs_out)
    feat = _butter_filter(feat, cfg.fs_out, low=cfg.bp_low, high=cfg.bp_high, order=cfg.filt_order)

    start = int(round(cfg.crop_tmin * cfg.fs_out))
    stop = int(round(cfg.crop_tmax * cfg.fs_out))
    feat = feat[start:stop]
    return _as_float64(feat)



def pad_masker_with_silence(wav: np.ndarray, fs: float, silence_seconds: float) -> np.ndarray:
    wav = np.asarray(wav)
    if wav.ndim == 1:
        wav = wav[:, None]
    n_pad = int(round(max(0.0, silence_seconds) * fs))
    pad = np.zeros((n_pad, wav.shape[1]), dtype=wav.dtype)
    return np.concatenate([pad, wav], axis=0)


# -----------------------------------------------------------------------------
# EEG preprocessing
# -----------------------------------------------------------------------------


def preprocess_eeg_run(
    eeg_path: Path,
    events_tsv: Path,
    cfg: EEGConfig,
    target_trigger_type: str = "targetonset",
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """
    Segment one EEG run into target-locked trials.

    Returns
    -------
    trials : list of arrays, each shaped (time, channels)
    info   : per-trial metadata
    """
    raw = _load_raw_eeg(eeg_path)
    events = pd.read_csv(events_tsv, sep="\t")

    missing_refs = [ch for ch in cfg.ref_channels if ch not in raw.ch_names]
    if missing_refs:
        raise ValueError(f"Reference channels not found in EEG data: {missing_refs}")

    raw = raw.copy().set_eeg_reference(ref_channels=list(cfg.ref_channels), verbose="ERROR")
    raw.resample(cfg.fs_out, verbose="ERROR")
    raw.filter(l_freq=cfg.hp, h_freq=None, method="iir", phase="zero", verbose="ERROR")
    raw.filter(l_freq=None, h_freq=cfg.lp, method="iir", phase="zero", verbose="ERROR")

    if cfg.keep_eeg_only:
        picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, misc=False)
        raw.pick(picks)

    fs = float(raw.info["sfreq"])
    target_events = events.loc[events["trigger_type"] == target_trigger_type].copy()

    trials: list[np.ndarray] = []
    trial_meta: list[dict[str, Any]] = []

    for _, row in target_events.iterrows():
        if "sample" not in row:
            raise ValueError("Events TSV must contain a 'sample' column.")

        onset_sample_orig = float(row["sample"])
        onset_sec = onset_sample_orig / raw.info["sfreq"] if False else None
        # Use original sampling conversion from file metadata when possible.
        # Because we resampled after loading, convert using raw.times duration mapping.
        # Here we infer onset time from original sample using original sfreq.
        # MNE does not expose original sfreq after resampling, so read it from file once.
        # Simpler and robust approach: reload sfreq from header metadata before resampling.
        # To avoid a second file read, we instead use event onset in seconds if present.
        if "onset" in row and not pd.isna(row["onset"]):
            onset_sec = float(row["onset"])
        else:
            raise ValueError("Events TSV must contain an 'onset' column in seconds when using this script.")

        start_sec = onset_sec + cfg.segment_tmin + cfg.crop_tmin
        stop_sec = onset_sec + cfg.crop_tmin + cfg.crop_tmax
        if start_sec < 0:
            raise ValueError(f"Negative start time encountered for trial onset {onset_sec:.3f}s")

        start_idx = int(round(start_sec * fs))
        stop_idx = int(round(stop_sec * fs))
        data = raw.get_data(start=start_idx, stop=stop_idx).T
        trials.append(_as_float64(data))
        trial_meta.append({
            "onset_sec": onset_sec,
            "start_sec": start_sec,
            "stop_sec": stop_sec,
            "n_samples": int(data.shape[0]),
            "n_channels": int(data.shape[1]),
            "trialinfo": _jsonable(row.to_dict()),
        })

    return trials, trial_meta


# -----------------------------------------------------------------------------
# BIDS helper logic for target/masker audio lookup
# -----------------------------------------------------------------------------


def _infer_audio_type(hearing_status: str | None) -> str:
    # Mirrors MATLAB: woa for HI, empty suffix for NH.
    if hearing_status is None:
        return ""
    return "" if str(hearing_status).lower() == "nh" else "woa"



def collect_wav_files_for_run(
    bidsdir: Path,
    events_tsv: Path,
    hearing_status: str | None,
) -> list[dict[str, Any]]:
    """
    Build per-trial {target, masker} audio file mapping like examplescript2.m.
    """
    events = pd.read_csv(events_tsv, sep="\t")
    tm = events.loc[events["trigger_type"].isin(["targetonset", "maskeronset"])].copy()
    audio_type = _infer_audio_type(hearing_status)

    trials: list[dict[str, Any]] = []
    current_trial: dict[str, Any] | None = None

    for i, row in tm.iterrows():
        stim_file = Path(str(row["stim_file"]))
        audio_path = bidsdir / "stimuli" / stim_file.parent / f"{stim_file.stem}{audio_type}{stim_file.suffix}"

        if row["trigger_type"] == "targetonset":
            current_trial = {
                "target_path": audio_path,
                "masker_path": None,
                "masker_delay_sec": None,
                "target_onset_sec": float(row["onset"]) if "onset" in row else None,
            }
            trials.append(current_trial)
        else:
            if current_trial is None:
                raise ValueError("Encountered maskeronset before any targetonset in events TSV")
            masker_onset = float(row["onset"]) if "onset" in row else None
            target_onset = current_trial["target_onset_sec"]
            delay = None if masker_onset is None or target_onset is None else masker_onset - target_onset
            current_trial["masker_path"] = audio_path
            current_trial["masker_delay_sec"] = delay

    return trials


# -----------------------------------------------------------------------------
# Subject-level processing
# -----------------------------------------------------------------------------


def process_subject(
    bidsdir: Path,
    subid: int,
    audio_cfg: AudioConfig,
    eeg_cfg: EEGConfig,
    participants: pd.DataFrame | None = None,
) -> dict[str, Any]:
    sub = f"sub-{subid:03d}"
    eeg_dir = bidsdir / sub / "eeg"

    run_specs: list[tuple[Path, Path]] = []
    run_specs.append(
        (
            eeg_dir / f"{sub}_task-selectiveattention_eeg.bdf",
            eeg_dir / f"{sub}_task-selectiveattention_events.tsv",
        )
    )
    run2_bdf = eeg_dir / f"{sub}_task-selectiveattention_run-2_eeg.bdf"
    run2_evt = eeg_dir / f"{sub}_task-selectiveattention_run-2_events.tsv"
    if run2_bdf.exists() and run2_evt.exists():
        run_specs.append((run2_bdf, run2_evt))

    hearing_status = None
    if participants is not None and "participant_id" in participants.columns:
        row = participants.loc[participants["participant_id"] == sub]
        if len(row) and "hearing_status" in row.columns:
            hearing_status = row.iloc[0]["hearing_status"]

    eeg_trials_all: list[np.ndarray] = []
    eeg_meta_all: list[dict[str, Any]] = []
    aud_trials_all: list[dict[str, np.ndarray | None]] = []
    aud_meta_all: list[dict[str, Any]] = []

    for eeg_path, events_path in run_specs:
        eeg_trials, eeg_meta = preprocess_eeg_run(eeg_path, events_path, eeg_cfg)
        wav_specs = collect_wav_files_for_run(bidsdir, events_path, hearing_status)

        if len(eeg_trials) != len(wav_specs):
            raise ValueError(
                f"Mismatch for {sub} {eeg_path.name}: {len(eeg_trials)} EEG trials vs {len(wav_specs)} target audio trials"
            )

        for eeg_trial, eeg_info, wav_spec in zip(eeg_trials, eeg_meta, wav_specs):
            wav_t, fs_t = sf.read(wav_spec["target_path"])
            feat_t = preprocess_audio_for_mtrf(wav_t, fs_t, audio_cfg)

            feat_m = None
            if wav_spec["masker_path"] is not None:
                wav_m, fs_m = sf.read(wav_spec["masker_path"])
                if fs_m != fs_t:
                    raise ValueError("Target and masker sampling rates differ within trial")
                wav_m = pad_masker_with_silence(wav_m, fs_m, wav_spec["masker_delay_sec"] or 0.0)
                feat_m = preprocess_audio_for_mtrf(wav_m, fs_m, audio_cfg)

            # Align lengths conservatively by truncating to common minimum.
            lengths = [eeg_trial.shape[0], len(feat_t)]
            if feat_m is not None:
                lengths.append(len(feat_m))
            n = min(lengths)

            eeg_trials_all.append(eeg_trial[:n])
            eeg_meta_all.append(eeg_info)
            aud_trials_all.append({
                "target": feat_t[:n],
                "masker": None if feat_m is None else feat_m[:n],
            })
            aud_meta_all.append({
                "target_path": wav_spec["target_path"],
                "masker_path": wav_spec["masker_path"],
                "masker_delay_sec": wav_spec["masker_delay_sec"],
                "aligned_samples": n,
            })

    single_idx = [i for i, a in enumerate(aud_trials_all) if a["masker"] is None]
    two_idx = [i for i, a in enumerate(aud_trials_all) if a["masker"] is not None]

    stim_st = [aud_trials_all[i]["target"] for i in single_idx]
    resp_st = [eeg_trials_all[i] for i in single_idx]
    stim_att = [aud_trials_all[i]["target"] for i in two_idx]
    stim_itt = [aud_trials_all[i]["masker"] for i in two_idx]
    resp_tt = [eeg_trials_all[i] for i in two_idx]

    return {
        "subject": sub,
        "hearing_status": hearing_status,
        "stim_st": np.array(stim_st, dtype=object),
        "resp_st": np.array(resp_st, dtype=object),
        "stim_att": np.array(stim_att, dtype=object),
        "stim_itt": np.array(stim_itt, dtype=object),
        "resp_tt": np.array(resp_tt, dtype=object),
        "eeg_meta": np.array(eeg_meta_all, dtype=object),
        "audio_meta": np.array(aud_meta_all, dtype=object),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Fuglsang-style mTRF inputs from a BIDS dataset")
    p.add_argument("--bidsdir", type=Path, required=True, help="Path to BIDS root directory")
    p.add_argument("--subjects", type=int, nargs="*", default=None, help="Subject IDs, e.g. 1 2 24")
    p.add_argument("--out", type=Path, required=True, help="Output .npz file")
    p.add_argument("--save-mat", action="store_true", help="Also save .mat file")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    bidsdir = args.bidsdir
    participants_path = bidsdir / "participants.tsv"
    participants = pd.read_csv(participants_path, sep="\t") if participants_path.exists() else None

    audio_cfg = AudioConfig()
    eeg_cfg = EEGConfig()

    if args.subjects is None or len(args.subjects) == 0:
        if participants is not None and "participant_id" in participants.columns:
            subjects = [int(str(pid).split("-")[-1]) for pid in participants["participant_id"]]
        else:
            raise ValueError("No --subjects given and participants.tsv was not available")
    else:
        subjects = args.subjects

    out_payload: dict[str, Any] = {
        "meta": np.array(json.dumps(_jsonable({
            "audio_config": asdict(audio_cfg),
            "eeg_config": asdict(eeg_cfg),
            "subjects": subjects,
            "notes": [
                "Python approximation of examplescript2.m",
                "Outputs are mTRF-style per-trial stimulus/response arrays",
                "resp_* trials have shape (time, channels)",
                "stim_* trials have shape (time,)",
            ],
        })), dtype=object)
    }

    for subid in subjects:
        result = process_subject(bidsdir, subid, audio_cfg, eeg_cfg, participants)
        key = f"sub_{subid:03d}"
        out_payload[key] = np.array(result, dtype=object)
        print(f"Processed subject {subid:03d}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out_payload)
    print(f"Saved: {args.out}")

    if args.save_mat:
        mat_payload = {k: (v.item() if isinstance(v, np.ndarray) and v.dtype == object and v.shape == () else v)
                       for k, v in out_payload.items()}
        mat_payload = {k: (_jsonable(v) if k == "meta" else v) for k, v in mat_payload.items()}
        savemat(args.out.with_suffix(".mat"), mat_payload, do_compression=True)
        print(f"Saved: {args.out.with_suffix('.mat')}")


if __name__ == "__main__":
    main()
