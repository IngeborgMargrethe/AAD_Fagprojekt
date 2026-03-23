from __future__ import annotations

"""
Preprocess audio and EEG data in a way that closely follows the Fuglsang
selective-attention pipeline shown by the user.

This script is a practical Python approximation of the MATLAB processing chain:

Audio (raw audio version)
-------------------------
average -> lowpass 6000 Hz -> resample 12 kHz -> gammatone/ERB filterbank
-> full-wave rectify / Hilbert envelope -> power-law compression (0.3)
-> average across bands -> lowpass 30 Hz -> resample 64 Hz
-> highpass 1 Hz -> lowpass 9 Hz

Audio (derived-envelope version)
--------------------------------
If the raw audio is not available, the script can start from a derived envelope
sampled at 512 Hz and then apply:
lowpass 30 Hz -> resample 64 Hz -> highpass 1 Hz -> lowpass 9 Hz

EEG
---
raw -> rereference to mastoids -> lowpass 30 Hz -> resample 64 Hz
-> EOG regression denoising (optional) -> highpass 1 Hz -> lowpass 9 Hz
-> crop time-of-interest

Important note
--------------
The original MATLAB functions used in Fuglsang's dataset codebase
(build_aud_features, build_eeg_features, and the exact FieldTrip helpers) are
not available here. Therefore, this script reproduces the *reported signal
processing logic* and the preprocessing sequence from the example script, but it
cannot guarantee bit-identical outputs.

References used for this approximation:
- Fuglsang et al. describe speech features as ERB/gammatone-like subband
  envelopes, Hilbert envelope extraction, power-law compression with exponent
  0.3, and final low-pass filtering/averaging across filters.
- They also describe EEG preprocessing with mastoid rereferencing, artifact
  cleaning, and a final low-frequency band emphasizing the delta/theta range.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import mne
import numpy as np
import soundfile as sf
from scipy.io import savemat
from scipy.signal import butter, filtfilt, hilbert, resample_poly

try:
    from gammatone.filters import centre_freqs, make_erb_filters, erb_filterbank
    HAVE_GAMMATONE = True
except Exception:
    HAVE_GAMMATONE = False


# -----------------------------------------------------------------------------
# Signal utilities
# -----------------------------------------------------------------------------


def _ensure_2d_audio(x: np.ndarray) -> np.ndarray:
    """Return audio as shape (samples, channels)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError(f"Audio must be 1D or 2D, got shape={x.shape}")



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
        btype = "band"
        wn = [low / nyq, high / nyq]
    elif low is not None:
        btype = "high"
        wn = low / nyq
    elif high is not None:
        btype = "low"
        wn = high / nyq
    else:
        raise ValueError("At least one of low/high must be specified")

    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, x, axis=axis)



def _resample(x: np.ndarray, fs_in: float, fs_out: float, axis: int = 0) -> np.ndarray:
    if fs_in == fs_out:
        return np.asarray(x)
    # Rational approximation robust enough for standard sampling rates.
    from fractions import Fraction

    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    return resample_poly(x, up, down, axis=axis)



def _average_channels(x: np.ndarray) -> np.ndarray:
    x = _ensure_2d_audio(x)
    return x.mean(axis=1)



def _full_wave_rectify(x: np.ndarray) -> np.ndarray:
    return np.abs(x)



def _hilbert_envelope(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.abs(hilbert(x, axis=axis))


# -----------------------------------------------------------------------------
# Audio feature extraction
# -----------------------------------------------------------------------------


@dataclass
class AudioConfig:
    fs_lowpass_1: float = 6000.0
    fs_resample_1: float = 12000.0
    n_bands: int = 28
    f_min: float = 150.0
    f_max: float = 8000.0
    compression: float = 0.3
    fs_lowpass_2: float = 30.0
    fs_mid: float = 512.0
    fs_out: float = 64.0
    hp_out: float = 1.0
    lp_out: float = 9.0
    filt_order: int = 4



def erb_envelope_feature(
    audio: np.ndarray,
    fs: float,
    cfg: AudioConfig,
) -> tuple[np.ndarray, dict]:
    """
    Extract Fuglsang-style broadband envelope from raw audio.

    Returns
    -------
    env : ndarray, shape (samples_out,)
        Final envelope sampled at cfg.fs_out.
    info : dict
        Metadata about the processing steps.
    """
    audio = _ensure_2d_audio(audio)
    mono = _average_channels(audio)

    # MATLAB-like preprocessing chain from the provided script.
    mono = _butter_filter(mono, fs, high=cfg.fs_lowpass_1, order=cfg.filt_order)
    mono = _resample(mono, fs, cfg.fs_resample_1)
    fs1 = cfg.fs_resample_1

    if not HAVE_GAMMATONE:
        raise ImportError(
            "gammatone package is not installed. Install `gammatone` to use raw "
            "audio feature extraction, or start from derived envelopes with "
            "--derived-envelope."
        )

    cfs = centre_freqs(cfg.fs_resample_1, cfg.n_bands, cfg.f_min)
    cfs = cfs[(cfs >= cfg.f_min) & (cfs <= cfg.f_max)]
    fcoefs = make_erb_filters(fs1, cfs)
    bands = erb_filterbank(mono, fcoefs)  # shape: (bands, samples)

    # User's MATLAB chain lists full-wave rectify; the paper describes Hilbert.
    # We use Hilbert envelopes because that is the more explicit method reported.
    env_bands = _hilbert_envelope(bands, axis=1)
    env_bands = env_bands ** cfg.compression

    # Average across bands *after* extracting/compressing narrowband envelopes.
    broadband = env_bands.mean(axis=0)

    # Match the MATLAB-style later envelope processing.
    broadband = _butter_filter(broadband, fs1, high=cfg.fs_lowpass_2, order=cfg.filt_order)
    broadband = _resample(broadband, fs1, cfg.fs_mid)
    broadband = _butter_filter(broadband, cfg.fs_mid, low=cfg.hp_out, order=cfg.filt_order)
    broadband = _butter_filter(broadband, cfg.fs_mid, high=cfg.lp_out, order=cfg.filt_order)
    broadband = _resample(broadband, cfg.fs_mid, cfg.fs_out)

    info = {
        "input_fs": fs,
        "mono": True,
        "fs_after_first_resample": cfg.fs_resample_1,
        "fs_after_second_resample": cfg.fs_mid,
        "fs_out": cfg.fs_out,
        "n_bands": int(len(cfs)),
        "band_centers_hz": cfs.tolist(),
        "compression": cfg.compression,
        "processing": [
            "average_channels",
            f"lowpass_{cfg.fs_lowpass_1}",
            f"resample_{cfg.fs_resample_1}",
            "erb_filterbank",
            "hilbert_envelope",
            f"powerlaw_{cfg.compression}",
            "average_bands",
            f"lowpass_{cfg.fs_lowpass_2}",
            f"resample_{cfg.fs_mid}",
            f"highpass_{cfg.hp_out}",
            f"lowpass_{cfg.lp_out}",
            f"resample_{cfg.fs_out}",
        ],
    }
    return _as_float64(broadband), info



def preprocess_derived_envelope(
    env_512: np.ndarray,
    fs_in: float,
    cfg: AudioConfig,
) -> tuple[np.ndarray, dict]:
    """Process already-derived envelope features, like the Zenodo derivatives."""
    x = np.asarray(env_512).squeeze()
    if x.ndim != 1:
        raise ValueError("Derived envelope must be one-dimensional after squeeze().")

    x = _butter_filter(x, fs_in, high=cfg.fs_lowpass_2, order=cfg.filt_order)
    x = _resample(x, fs_in, cfg.fs_out)
    x = _butter_filter(x, cfg.fs_out, low=cfg.hp_out, order=cfg.filt_order)
    x = _butter_filter(x, cfg.fs_out, high=cfg.lp_out, order=cfg.filt_order)

    info = {
        "input_fs": fs_in,
        "fs_out": cfg.fs_out,
        "processing": [
            f"lowpass_{cfg.fs_lowpass_2}",
            f"resample_{cfg.fs_out}",
            f"highpass_{cfg.hp_out}",
            f"lowpass_{cfg.lp_out}",
        ],
    }
    return _as_float64(x), info


# -----------------------------------------------------------------------------
# EEG preprocessing
# -----------------------------------------------------------------------------


@dataclass
class EEGConfig:
    lowpass_30_before_resample: float = 30.0
    resample_to: float = 64.0
    highpass_final: float = 1.0
    lowpass_final: float = 9.0
    eog_regression: bool = True
    crop_tmin: float | None = None
    crop_tmax: float | None = None



def _load_raw_eeg(path: Path, preload: bool = True) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=preload, verbose="ERROR")
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=preload, verbose="ERROR")
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(path, preload=preload, verbose="ERROR")
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=preload, verbose="ERROR")
    if suffix == ".set":
        return mne.io.read_raw_eeglab(path, preload=preload, verbose="ERROR")
    raise ValueError(
        f"Unsupported EEG format '{suffix}'. Supported: .fif, .edf, .bdf, .vhdr, .set"
    )



def _apply_mastoid_reference(raw: mne.io.BaseRaw, mastoids: Sequence[str] | None) -> mne.io.BaseRaw:
    if mastoids:
        missing = [ch for ch in mastoids if ch not in raw.ch_names]
        if missing:
            raise ValueError(f"Mastoid channels not found: {missing}")
        raw = raw.copy().set_eeg_reference(ref_channels=list(mastoids), verbose="ERROR")
    return raw



def _run_eog_regression(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw = raw.copy()
    eog_picks = mne.pick_types(raw.info, eog=True)
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False)
    if len(eog_picks) == 0 or len(eeg_picks) == 0:
        return raw

    data = raw.get_data()
    X = data[eog_picks].T  # (time, n_eog)
    X = np.column_stack([X, np.ones(len(X))])

    for pick in eeg_picks:
        y = data[pick]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X[:, :-1] @ beta[:-1] + beta[-1]
        data[pick] = y - y_hat

    raw._data = data
    return raw



def preprocess_eeg(
    eeg_path: Path,
    cfg: EEGConfig,
    mastoids: Sequence[str] | None = None,
) -> tuple[np.ndarray, float, dict]:
    """
    Return EEG as (time, channels) after Fuglsang-like preprocessing.
    """
    raw = _load_raw_eeg(eeg_path)
    original_fs = float(raw.info["sfreq"])

    raw = _apply_mastoid_reference(raw, mastoids)

    # MATLAB example script first low-pass filters at 30 Hz, then resamples to 64 Hz.
    raw.filter(
        l_freq=None,
        h_freq=cfg.lowpass_30_before_resample,
        method="iir",
        phase="zero",
        verbose="ERROR",
    )
    raw.resample(cfg.resample_to, verbose="ERROR")

    if cfg.eog_regression:
        raw = _run_eog_regression(raw)

    raw.filter(
        l_freq=cfg.highpass_final,
        h_freq=None,
        method="iir",
        phase="zero",
        verbose="ERROR",
    )
    raw.filter(
        l_freq=None,
        h_freq=cfg.lowpass_final,
        method="iir",
        phase="zero",
        verbose="ERROR",
    )

    if cfg.crop_tmin is not None or cfg.crop_tmax is not None:
        raw.crop(tmin=cfg.crop_tmin or 0.0, tmax=cfg.crop_tmax, include_tmax=False)

    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False)
    eeg = raw.get_data(picks=eeg_picks).T  # (time, channels)
    ch_names = [raw.ch_names[i] for i in eeg_picks]

    info = {
        "input_fs": original_fs,
        "fs_out": cfg.resample_to,
        "n_channels": len(ch_names),
        "channels": ch_names,
        "mastoids": list(mastoids) if mastoids else None,
        "eog_regression": cfg.eog_regression,
        "crop_tmin": cfg.crop_tmin,
        "crop_tmax": cfg.crop_tmax,
        "processing": [
            "reref_mastoids" if mastoids else "reference_unchanged",
            f"lowpass_{cfg.lowpass_30_before_resample}",
            f"resample_{cfg.resample_to}",
            "eog_regression" if cfg.eog_regression else "no_eog_regression",
            f"highpass_{cfg.highpass_final}",
            f"lowpass_{cfg.lowpass_final}",
        ],
    }
    return _as_float64(eeg), float(cfg.resample_to), info


# -----------------------------------------------------------------------------
# Trial struct assembly
# -----------------------------------------------------------------------------



def stack_trials_3d(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Stack equal-length arrays into (time, features, trials)."""
    if not arrays:
        raise ValueError("No arrays supplied")
    arrays = [np.asarray(a) for a in arrays]
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"All trials must have equal length, got lengths={sorted(lengths)}")

    fixed = []
    for a in arrays:
        if a.ndim == 1:
            fixed.append(a[:, None])
        elif a.ndim == 2:
            fixed.append(a)
        else:
            raise ValueError(f"Each trial must be 1D or 2D, got shape={a.shape}")
    return np.stack(fixed, axis=2)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuglsang-style preprocessing for audio and EEG.")

    p.add_argument("--audio", type=Path, nargs="*", default=None, help="Raw audio file(s), one per trial.")
    p.add_argument(
        "--derived-envelope",
        type=Path,
        nargs="*",
        default=None,
        help="Envelope file(s) instead of raw audio. Supported: .npy, .npz, .txt, .csv.",
    )
    p.add_argument("--eeg", type=Path, nargs="*", default=None, help="Raw EEG file(s), one per trial or recording.")
    p.add_argument("--out", type=Path, required=True, help="Output .npz or .mat file.")
    p.add_argument("--audio-key", type=str, default=None, help="Key to read from .npz derived envelope files.")
    p.add_argument("--derived-fs", type=float, default=512.0, help="Sampling rate of derived envelopes.")
    p.add_argument("--mastoids", nargs="*", default=["M1", "M2"], help="Mastoid channel names for rereferencing.")
    p.add_argument("--crop-tmin", type=float, default=None, help="Crop EEG start time in seconds.")
    p.add_argument("--crop-tmax", type=float, default=None, help="Crop EEG end time in seconds.")
    p.add_argument("--no-eog-regression", action="store_true", help="Disable simple EOG regression denoising.")
    p.add_argument("--save-mat", action="store_true", help="Also save MATLAB .mat alongside the .npz output.")
    return p.parse_args()



def _load_derived_env(path: Path, key: str | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        z = np.load(path)
        if key is None:
            if len(z.files) != 1:
                raise ValueError(f"{path} has multiple arrays; specify --audio-key.")
            key = z.files[0]
        return z[key]
    if suffix in {".txt", ".csv"}:
        return np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    raise ValueError(f"Unsupported derived-envelope format: {suffix}")



def main() -> None:
    args = parse_args()
    audio_cfg = AudioConfig()
    eeg_cfg = EEGConfig(
        eog_regression=not args.no_eog_regression,
        crop_tmin=args.crop_tmin,
        crop_tmax=args.crop_tmax,
    )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "meta": {
            "audio_config": asdict(audio_cfg),
            "eeg_config": asdict(eeg_cfg),
            "notes": [
                "Python approximation of Fuglsang-style preprocessing",
                "not guaranteed to be numerically identical to the original MATLAB code",
            ],
        }
    }

    # Audio trials
    audio_trials = []
    audio_infos = []
    if args.audio:
        for path in args.audio:
            wav, fs = sf.read(path)
            feat, info = erb_envelope_feature(wav, fs, audio_cfg)
            audio_trials.append(feat)
            info["source"] = str(path)
            audio_infos.append(info)
        result["aud_feature"] = stack_trials_3d(audio_trials)
        result["aud_fs"] = audio_cfg.fs_out
        result["aud_info"] = audio_infos

    elif args.derived_envelope:
        for path in args.derived_envelope:
            env = _load_derived_env(path, args.audio_key)
            feat, info = preprocess_derived_envelope(env, args.derived_fs, audio_cfg)
            audio_trials.append(feat)
            info["source"] = str(path)
            audio_infos.append(info)
        result["aud_feature"] = stack_trials_3d(audio_trials)
        result["aud_fs"] = audio_cfg.fs_out
        result["aud_info"] = audio_infos

    # EEG trials
    eeg_trials = []
    eeg_infos = []
    if args.eeg:
        for path in args.eeg:
            eeg, fs_out, info = preprocess_eeg(path, eeg_cfg, mastoids=args.mastoids)
            eeg_trials.append(eeg)
            info["source"] = str(path)
            eeg_infos.append(info)
        result["eeg_feature"] = stack_trials_3d(eeg_trials)
        result["eeg_fs"] = fs_out
        result["eeg_info"] = eeg_infos

    if "aud_feature" not in result and "eeg_feature" not in result:
        raise SystemExit("Nothing to do. Provide --audio and/or --derived-envelope and/or --eeg.")

    # Save NPZ
    npz_payload = {}
    for key, value in result.items():
        if isinstance(value, (dict, list)):
            npz_payload[key] = np.array(json.dumps(value), dtype=object)
        else:
            npz_payload[key] = value
    np.savez(out, **npz_payload)

    # Optional MATLAB export
    if args.save_mat:
        mat_path = out.with_suffix(".mat")
        mat_payload = {}
        for key, value in result.items():
            if isinstance(value, (dict, list)):
                mat_payload[key] = json.dumps(value)
            else:
                mat_payload[key] = value
        savemat(mat_path, mat_payload, do_compression=True)

    print(f"Saved: {out}")
    if args.save_mat:
        print(f"Saved: {out.with_suffix('.mat')}")


if __name__ == "__main__":
    main()
