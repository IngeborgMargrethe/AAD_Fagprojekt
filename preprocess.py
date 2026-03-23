from __future__ import annotations

"""
Unified Fuglsang-style preprocessing runner.

This script organizes outputs into a single `data/` folder with two subfolders:

    data/
      example1/
        preprocessed.npz
        preprocessed.mat   (optional)
      example2_mtrf/
        preprocessed.npz
        preprocessed.mat   (optional)

It wraps the two earlier preprocessing implementations:
- fuglsang_preprocess.py          -> example1-style feature tensors
- fuglsang_preprocess_mtrf.py     -> example2 / mTRF-style trial lists

Typical use
-----------
python fuglsang_preprocess_dataset.py \
    --bidsdir /path/to/ds-eeg-nhhi \
    --subject 1 \
    --data-dir ./data \
    --save-mat

Notes
-----
- This script creates ONE preprocessed file inside each folder.
- For example1, it builds an `srdat`-like structure for a single subject:
    target_st, target_tt, masker_tt, eeg_st, eeg_tt
- For example2, it stores mTRF-ready trial arrays for the same subject.
- This is a practical Python approximation of the MATLAB pipelines.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import savemat

# Import the two earlier scripts as modules.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fuglsang_preprocess as ex1
import fuglsang_preprocess_mtrf as ex2


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


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



def _save_npz_and_optional_mat(out_base: Path, payload: dict[str, Any], save_mat: bool) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)

    npz_payload: dict[str, Any] = {}
    mat_payload: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            npz_payload[key] = np.array(json.dumps(_jsonable(value)), dtype=object)
            mat_payload[key] = json.dumps(_jsonable(value))
        else:
            npz_payload[key] = value
            mat_payload[key] = value

    np.savez(out_base.with_suffix(".npz"), **npz_payload)
    if save_mat:
        savemat(out_base.with_suffix(".mat"), mat_payload, do_compression=True)



def _load_participants(bidsdir: Path) -> pd.DataFrame | None:
    p = bidsdir / "participants.tsv"
    return pd.read_csv(p, sep="\t") if p.exists() else None


# -----------------------------------------------------------------------------
# Example 1 builder: create srdat-like arrays for one subject
# -----------------------------------------------------------------------------


def build_example1_subject(bidsdir: Path, subject: int) -> dict[str, Any]:
    audio_cfg = ex1.AudioConfig()
    eeg_cfg = ex1.EEGConfig(
        eog_regression=True,
        crop_tmin=6.0,
        crop_tmax=43.0,
    )

    participants = _load_participants(bidsdir)
    sub = f"sub-{subject:03d}"
    eeg_dir = bidsdir / sub / "eeg"

    run_specs: list[tuple[Path, Path]] = [
        (
            eeg_dir / f"{sub}_task-selectiveattention_eeg.bdf",
            eeg_dir / f"{sub}_task-selectiveattention_events.tsv",
        )
    ]
    run2_bdf = eeg_dir / f"{sub}_task-selectiveattention_run-2_eeg.bdf"
    run2_evt = eeg_dir / f"{sub}_task-selectiveattention_run-2_events.tsv"
    if run2_bdf.exists() and run2_evt.exists():
        run_specs.append((run2_bdf, run2_evt))

    hearing_status = None
    if participants is not None and "participant_id" in participants.columns:
        row = participants.loc[participants["participant_id"] == sub]
        if len(row) and "hearing_status" in row.columns:
            hearing_status = row.iloc[0]["hearing_status"]

    target_st_trials: list[np.ndarray] = []
    target_tt_trials: list[np.ndarray] = []
    masker_tt_trials: list[np.ndarray] = []
    eeg_st_trials: list[np.ndarray] = []
    eeg_tt_trials: list[np.ndarray] = []
    trial_meta: list[dict[str, Any]] = []

    for eeg_path, events_path in run_specs:
        eeg_trials, eeg_meta = ex2.preprocess_eeg_run(eeg_path, events_path, ex2.EEGConfig())
        wav_specs = ex2.collect_wav_files_for_run(bidsdir, events_path, hearing_status)

        if len(eeg_trials) != len(wav_specs):
            raise ValueError(
                f"Mismatch for {sub} {eeg_path.name}: {len(eeg_trials)} EEG trials vs {len(wav_specs)} audio trials"
            )

        for eeg_trial, eeg_info, wav_spec in zip(eeg_trials, eeg_meta, wav_specs):
            # Audio processing using the richer example1-style front-end.
            import soundfile as sf
            wav_t, fs_t = sf.read(wav_spec["target_path"])
            feat_t, _ = ex1.erb_envelope_feature(wav_t, fs_t, audio_cfg)

            feat_m = None
            if wav_spec["masker_path"] is not None:
                wav_m, fs_m = sf.read(wav_spec["masker_path"])
                if fs_m != fs_t:
                    raise ValueError("Target and masker sampling rates differ within trial")
                wav_m = ex2.pad_masker_with_silence(wav_m, fs_m, wav_spec["masker_delay_sec"] or 0.0)
                feat_m, _ = ex1.erb_envelope_feature(wav_m, fs_m, audio_cfg)

            # Align to common sample count after preprocessing.
            lengths = [eeg_trial.shape[0], len(feat_t)]
            if feat_m is not None:
                lengths.append(len(feat_m))
            n = min(lengths)

            eeg_trial = eeg_trial[:n]
            feat_t = feat_t[:n]
            if feat_m is not None:
                feat_m = feat_m[:n]

            if feat_m is None:
                target_st_trials.append(feat_t)
                eeg_st_trials.append(eeg_trial)
                trial_kind = "singletalker"
            else:
                target_tt_trials.append(feat_t)
                masker_tt_trials.append(feat_m)
                eeg_tt_trials.append(eeg_trial)
                trial_kind = "twotalker"

            trial_meta.append({
                "subject": sub,
                "trial_kind": trial_kind,
                "target_path": wav_spec["target_path"],
                "masker_path": wav_spec["masker_path"],
                "masker_delay_sec": wav_spec["masker_delay_sec"],
                "aligned_samples": int(n),
                "eeg_info": eeg_info,
            })

    result = {
        "meta": {
            "subject": sub,
            "hearing_status": hearing_status,
            "audio_config": asdict(audio_cfg),
            "eeg_config": asdict(eeg_cfg),
            "notes": [
                "Approximation of examplescript1.m",
                "Arrays are organized to resemble srdat",
                "audio arrays are (time, features, trials)",
                "eeg arrays are (time, channels, trials)",
            ],
        },
        "target_st": ex1.stack_trials_3d(target_st_trials) if target_st_trials else np.empty((0, 0, 0)),
        "target_tt": ex1.stack_trials_3d(target_tt_trials) if target_tt_trials else np.empty((0, 0, 0)),
        "masker_tt": ex1.stack_trials_3d(masker_tt_trials) if masker_tt_trials else np.empty((0, 0, 0)),
        "eeg_st": ex1.stack_trials_3d(eeg_st_trials) if eeg_st_trials else np.empty((0, 0, 0)),
        "eeg_tt": ex1.stack_trials_3d(eeg_tt_trials) if eeg_tt_trials else np.empty((0, 0, 0)),
        "fs_audio": audio_cfg.fs_out,
        "fs_eeg": ex2.EEGConfig().fs_out,
        "trial_meta": trial_meta,
    }
    return result


# -----------------------------------------------------------------------------
# Example 2 builder: create one mTRF file for one subject
# -----------------------------------------------------------------------------


def build_example2_subject(bidsdir: Path, subject: int) -> dict[str, Any]:
    participants = _load_participants(bidsdir)
    audio_cfg = ex2.AudioConfig()
    eeg_cfg = ex2.EEGConfig()
    result = ex2.process_subject(bidsdir, subject, audio_cfg, eeg_cfg, participants)
    return {
        "meta": {
            "subject": result["subject"],
            "hearing_status": result["hearing_status"],
            "audio_config": asdict(audio_cfg),
            "eeg_config": asdict(eeg_cfg),
            "notes": [
                "Approximation of examplescript2.m",
                "mTRF-ready per-trial data",
                "stim_* arrays are object arrays of shape (n_trials,)",
                "resp_* arrays are object arrays of shape (n_trials,) with each trial shaped (time, channels)",
            ],
        },
        "stim_st": result["stim_st"],
        "resp_st": result["resp_st"],
        "stim_att": result["stim_att"],
        "stim_itt": result["stim_itt"],
        "resp_tt": result["resp_tt"],
        "eeg_meta": result["eeg_meta"],
        "audio_meta": result["audio_meta"],
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create organized Fuglsang preprocessing outputs in data/example1 and data/example2_mtrf")
    p.add_argument("--bidsdir", type=Path, required=True, help="Path to ds-eeg-nhhi BIDS root")
    p.add_argument("--subject", type=int, required=True, help="Single subject ID, e.g. 1")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Root output folder")
    p.add_argument("--mode", choices=["all", "example1", "example2"], default="all")
    p.add_argument("--save-mat", action="store_true", help="Also save .mat files")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"all", "example1"}:
        out1 = data_dir / "example1" / "preprocessed"
        payload1 = build_example1_subject(args.bidsdir, args.subject)
        _save_npz_and_optional_mat(out1, payload1, args.save_mat)
        print(f"Saved: {out1.with_suffix('.npz')}")
        if args.save_mat:
            print(f"Saved: {out1.with_suffix('.mat')}")

    if args.mode in {"all", "example2"}:
        out2 = data_dir / "example2_mtrf" / "preprocessed"
        payload2 = build_example2_subject(args.bidsdir, args.subject)
        _save_npz_and_optional_mat(out2, payload2, args.save_mat)
        print(f"Saved: {out2.with_suffix('.npz')}")
        if args.save_mat:
            print(f"Saved: {out2.with_suffix('.mat')}")


if __name__ == "__main__":
    main()
