from __future__ import annotations

"""
Create mTRF-ready trial lists from Fuglsang selective-attention data when the
stimuli available in the subset are MAT files with derived stimulus features.

This is the MAT-stimulus counterpart of the previous mTRF-oriented script.
Unlike the WAV-based version, it does not run a full auditory front-end. It
loads the most plausible numeric timeseries from each MAT file, assumes it is a
precomputed stimulus feature (typically around 512 Hz), then applies the later
steps:
    lowpass 30 Hz -> resample 64 Hz -> highpass 1 Hz -> lowpass 9 Hz
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

import preprocess_trf as core


def process_subject_mtrf(bidsdir: Path, subject: int) -> dict[str, Any]:
    out = core.process_subject(bidsdir, subject, core.DerivedStimulusConfig(), core.EEGConfig())
    target_st = out["target_st"]
    target_tt = out["target_tt"]
    masker_tt = out["masker_tt"]
    eeg_st = out["eeg_st"]
    eeg_tt = out["eeg_tt"]

    stim_st = np.empty((target_st.shape[2],), dtype=object) if target_st.size else np.empty((0,), dtype=object)
    resp_st = np.empty((eeg_st.shape[2],), dtype=object) if eeg_st.size else np.empty((0,), dtype=object)
    stim_att = np.empty((target_tt.shape[2],), dtype=object) if target_tt.size else np.empty((0,), dtype=object)
    stim_itt = np.empty((masker_tt.shape[2],), dtype=object) if masker_tt.size else np.empty((0,), dtype=object)
    resp_tt = np.empty((eeg_tt.shape[2],), dtype=object) if eeg_tt.size else np.empty((0,), dtype=object)

    for i in range(len(stim_st)):
        stim_st[i] = np.asarray(target_st[:, 0, i], dtype=np.float64)
        resp_st[i] = np.asarray(eeg_st[:, :, i], dtype=np.float64)
    for i in range(len(stim_att)):
        stim_att[i] = np.asarray(target_tt[:, 0, i], dtype=np.float64)
        stim_itt[i] = np.asarray(masker_tt[:, 0, i], dtype=np.float64)
        resp_tt[i] = np.asarray(eeg_tt[:, :, i], dtype=np.float64)

    return {
        "meta": {
            "subject": out["meta"]["subject"],
            "hearing_status": out["meta"]["hearing_status"],
            "notes": [
                "mTRF-ready trial lists built from MAT stimulus derivatives",
                "Each stim_* entry is a 1D time series",
                "Each resp_* entry is a 2D array (time, channels)",
            ],
        },
        "stim_st": stim_st,
        "resp_st": resp_st,
        "stim_att": stim_att,
        "stim_itt": stim_itt,
        "resp_tt": resp_tt,
        "trial_meta": np.array(json.dumps(core._jsonable(out["trial_meta"])), dtype=object),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess Fuglsang MAT stimuli into mTRF-ready trial lists")
    p.add_argument("--bidsdir", type=Path, required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--save-mat", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = process_subject_mtrf(args.bidsdir, args.subject)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    print(f"Saved {args.out}")
    if args.save_mat:
        savemat(args.out.with_suffix('.mat'), {k: (v if not isinstance(v, np.ndarray) or v.dtype != object else v) for k, v in payload.items()}, do_compression=True)
        print(f"Saved {args.out.with_suffix('.mat')}")


if __name__ == "__main__":
    main()
