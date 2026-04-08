from __future__ import annotations

"""
Organize MAT-stimulus Fuglsang preprocessing outputs into:
    data/example1/preprocessed.npz
    data/example2_mtrf/preprocessed.npz

This wrapper is the MAT-stimulus replacement for the previous dataset script.
It should be used when the subset contains stimuli/subXXX/target/*.mat and
stimuli/subXXX/masker/*.mat instead of raw WAV files.
"""

import argparse
from pathlib import Path
from scipy.io import savemat
import numpy as np

import preprocess_trf as ex1
import preprocess_mtrf as ex2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create organized Fuglsang preprocessing outputs for MAT stimuli")
    p.add_argument("--bidsdir", type=Path, required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--mode", choices=["all", "example1", "example2"], default="all")
    p.add_argument("--save-mat", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"all", "example1"}:
        out1 = args.data_dir / "example1" / "preprocessed.npz"
        out1.parent.mkdir(parents=True, exist_ok=True)
        payload1 = ex1.process_subject(args.bidsdir, args.subject, ex1.DerivedStimulusConfig(), ex1.EEGConfig())
        np.savez(out1, **{k: (np.array(ex1._jsonable(v), dtype=object) if isinstance(v, (dict, list)) else v) for k, v in payload1.items()})
        print(f"Saved {out1}")
        if args.save_mat:
            savemat(out1.with_suffix('.mat'), {k: (str(ex1._jsonable(v)) if isinstance(v, (dict, list)) else v) for k, v in payload1.items()}, do_compression=True)
            print(f"Saved {out1.with_suffix('.mat')}")

    if args.mode in {"all", "example2"}:
        out2 = args.data_dir / "example2_mtrf" / "preprocessed.npz"
        out2.parent.mkdir(parents=True, exist_ok=True)
        payload2 = ex2.process_subject_mtrf(args.bidsdir, args.subject)
        np.savez(out2, **payload2)
        print(f"Saved {out2}")
        if args.save_mat:
            savemat(out2.with_suffix('.mat'), payload2, do_compression=True)
            print(f"Saved {out2.with_suffix('.mat')}")


if __name__ == "__main__":
    main()
