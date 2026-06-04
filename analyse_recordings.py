"""
Sleep metrics calculation for harmonized PSG datasets (HDF5 format).

Metrics computed per recording:
  - Time in Bed (TIB):            first epoch to last epoch = recording duration
                                  (lights-off = recording start, lights-on = recording end)
  - Total Sleep Time (TST):       sum of all non-wake, non-movement, non-unknown epochs (minutes)
  - Sleep Onset Latency (SOL):    time from start of recording (lights-out) to the FIRST
                                  epoch of any sleep stage (N1/N2/N3/REM).
                                  Returns recording_min if no sleep occurs.
  - Sleep Efficiency (SE):        TST / TIB (%)
  - WASO (min):                   wake time after sleep onset until END OF RECORDING (lights-on),
                                  i.e. includes the final awakening period.
  - N1%:                          N1 time / TST
  - N2%:                          N2 time / TST
  - N3%:                          N3 time / TST
  - REM%:                         REM time / TST

Definitions
  - SOL: "time from lights-out to the first epoch scored as any sleep stage"
  - WASO: "total wake time after sleep onset until lights-on"
    (includes the final awakening / WAFA, not just inter-sleep wake)

Stage label mapping:
  W=0, N1=1, N2=2, N3=3, REM=4, MOVE=5, UNK=6
"""

import os
import glob
import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

STAGE_DICT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "MOVE": 5, "UNK": 6}
SLEEP_STAGES = {STAGE_DICT["N1"], STAGE_DICT["N2"], STAGE_DICT["N3"], STAGE_DICT["REM"]}

def extract_channels(h5f):
    if "signals" in h5f:
        return [str(c) for c in h5f["signals"].keys()]
    return []


def compute_metrics(h5_path):
    """
    Returns:
        Dictionary with keys: file, tst_min, sol_min, se_pct, tib_min,
        n1_pct, n2_pct, n3_pct, rem_pct,
        recording_min, n_epochs, epoch_dur_s,
        n_wake, n_n1, n_n2, n_n3, n_rem, n_move, n_unk
    """
    # Initialize everything as NaN/Empty
    res = {
        "file": Path(h5_path).name, "tib_min": np.nan, "tst_min": np.nan, "sol_min": np.nan,
        "se_pct": np.nan, "waso_min": np.nan, "n1_pct": np.nan, "n2_pct": np.nan,
        "n3_pct": np.nan, "rem_pct": np.nan, "recording_min": np.nan, "n_epochs": np.nan,
        "epoch_dur_s": np.nan, "n_wake": np.nan, "n_n1": np.nan, "n_n2": np.nan,
        "n_n3": np.nan, "n_rem": np.nan, "n_move": np.nan, "n_unk": np.nan,
        "n_waso": np.nan, "channels": []
    }

    with h5py.File(h5_path, "r") as h5f:
        # Channels
        res["channels"] = extract_channels(h5f)

        # Epoch duration
        if "epoch_duration" not in h5f.attrs:
            return res
        res["epoch_dur_s"] = float(h5f.attrs["epoch_duration"])

        # Hypnogram
        if "y" not in h5f:
            return res
        labels = h5f["y"][:]
        epoch_dur_s = res["epoch_dur_s"]
        epoch_dur_min = epoch_dur_s / 60.0

        #  Channels
        channels = extract_channels(h5f)

        # Per-stage epoch counts
        n_wake  = int(np.sum(labels == STAGE_DICT["W"]))
        n_n1    = int(np.sum(labels == STAGE_DICT["N1"]))
        n_n2    = int(np.sum(labels == STAGE_DICT["N2"]))
        n_n3    = int(np.sum(labels == STAGE_DICT["N3"]))
        n_rem   = int(np.sum(labels == STAGE_DICT["REM"]))
        n_move  = int(np.sum(labels == STAGE_DICT["MOVE"]))
        n_unk   = int(np.sum(labels == STAGE_DICT["UNK"]))
        n_sleep = n_n1 + n_n2 + n_n3 + n_rem  # MOVE and UNK excluded from TST

        n_epochs      = len(labels)
        recording_min = n_epochs * epoch_dur_min

        # Total Sleep Time
        tst_min = n_sleep * epoch_dur_min

        # SOL: time from lights-out (epoch 0) to the FIRST sleep epoch
        # lights-out to first epoch scored as sleep
        sleep_epoch_indices = np.where(np.isin(labels, list(SLEEP_STAGES)))[0]
        if len(sleep_epoch_indices) >= 1:
            first_sleep_idx = sleep_epoch_indices[0]
            sol_min = round(first_sleep_idx * epoch_dur_min, 2)
        else:
            # No sleep detected: SOL equals the full recording duration
            first_sleep_idx = None
            sol_min = round(recording_min, 2)

        se_pct  = (tst_min / recording_min * 100.0) if recording_min > 0 else np.nan

        # TIB: lights-off = recording start, lights-on = recording end
        tib_min = recording_min

        #  WASO: ALL wake epochs from sleep onset to final awakening (end of recording)
        # includes the final awakening period
        if first_sleep_idx is not None:
            post_onset_labels = labels[first_sleep_idx:]
            n_waso  = int(np.sum(post_onset_labels == STAGE_DICT["W"]))
            waso_min = round(n_waso * epoch_dur_min, 2)
        else:
            n_waso   = 0
            waso_min = 0.0

        # Stage percentages (relative to TST; NaN if TST == 0)
        def stage_pct(n_stage):
            return round(n_stage / n_sleep * 100.0, 2) if n_sleep > 0 else np.nan

        res.update({
            "tib_min": round(tib_min, 2), "tst_min": round(tst_min, 2), "sol_min": sol_min,
            "se_pct": round(se_pct, 2), "waso_min": waso_min, "n1_pct": stage_pct(n_n1),
            "n2_pct": stage_pct(n_n2), "n3_pct": stage_pct(n_n3), "rem_pct": stage_pct(n_rem),
            "recording_min": round(recording_min, 2), "n_epochs": n_epochs,
            "n_wake": n_wake, "n_n1": n_n1, "n_n2": n_n2, "n_n3": n_n3,
            "n_rem": n_rem, "n_move": n_move, "n_unk": n_unk, "n_waso": n_waso
        })

        return res


def summarize(df):
    """Compute aggregate statistics over per-file metric columns."""
    metric_cols = ["tib_min", "tst_min", "se_pct", "sol_min", "waso_min",
                   "n1_pct", "n2_pct", "n3_pct", "rem_pct"]
    # Only attempt to summarize columns that exist in the dataframe
    existing_cols = [c for c in metric_cols if c in df.columns]
    if not existing_cols:
        return pd.DataFrame()

    stats = df[metric_cols].agg(
        ["mean", "std", "min",
         lambda x: x.quantile(0.25),
         "median",
         lambda x: x.quantile(0.75),
         "max"]
    )
    stats.index = ["mean", "std", "min", "q1", "median", "q3", "max"]
    return stats.round(2)

def print_channel_summary(records, num_files):
    """
    Print channel occurrence statistics across all recordings.
    """
    counter = Counter()

    for r in records:
        unique_channels = set(r.get("channels", []))
        counter.update(unique_channels)

    print(f"\n{'='*76}")
    print("  Channel occurrence summary")
    print(f"{'='*76}")

    if not counter:
        print("No channel information found.")
        return

    print(f"{'Channel':<30} {'%':>20}")
    print("-" * 52)

    for ch, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"{ch:<30} {100*count/num_files:>20.1f}")

    print(f"{'='*76}\n")

def print_summary(df, stats):
    if stats.empty:
        print("\n[INFO] No numeric sleep metrics could be calculated (missing hypnograms in all files).")
        return

    print(f"\n{'='*76}")
    print(f"  Files processed: {len(df)}")
    print(f"{'='*76}")
    col_w = 18
    header = (f"{'Metric':<{col_w}} {'Mean':>8} {'Std':>8} {'Min':>8}"
              f" {'Q1':>8} {'Median':>8} {'Q3':>8} {'Max':>8}")
    print(header)
    print("-" * len(header))
    labels_map = {
        "tib_min":  "TIB (min)",
        "tst_min":  "TST (min)",
        "se_pct":   "Sleep Eff. (%)",
        "sol_min":  "SOL (min)",
        "waso_min": "WASO (min)",
        "n1_pct":   "N1 (%TST)",
        "n2_pct":   "N2 (%TST)",
        "n3_pct":   "N3 (%TST)",
        "rem_pct":  "REM (%TST)",
    }
    for col, label in labels_map.items():
        row = stats[col]
        print(
            f"{label:<{col_w}} "
            f"{row['mean']:>8.1f} "
            f"{row['std']:>8.1f} "
            f"{row['min']:>8.1f} "
            f"{row['q1']:>8.1f} "
            f"{row['median']:>8.1f} "
            f"{row['q3']:>8.1f} "
            f"{row['max']:>8.1f}"
        )
    print(f"{'='*76}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute sleep metrics from harmonized HDF5 files."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing harmonized .hdf5 files (searched recursively).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path for output CSV file. Defaults to <input_dir>/sleep_metrics.csv",
    )
    args = parser.parse_args()

    h5_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.hdf5"), recursive=True))
    if not h5_files:
        raise FileNotFoundError(f"No .hdf5 files found under: {args.input_dir}")
    print(f"Found {len(h5_files)} HDF5 file(s) in: {args.input_dir}")

    records = []
    failed  = []
    for h5_path in h5_files:
        try:
            metrics = compute_metrics(h5_path)
            records.append(metrics)
        except Exception as e:
            print(f"  [WARNING] Skipping {Path(h5_path).name}: {e}")
            failed.append(h5_path)

    if not records:
        raise RuntimeError("No files could be processed successfully.")

    df = pd.DataFrame(records)

    stats = summarize(df)
    print_channel_summary(records, len(h5_files))
    print_summary(df, stats)

    output_csv = args.output_csv or os.path.join(args.input_dir, "sleep_metrics.csv")
    df.to_csv(output_csv, index=False)
    print(f"Per-file metrics saved to: {output_csv}")

    if failed:
        print(f"\n[WARNING] {len(failed)} file(s) failed to process:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
