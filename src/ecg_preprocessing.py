"""
src/ecg_preprocessing.py

ECG preprocessing for the PTB-XL dataset.

Dataset: PTB-XL — 21,799 clinical 12-lead ECG records from 18,869 patients
Sampling rate: 100 Hz (records100/ folder)
Record length: 10 seconds → 1000 samples per lead
Output shape: [N, 12, 1000] float32

Preprocessing steps:
    1. Load WFDB records using filename_lr from ptbxl_database.csv
    2. Transpose from [T, C] to [C, T]
    3. Bandpass filter 0.5-40 Hz — removes baseline wander and noise
    4. Z-score normalise per lead per record
    5. Patient-level train/val/test split using strat_fold column
    6. Save outputs with full metadata

Split strategy (as recommended by dataset authors):
    Folds 1-8 → training
    Fold 9    → validation (human-validated labels)
    Fold 10   → test (human-validated labels)
"""

import os
import ast
import glob
import logging
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Lead names in the order they appear in PTB-XL records
LEAD_NAMES = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# =============================================================================
# Signal preprocessing
# =============================================================================

def bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase bandpass filter using second-order sections.

    Args:
        signal:  [n_leads, n_samples]
        lowcut:  Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        fs:      Sampling frequency in Hz
        order:   Filter order

    Returns:
        Filtered signal [n_leads, n_samples]
    """
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    filtered = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        filtered[i] = sosfiltfilt(sos, signal[i])
    return filtered


def zscore_normalise(signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalise each lead independently.
    Removes amplitude scale differences between patients and leads.

    Args:
        signal: [n_leads, n_samples]

    Returns:
        Normalised signal [n_leads, n_samples]
    """
    normalised = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        lead = signal[i]
        std = lead.std()
        if std > 0:
            normalised[i] = (lead - lead.mean()) / std
        else:
            normalised[i] = lead - lead.mean()
    return normalised


# =============================================================================
# Label parsing
# =============================================================================

def parse_scp_codes(scp_codes_str: str) -> dict:
    """
    Parse the scp_codes string from ptbxl_database.csv.

    The field is stored as a string representation of a dict e.g.:
    "{'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}"

    Returns a dict of {statement: likelihood}
    """
    try:
        return ast.literal_eval(scp_codes_str)
    except (ValueError, SyntaxError):
        return {}


def get_primary_label(scp_codes: dict) -> str:
    """
    Return the SCP statement with the highest likelihood.
    If all likelihoods are 0, return the first key.
    """
    if not scp_codes:
        return "UNKNOWN"
    return max(scp_codes, key=scp_codes.get)


# =============================================================================
# Main ECG pipeline
# =============================================================================

def run_ecg_pipeline(config: dict) -> None:
    """
    Full ECG preprocessing pipeline:
    1. Load ptbxl_database.csv
    2. For each record load WFDB signal
    3. Bandpass filter and z-score normalise
    4. Assign patient-level splits using strat_fold
    5. Save outputs as .npz with metadata CSV

    Processes in batches to keep RAM usage manageable.
    """
    ecg_cfg = config["ecg"]
    paths   = config["paths"]

    # Find PTB-XL root directory
    ptbxl_root = paths["ptbxl_raw"]
    out_dir    = os.path.join(paths["processed"], "ecg")
    os.makedirs(out_dir, exist_ok=True)

    # Load metadata CSV
    csv_path = os.path.join(ptbxl_root, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        # Search recursively in case of nested folder
        csvs = glob.glob(
            os.path.join(paths["raw"], "ptbxl", "**", "ptbxl_database.csv"),
            recursive=True
        )
        if not csvs:
            raise FileNotFoundError("ptbxl_database.csv not found")
        csv_path = csvs[0]
        ptbxl_root = os.path.dirname(csv_path)

    logger.info(f"Loading PTB-XL metadata from {csv_path}")
    meta_df = pd.read_csv(csv_path)
    logger.info(f"Total records: {len(meta_df)}")

    # Assign splits
    val_fold  = ecg_cfg["val_fold"]   # 9
    test_fold = ecg_cfg["test_fold"]  # 10

    def get_split(fold: int) -> str:
        if fold == test_fold:
            return "test"
        elif fold == val_fold:
            return "val"
        else:
            return "train"

    meta_df["split"] = meta_df["strat_fold"].apply(get_split)

    logger.info(
        f"Split distribution:\n"
        f"{meta_df['split'].value_counts().to_string()}"
    )

    # Parse SCP codes
    meta_df["scp_codes_parsed"] = meta_df["scp_codes"].apply(parse_scp_codes)
    meta_df["primary_label"]    = meta_df["scp_codes_parsed"].apply(get_primary_label)

    # Processing parameters
    target_hz = ecg_cfg["target_hz"]       # 100
    bp_low    = ecg_cfg["bandpass_low_hz"] # 0.5
    bp_high   = ecg_cfg["bandpass_high_hz"]# 40.0
    bp_order  = ecg_cfg["bandpass_order"]  # 4
    n_leads   = ecg_cfg["n_leads"]         # 12
    n_samples = ecg_cfg["window_samples"]  # 1000

    all_signals  = []
    all_metadata = []
    skipped      = 0

    logger.info("Processing ECG records...")

    for idx, row in meta_df.iterrows():
        record_path = os.path.join(ptbxl_root, row["filename_lr"])

        try:
            # Load WFDB record
            record = wfdb.rdrecord(record_path)

            # Signal shape from WFDB is [T, C] — transpose to [C, T]
            signal = record.p_signal.T.astype(np.float32)  # [12, 1000]

            # Skip if shape is unexpected
            if signal.shape != (n_leads, n_samples):
                logger.debug(
                    f"Skipping record {row['ecg_id']} — "
                    f"unexpected shape {signal.shape}"
                )
                skipped += 1
                continue

            # Skip if all NaN
            if np.isnan(signal).all():
                logger.debug(f"Skipping record {row['ecg_id']} — all NaN")
                skipped += 1
                continue

            # Replace NaN with 0 before filtering
            signal = np.nan_to_num(signal, nan=0.0)

            # Bandpass filter
            signal = bandpass_filter(
                signal, bp_low, bp_high, target_hz, bp_order
            )

            # Z-score normalise per lead
            signal = zscore_normalise(signal)

            all_signals.append(signal)

            all_metadata.append({
                "sample_id":        f"ptbxl_{row['ecg_id']:06d}",
                "ecg_id":           int(row["ecg_id"]),
                "patient_id":       int(row["patient_id"]) if pd.notna(row["patient_id"]) else -1,
                "dataset_name":     "ptbxl",
                "modality":         "ECG",
                "source_file":      row["filename_lr"],
                "split":            row["split"],
                "strat_fold":       int(row["strat_fold"]),
                "primary_label":    row["primary_label"],
                "scp_codes":        str(row["scp_codes"]),
                "age":              row["age"] if pd.notna(row["age"]) else -1,
                "sex":              int(row["sex"]) if pd.notna(row["sex"]) else -1,
                "sampling_rate_hz": target_hz,
                "n_channels":       n_leads,
                "n_samples":        n_samples,
                "lead_names":       ",".join(LEAD_NAMES),
                "validated_by_human": bool(row["validated_by_human"]),
                "qc_flags":         _get_qc_flags(row),
            })

        except Exception as e:
            logger.debug(f"Error loading record {row['ecg_id']}: {e}")
            skipped += 1
            continue

        # Log progress every 1000 records
        if (idx + 1) % 1000 == 0:
            logger.info(f"  Processed {idx + 1}/{len(meta_df)} records...")

    logger.info(f"Skipped {skipped} records")
    logger.info(f"Successfully processed {len(all_signals)} records")

    if not all_signals:
        logger.error("No ECG records processed — check raw data")
        return

    # Stack all signals
    signals_arr = np.stack(all_signals, axis=0)  # [N, 12, 1000]
    metadata_df = pd.DataFrame(all_metadata)

    # Save outputs
    npz_path = os.path.join(out_dir, "ptbxl.npz")
    csv_path = os.path.join(out_dir, "ptbxl_metadata.csv")

    np.savez_compressed(npz_path, signals=signals_arr)
    metadata_df.to_csv(csv_path, index=False)

    logger.info("=" * 60)
    logger.info("ECG pipeline complete.")
    logger.info(f"Total records:    {len(signals_arr)}")
    logger.info(f"Shape:            {signals_arr.shape}")
    logger.info(f"Dtype:            {signals_arr.dtype}")
    logger.info(f"NaNs:             {np.isnan(signals_arr).sum()}")
    logger.info(
        f"Split distribution:\n"
        f"{metadata_df['split'].value_counts().to_string()}"
    )
    logger.info(
        f"Top 10 primary labels:\n"
        f"{metadata_df['primary_label'].value_counts().head(10).to_string()}"
    )
    logger.info("=" * 60)


def _get_qc_flags(row: pd.Series) -> str:
    """
    Generate a QC flag string from signal quality metadata.
    Flags any known signal issues for downstream filtering.
    """
    flags = []
    if pd.notna(row.get("static_noise")) and row["static_noise"]:
        flags.append("static_noise")
    if pd.notna(row.get("burst_noise")) and row["burst_noise"]:
        flags.append("burst_noise")
    if pd.notna(row.get("baseline_drift")) and row["baseline_drift"]:
        flags.append("baseline_drift")
    if pd.notna(row.get("electrodes_problems")) and row["electrodes_problems"]:
        flags.append("electrodes_problems")
    return ",".join(flags) if flags else "none"