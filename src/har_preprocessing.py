"""
src/har_preprocessing.py

HAR preprocessing for PAMAP2 and WISDM datasets.
Produces harmonised windowed arrays for:
  - Pretraining: 10s windows, no overlap, no labels
  - Supervised:  5s windows, 50% overlap, majority-vote labels

Output shape: [N, 6, T] float32
  N = number of windows
  6 = channels [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
  T = samples per window (200 for pretraining, 100 for supervised)
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
from scipy.signal import decimate
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)


# =============================================================================
# PAMAP2 Parsing
# =============================================================================

# Column indices (0-indexed) for wrist IMU within the 54-column file
PAMAP2_COLS = {
    "timestamp": 0,
    "activity_id": 1,
    "heart_rate": 2,
    "acc_x": 3,   # ±16g accelerometer — recommended by dataset authors
    "acc_y": 4,
    "acc_z": 5,
    "gyr_x": 9,   # gyroscope
    "gyr_y": 10,
    "gyr_z": 11,
}

PAMAP2_SIGNAL_COLS = [
    PAMAP2_COLS["acc_x"],
    PAMAP2_COLS["acc_y"],
    PAMAP2_COLS["acc_z"],
    PAMAP2_COLS["gyr_x"],
    PAMAP2_COLS["gyr_y"],
    PAMAP2_COLS["gyr_z"],
]

# Activity label mapping — PAMAP2 integer ID → unified string label
PAMAP2_LABEL_MAP = {
    1:  "lying",
    2:  "sitting",
    3:  "standing",
    4:  "walking",
    5:  "running",
    6:  "cycling",
    7:  "nordic_walking",
    9:  "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
    # 0 = transient — explicitly excluded
}


def parse_pamap2_file(filepath: str, subject_id: str) -> pd.DataFrame:
    """
    Parse a single PAMAP2 .dat file.

    Returns a DataFrame with columns:
        timestamp, activity_id, label, subject_id,
        acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
    """
    logger.info(f"Parsing PAMAP2 file: {filepath}")

    # Load space-separated file — no header
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        na_values=["NaN"]
    )

    # Extract relevant columns
    out = pd.DataFrame()
    out["timestamp"]   = df.iloc[:, PAMAP2_COLS["timestamp"]]
    out["activity_id"] = df.iloc[:, PAMAP2_COLS["activity_id"]]
    out["acc_x"]       = df.iloc[:, PAMAP2_COLS["acc_x"]]
    out["acc_y"]       = df.iloc[:, PAMAP2_COLS["acc_y"]]
    out["acc_z"]       = df.iloc[:, PAMAP2_COLS["acc_z"]]
    out["gyr_x"]       = df.iloc[:, PAMAP2_COLS["gyr_x"]]
    out["gyr_y"]       = df.iloc[:, PAMAP2_COLS["gyr_y"]]
    out["gyr_z"]       = df.iloc[:, PAMAP2_COLS["gyr_z"]]
    out["subject_id"]  = subject_id
    out["source_file"] = os.path.basename(filepath)
    out["dataset"]     = "pamap2"

    # Map activity IDs to unified labels
    out["label"] = out["activity_id"].map(PAMAP2_LABEL_MAP)

    # Discard transient label (activity_id == 0) and unmapped rows
    before = len(out)
    out = out[out["activity_id"] != 0].copy()
    out = out[out["label"].notna()].copy()
    after = len(out)
    logger.info(f"  Discarded {before - after} transient/unmapped rows")

    return out


def load_pamap2(raw_dir: str) -> pd.DataFrame:
    """
    Load all PAMAP2 Protocol and Optional .dat files.

    Args:
        raw_dir: Path to PAMAP2_Dataset folder

    Returns:
        Combined DataFrame for all subjects
    """
    all_dfs = []

    for folder in ["Protocol", "Optional"]:
        folder_path = os.path.join(raw_dir, folder)
        if not os.path.exists(folder_path):
            logger.warning(f"Folder not found: {folder_path}")
            continue

        dat_files = sorted(glob.glob(os.path.join(folder_path, "subject*.dat")))
        logger.info(f"Found {len(dat_files)} files in {folder}")

        for filepath in dat_files:
            # Extract subject ID from filename e.g. subject101 → S101
            basename = os.path.splitext(os.path.basename(filepath))[0]
            subject_id = "S" + basename.replace("subject", "")
            df = parse_pamap2_file(filepath, subject_id)
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"PAMAP2 total rows after loading: {len(combined)}")
    return combined


# =============================================================================
# PAMAP2 Resampling
# =============================================================================

def resample_pamap2(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Resample PAMAP2 from 100Hz to 20Hz using scipy.signal.decimate.
    Decimation is preferred over naive downsampling to avoid aliasing.

    Resampling is applied per subject per activity segment to avoid
    boundary artefacts between different activities.
    """
    factor = config["har"]["pamap2_decimate_factor"]  # 5
    signal_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    meta_cols = ["subject_id", "source_file", "dataset", "label", "activity_id"]

    resampled_dfs = []

    for (subject_id, label), group in df.groupby(["subject_id", "label"]):
        group = group.copy().reset_index(drop=True)

        # Interpolate NaNs before resampling
        group[signal_cols] = group[signal_cols].interpolate(
            method="linear",
            limit=int(config["har"]["max_interp_gap_sec"] * 100),
            limit_direction="both"
        )

        # Drop rows still containing NaN after interpolation
        group = group.dropna(subset=signal_cols)

        if len(group) < factor:
            logger.warning(
                f"Skipping {subject_id}/{label} — too few samples ({len(group)})"
            )
            continue

        # Decimate each signal channel — output is shorter than input
        decimated_signals = {}
        for col in signal_cols:
            decimated_signals[col] = decimate(
                group[col].values, factor, zero_phase=True
            )

        # Downsample metadata at same rate by taking every nth row
        meta = group[meta_cols].iloc[::factor].reset_index(drop=True)

        # Build new DataFrame with decimated signals + downsampled metadata
        n_out = len(meta)
        new_df = meta.copy()
        for col in signal_cols:
            # Trim to match metadata length in case of rounding differences
            new_df[col] = decimated_signals[col][:n_out]

        resampled_dfs.append(new_df)

    result = pd.concat(resampled_dfs, ignore_index=True)
    logger.info(f"PAMAP2 rows after resampling to 20Hz: {len(result)}")
    return result


# =============================================================================
# WISDM Parsing
# =============================================================================

# Activity label mapping — WISDM letter code → unified string label
WISDM_LABEL_MAP = {
    "A": "walking",
    "B": "running",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "computer_work",
    "G": "brushing_teeth",
    "H": "eating_soup",
    "I": "eating_chips",
    "J": "eating_pasta",
    "K": "drinking",
    "L": "eating_sandwich",
    "M": "kicking",
    "O": "playing_catch",
    "P": "dribbling",
    "Q": "writing",
    "R": "clapping",
    "S": "folding_laundry",
}


def parse_wisdm_file(filepath: str) -> pd.DataFrame:
    """
    Parse a single WISDM watch sensor .txt file.

    Format per line: subject_id,activity_code,timestamp,x,y,z;
    Note: each line ends with a semicolon which must be stripped.
    """
    rows = []

    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            # Strip whitespace and trailing semicolon
            line = line.strip().rstrip(";")
            if not line:
                continue

            parts = line.split(",")
            if len(parts) != 6:
                logger.debug(
                    f"Skipping malformed line {line_num} in {filepath}: "
                    f"expected 6 fields, got {len(parts)}"
                )
                continue

            try:
                rows.append({
                    "subject_id": f"W{parts[0].strip()}",  # prefix W to distinguish from PAMAP2
                    "activity_code": parts[1].strip(),
                    "timestamp": int(parts[2].strip()),
                    "acc_x": float(parts[3].strip()),
                    "acc_y": float(parts[4].strip()),
                    "acc_z": float(parts[5].strip()),
                })
            except (ValueError, IndexError):
                logger.debug(f"Skipping unparseable line {line_num} in {filepath}")
                continue

    return pd.DataFrame(rows)


def load_wisdm(raw_dir: str, sensor: str = "watch") -> pd.DataFrame:
    """
    Load WISDM watch accelerometer and gyroscope data.
    Merges accel and gyro by subject_id, activity_code, and timestamp.

    Args:
        raw_dir: Path to wisdm-dataset/raw folder
        sensor:  'watch' (default) or 'phone'

    Returns:
        Combined DataFrame with 6 signal channels
    """
    accel_dir = os.path.join(raw_dir, sensor, "accel")
    gyro_dir  = os.path.join(raw_dir, sensor, "gyro")

    accel_files = sorted(glob.glob(os.path.join(accel_dir, "*.txt")))
    gyro_files  = sorted(glob.glob(os.path.join(gyro_dir,  "*.txt")))

    logger.info(f"Found {len(accel_files)} accel + {len(gyro_files)} gyro files")

    accel_dfs = []
    gyro_dfs  = []

    for filepath in accel_files:
        df = parse_wisdm_file(filepath)
        df = df.rename(columns={"acc_x": "acc_x", "acc_y": "acc_y", "acc_z": "acc_z"})
        df["source_file"] = os.path.basename(filepath)
        accel_dfs.append(df)

    for filepath in gyro_files:
        df = parse_wisdm_file(filepath)
        df = df.rename(columns={"acc_x": "gyr_x", "acc_y": "gyr_y", "acc_z": "gyr_z"})
        gyro_dfs.append(df)

    accel_all = pd.concat(accel_dfs, ignore_index=True)
    gyro_all  = pd.concat(gyro_dfs,  ignore_index=True)

    # Merge accel and gyro on subject_id + activity_code + timestamp
    merged = pd.merge(
        accel_all,
        gyro_all[["subject_id", "activity_code", "timestamp", "gyr_x", "gyr_y", "gyr_z"]],
        on=["subject_id", "activity_code", "timestamp"],
        how="inner"
    )

    # Map activity codes to unified labels
    merged["label"] = merged["activity_code"].map(WISDM_LABEL_MAP)
    merged["dataset"] = "wisdm"

    # Drop rows with unmapped labels
    before = len(merged)
    merged = merged[merged["label"].notna()].copy()
    after  = len(merged)
    logger.info(f"WISDM: dropped {before - after} rows with unmapped labels")
    logger.info(f"WISDM total rows after loading: {len(merged)}")

    return merged


# =============================================================================
# Signal Cleaning
# =============================================================================

def clip_outliers(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clip accelerometer and gyroscope values to physically plausible ranges.
    PAMAP2: acc ±16g (ms-2 scale), gyr ±2000 deg/s
    WISDM:  acc in m/s², gyr in rad/s
    """
    acc_limit = config["har"]["outlier_acc_g"]
    gyr_limit = config["har"]["outlier_gyr_dps"]

    df = df.copy()
    for col in ["acc_x", "acc_y", "acc_z"]:
        df[col] = df[col].clip(-acc_limit, acc_limit)
    for col in ["gyr_x", "gyr_y", "gyr_z"]:
        df[col] = df[col].clip(-gyr_limit, gyr_limit)

    return df


# =============================================================================
# Windowing
# =============================================================================

SHARED_LABELS = [
    "walking", "running", "stairs", "sitting", "standing",
    "computer_work", "folding_laundry"
]

SIGNAL_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]


def majority_vote_label(labels: np.ndarray) -> str:
    """Return the most frequent label in a window."""
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


def create_windows(
    df: pd.DataFrame,
    window_samples: int,
    step_samples: int,
    include_labels: bool,
    shared_only: bool,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Slide a window over a continuous signal segment.

    Args:
        df:              DataFrame with signal columns and metadata
        window_samples:  Number of samples per window
        step_samples:    Step size between windows
        include_labels:  Whether to assign labels to windows
        shared_only:     If True, only keep windows with shared labels

    Returns:
        signals:  np.ndarray of shape [N, 6, window_samples] float32
        metadata: List of dicts with one entry per window
    """
    signals  = []
    metadata = []

    signal_data = df[SIGNAL_COLS].values
    label_data  = df["label"].values
    subject_ids = df["subject_id"].values
    source_files = df["source_file"].values if "source_file" in df.columns else ["unknown"] * len(df)
    datasets    = df["dataset"].values

    n_samples = len(signal_data)

    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples

        window_signal = signal_data[start:end]
        window_labels = label_data[start:end]

        # Skip windows with NaN values
        if np.isnan(window_signal).any():
            continue

        # Assign label
        if include_labels:
            window_label = majority_vote_label(window_labels)
            if shared_only and window_label not in SHARED_LABELS:
                continue
        else:
            window_label = "unlabelled"

        signals.append(window_signal.T.astype(np.float32))  # [6, T]

        metadata.append({
            "subject_id":    subject_ids[start],
            "dataset":       datasets[start],
            "source_file":   source_files[start],
            "label":         window_label,
            "window_start":  start,
            "window_end":    end,
            "n_samples":     window_samples,
            "n_channels":    6,
            "sampling_rate": 20,
            "channel_schema": "acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z",
        })

    if not signals:
        return np.empty((0, 6, window_samples), dtype=np.float32), []

    return np.stack(signals, axis=0), metadata


def window_dataset(
    df: pd.DataFrame,
    config: dict,
    mode: str,  # 'pretrain' or 'supervised'
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Apply windowing to an entire dataset grouped by subject and label segment.

    Args:
        df:     Preprocessed DataFrame
        config: Pipeline config
        mode:   'pretrain' or 'supervised'

    Returns:
        all_signals:  np.ndarray [N, 6, T]
        all_metadata: pd.DataFrame with one row per window
    """
    har_cfg = config["har"]

    if mode == "pretrain":
        window_samples = har_cfg["pretrain_window_sec"] * har_cfg["target_hz"]  # 200
        step_samples   = window_samples  # 0% overlap
        include_labels = False
        shared_only    = False
    else:
        window_samples = har_cfg["supervised_window_sec"] * har_cfg["target_hz"]  # 100
        step_samples   = int(window_samples * (1 - har_cfg["supervised_overlap"]))  # 50
        include_labels = True
        shared_only    = True

    all_signals  = []
    all_metadata = []

    # Window per subject per continuous label segment
    for subject_id, subj_df in df.groupby("subject_id"):
        subj_df = subj_df.reset_index(drop=True)

        signals, metadata = create_windows(
            subj_df,
            window_samples=window_samples,
            step_samples=step_samples,
            include_labels=include_labels,
            shared_only=shared_only,
        )

        if len(signals) > 0:
            all_signals.append(signals)
            all_metadata.extend(metadata)

    if not all_signals:
        return np.empty((0, 6, window_samples), dtype=np.float32), pd.DataFrame()

    return np.concatenate(all_signals, axis=0), pd.DataFrame(all_metadata)


# =============================================================================
# Subject-level splits
# =============================================================================

def assign_splits(
    metadata: pd.DataFrame,
    config: dict,
    dataset: str,
) -> pd.DataFrame:
    """
    Assign train/val/test splits based on subject ID.
    Splits are at the subject level to prevent data leakage.
    """
    har_cfg = config["har"]
    metadata = metadata.copy()

    if dataset == "pamap2":
        val_subjects  = [f"S{s}" for s in har_cfg["pamap2_val_subjects"]]
        test_subjects = [f"S{s}" for s in har_cfg["pamap2_test_subjects"]]
    else:
        val_subjects  = [f"W{s}" for s in har_cfg["wisdm_val_subjects"]]
        test_subjects = [f"W{s}" for s in har_cfg["wisdm_test_subjects"]]

    def get_split(subject_id):
        if subject_id in test_subjects:
            return "test"
        elif subject_id in val_subjects:
            return "val"
        else:
            return "train"

    metadata["split"] = metadata["subject_id"].apply(get_split)
    return metadata


# =============================================================================
# Save outputs
# =============================================================================

def save_har_outputs(
    signals: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: str,
    dataset: str,
    mode: str,
) -> str:
    """
    Save processed HAR windows as .npz with metadata CSV.

    Args:
        signals:    [N, 6, T] float32 array
        metadata:   DataFrame with one row per window
        output_dir: Path to processed/har/
        dataset:    'pamap2', 'wisdm', or 'combined'
        mode:       'pretrain' or 'supervised'

    Returns:
        Path to saved .npz file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add sample IDs
    metadata = metadata.copy()
    metadata["sample_id"] = [
        f"{dataset}_{mode}_{i:06d}" for i in range(len(metadata))
    ]
    metadata["modality"]     = "HAR"
    metadata["dataset_name"] = dataset

    # Save signals
    npz_path = os.path.join(output_dir, f"{dataset}_{mode}.npz")
    np.savez_compressed(npz_path, signals=signals)

    # Save metadata
    csv_path = os.path.join(output_dir, f"{dataset}_{mode}_metadata.csv")
    metadata.to_csv(csv_path, index=False)

    logger.info(
        f"Saved {dataset}/{mode}: {signals.shape} → {npz_path}"
    )
    return npz_path


# =============================================================================
# Main HAR pipeline
# =============================================================================

def run_har_pipeline(config: dict) -> None:
    """
    Full HAR preprocessing pipeline:
    1. Load PAMAP2 and WISDM
    2. Resample PAMAP2 to 20Hz
    3. Clean signals
    4. Window for pretraining and supervised
    5. Assign splits
    6. Save outputs
    """
    paths   = config["paths"]
    har_cfg = config["har"]

    processed_dir = os.path.join(paths["processed"], "har")
    os.makedirs(processed_dir, exist_ok=True)

    # ── PAMAP2 ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Processing PAMAP2")
    logger.info("=" * 60)

    pamap2_raw = paths["pamap2_raw"]
    pamap2_df  = load_pamap2(pamap2_raw)
    pamap2_df  = resample_pamap2(pamap2_df, config)
    pamap2_df  = clip_outliers(pamap2_df, config)

    for mode in ["pretrain", "supervised"]:
        signals, metadata = window_dataset(pamap2_df, config, mode)
        if len(signals) == 0:
            logger.warning(f"PAMAP2 {mode}: no windows generated")
            continue
        metadata = assign_splits(metadata, config, "pamap2")
        save_har_outputs(signals, metadata, processed_dir, "pamap2", mode)

    # ── WISDM ─────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Processing WISDM")
    logger.info("=" * 60)

    wisdm_raw = paths["wisdm_raw"]
    wisdm_df  = load_wisdm(wisdm_raw)
    wisdm_df  = clip_outliers(wisdm_df, config)

    for mode in ["pretrain", "supervised"]:
        signals, metadata = window_dataset(wisdm_df, config, mode)
        if len(signals) == 0:
            logger.warning(f"WISDM {mode}: no windows generated")
            continue
        metadata = assign_splits(metadata, config, "wisdm")
        save_har_outputs(signals, metadata, processed_dir, "wisdm", mode)

    # ── Combined ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Creating combined HAR dataset")
    logger.info("=" * 60)

    combined_df = pd.concat([pamap2_df, wisdm_df], ignore_index=True)

    for mode in ["pretrain", "supervised"]:
        signals, metadata = window_dataset(combined_df, config, mode)
        if len(signals) == 0:
            logger.warning(f"Combined {mode}: no windows generated")
            continue

        # Assign splits per dataset
        pamap2_mask = metadata["dataset"] == "pamap2"
        wisdm_mask  = metadata["dataset"] == "wisdm"
        metadata.loc[pamap2_mask] = assign_splits(
            metadata[pamap2_mask], config, "pamap2"
        )
        metadata.loc[wisdm_mask] = assign_splits(
            metadata[wisdm_mask], config, "wisdm"
        )

        save_har_outputs(signals, metadata, processed_dir, "combined", mode)

    logger.info("HAR pipeline complete.")