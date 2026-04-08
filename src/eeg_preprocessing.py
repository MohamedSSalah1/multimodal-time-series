"""
src/eeg_preprocessing.py

EEG preprocessing for the EEGMMIDB dataset.
Uses motor imagery runs 4, 8, and 12 only as mandated by the assessment brief.

Annotation scheme for runs 4, 8, 12:
    T0 = rest
    T1 = left fist imagery
    T2 = right fist imagery

Preprocessing steps:
    1. Parse EDF+ files with MNE
    2. Bandpass filter 1-40 Hz (4th order Butterworth, zero-phase)
    3. Notch filter at 50 Hz (UK mains interference)
    4. Average re-reference
    5. Artefact rejection (amplitude threshold 500 uV)
    6. Event-aligned windowing — 4 seconds at T0/T1/T2 onset
    7. Subject-level splits

Output shape: [N, 64, 640] float32
    N   = number of windows
    64  = EEG channels
    640 = 4 seconds x 160 Hz
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import mne
from typing import Tuple, List

logger = logging.getLogger(__name__)

# Suppress MNE verbose output
mne.set_log_level("WARNING")

# Event code mapping
EVENT_MAP = {
    "T0": 0,  # rest
    "T1": 1,  # left fist imagery
    "T2": 2,  # right fist imagery
}

EVENT_LABELS = {
    0: "rest",
    1: "left_fist",
    2: "right_fist",
}


# =============================================================================
# Single file preprocessing
# =============================================================================

def preprocess_raw(raw: mne.io.BaseRaw, config: dict) -> mne.io.BaseRaw:
    """
    Apply light preprocessing to a raw EEG recording.

    Steps:
    1. Bandpass filter 1-40 Hz — removes slow drift and high frequency noise
    2. Notch filter at 50 Hz — removes UK mains electrical interference
    3. Average re-reference — standard for motor imagery EEG

    Args:
        raw:    MNE Raw object
        config: Pipeline config

    Returns:
        Preprocessed Raw object
    """
    eeg_cfg = config["eeg"]

    # Bandpass filter — zero-phase 4th order Butterworth
    raw = raw.filter(
        l_freq=eeg_cfg["bandpass_low_hz"],
        h_freq=eeg_cfg["bandpass_high_hz"],
        method="fir",
        fir_window="hamming",
        verbose=False,
    )

    # Notch filter — remove mains interference
    raw = raw.notch_filter(
        freqs=eeg_cfg["notch_hz"],
        verbose=False,
    )

    # Average re-reference
    raw = raw.set_eeg_reference("average", verbose=False)

    return raw


def is_corrupted_segment(
    segment: np.ndarray,
    threshold_uv: float = 500.0,
) -> bool:
    """
    Check if a segment is corrupted.

    Criteria:
    - Any channel exceeds amplitude threshold (default 500 uV)
    - Any channel is completely flat (std == 0)

    Args:
        segment:      np.ndarray [n_channels, n_samples]
        threshold_uv: Amplitude threshold in microvolts

    Returns:
        True if corrupted, False if clean
    """
    # Convert from volts to microvolts
    segment_uv = segment * 1e6

    # Check amplitude threshold
    if np.abs(segment_uv).max() > threshold_uv:
        return True

    # Check for flat channels
    if np.any(segment_uv.std(axis=1) == 0):
        return True

    return False


# =============================================================================
# Windowing
# =============================================================================

def extract_windows(
    raw: mne.io.BaseRaw,
    subject_id: str,
    run_id: int,
    config: dict,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Extract event-aligned 4-second windows from a preprocessed Raw object.

    Windows are aligned to the onset of T0, T1, and T2 annotations.
    Corrupted segments are excluded based on amplitude threshold.

    Args:
        raw:        Preprocessed MNE Raw object
        subject_id: Subject identifier e.g. 'S001'
        run_id:     Run number e.g. 4
        config:     Pipeline config

    Returns:
        signals:  np.ndarray [N, 64, 640] float32
        metadata: List of dicts
    """
    eeg_cfg    = config["eeg"]
    sfreq      = raw.info["sfreq"]  # 160 Hz
    window_sec = eeg_cfg["window_sec"]  # 4 seconds
    window_samples = int(window_sec * sfreq)  # 640
    threshold_uv   = eeg_cfg["amplitude_threshold_uv"]

    signals  = []
    metadata = []

    for ann in raw.annotations:
        description = ann["description"]
        onset_sec   = ann["onset"]
        duration    = ann["duration"]

        # Only process T0, T1, T2 events
        if description not in EVENT_MAP:
            continue

        event_code  = EVENT_MAP[description]
        event_label = EVENT_LABELS[event_code]

        # Convert onset to sample index
        onset_sample = int(onset_sec * sfreq)
        end_sample   = onset_sample + window_samples

        # Check window fits within recording
        if end_sample > len(raw.times):
            logger.debug(
                f"Skipping {subject_id}/R{run_id:02d}/{description} "
                f"— window extends beyond recording"
            )
            continue

        # Extract window — shape [n_channels, n_samples]
        segment, _ = raw[:, onset_sample:end_sample]

        # Skip if window is shorter than expected
        if segment.shape[1] != window_samples:
            logger.debug(
                f"Skipping {subject_id}/R{run_id:02d}/{description} "
                f"at {onset_sec:.2f}s — wrong shape: {segment.shape[1]} != {window_samples}"
            )
            continue

        # Check for corruption
        if is_corrupted_segment(segment, threshold_uv):
            logger.debug(
                f"Skipping {subject_id}/R{run_id:02d}/{description} "
                f"at {onset_sec:.2f}s — corrupted segment"
            )
            continue

        signals.append(segment.astype(np.float32))  # [64, 640]

        metadata.append({
            "subject_id":      subject_id,
            "run_id":          run_id,
            "event_code":      event_code,
            "event_label":     event_label,
            "onset_sec":       round(onset_sec, 4),
            "onset_sample":    onset_sample,
            "duration_sec":    round(duration, 4),
            "n_channels":      segment.shape[0],
            "n_samples":       window_samples,
            "sampling_rate":   int(sfreq),
            "channel_schema":  ",".join(raw.ch_names),
            "dataset":         "eegmmidb",
            "source_file":     f"{subject_id}R{run_id:02d}.edf",
            "modality":        "EEG",
        })

    if not signals:
        return np.empty((0, 64, window_samples), dtype=np.float32), []

    return np.stack(signals, axis=0), metadata


# =============================================================================
# Subject-level splits
# =============================================================================

def assign_eeg_splits(
    metadata: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Assign train/val/test splits based on subject ID.
    Splits are at the subject level to prevent data leakage.

    Split strategy:
        Subjects 1-87:   train
        Subjects 88-98:  val
        Subjects 99-109: test
    """
    eeg_cfg = config["eeg"]
    metadata = metadata.copy()

    def get_split(subject_id: str) -> str:
        # Extract numeric subject index e.g. S001 → 1
        subj_num = int(subject_id[1:])
        if subj_num >= eeg_cfg["test_subjects_start"]:
            return "test"
        elif subj_num >= eeg_cfg["val_subjects_start"]:
            return "val"
        else:
            return "train"

    metadata["split"] = metadata["subject_id"].apply(get_split)
    return metadata


# =============================================================================
# Save outputs
# =============================================================================

def save_eeg_outputs(
    signals: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: str,
) -> str:
    """
    Save processed EEG windows as .npz with metadata CSV.

    Args:
        signals:    [N, 64, 640] float32
        metadata:   DataFrame with one row per window
        output_dir: Path to processed/eeg/

    Returns:
        Path to saved .npz file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add sample IDs
    metadata = metadata.copy()
    metadata["sample_id"] = [
        f"eegmmidb_{i:06d}" for i in range(len(metadata))
    ]

    # Save signals
    npz_path = os.path.join(output_dir, "eegmmidb.npz")
    np.savez_compressed(npz_path, signals=signals)

    # Save metadata
    csv_path = os.path.join(output_dir, "eegmmidb_metadata.csv")
    metadata.to_csv(csv_path, index=False)

    logger.info(f"Saved EEG: {signals.shape} → {npz_path}")
    return npz_path


# =============================================================================
# Main EEG pipeline
# =============================================================================

def run_eeg_pipeline(config: dict) -> None:
    """
    Full EEG preprocessing pipeline:
    1. Load EDF+ files for all 109 subjects, runs 4, 8, 12
    2. Apply bandpass, notch, re-reference
    3. Extract event-aligned windows
    4. Reject corrupted segments
    5. Assign subject-level splits
    6. Save outputs
    """
    eeg_cfg   = config["eeg"]
    paths     = config["paths"]
    raw_dir   = paths["eegmmidb_raw"]
    out_dir   = os.path.join(paths["processed"], "eeg")
    runs      = eeg_cfg["runs"]  # [4, 8, 12]

    all_signals  = []
    all_metadata = []

    # Find all subject folders
    subject_dirs = sorted([
        d for d in glob.glob(os.path.join(raw_dir, "S*"))
        if os.path.isdir(d) and os.path.basename(d).startswith("S")
        and os.path.basename(d)[1:].isdigit()
    ])

    logger.info(f"Found {len(subject_dirs)} subject folders")

    for subj_dir in subject_dirs:
        subject_id = os.path.basename(subj_dir)  # e.g. S001

        for run_id in runs:
            edf_path = os.path.join(
                subj_dir, f"{subject_id}R{run_id:02d}.edf"
            )

            if not os.path.exists(edf_path):
                logger.warning(f"Missing: {edf_path}")
                continue

            try:
                # Load EDF+ file
                raw = mne.io.read_raw_edf(
                    edf_path,
                    preload=True,
                    verbose=False,
                )

                # Resample to target Hz if needed
                actual_sfreq = raw.info["sfreq"]
                target_sfreq = config["eeg"]["target_hz"]
                if actual_sfreq != target_sfreq:
                    logger.info(
                        f"{subject_id}/R{run_id:02d}: resampling "
                        f"{actual_sfreq}Hz → {target_sfreq}Hz"
                    )
                    raw = raw.resample(target_sfreq, verbose=False)

                # Preprocess
                raw = preprocess_raw(raw, config)

                # Extract windows
                signals, metadata = extract_windows(
                    raw, subject_id, run_id, config
                )

                if len(signals) > 0:
                    all_signals.append(signals)
                    all_metadata.extend(metadata)
                    logger.info(
                        f"{subject_id}/R{run_id:02d}: "
                        f"{len(signals)} windows extracted"
                    )

            except Exception as e:
                logger.error(f"Error processing {edf_path}: {e}")
                continue

    if not all_signals:
        logger.error("No EEG windows generated — check raw data")
        return

    # Stack all windows
    signals_arr = np.concatenate(all_signals, axis=0)
    metadata_df = pd.DataFrame(all_metadata)

    # Assign splits
    metadata_df = assign_eeg_splits(metadata_df, config)

    # Add sample IDs
    metadata_df["sample_id"] = [
        f"eegmmidb_{i:06d}" for i in range(len(metadata_df))
    ]

    # Save
    save_eeg_outputs(signals_arr, metadata_df, out_dir)

    # Summary
    logger.info("=" * 60)
    logger.info(f"EEG pipeline complete.")
    logger.info(f"Total windows:    {len(signals_arr)}")
    logger.info(f"Shape:            {signals_arr.shape}")
    logger.info(f"Dtype:            {signals_arr.dtype}")
    logger.info(f"NaNs:             {np.isnan(signals_arr).sum()}")
    logger.info(
        f"Label distribution:\n"
        f"{metadata_df['event_label'].value_counts().to_string()}"
    )
    logger.info(
        f"Split distribution:\n"
        f"{metadata_df['split'].value_counts().to_string()}"
    )
    logger.info("=" * 60)