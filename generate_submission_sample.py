"""
generate_submission_sample.py

Generates a representative processed sample pack for submission.
100 windows per dataset, stratified where possible.

Outputs to submission_sample/
"""

import os
import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

N_SAMPLES = 100


def stratified_sample(
    signals: np.ndarray,
    metadata: pd.DataFrame,
    n: int,
    stratify_col: str,
) -> tuple:
    """
    Sample n windows stratified by a column.
    Falls back to random sampling if stratification not possible.
    """
    groups = metadata[stratify_col].unique()
    per_group = max(1, n // len(groups))

    indices = []
    for group in groups:
        group_idx = metadata[metadata[stratify_col] == group].index.tolist()
        sampled = group_idx[:per_group]
        indices.extend(sampled)

    # Top up to exactly n if needed
    if len(indices) < n:
        remaining = list(set(metadata.index.tolist()) - set(indices))
        np.random.shuffle(remaining)
        indices.extend(remaining[:n - len(indices)])

    indices = sorted(indices[:n])
    return signals[indices], metadata.loc[indices].reset_index(drop=True)


def save_sample(
    signals: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: str,
    name: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, f"{name}.npz"), signals=signals)
    metadata.to_csv(os.path.join(output_dir, f"{name}_metadata.csv"), index=False)
    logger.info(f"Saved {name}: {signals.shape} → {output_dir}")


def main():
    np.random.seed(42)

    # ── HAR ──────────────────────────────────────────────────────────────────
    logger.info("Generating HAR samples...")

    for dataset in ["pamap2", "wisdm", "combined"]:
        npz_path = f"data/processed/har/{dataset}_supervised.npz"
        csv_path = f"data/processed/har/{dataset}_supervised_metadata.csv"

        if not os.path.exists(npz_path):
            logger.warning(f"Missing: {npz_path}")
            continue

        signals  = np.load(npz_path)["signals"]
        metadata = pd.read_csv(csv_path)

        sampled_signals, sampled_meta = stratified_sample(
            signals, metadata, N_SAMPLES, "label"
        )

        save_sample(
            sampled_signals, sampled_meta,
            "submission_sample/har", f"{dataset}_supervised_sample"
        )

    # ── EEG ──────────────────────────────────────────────────────────────────
    logger.info("Generating EEG samples...")

    npz_path = "data/processed/eeg/eegmmidb.npz"
    csv_path = "data/processed/eeg/eegmmidb_metadata.csv"

    if os.path.exists(npz_path):
        signals  = np.load(npz_path)["signals"]
        metadata = pd.read_csv(csv_path)

        sampled_signals, sampled_meta = stratified_sample(
            signals, metadata, N_SAMPLES, "event_label"
        )

        save_sample(
            sampled_signals, sampled_meta,
            "submission_sample/eeg", "eegmmidb_sample"
        )

    # ── ECG ──────────────────────────────────────────────────────────────────
    logger.info("Generating ECG samples...")

    npz_path = "data/processed/ecg/ptbxl.npz"
    csv_path = "data/processed/ecg/ptbxl_metadata.csv"

    if os.path.exists(npz_path):
        signals  = np.load(npz_path)["signals"]
        metadata = pd.read_csv(csv_path)

        sampled_signals, sampled_meta = stratified_sample(
            signals, metadata, N_SAMPLES, "split"
        )

        save_sample(
            sampled_signals, sampled_meta,
            "submission_sample/ecg", "ptbxl_sample"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Submission sample pack complete.")
    logger.info("Contents:")
    for root, dirs, files in os.walk("submission_sample"):
        for f in sorted(files):
            path = os.path.join(root, f)
            size = round(os.path.getsize(path) / (1024 ** 2), 2)
            logger.info(f"  {path} ({size} MB)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()