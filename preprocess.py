"""
preprocess.py

Main entrypoint for the multimodal biosignal preprocessing pipeline.

Usage:
    python preprocess.py           # Run all modalities
    python preprocess.py --har     # Run HAR only
    python preprocess.py --eeg     # Run EEG only
    python preprocess.py --ecg     # Run ECG only

Outputs:
    data/processed/har/   — HAR .npz arrays and metadata CSVs
    data/processed/eeg/   — EEG .npz arrays and metadata CSVs
    data/processed/ecg/   — ECG .npz arrays and metadata CSVs
"""

import argparse
import logging
import yaml

from src.har_preprocessing import run_har_pipeline
from src.eeg_preprocessing import run_eeg_pipeline
from src.ecg_preprocessing import run_ecg_pipeline


# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/pipeline.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal biosignal preprocessing pipeline"
    )
    parser.add_argument("--har", action="store_true", help="Run HAR preprocessing only")
    parser.add_argument("--eeg", action="store_true", help="Run EEG preprocessing only")
    parser.add_argument("--ecg", action="store_true", help="Run ECG preprocessing only")
    args = parser.parse_args()

    # If no flags given run everything
    run_all = not (args.har or args.eeg or args.ecg)

    config = load_config()

    if args.har or run_all:
        logger.info("Starting HAR preprocessing...")
        run_har_pipeline(config)
        logger.info("HAR preprocessing complete.")

    if args.eeg or run_all:
        logger.info("Starting EEG preprocessing...")
        run_eeg_pipeline(config)
        logger.info("EEG preprocessing complete.")

    if args.ecg or run_all:
        logger.info("Starting ECG preprocessing...")
        run_ecg_pipeline(config)
        logger.info("ECG preprocessing complete.")


if __name__ == "__main__":
    main()