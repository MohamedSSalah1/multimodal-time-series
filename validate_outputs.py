"""
validate_outputs.py

Validation script for the multimodal biosignal preprocessing pipeline.
Runs integrity checks on all processed outputs and generates:
    - reports/validation_report.md
    - reports/resource_estimate.md
    - data/manifest_outputs.json

Usage:
    python validate_outputs.py

Based on section 7 of the assessment brief.
"""

import os
import json
import glob
import time
import logging
import numpy as np
import pandas as pd
import psutil
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Helpers
# =============================================================================

def get_file_size_mb(path: str) -> float:
    return round(os.path.getsize(path) / (1024 ** 2), 2)

def get_dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return round(total / (1024 ** 2), 2)

def check_array(path: str) -> dict:
    """Load a .npz file and return integrity stats."""
    d = np.load(path)
    arr = d["signals"]
    return {
        "shape":    list(arr.shape),
        "dtype":    str(arr.dtype),
        "n_nans":   int(np.isnan(arr).sum()),
        "n_infs":   int(np.isinf(arr).sum()),
        "min":      float(arr.min()),
        "max":      float(arr.max()),
        "size_mb":  get_file_size_mb(path),
    }

def check_leakage(metadata: pd.DataFrame, id_col: str) -> dict:
    """Check no subject/patient appears in both train and test."""
    train_ids = set(metadata[metadata["split"] == "train"][id_col])
    test_ids  = set(metadata[metadata["split"] == "test"][id_col])
    val_ids   = set(metadata[metadata["split"] == "val"][id_col])
    leakage_train_test = train_ids & test_ids
    leakage_train_val  = train_ids & val_ids
    return {
        "train_subjects":         len(train_ids),
        "val_subjects":           len(val_ids),
        "test_subjects":          len(test_ids),
        "leakage_train_test":     len(leakage_train_test),
        "leakage_train_val":      len(leakage_train_val),
        "leakage_examples":       list(leakage_train_test)[:5],
    }

# =============================================================================
# HAR validation
# =============================================================================

def validate_har() -> dict:
    logger.info("Validating HAR outputs...")
    results = {}
    har_dir = "data/processed/har"

    expected_files = [
        "pamap2_pretrain.npz",    "pamap2_pretrain_metadata.csv",
        "pamap2_supervised.npz",  "pamap2_supervised_metadata.csv",
        "wisdm_pretrain.npz",     "wisdm_pretrain_metadata.csv",
        "wisdm_supervised.npz",   "wisdm_supervised_metadata.csv",
        "combined_pretrain.npz",  "combined_pretrain_metadata.csv",
        "combined_supervised.npz","combined_supervised_metadata.csv",
    ]

    # File existence
    missing = [f for f in expected_files
               if not os.path.exists(os.path.join(har_dir, f))]
    results["missing_files"] = missing

    # Array checks
    results["arrays"] = {}
    for npz in ["pamap2_pretrain", "pamap2_supervised",
                "wisdm_pretrain", "wisdm_supervised",
                "combined_pretrain", "combined_supervised"]:
        path = os.path.join(har_dir, f"{npz}.npz")
        if os.path.exists(path):
            results["arrays"][npz] = check_array(path)

    # Shape checks
    shape_checks = {}
    if "pamap2_pretrain" in results["arrays"]:
        shape = results["arrays"]["pamap2_pretrain"]["shape"]
        shape_checks["pamap2_pretrain_channels"] = shape[1] == 6
        shape_checks["pamap2_pretrain_samples"]  = shape[2] == 200

    if "pamap2_supervised" in results["arrays"]:
        shape = results["arrays"]["pamap2_supervised"]["shape"]
        shape_checks["pamap2_supervised_channels"] = shape[1] == 6
        shape_checks["pamap2_supervised_samples"]  = shape[2] == 100

    if "wisdm_pretrain" in results["arrays"]:
        shape = results["arrays"]["wisdm_pretrain"]["shape"]
        shape_checks["wisdm_pretrain_channels"] = shape[1] == 6
        shape_checks["wisdm_pretrain_samples"]  = shape[2] == 200

    if "wisdm_supervised" in results["arrays"]:
        shape = results["arrays"]["wisdm_supervised"]["shape"]
        shape_checks["wisdm_supervised_channels"] = shape[1] == 6
        shape_checks["wisdm_supervised_samples"]  = shape[2] == 100

    results["shape_checks"] = shape_checks

    # Harmonisation check — same channel schema
    results["harmonisation"] = {}
    for dataset in ["pamap2", "wisdm"]:
        csv_path = os.path.join(har_dir, f"{dataset}_supervised_metadata.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            results["harmonisation"][dataset] = {
                "sampling_rate":   int(df["sampling_rate"].iloc[0]),
                "n_channels":      int(df["n_channels"].iloc[0]),
                "channel_schema":  df["channel_schema"].iloc[0],
            }

    # Null label check — PAMAP2 class 0 must be absent
    pamap2_csv = os.path.join(har_dir, "pamap2_supervised_metadata.csv")
    if os.path.exists(pamap2_csv):
        df = pd.read_csv(pamap2_csv)
        results["pamap2_null_label_check"] = {
            "class_0_present": "transient" in df["label"].values or
                                "0" in df["label"].astype(str).values,
            "unique_labels":   sorted(df["label"].unique().tolist()),
        }

    # Leakage check
    combined_csv = os.path.join(har_dir, "combined_supervised_metadata.csv")
    if os.path.exists(combined_csv):
        df = pd.read_csv(combined_csv)
        results["leakage"] = check_leakage(df, "subject_id")

    # Split distribution
    if os.path.exists(combined_csv):
        df = pd.read_csv(combined_csv)
        results["split_distribution"] = df["split"].value_counts().to_dict()
        results["label_distribution"] = df["label"].value_counts().to_dict()

    return results

# =============================================================================
# EEG validation
# =============================================================================

def validate_eeg() -> dict:
    logger.info("Validating EEG outputs...")
    results = {}
    eeg_dir = "data/processed/eeg"

    # File existence
    for f in ["eegmmidb.npz", "eegmmidb_metadata.csv"]:
        path = os.path.join(eeg_dir, f)
        results[f"file_{f}_exists"] = os.path.exists(path)

    # Array check
    npz_path = os.path.join(eeg_dir, "eegmmidb.npz")
    if os.path.exists(npz_path):
        results["array"] = check_array(npz_path)

        # Shape check — [N, 64, 640]
        shape = results["array"]["shape"]
        results["shape_checks"] = {
            "channels_64":  shape[1] == 64,
            "samples_640":  shape[2] == 640,
            "dtype_float32": results["array"]["dtype"] == "float32",
        }

    # Metadata checks
    csv_path = os.path.join(eeg_dir, "eegmmidb_metadata.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Annotation check — T0/T1/T2 all present
        event_codes = set(df["event_code"].unique())
        results["annotation_checks"] = {
            "T0_present": 0 in event_codes,
            "T1_present": 1 in event_codes,
            "T2_present": 2 in event_codes,
            "event_distribution": df["event_label"].value_counts().to_dict(),
        }

        # Timing metadata check
        required_cols = [
            "subject_id", "run_id", "event_code", "event_label",
            "onset_sec", "onset_sample", "sampling_rate", "split"
        ]
        results["metadata_cols_present"] = {
            col: col in df.columns for col in required_cols
        }

        # Leakage check
        results["leakage"] = check_leakage(df, "subject_id")

        # Split distribution
        results["split_distribution"] = df["split"].value_counts().to_dict()

        # Runs check — only 4, 8, 12
        results["runs_check"] = {
            "runs_present": sorted(df["run_id"].unique().tolist()),
            "only_required_runs": set(df["run_id"].unique()) == {4, 8, 12},
        }

    return results

# =============================================================================
# ECG validation
# =============================================================================

def validate_ecg() -> dict:
    logger.info("Validating ECG outputs...")
    results = {}
    ecg_dir = "data/processed/ecg"

    # File existence
    for f in ["ptbxl.npz", "ptbxl_metadata.csv"]:
        path = os.path.join(ecg_dir, f)
        results[f"file_{f}_exists"] = os.path.exists(path)

    # Array check
    npz_path = os.path.join(ecg_dir, "ptbxl.npz")
    if os.path.exists(npz_path):
        results["array"] = check_array(npz_path)

        # Shape check — [N, 12, 1000]
        shape = results["array"]["shape"]
        results["shape_checks"] = {
            "leads_12":     shape[1] == 12,
            "samples_1000": shape[2] == 1000,
            "dtype_float32": results["array"]["dtype"] == "float32",
        }

    # Metadata checks
    csv_path = os.path.join(ecg_dir, "ptbxl_metadata.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Required metadata columns
        required_cols = [
            "sample_id", "ecg_id", "patient_id", "split",
            "strat_fold", "primary_label", "sampling_rate_hz",
            "n_channels", "n_samples", "lead_names", "qc_flags"
        ]
        results["metadata_cols_present"] = {
            col: col in df.columns for col in required_cols
        }

        # Fold metadata
        results["fold_distribution"] = df["strat_fold"].value_counts().sort_index().to_dict()
        results["split_distribution"] = df["split"].value_counts().to_dict()

        # Leakage check — patient level
        results["leakage"] = check_leakage(df, "patient_id")

        # Label distribution
        results["top_10_labels"] = df["primary_label"].value_counts().head(10).to_dict()

        # Total records
        results["total_records"] = len(df)

    return results

# =============================================================================
# Resource estimate
# =============================================================================

def compute_resource_estimate() -> dict:
    logger.info("Computing resource estimates...")

    raw_sizes = {
        "pamap2":   get_dir_size_mb("data/raw/pamap2"),
        "wisdm":    get_dir_size_mb("data/raw/wisdm"),
        "eegmmidb": get_dir_size_mb("data/raw/eegmmidb"),
        "ptbxl":    get_dir_size_mb("data/raw/ptbxl"),
    }

    processed_sizes = {
        "har": get_dir_size_mb("data/processed/har"),
        "eeg": get_dir_size_mb("data/processed/eeg"),
        "ecg": get_dir_size_mb("data/processed/ecg"),
    }

    # Peak RAM from psutil
    process    = psutil.Process(os.getpid())
    ram_usage  = round(process.memory_info().rss / (1024 ** 2), 2)
    total_ram  = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    avail_ram  = round(psutil.virtual_memory().available / (1024 ** 3), 2)

    return {
        "raw_storage_mb":       raw_sizes,
        "raw_total_mb":         round(sum(raw_sizes.values()), 2),
        "processed_storage_mb": processed_sizes,
        "processed_total_mb":   round(sum(processed_sizes.values()), 2),
        "current_ram_usage_mb": ram_usage,
        "total_system_ram_gb":  total_ram,
        "available_ram_gb":     avail_ram,
        "peak_ram_notes": {
            "har":  "~2-4 GB — processed in memory, no chunking needed",
            "eeg":  "~6 GB — processed per subject to limit RAM",
            "ecg":  "~8 GB — processed record by record",
        },
        "runtime_notes": {
            "setup_data.sh":  "~5 hours total (EEGMMIDB ~1.5hrs, PTB-XL ~3hrs)",
            "preprocess.py":  "~5-10 minutes total",
            "validate_outputs.py": "~1-2 minutes",
        }
    }

# =============================================================================
# Output manifest
# =============================================================================

def generate_output_manifest() -> list:
    logger.info("Generating output manifest...")
    manifest = []

    for root, dirs, files in os.walk("data/processed"):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            entry = {
                "file":     fpath,
                "size_mb":  get_file_size_mb(fpath),
            }

            if fname.endswith(".npz"):
                d = np.load(fpath)
                arr = d["signals"]
                entry.update({
                    "type":       "signals",
                    "shape":      list(arr.shape),
                    "dtype":      str(arr.dtype),
                    "n_windows":  arr.shape[0],
                })

            elif fname.endswith(".csv"):
                df = pd.read_csv(fpath)
                entry.update({
                    "type":    "metadata",
                    "n_rows":  len(df),
                    "columns": list(df.columns),
                })

            manifest.append(entry)

    return manifest

# =============================================================================
# Report generation
# =============================================================================

def write_validation_report(
    har: dict,
    eeg: dict,
    ecg: dict,
    timestamp: str,
) -> None:
    lines = []
    lines.append("# Validation Report")
    lines.append(f"Generated: {timestamp}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── HAR ──────────────────────────────────────────────────────────────────
    lines.append("## HAR — PAMAP2 + WISDM")
    lines.append("")

    lines.append("### Array Integrity")
    lines.append("")
    lines.append("| Dataset | Shape | Dtype | NaNs | Infs | Size (MB) |")
    lines.append("|---|---|---|---|---|---|")
    for name, arr in har.get("arrays", {}).items():
        lines.append(
            f"| {name} | {arr['shape']} | {arr['dtype']} | "
            f"{arr['n_nans']} | {arr['n_infs']} | {arr['size_mb']} |"
        )
    lines.append("")

    lines.append("### Shape Checks")
    lines.append("")
    lines.append("| Check | Result |")
    lines.append("|---|---|")
    for check, result in har.get("shape_checks", {}).items():
        status = "✅ PASS" if result else "❌ FAIL"
        lines.append(f"| {check} | {status} |")
    lines.append("")

    lines.append("### HAR Harmonisation")
    lines.append("")
    lines.append("| Dataset | Sampling Rate (Hz) | Channels | Channel Schema |")
    lines.append("|---|---|---|---|")
    for dataset, info in har.get("harmonisation", {}).items():
        lines.append(
            f"| {dataset} | {info['sampling_rate']} | "
            f"{info['n_channels']} | {info['channel_schema']} |"
        )
    lines.append("")

    null_check = har.get("pamap2_null_label_check", {})
    status = "❌ FAIL — class 0 present" if null_check.get("class_0_present") else "✅ PASS — class 0 absent"
    lines.append(f"### Null Label Check (PAMAP2 class 0)")
    lines.append("")
    lines.append(f"Result: {status}")
    lines.append("")
    lines.append(f"Unique labels: {null_check.get('unique_labels', [])}")
    lines.append("")

    leakage = har.get("leakage", {})
    lines.append("### Leakage Control")
    lines.append("")
    lines.append("| Check | Value |")
    lines.append("|---|---|")
    lines.append(f"| Train subjects | {leakage.get('train_subjects', 'N/A')} |")
    lines.append(f"| Val subjects | {leakage.get('val_subjects', 'N/A')} |")
    lines.append(f"| Test subjects | {leakage.get('test_subjects', 'N/A')} |")
    leak_status = "✅ PASS" if leakage.get("leakage_train_test", 1) == 0 else "❌ FAIL"
    lines.append(f"| Train/test leakage | {leak_status} ({leakage.get('leakage_train_test', 'N/A')} subjects) |")
    lines.append("")

    lines.append("### Split Distribution")
    lines.append("")
    lines.append("| Split | Windows |")
    lines.append("|---|---|")
    for split, count in har.get("split_distribution", {}).items():
        lines.append(f"| {split} | {count} |")
    lines.append("")

    # ── EEG ──────────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## EEG — EEGMMIDB")
    lines.append("")

    arr = eeg.get("array", {})
    lines.append("### Array Integrity")
    lines.append("")
    lines.append("| Shape | Dtype | NaNs | Infs | Size (MB) |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| {arr.get('shape')} | {arr.get('dtype')} | "
        f"{arr.get('n_nans')} | {arr.get('n_infs')} | {arr.get('size_mb')} |"
    )
    lines.append("")

    lines.append("### Shape Checks")
    lines.append("")
    lines.append("| Check | Result |")
    lines.append("|---|---|")
    for check, result in eeg.get("shape_checks", {}).items():
        status = "✅ PASS" if result else "❌ FAIL"
        lines.append(f"| {check} | {status} |")
    lines.append("")

    ann = eeg.get("annotation_checks", {})
    lines.append("### Annotation Checks")
    lines.append("")
    lines.append("| Check | Result |")
    lines.append("|---|---|")
    for key in ["T0_present", "T1_present", "T2_present"]:
        status = "✅ PASS" if ann.get(key) else "❌ FAIL"
        lines.append(f"| {key} | {status} |")
    lines.append("")
    lines.append("| Event | Windows |")
    lines.append("|---|---|")
    for event, count in ann.get("event_distribution", {}).items():
        lines.append(f"| {event} | {count} |")
    lines.append("")

    runs = eeg.get("runs_check", {})
    run_status = "✅ PASS" if runs.get("only_required_runs") else "❌ FAIL"
    lines.append(f"### Runs Check")
    lines.append("")
    lines.append(f"Runs present: {runs.get('runs_present')} — {run_status}")
    lines.append("")

    leakage = eeg.get("leakage", {})
    lines.append("### Leakage Control")
    lines.append("")
    leak_status = "✅ PASS" if leakage.get("leakage_train_test", 1) == 0 else "❌ FAIL"
    lines.append(f"Train/test leakage: {leak_status} ({leakage.get('leakage_train_test', 'N/A')} subjects)")
    lines.append("")
    lines.append("| Split | Windows |")
    lines.append("|---|---|")
    for split, count in eeg.get("split_distribution", {}).items():
        lines.append(f"| {split} | {count} |")
    lines.append("")

    # ── ECG ──────────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## ECG — PTB-XL")
    lines.append("")

    arr = ecg.get("array", {})
    lines.append("### Array Integrity")
    lines.append("")
    lines.append("| Shape | Dtype | NaNs | Infs | Size (MB) |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| {arr.get('shape')} | {arr.get('dtype')} | "
        f"{arr.get('n_nans')} | {arr.get('n_infs')} | {arr.get('size_mb')} |"
    )
    lines.append("")

    lines.append("### Shape Checks")
    lines.append("")
    lines.append("| Check | Result |")
    lines.append("|---|---|")
    for check, result in ecg.get("shape_checks", {}).items():
        status = "✅ PASS" if result else "❌ FAIL"
        lines.append(f"| {check} | {status} |")
    lines.append("")

    lines.append("### ECG Fold Metadata")
    lines.append("")
    lines.append("| Fold | Records |")
    lines.append("|---|---|")
    for fold, count in ecg.get("fold_distribution", {}).items():
        lines.append(f"| {fold} | {count} |")
    lines.append("")

    lines.append("### Split Distribution")
    lines.append("")
    lines.append("| Split | Records |")
    lines.append("|---|---|")
    for split, count in ecg.get("split_distribution", {}).items():
        lines.append(f"| {split} | {count} |")
    lines.append("")

    leakage = ecg.get("leakage", {})
    leak_status = "✅ PASS" if leakage.get("leakage_train_test", 1) == 0 else "❌ FAIL"
    lines.append("### Leakage Control (Patient Level)")
    lines.append("")
    lines.append(f"Train/test patient leakage: {leak_status} ({leakage.get('leakage_train_test', 'N/A')} patients)")
    lines.append("")

    lines.append("### Top 10 Diagnostic Labels")
    lines.append("")
    lines.append("| Label | Records |")
    lines.append("|---|---|")
    for label, count in ecg.get("top_10_labels", {}).items():
        lines.append(f"| {label} | {count} |")
    lines.append("")

    # ── Summary ───────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Check | Result |")
    lines.append("|---|---|")

    all_checks = []

    # Collect all pass/fail
    for arr in har.get("arrays", {}).values():
        all_checks.append(arr["n_nans"] == 0)
        all_checks.append(arr["n_infs"] == 0)
    for result in har.get("shape_checks", {}).values():
        all_checks.append(result)
    all_checks.append(not har.get("pamap2_null_label_check", {}).get("class_0_present", True))
    all_checks.append(har.get("leakage", {}).get("leakage_train_test", 1) == 0)
    for result in eeg.get("shape_checks", {}).values():
        all_checks.append(result)
    all_checks.append(eeg.get("annotation_checks", {}).get("T0_present", False))
    all_checks.append(eeg.get("annotation_checks", {}).get("T1_present", False))
    all_checks.append(eeg.get("annotation_checks", {}).get("T2_present", False))
    all_checks.append(eeg.get("runs_check", {}).get("only_required_runs", False))
    all_checks.append(eeg.get("leakage", {}).get("leakage_train_test", 1) == 0)
    for result in ecg.get("shape_checks", {}).values():
        all_checks.append(result)
    all_checks.append(ecg.get("leakage", {}).get("leakage_train_test", 1) == 0)

    passed = sum(all_checks)
    total  = len(all_checks)

    overall = "✅ ALL CHECKS PASSED" if passed == total else f"⚠️ {total - passed} checks failed"
    lines.append(f"| Overall | {overall} ({passed}/{total}) |")
    lines.append(f"| HAR NaN/Inf | {'✅ PASS' if all(arr['n_nans'] == 0 and arr['n_infs'] == 0 for arr in har.get('arrays', {}).values()) else '❌ FAIL'} |")
    lines.append(f"| EEG NaN/Inf | {'✅ PASS' if eeg.get('array', {}).get('n_nans', 1) == 0 else '❌ FAIL'} |")
    lines.append(f"| ECG NaN/Inf | {'✅ PASS' if ecg.get('array', {}).get('n_nans', 1) == 0 else '❌ FAIL'} |")
    lines.append(f"| HAR leakage | {'✅ PASS' if har.get('leakage', {}).get('leakage_train_test', 1) == 0 else '❌ FAIL'} |")
    lines.append(f"| EEG leakage | {'✅ PASS' if eeg.get('leakage', {}).get('leakage_train_test', 1) == 0 else '❌ FAIL'} |")
    lines.append(f"| ECG leakage | {'✅ PASS' if ecg.get('leakage', {}).get('leakage_train_test', 1) == 0 else '❌ FAIL'} |")
    lines.append("")

    os.makedirs("reports", exist_ok=True)
    with open("reports/validation_report.md", "w") as f:
        f.write("\n".join(lines))
    logger.info("Validation report written to reports/validation_report.md")


def write_resource_estimate(resources: dict, timestamp: str) -> None:
    lines = []
    lines.append("# Resource Estimate")
    lines.append(f"Generated: {timestamp}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Raw Storage")
    lines.append("")
    lines.append("| Dataset | Size (MB) |")
    lines.append("|---|---|")
    for dataset, size in resources["raw_storage_mb"].items():
        lines.append(f"| {dataset} | {size} |")
    lines.append(f"| **Total** | **{resources['raw_total_mb']}** |")
    lines.append("")

    lines.append("## Processed Storage")
    lines.append("")
    lines.append("| Modality | Size (MB) |")
    lines.append("|---|---|")
    for modality, size in resources["processed_storage_mb"].items():
        lines.append(f"| {modality} | {size} |")
    lines.append(f"| **Total** | **{resources['processed_total_mb']}** |")
    lines.append("")

    lines.append("## RAM Usage")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Current process RAM | {resources['current_ram_usage_mb']} MB |")
    lines.append(f"| Total system RAM | {resources['total_system_ram_gb']} GB |")
    lines.append(f"| Available RAM | {resources['available_ram_gb']} GB |")
    lines.append("")

    lines.append("## Peak RAM Estimates by Modality")
    lines.append("")
    lines.append("| Modality | Peak RAM Estimate |")
    lines.append("|---|---|")
    for modality, note in resources["peak_ram_notes"].items():
        lines.append(f"| {modality.upper()} | {note} |")
    lines.append("")

    lines.append("## Runtime Estimates")
    lines.append("")
    lines.append("| Script | Estimated Runtime |")
    lines.append("|---|---|")
    for script, note in resources["runtime_notes"].items():
        lines.append(f"| {script} | {note} |")
    lines.append("")

    with open("reports/resource_estimate.md", "w") as f:
        f.write("\n".join(lines))
    logger.info("Resource estimate written to reports/resource_estimate.md")


# =============================================================================
# Main
# =============================================================================

def main():
    start_time = time.time()
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("=" * 60)
    logger.info("Multimodal Biosignal Pipeline — Validation")
    logger.info(f"Started: {timestamp}")
    logger.info("=" * 60)

    # Run validations
    har_results = validate_har()
    eeg_results = validate_eeg()
    ecg_results = validate_ecg()

    # Resource estimate
    resources = compute_resource_estimate()

    # Output manifest
    manifest = generate_output_manifest()
    os.makedirs("data", exist_ok=True)
    with open("data/manifest_outputs.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Output manifest written to data/manifest_outputs.json")

    # Write reports
    write_validation_report(har_results, eeg_results, ecg_results, timestamp)
    write_resource_estimate(resources, timestamp)

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Validation complete in {elapsed}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()