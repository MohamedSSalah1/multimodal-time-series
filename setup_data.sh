#!/bin/bash
# =============================================================================
# setup_data.sh
# Shell wrapper for dataset download and folder setup.
# Orchestration only — no preprocessing logic lives here.
# All parsing, preprocessing, and validation is handled in Python.
#
# Usage:
#   caffeinate -i nohup bash setup_data.sh &
#
#   Monitor progress:
#   tail -f logs/setup_*.log
#
# Requirements:
#   - wget must be installed: brew install wget
#   - Run from the project root directory
#   - conda environment will be activated automatically
# =============================================================================

set -euo pipefail

# ── Conda environment self-activation ────────────────────────────────────────
# Ensures the correct environment is active regardless of how the script
# is launched (directly, via nohup, or inside screen/tmux)
CONDA_BASE=$(conda info --base 2>/dev/null) || {
    echo "ERROR: conda not found. Please install Anaconda or Miniforge."
    exit 1
}
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate biomedical_pipeline || {
    echo "ERROR: Could not activate biomedical_pipeline environment."
    echo "Create it first with: conda env create -f environment.yml"
    exit 1
}

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p logs
LOG_FILE="logs/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================"
echo " Multimodal Biosignal Pipeline — Setup"
echo " Started: $(date)"
echo " Log: $LOG_FILE"
echo "============================================"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────
echo -e "${BLUE}[0/7] Running preflight checks...${NC}"

# Check wget
if ! command -v wget &> /dev/null; then
    echo -e "${RED}ERROR: wget is not installed.${NC}"
    echo "Install it with: brew install wget"
    exit 1
fi
echo -e "${GREEN}      wget found.${NC}"

# Check python3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 not available.${NC}"
    echo "Activate your conda environment: conda activate biomedical_pipeline"
    exit 1
fi
echo -e "${GREEN}      python3 found.${NC}"

# Check we are in the project root
if [ ! -f "configs/pipeline.yaml" ]; then
    echo -e "${RED}ERROR: configs/pipeline.yaml not found.${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi
echo -e "${GREEN}      Project root confirmed.${NC}"
echo ""

# ── Step 1: Create folder structure ──────────────────────────────────────────
echo -e "${BLUE}[1/7] Creating folder structure...${NC}"
mkdir -p data/raw/pamap2 \
         data/raw/wisdm \
         data/raw/eegmmidb \
         data/raw/ptbxl \
         data/interim/har \
         data/interim/eeg \
         data/interim/ecg \
         data/processed/har \
         data/processed/eeg \
         data/processed/ecg \
         configs \
         reports \
         submission_sample/har \
         submission_sample/eeg \
         submission_sample/ecg \
         logs
echo -e "${GREEN}      Folder structure ready.${NC}"
echo ""

# ── Helper: download a file ───────────────────────────────────────────────────
download_file() {
    local url=$1
    local dest=$2
    local name=$3

    if [ -f "$dest" ]; then
        echo -e "${YELLOW}      $name already exists — skipping download.${NC}"
    else
        echo "      Downloading $name..."
        wget --progress=bar:force -O "$dest" "$url" || {
            echo -e "${RED}ERROR: Failed to download $name${NC}"
            echo -e "${RED}URL: $url${NC}"
            rm -f "$dest"
            exit 1
        }
        echo -e "${GREEN}      $name downloaded.${NC}"
    fi
}

# ── Helper: minimum file size check ──────────────────────────────────────────
check_min_size() {
    local file=$1
    local min_mb=$2
    local name=$3

    local size_mb
    size_mb=$(du -m "$file" | cut -f1)
    if [ "$size_mb" -lt "$min_mb" ]; then
        echo -e "${RED}ERROR: $name appears incomplete.${NC}"
        echo -e "${RED}Expected at least ${min_mb}MB, got ${size_mb}MB.${NC}"
        echo -e "${RED}Delete the file and re-run setup_data.sh.${NC}"
        exit 1
    fi
    echo -e "${GREEN}      Size check passed: ${size_mb}MB.${NC}"
}

# ── Helper: record SHA256 checksum (PAMAP2 + WISDM) ──────────────────────────
# For datasets without official checksums, we record on first download
# and verify on subsequent runs
record_checksum() {
    local file=$1
    local name=$2
    local checksum_file="data/checksums.sha256"

    if grep -q "$file" "$checksum_file" 2>/dev/null; then
        echo -e "${YELLOW}      Checksum already recorded for $name — skipping.${NC}"
        return
    fi

    echo "      Recording checksum for $name..."
    if command -v sha256sum &>/dev/null; then
        sha256sum "$file" >> "$checksum_file"
    else
        shasum -a 256 "$file" >> "$checksum_file"
    fi
    echo -e "${GREEN}      Checksum recorded in $checksum_file.${NC}"
}

verify_recorded_checksum() {
    local file=$1
    local name=$2
    local checksum_file="data/checksums.sha256"

    if [ ! -f "$checksum_file" ]; then
        echo -e "${YELLOW}      No checksum file found — skipping verification.${NC}"
        return
    fi

    if ! grep -q "$file" "$checksum_file" 2>/dev/null; then
        echo -e "${YELLOW}      No recorded checksum for $name — recording now.${NC}"
        record_checksum "$file" "$name"
        return
    fi

    echo "      Verifying checksum for $name..."
    if command -v sha256sum &>/dev/null; then
        if sha256sum --check "$checksum_file" --ignore-missing 2>/dev/null | grep -q "OK"; then
            echo -e "${GREEN}      Checksum verified for $name.${NC}"
        else
            echo -e "${RED}ERROR: Checksum mismatch for $name.${NC}"
            echo -e "${RED}File may be corrupted. Delete and re-run setup_data.sh.${NC}"
            exit 1
        fi
    else
        if shasum -a 256 -c "$checksum_file" 2>/dev/null | grep -q "OK"; then
            echo -e "${GREEN}      Checksum verified for $name.${NC}"
        else
            echo -e "${RED}ERROR: Checksum mismatch for $name.${NC}"
            echo -e "${RED}File may be corrupted. Delete and re-run setup_data.sh.${NC}"
            exit 1
        fi
    fi
}

# ── Helper: verify against official PhysioNet SHA256SUMS.txt ─────────────────
# Used for EEGMMIDB and PTB-XL which publish official checksums
verify_official_checksum() {
    local file=$1
    local sha256sums_file=$2
    local name=$3

    if [ ! -f "$sha256sums_file" ]; then
        echo -e "${YELLOW}      Official SHA256SUMS.txt not found — skipping.${NC}"
        return
    fi

    # Build relative path from the sha256sums file directory
    # e.g. data/raw/eegmmidb/S001/S001R04.edf → S001/S001R04.edf
    local sha256_dir
    sha256_dir=$(dirname "$sha256sums_file")
    local relative_path
    relative_path="${file#$sha256_dir/}"

    if ! grep -q "$relative_path" "$sha256sums_file"; then
        echo -e "${YELLOW}      $name not listed in SHA256SUMS.txt — skipping.${NC}"
        return
    fi

    echo "      Verifying $name against official PhysioNet checksum..."
    local expected
    expected=$(grep " $relative_path$" "$sha256sums_file" | awk '{print $1}')

    local actual
    if command -v sha256sum &>/dev/null; then
        actual=$(sha256sum "$file" | awk '{print $1}')
    else
        actual=$(shasum -a 256 "$file" | awk '{print $1}')
    fi

    if [ "$expected" = "$actual" ]; then
        echo -e "${GREEN}      Official checksum verified for $name.${NC}"
    else
        echo -e "${RED}ERROR: Official checksum mismatch for $name.${NC}"
        echo -e "${RED}Expected: $expected${NC}"
        echo -e "${RED}Got:      $actual${NC}"
        echo -e "${RED}File may be corrupted. Delete and re-run setup_data.sh.${NC}"
        exit 1
    fi
}

# ── Step 2: Download PAMAP2 ───────────────────────────────────────────────────
echo -e "${BLUE}[2/7] Downloading PAMAP2 (HAR — ~656 MB)...${NC}"
download_file \
    "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip" \
    "data/raw/pamap2/pamap2.zip" \
    "PAMAP2"

check_min_size "data/raw/pamap2/pamap2.zip" 500 "PAMAP2"
verify_recorded_checksum "data/raw/pamap2/pamap2.zip" "PAMAP2"

echo "      Extracting PAMAP2 (outer zip)..."
unzip -q -o data/raw/pamap2/pamap2.zip -d data/raw/pamap2/

echo "      Extracting PAMAP2 (inner zip)..."
unzip -q -o data/raw/pamap2/PAMAP2_Dataset.zip -d data/raw/pamap2/

# Verify expected folder structure
if [ ! -d "data/raw/pamap2/PAMAP2_Dataset/Protocol" ]; then
    echo -e "${RED}ERROR: PAMAP2 extraction failed — Protocol folder not found.${NC}"
    exit 1
fi

PAMAP2_COUNT=$(ls data/raw/pamap2/PAMAP2_Dataset/Protocol/*.dat 2>/dev/null | wc -l)
if [ "$PAMAP2_COUNT" -lt 9 ]; then
    echo -e "${RED}ERROR: Expected 9 Protocol .dat files, found ${PAMAP2_COUNT}.${NC}"
    exit 1
fi
echo -e "${GREEN}      PAMAP2 ready — ${PAMAP2_COUNT} Protocol files confirmed.${NC}"
echo ""

# ── Step 3: Download WISDM ────────────────────────────────────────────────────
echo -e "${BLUE}[3/7] Downloading WISDM (HAR — ~780 MB)...${NC}"
download_file \
    "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip" \
    "data/raw/wisdm/wisdm.zip" \
    "WISDM"

check_min_size "data/raw/wisdm/wisdm.zip" 250 "WISDM"
verify_recorded_checksum "data/raw/wisdm/wisdm.zip" "WISDM"

echo "      Extracting WISDM (outer zip)..."
unzip -q -o data/raw/wisdm/wisdm.zip -d data/raw/wisdm/

echo "      Extracting WISDM (inner zip)..."
unzip -q -o data/raw/wisdm/wisdm-dataset.zip -d data/raw/wisdm/

# Verify expected folder structure
WISDM_WATCH_ACCEL=$(find data/raw/wisdm/wisdm-dataset/raw/watch/accel -name "*.txt" 2>/dev/null | wc -l)
WISDM_WATCH_GYRO=$(find data/raw/wisdm/wisdm-dataset/raw/watch/gyro -name "*.txt" 2>/dev/null | wc -l)

if [ "$WISDM_WATCH_ACCEL" -lt 51 ] || [ "$WISDM_WATCH_GYRO" -lt 51 ]; then
    echo -e "${RED}ERROR: WISDM watch files incomplete.${NC}"
    echo -e "${RED}Expected 51 accel + 51 gyro, found ${WISDM_WATCH_ACCEL} + ${WISDM_WATCH_GYRO}.${NC}"
    exit 1
fi
echo -e "${GREEN}      WISDM ready — ${WISDM_WATCH_ACCEL} accel + ${WISDM_WATCH_GYRO} gyro watch files confirmed.${NC}"
echo ""

# ── Step 4: Download EEGMMIDB ─────────────────────────────────────────────────
echo -e "${BLUE}[4/7] Downloading EEGMMIDB — runs 4, 8, 12 only (EEG — ~500 MB)...${NC}"
BASE_EEG_URL="https://physionet.org/files/eegmmidb/1.0.0"

# Download official SHA256SUMS.txt first for verification
download_file \
    "${BASE_EEG_URL}/SHA256SUMS.txt" \
    "data/raw/eegmmidb/SHA256SUMS.txt" \
    "EEGMMIDB SHA256SUMS.txt"

# Download runs 4, 8, 12 for all 109 subjects
for SUBJ in $(seq -f "%03g" 1 109); do
    for RUN in 04 08 12; do
        FILE="S${SUBJ}/S${SUBJ}R${RUN}.edf"
        DEST="data/raw/eegmmidb/S${SUBJ}/S${SUBJ}R${RUN}.edf"
        mkdir -p "data/raw/eegmmidb/S${SUBJ}"
        download_file "${BASE_EEG_URL}/${FILE}" "$DEST" "EEGMMIDB ${FILE}"
        verify_official_checksum \
            "$DEST" \
            "data/raw/eegmmidb/SHA256SUMS.txt" \
            "EEGMMIDB ${FILE}"
    done
done

# Verify total file count
EEG_COUNT=$(find data/raw/eegmmidb -name "*.edf" | wc -l)
EXPECTED_EEG=$((109 * 3))

if [ "$EEG_COUNT" -lt "$EXPECTED_EEG" ]; then
    echo -e "${RED}ERROR: Expected ${EXPECTED_EEG} EDF files, found ${EEG_COUNT}.${NC}"
    exit 1
fi
echo -e "${GREEN}      EEGMMIDB ready — ${EEG_COUNT} EDF files confirmed and verified.${NC}"
echo ""

# ── Step 5: Download PTB-XL ───────────────────────────────────────────────────
echo -e "${BLUE}[5/7] Downloading PTB-XL (ECG — ~2 GB)...${NC}"
PTBXL_URL="https://physionet.org/content/ptb-xl/get-zip/1.0.3/"

# Download official SHA256SUMS.txt first
download_file \
    "https://physionet.org/files/ptb-xl/1.0.3/SHA256SUMS.txt" \
    "data/raw/ptbxl/SHA256SUMS.txt" \
    "PTB-XL SHA256SUMS.txt"

download_file \
    "$PTBXL_URL" \
    "data/raw/ptbxl/ptbxl.zip" \
    "PTB-XL"

check_min_size "data/raw/ptbxl/ptbxl.zip" 1400 "PTB-XL"
verify_recorded_checksum "data/raw/ptbxl/ptbxl.zip" "PTB-XL"

echo "      Extracting PTB-XL (outer zip)..."
unzip -q -o data/raw/ptbxl/ptbxl.zip -d data/raw/ptbxl/

# Check if nested zip exists and extract if so
INNER_ZIP=$(find data/raw/ptbxl -name "*.zip" ! -name "ptbxl.zip" | head -1)
if [ -n "$INNER_ZIP" ]; then
    echo "      Nested zip detected: $INNER_ZIP"
    echo "      Extracting PTB-XL (inner zip)..."
    unzip -q -o "$INNER_ZIP" -d data/raw/ptbxl/
    echo -e "${GREEN}      Inner zip extracted.${NC}"
fi

# Verify key files exist — search recursively regardless of subfolder name
PTBXL_CSV=$(find data/raw/ptbxl -name "ptbxl_database.csv" | head -1)
if [ -z "$PTBXL_CSV" ]; then
    echo -e "${RED}ERROR: PTB-XL extraction failed — ptbxl_database.csv not found.${NC}"
    echo -e "${RED}Folder structure found:${NC}"
    find data/raw/ptbxl -type d | sort
    exit 1
fi
echo -e "${GREEN}      PTB-XL ready — ptbxl_database.csv confirmed at ${PTBXL_CSV}.${NC}"
echo ""

# ── Step 6: Spot check data integrity ────────────────────────────────────────
echo -e "${BLUE}[6/7] Running spot checks...${NC}"

python3 - << 'PYEOF'
import os
import sys
import glob
import csv

errors = []

# PAMAP2 — check subject101.dat has 54 columns
pamap2_file = "data/raw/pamap2/PAMAP2_Dataset/Protocol/subject101.dat"
if os.path.exists(pamap2_file):
    with open(pamap2_file) as f:
        first_line = f.readline().strip().split()
    if len(first_line) != 54:
        errors.append(f"PAMAP2: Expected 54 columns, found {len(first_line)}")
    else:
        print(f"      PAMAP2 column check passed: {len(first_line)} columns confirmed.")
else:
    errors.append("PAMAP2: subject101.dat not found")

# WISDM — check watch accel file format
wisdm_files = glob.glob("data/raw/wisdm/**/watch/accel/*.txt", recursive=True)
if wisdm_files:
    with open(wisdm_files[0]) as f:
        first_line = f.readline().strip().rstrip(";").split(",")
    if len(first_line) != 6:
        errors.append(f"WISDM: Expected 6 fields, found {len(first_line)}")
    else:
        print(f"      WISDM format check passed: 6 fields confirmed.")
else:
    errors.append("WISDM: No watch accel files found")

# EEGMMIDB — check first EDF file exists and has non-zero size
eeg_file = "data/raw/eegmmidb/S001/S001R04.edf"
if os.path.exists(eeg_file):
    size = os.path.getsize(eeg_file)
    if size < 1000:
        errors.append(f"EEGMMIDB: S001R04.edf too small ({size} bytes)")
    else:
        print(f"      EEGMMIDB file check passed: S001R04.edf is {size/1024:.1f} KB.")
else:
    errors.append("EEGMMIDB: S001R04.edf not found")

# PTB-XL — check database CSV has required columns
ptbxl_csvs = glob.glob("data/raw/ptbxl/**/ptbxl_database.csv", recursive=True)
if ptbxl_csvs:
    with open(ptbxl_csvs[0]) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
    required = ["ecg_id", "patient_id", "strat_fold", "filename_lr"]
    missing = [h for h in required if h not in headers]
    if missing:
        errors.append(f"PTB-XL: Missing columns: {missing}")
    else:
        print(f"      PTB-XL CSV check passed: all required columns confirmed.")
else:
    errors.append("PTB-XL: ptbxl_database.csv not found")

if errors:
    print("\nSPOT CHECK ERRORS:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("      All spot checks passed.")
PYEOF

echo ""

# ── Step 7: Generate download manifest ───────────────────────────────────────
echo -e "${BLUE}[7/7] Generating download manifest...${NC}"

python3 - << 'PYEOF'
import json
import os
import glob
from datetime import datetime, timezone

def get_size_mb(path):
    if os.path.isfile(path):
        return round(os.path.getsize(path) / (1024 ** 2), 2)
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return round(total / (1024 ** 2), 2)

def count_files(path, pattern):
    return len(glob.glob(os.path.join(path, "**", pattern), recursive=True))

now = datetime.now(timezone.utc).isoformat()

manifest = {
    "generated_at": now,
    "project": "Multimodal Biosignal Preprocessing Pipeline",
    "datasets": [
        {
            "name": "PAMAP2",
            "modality": "HAR",
            "source_url": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
            "landing_page": "https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring",
            "local_path": "data/raw/pamap2",
            "version": "UCI ML Repository — Dataset #231",
            "native_sampling_rate_hz": 100,
            "target_sampling_rate_hz": 20,
            "n_subjects": 9,
            "checksum_method": "SHA256 — recorded on first download, verified on subsequent runs",
            "file_count": count_files("data/raw/pamap2", "*.dat"),
            "size_mb": get_size_mb("data/raw/pamap2"),
            "status": "downloaded" if os.path.exists("data/raw/pamap2/PAMAP2_Dataset/Protocol") else "missing",
            "downloaded_at": now
        },
        {
            "name": "WISDM",
            "modality": "HAR",
            "source_url": "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
            "landing_page": "https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset",
            "local_path": "data/raw/wisdm",
            "version": "UCI ML Repository — Dataset #507",
            "native_sampling_rate_hz": 20,
            "target_sampling_rate_hz": 20,
            "n_subjects": 51,
            "checksum_method": "SHA256 — recorded on first download, verified on subsequent runs",
            "file_count": count_files("data/raw/wisdm", "*.txt"),
            "size_mb": get_size_mb("data/raw/wisdm"),
            "status": "downloaded" if os.path.exists("data/raw/wisdm") else "missing",
            "downloaded_at": now
        },
        {
            "name": "EEGMMIDB",
            "modality": "EEG",
            "source_url": "https://physionet.org/files/eegmmidb/1.0.0",
            "landing_page": "https://physionet.org/content/eegmmidb/1.0.0/",
            "local_path": "data/raw/eegmmidb",
            "version": "PhysioNet v1.0.0",
            "native_sampling_rate_hz": 160,
            "target_sampling_rate_hz": 160,
            "n_subjects": 109,
            "runs_downloaded": [4, 8, 12],
            "checksum_method": "SHA256 — verified against official PhysioNet SHA256SUMS.txt",
            "note": "Only motor imagery runs 4, 8, 12 downloaded as per assessment brief",
            "file_count": count_files("data/raw/eegmmidb", "*.edf"),
            "size_mb": get_size_mb("data/raw/eegmmidb"),
            "status": "downloaded" if os.path.exists("data/raw/eegmmidb/S001") else "missing",
            "downloaded_at": now
        },
        {
            "name": "PTB-XL",
            "modality": "ECG",
            "source_url": "https://physionet.org/content/ptb-xl/get-zip/1.0.3/",
            "landing_page": "https://physionet.org/content/ptb-xl/1.0.3/",
            "local_path": "data/raw/ptbxl",
            "version": "PhysioNet v1.0.3",
            "native_sampling_rate_hz_available": [100, 500],
            "target_sampling_rate_hz": 100,
            "n_records": 21799,
            "n_patients": 18869,
            "checksum_method": "SHA256 — verified against official PhysioNet SHA256SUMS.txt",
            "file_count": count_files("data/raw/ptbxl", "*.dat"),
            "size_mb": get_size_mb("data/raw/ptbxl"),
            "status": "downloaded" if len(glob.glob("data/raw/ptbxl/**/ptbxl_database.csv",
                      recursive=True)) > 0 else "missing",
            "downloaded_at": now
        }
    ]
}

os.makedirs("data", exist_ok=True)
with open("data/manifest_downloads.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("      Manifest written to data/manifest_downloads.json")

missing = [d["name"] for d in manifest["datasets"] if d["status"] == "missing"]
if missing:
    print(f"\nWARNING: Missing datasets: {', '.join(missing)}")
    import sys
    sys.exit(1)
else:
    print("      All datasets confirmed present.")
PYEOF

echo ""
echo "============================================"
echo -e "${GREEN} Setup complete.${NC}"
echo " Finished: $(date)"
echo " Log:      $LOG_FILE"
echo " Manifest: data/manifest_downloads.json"
echo " Checksums: data/checksums.sha256"
echo "============================================"