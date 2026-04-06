# Multimodal Biosignal Preprocessing Pipeline

Reproducible pipeline for downloading, preprocessing, and validating multimodal time-series datasets (HAR, EEG, ECG) in preparation for downstream self-supervised learning workflows.

Built as part of a technical assessment for the Research Assistant in Biomedical Data Integration role at Imperial College London (UK Dementia Research Institute).

---

## Environment Setup

This project uses a conda environment for full reproducibility.

### Requirements
- [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda
- Python 3.11

### Create and activate the environment
```bash
conda env create -f environment.yml
conda activate biomedical_pipeline
```

### Packages and rationale

| Package | Role in this project |
|---|---|
| `numpy` | Storing signals as arrays, windowing, saving `.npy` files |
| `pandas` | Metadata CSVs, manifest files, label tables |
| `scipy` | Resampling HAR signals, bandpass/notch filtering |
| `mne` | Parsing EDF+ files, EEG preprocessing |
| `wfdb` | Reading PTB-XL ECG records (WFDB format) |
| `scikit-learn` | Utility functions for splitting |
| `matplotlib` | QC plots |
| `pyyaml` | Reading the `configs/pipeline.yaml` config file |
| `tqdm` | Progress bars when looping over subjects/files |
| `psutil` | Monitoring peak RAM usage for resource estimate report |
| `tabulate` | Formatting validation report and manifest as markdown tables |

---

## Project Structure
```
multimodal-time-series/
├── setup_data.sh              # Shell wrapper — folder setup and dataset downloads
├── preprocess.py              # Main preprocessing entrypoint
├── validate_outputs.py        # Validation and report generation
├── configs/
│   └── pipeline.yaml          # All pipeline parameters — sampling rates, window sizes, channel schemas
├── data/
│   ├── raw/                   # Downloaded datasets (gitignored — see Data section below)
│   │   ├── pamap2/
│   │   ├── wisdm/
│   │   ├── eegmmidb/
│   │   └── ptbxl/
│   ├── interim/               # Parsed but not yet windowed
│   │   ├── har/
│   │   ├── eeg/
│   │   └── ecg/
│   └── processed/             # Final .npy arrays and metadata CSVs
│       ├── har/
│       ├── eeg/
│       └── ecg/
├── reports/                   # Validation report, resource estimate, preprocessing plan
├── submission_sample/         # 100 representative windows per dataset
│   ├── har/
│   ├── eeg/
│   └── ecg/
├── logs/                      # Download and pipeline logs
├── environment.yml            # Conda environment for full reproducibility
└── README.md
```

---

## Data

### Why data is not in this repository

Raw and processed data files are excluded from this repository via `.gitignore` for two reasons:
- The full datasets are several gigabytes in size and not suitable for git
- Raw data should always be downloaded from official sources to ensure integrity and reproducibility

### How the folder structure is preserved

Each data folder contains a `.gitkeep` file — a tiny empty placeholder that allows git to track the folder structure without tracking any data inside it. When you clone this repo, the full folder structure is immediately available and ready for `setup_data.sh` to populate.

### Where to find the processed sample pack

The representative processed sample pack (100 windows per dataset) is available at:
> link — to be added upon submission

---

## Dataset Notes

The following findings were gathered by exploring each dataset before writing any preprocessing code. These observations directly informed the preprocessing plan and configuration choices.

---

### PAMAP2 — Physical Activity Monitoring

**Folder structure after extraction:**
```
PAMAP2_Dataset/
├── Protocol/
│   ├── subject101.dat     # 9 subjects: 101–109
│   └── ...
└── Optional/
├── subject101.dat     # 6 subjects with optional activities
└── ...
```

**Key findings:**

| Detail | Implication |
|---|---|
| 54 columns per row | Precise column selection required |
| Column 1: timestamp, Column 2: activity ID | Activity label is always second column |
| Columns 4–20: IMU hand (wrist) | Our target sensor block |
| Within wrist block — cols 2–4: ±16g accelerometer | Use these — recommended by dataset authors |
| Within wrist block — cols 5–7: ±6g accelerometer | Discard — saturates at high impact |
| Within wrist block — cols 8–10: gyroscope | Use these |
| Within wrist block — cols 14–17: orientation | Discard — invalid in this collection |
| 100 Hz native sampling rate | Must resample to 20 Hz (target) |
| Missing values marked as NaN | Interpolation strategy required |
| Activity ID 0 = transient | Must be explicitly discarded per dataset readme |
| 9 subjects (101–109) | Small — subject-level splits must be careful |
| Protocol + Optional folders | Both parsed — Optional adds coverage for some subjects |

**Wrist IMU 6-channel schema (0-indexed column positions):**

| Channel | Column (0-indexed) | Description |
|---|---|---|
| acc_x | 3 | Wrist accelerometer X ±16g |
| acc_y | 4 | Wrist accelerometer Y ±16g |
| acc_z | 5 | Wrist accelerometer Z ±16g |
| gyr_x | 9 | Wrist gyroscope X |
| gyr_y | 10 | Wrist gyroscope Y |
| gyr_z | 11 | Wrist gyroscope Z |

**Activity label mapping:**

| ID | Activity | Kept in unified schema |
|---|---|---|
| 0 | Transient | ❌ Discarded |
| 1 | Lying | ✅ |
| 2 | Sitting | ✅ |
| 3 | Standing | ✅ |
| 4 | Walking | ✅ |
| 5 | Running | ✅ |
| 6 | Cycling | ✅ |
| 7 | Nordic walking | ✅ |
| 9 | Watching TV | ✅ |
| 10 | Computer work | ✅ |
| 11 | Car driving | ✅ |
| 12 | Ascending stairs | ✅ |
| 13 | Descending stairs | ✅ |
| 16 | Vacuum cleaning | ✅ |
| 17 | Ironing | ✅ |
| 18 | Folding laundry | ✅ |
| 19 | House cleaning | ✅ |
| 20 | Playing soccer | ✅ |
| 24 | Rope jumping | ✅ |

---

### WISDM — Smartphone and Smartwatch Activity Dataset

**Folder structure after extraction:**
```
wisdm-dataset/
├── raw/
│   ├── phone/
│   │   ├── accel/     # 51 files — not used
│   │   └── gyro/      # 51 files — not used
│   └── watch/
│       ├── accel/     # 51 files — USED
│       └── gyro/      # 51 files — USED
├── arff_files/        # Pre-transformed features — not used
└── arffmagic-master/  # Transformation scripts — not used
```

**Key findings:**

| Detail | Implication |
|---|---|
| 51 subjects (IDs 1600–1650) | Larger than PAMAP2 — better generalisation |
| 20 Hz native sampling rate | No resampling needed — already at target |
| Watch accelerometer + gyroscope used only | Matches PAMAP2 wrist IMU pairing |
| Each line ends with semicolon `;` | Must strip during parsing |
| Activity labels are letters (A–S) | Must map to unified integer/string schema |
| Format: `subject_id, activity, timestamp, x, y, z;` | Simple CSV with semicolon terminator |
| No explicit missing value indicator | Verify during parsing |
| 51 files per subdirectory | Expect 102 files total for watch data |
| Units: accelerometer m/s², gyroscope rad/s | Consistent with PAMAP2 |

**Activity label mapping to unified schema:**

| WISDM Code | WISDM Activity | PAMAP2 Equivalent | Unified Label | Kept |
|---|---|---|---|---|
| A | Walking | Walking (4) | walking | ✅ |
| B | Jogging | Running (5) | running | ✅ |
| C | Stairs | Stairs (12/13) | stairs | ✅ |
| D | Sitting | Sitting (2) | sitting | ✅ |
| E | Standing | Standing (3) | standing | ✅ |
| F | Typing | Computer work (10) | computer_work | ✅ |
| G | Brushing Teeth | No equivalent | brushing_teeth | ✅ pretraining only |
| H | Eating Soup | No equivalent | eating_soup | ✅ pretraining only |
| I | Eating Chips | No equivalent | eating_chips | ✅ pretraining only |
| J | Eating Pasta | No equivalent | eating_pasta | ✅ pretraining only |
| K | Drinking | No equivalent | drinking | ✅ pretraining only |
| L | Eating Sandwich | No equivalent | eating_sandwich | ✅ pretraining only |
| M | Kicking | Playing soccer (20) | kicking | ✅ pretraining only |
| O | Playing Catch | No equivalent | playing_catch | ✅ pretraining only |
| P | Dribbling | Playing soccer (20) | dribbling | ✅ pretraining only |
| Q | Writing | No equivalent | writing | ✅ pretraining only |
| R | Clapping | No equivalent | clapping | ✅ pretraining only |
| S | Folding Clothes | Folding laundry (18) | folding_laundry | ✅ |

**Label strategy:**
- **Pretraining output:** all activities included (no labels used)
- **Supervised output:** only shared activities used across both datasets

---

### EEGMMIDB — EEG Motor Movement/Imagery Dataset

**Folder structure after extraction:**
```
eegmmidb/
├── S001/
│   ├── S001R01.edf        # Run 1 — baseline eyes open
│   ├── S001R01.edf.event  # Separate annotation file
│   └── ... R02–R14
├── ...
├── S109/
├── SHA256SUMS.txt         # Official checksums — used for verification
└── RECORDS
```
**Key findings:**

| Detail | Implication |
|---|---|
| 109 subjects (S001–S109) | Large dataset |
| 64 channels at 160 Hz | Keep native rate — no resampling |
| EDF+ format with embedded annotation channel | MNE parses both signal and annotations |
| Separate `.edf.event` files also exist | We use annotations embedded in EDF+ |
| 14 runs per subject total | We only use runs 4, 8, 12 |
| SHA256SUMS.txt provided officially | Used for download verification |
| Total size: 3.4 GB uncompressed | We download 327 EDF files only |

**Why runs 4, 8, 12:**

| Run | Task | T1 | T2 |
|---|---|---|---|
| 4 | Imagined movement | Left fist imagery | Right fist imagery |
| 8 | Imagined movement | Left fist imagery | Right fist imagery |
| 12 | Imagined movement | Left fist imagery | Right fist imagery |

Consistent 3-class problem: T0 (rest), T1 (left fist), T2 (right fist)

**Annotation codes:**

| Code | Meaning |
|---|---|
| T0 | Rest |
| T1 | Left fist imagery (runs 4, 8, 12) |
| T2 | Right fist imagery (runs 4, 8, 12) |

---

### PTB-XL — Large Publicly Available ECG Dataset

**Folder structure after extraction:**
```
ptbxl/
├── ptbxl_database.csv     # Master metadata — one row per record
├── scp_statements.csv     # ECG statement definitions
├── example_physionet.py   # Official usage example
├── SHA256SUMS.txt         # Official checksums
├── records100/            # 100 Hz waveforms — USED
│   ├── 00000/
│   │   ├── 00001_lr.dat
│   │   ├── 00001_lr.hea
│   │   └── ...
│   └── 21000/
└── records500/            # 500 Hz waveforms — not used
```
**Key findings:**

| Detail | Implication |
|---|---|
| 21,799 records from 18,869 patients | Multiple records per patient possible |
| 12-lead ECG, 10 seconds per record | Output shape: `[N, 12, 1000]` at 100 Hz |
| Two sampling rates: 100 Hz and 500 Hz | We use 100 Hz — sufficient for 0.5–40 Hz bandpass |
| WFDB format: `.dat` + `.hea` pairs | `wfdb` library handles parsing |
| `ptbxl_database.csv` contains `strat_fold` | Pre-defined patient-level splits |
| Folds 1–8: training, Fold 9: validation, Fold 10: test | Recommended by dataset authors |
| Folds 9 and 10 are human-validated | Highest label quality — ideal for evaluation |
| Records in subfolders of 1000 | `00000/`, `01000/` ... `21000/` |
| `_lr` suffix = 100 Hz, `_hr` suffix = 500 Hz | `filename_lr` column in CSV points to correct file |
| SHA256SUMS.txt provided officially | Used for download verification |

**Why 100 Hz over 500 Hz:**

| Factor | 100 Hz | 500 Hz |
|---|---|---|
| Storage per record | ~23 KB | ~115 KB |
| Total dataset size | ~500 MB | ~2.5 GB |
| Sufficient for 40 Hz bandpass | ✅ | ✅ |
| Standard in ECG ML literature | ✅ | Less common |

---

## Pipeline Overview

| Script | Purpose |
|---|---|
| `setup_data.sh` | Creates folder structure, downloads all datasets, verifies checksums, generates download manifest |
| `preprocess.py` | Parses, cleans, resamples, windows, and saves all modalities as `.npy` arrays with metadata CSVs |
| `validate_outputs.py` | Checks array integrity, label distributions, subject leakage, and generates validation report |

---

## Usage
```bash
# Step 1 — Clone and set up environment
git clone https://github.com/MohamedSSalah1/multimodal-time-series.git
cd multimodal-time-series
conda env create -f environment.yml
conda activate biomedical_pipeline

# Step 2 — Download all datasets
bash setup_data.sh

# Step 3 — Run preprocessing
python preprocess.py

# Step 4 — Run validation
python validate_outputs.py
```

---

## Dataset Sources

| Dataset | Modality | Source |
|---|---|---|
| PAMAP2 | HAR | [UCI ML Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) |
| WISDM | HAR | [UCI ML Repository](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset) |
| EEGMMIDB | EEG | [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) |
| PTB-XL | ECG | [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) |