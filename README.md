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

Each data folder contains a `.gitkeep` file — an empty placeholder that allows git to track the folder structure without tracking any data inside it. When you clone this repo, the full folder structure is immediately available and ready for `setup_data.sh` to populate.

### Where to find the processed sample pack

The representative processed sample pack (100 windows per dataset) is available at:
> 

---

## Pipeline Overview

## Usage

## Dataset Sources

| Dataset | Modality |
|---|---|
| PAMAP2 | HAR |
| WISDM | HAR |
| EEGMMIDB | EEG |
| PTB-XL | ECG |