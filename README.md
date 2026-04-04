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

## Pipeline Overview

## Usage

## Dataset Sources

| Dataset | Modality |
|---|---|
| PAMAP2 | HAR |
| WISDM | HAR |
| EEGMMIDB | EEG |
| PTB-XL | ECG |

## Project Structure

