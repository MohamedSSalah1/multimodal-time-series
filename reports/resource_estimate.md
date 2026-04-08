# Resource Estimate
Generated: 2026-04-08 18:59:42

---

## Raw Storage

| Dataset | Size (MB) |
|---|---|
| pamap2 | 2965.16 |
| wisdm | 1486.46 |
| eegmmidb | 795.56 |
| ptbxl | 4796.33 |
| **Total** | **10043.51** |

## Processed Storage

| Modality | Size (MB) |
|---|---|
| har | 214.33 |
| eeg | 1202.42 |
| ecg | 931.65 |
| **Total** | **2348.4** |

## RAM Usage

| Metric | Value |
|---|---|
| Current process RAM | 107.84 MB |
| Total system RAM | 18.0 GB |
| Available RAM | 3.63 GB |

## Peak RAM Estimates by Modality

| Modality | Peak RAM Estimate |
|---|---|
| HAR | ~2-4 GB — processed in memory, no chunking needed |
| EEG | ~6 GB — processed per subject to limit RAM |
| ECG | ~8 GB — processed record by record |

## Runtime Estimates

| Script | Estimated Runtime |
|---|---|
| setup_data.sh | ~5 hours total (EEGMMIDB ~1.5hrs, PTB-XL ~3hrs) |
| preprocess.py | ~5-10 minutes total |
| validate_outputs.py | ~1-2 minutes |
