# Resource Estimate
Generated: 2026-04-08 18:59:42

---

## Raw Storage

| Dataset | Size (MB) |
|---|---|
| pamap2 | 2,965 |
| wisdm | 1,486 |
| eegmmidb | 795 |
| ptbxl | 4,796 |
| **Total** | **10,042** |

Note: Raw sizes include both zip archives and extracted files. The zip files can be deleted after extraction to recover approximately half the space.

---

## Processed Storage

| Modality | Size (MB) |
|---|---|
| har | 214 |
| eeg | 1,202 |
| ecg | 931 |
| **Total** | **2,347** |

---

## RAM Usage

The table below shows RAM metrics captured during validation. These are provided for context — only peak RAM per modality is directly relevant to running the pipeline.

| Metric | Value | Relevance |
|---|---|---|
| Current process RAM | 107 MB | RAM used by the validation script itself — not representative of preprocessing peak |
| Total system RAM | 18 GB | The machine this pipeline was developed and tested on |
| Available RAM at validation time | 3.63 GB | Snapshot during validation — not the peak during preprocessing |

**Important:** The figures above were captured during `validate_outputs.py` which is lightweight. The peak RAM figures below are the relevant ones for running the full pipeline.

---

## Peak RAM by Modality

These are estimates based on observed behaviour during preprocessing runs.

| Modality | Script | Peak RAM Estimate | Notes |
|---|---|---|---|
| HAR | `preprocess.py --har` | ~2–4 GB | PAMAP2 and WISDM loaded fully into memory; no chunking needed given dataset size |
| EEG | `preprocess.py --eeg` | ~4–6 GB | Processed subject by subject to limit RAM; loading all 327 EDF files at once would exceed 8 GB |
| ECG | `preprocess.py --ecg` | ~6–8 GB | 21,799 records processed sequentially; peak occurs when stacking final array before saving |

**Minimum recommended RAM to run full pipeline: 8 GB**
**Recommended RAM for comfortable headroom: 16 GB**

---

## Runtime Estimates

These are based on actual measured runtimes on an Apple M-series MacBook with standard broadband.

| Script | Measured / Estimated Runtime | Notes |
|---|---|---|
| `bash setup_data.sh` | ~5 hours total | EEGMMIDB ~1.5 hrs (327 individual file downloads), PTB-XL ~3 hrs (1.7 GB zip), PAMAP2 + WISDM ~15 mins |
| `python preprocess.py` | ~5–10 minutes total | HAR ~2 mins, EEG ~2 mins, ECG ~5 mins |
| `python validate_outputs.py` | ~22 seconds | Measured: 21.61s |
| `python generate_submission_sample.py` | ~10 seconds | |

---

## Chunking and Streaming

| Modality | Approach | Reason |
|---|---|---|
| HAR | Full in-memory | Dataset small enough — PAMAP2 545K rows, WISDM 3.4M rows fit comfortably in RAM |
| EEG | Per-subject chunking | 109 subjects × 3 runs processed one at a time; avoids loading all EDF files simultaneously |
| ECG | Per-record streaming | 21,799 records loaded and processed individually; final array stacked once at end |