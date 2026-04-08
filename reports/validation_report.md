# Validation Report
Generated: 2026-04-08 18:59:42

---

## HAR — PAMAP2 + WISDM

### Array Integrity

| Dataset | Shape | Dtype | NaNs | Infs | Size (MB) |
|---|---|---|---|---|---|
| pamap2_pretrain | [2721, 6, 200] | float32 | 0 | 0 | 9.67 |
| pamap2_supervised | [4478, 6, 100] | float32 | 0 | 0 | 4.07 |
| wisdm_pretrain | [16821, 6, 200] | float32 | 0 | 0 | 61.89 |
| wisdm_supervised | [26021, 6, 100] | float32 | 0 | 0 | 24.5 |
| combined_pretrain | [19542, 6, 200] | float32 | 0 | 0 | 71.56 |
| combined_supervised | [30499, 6, 100] | float32 | 0 | 0 | 28.57 |

### Shape Checks

| Check | Result |
|---|---|
| pamap2_pretrain_channels | ✅ PASS |
| pamap2_pretrain_samples | ✅ PASS |
| pamap2_supervised_channels | ✅ PASS |
| pamap2_supervised_samples | ✅ PASS |
| wisdm_pretrain_channels | ✅ PASS |
| wisdm_pretrain_samples | ✅ PASS |
| wisdm_supervised_channels | ✅ PASS |
| wisdm_supervised_samples | ✅ PASS |

### HAR Harmonisation

| Dataset | Sampling Rate (Hz) | Channels | Channel Schema |
|---|---|---|---|
| pamap2 | 20 | 6 | acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z |
| wisdm | 20 | 6 | acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z |

### Null Label Check (PAMAP2 class 0)

Result: ✅ PASS — class 0 absent

Unique labels: ['computer_work', 'folding_laundry', 'running', 'sitting', 'standing', 'walking']

### Leakage Control

| Check | Value |
|---|---|
| Train subjects | 48 |
| Val subjects | 6 |
| Test subjects | 6 |
| Train/test leakage | ✅ PASS (0 subjects) |

### Split Distribution

| Split | Windows |
|---|---|
| train | 24391 |
| val | 3314 |
| test | 2794 |

---

## EEG — EEGMMIDB

### Array Integrity

| Shape | Dtype | NaNs | Infs | Size (MB) |
|---|---|---|---|---|
| [8254, 64, 640] | float32 | 0 | 0 | 1199.18 |

### Shape Checks

| Check | Result |
|---|---|
| channels_64 | ✅ PASS |
| samples_640 | ✅ PASS |
| dtype_float32 | ✅ PASS |

### Annotation Checks

| Check | Result |
|---|---|
| T0_present | ✅ PASS |
| T1_present | ✅ PASS |
| T2_present | ✅ PASS |

| Event | Windows |
|---|---|
| rest | 4007 |
| left_fist | 2138 |
| right_fist | 2109 |

### Runs Check

Runs present: [4, 8, 12] — ✅ PASS

### Leakage Control

Train/test leakage: ✅ PASS (0 subjects)

| Split | Windows |
|---|---|
| train | 7357 |
| val | 484 |
| test | 413 |

---

## ECG — PTB-XL

### Array Integrity

| Shape | Dtype | NaNs | Infs | Size (MB) |
|---|---|---|---|---|
| [21799, 12, 1000] | float32 | 0 | 0 | 927.78 |

### Shape Checks

| Check | Result |
|---|---|
| leads_12 | ✅ PASS |
| samples_1000 | ✅ PASS |
| dtype_float32 | ✅ PASS |

### ECG Fold Metadata

| Fold | Records |
|---|---|
| 1 | 2175 |
| 2 | 2181 |
| 3 | 2192 |
| 4 | 2174 |
| 5 | 2174 |
| 6 | 2173 |
| 7 | 2176 |
| 8 | 2173 |
| 9 | 2183 |
| 10 | 2198 |

### Split Distribution

| Split | Records |
|---|---|
| train | 17418 |
| test | 2198 |
| val | 2183 |

### Leakage Control (Patient Level)

Train/test patient leakage: ✅ PASS (0 patients)

### Top 10 Diagnostic Labels

| Label | Records |
|---|---|
| NORM | 9134 |
| IMI | 1677 |
| NDT | 1613 |
| ASMI | 1457 |
| LVH | 1141 |
| LAFB | 888 |
| IRBBB | 831 |
| CLBBB | 527 |
| NST_ | 502 |
| CRBBB | 385 |

---

## Summary

| Check | Result |
|---|---|
| Overall | ✅ ALL CHECKS PASSED (34/34) |
| HAR NaN/Inf | ✅ PASS |
| EEG NaN/Inf | ✅ PASS |
| ECG NaN/Inf | ✅ PASS |
| HAR leakage | ✅ PASS |
| EEG leakage | ✅ PASS |
| ECG leakage | ✅ PASS |
