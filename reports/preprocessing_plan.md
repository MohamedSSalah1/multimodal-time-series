# Preprocessing Plan
## Multimodal Biosignal Pipeline — HAR, EEG, ECG

---

## 1. Channel Schema

| Modality | Dataset | Channels | Justification |
|---|---|---|---|
| HAR | PAMAP2 | Wrist acc_x/y/z (±16g), gyr_x/y/z | ±16g recommended by dataset authors; ±6g saturates at high impact |
| HAR | WISDM | Watch acc_x/y/z, gyr_x/y/z | Watch sensor is anatomically comparable to PAMAP2 wrist IMU |
| HAR | Combined | 6 channels: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z] | Shared schema enables single downstream model across both datasets |
| EEG | EEGMMIDB | All 64 channels (10-10 system) | Full montage retained — channel selection deferred to downstream model |
| ECG | PTB-XL | All 12 leads (I, II, III, AVL, AVR, AVF, V1–V6) | Standard clinical lead set — no justification for dropping leads |

---

## 2. Sampling Rate and Resampling

| Dataset | Native Hz | Target Hz | Method | Justification |
|---|---|---|---|---|
| PAMAP2 | 100 Hz | 20 Hz | Scipy decimate (factor 5) | Brief mandates 20 Hz; decimation preferred over naive downsampling to avoid aliasing |
| WISDM | 20 Hz | 20 Hz | None | Already at target rate |
| EEGMMIDB | 160 Hz | 160 Hz | None | Native rate retained; Nyquist (80 Hz) comfortably covers 40 Hz bandpass |
| EEGMMIDB (S088, S092, S100) | 128 Hz | 160 Hz | MNE resample | Three subjects recorded at non-standard rate; resampled to ensure consistent output shape |
| PTB-XL | 100/500 Hz | 100 Hz | None | 100 Hz sufficient for 0.5–40 Hz bandpass; halves storage vs 500 Hz |

---

## 3. Window Definitions, Overlap, and Split Strategy

### HAR
| Output | Window | Overlap | Labels | Shape | Windows Generated |
|---|---|---|---|---|---|
| Pretraining | 10 seconds | 0% | None | [N, 6, 200] | 19,542 combined (2,721 PAMAP2 + 16,821 WISDM) |
| Supervised | 5 seconds | 50% | Majority vote | [N, 6, 100] | 30,499 combined (4,478 PAMAP2 + 26,021 WISDM) |

Label assignment for supervised windows uses majority vote, the most frequent label among all samples within a window is assigned as the window label. This is robust to short label transitions at window boundaries.

**Split strategy — HAR:**
PAMAP2 (9 subjects): 7 train, 1 val (S108), 1 test (S109).
WISDM (51 subjects): 41 train, 5 val (IDs 1646–1650), 5 test (IDs 1641–1645).

Splits are assigned at the subject level to prevent data leakage, all windows from a given subject appear in exactly one split. For PAMAP2, the small subject count (9) means a single subject per split is the only practical option. For WISDM, approximately 10% of subjects are held out for val and test respectively, which is standard practice for activity recognition benchmarks.

**Edge case — S104/running:** Subject 104 had only 1 sample of running after decimation — too few to process. Segment skipped and logged as a warning.

### EEG
| Output | Window | Overlap | Labels | Shape | Windows Generated |
|---|---|---|---|---|---|
| Event-aligned | 4 seconds | None | T0/T1/T2 event code | [N, 64, 640] | 8,254 total |

**T0 sampling strategy:** T0 (rest) windows are included alongside T1 and T2 imagery windows. Each T0 annotation is treated identically to T1/T2 — a fixed 4-second window is extracted at T0 onset. T0 naturally alternates with every imagery event in runs 4, 8, and 12, resulting in roughly twice as many rest windows as imagery windows (rest: 4,007, left_fist: 2,138, right_fist: 2,109).

**Sampling rate note:** Three subjects (S088, S092, S100) had recordings at 128 Hz rather than the expected 160 Hz. These were automatically resampled to 160 Hz before preprocessing to ensure consistent output shape [N, 64, 640].

**Split strategy — EEG:**
Subjects 1–87: train (7,357 windows), Subjects 88–98: val (484 windows), Subjects 99–109: test (413 windows).

An 80/10/10 subject-level split was applied. Subjects are treated as independent units — all runs from a given subject appear in exactly one split. The slight imbalance between val and test window counts reflects natural variation in the number of clean windows extracted per subject after artefact rejection.

### ECG
| Output | Window | Overlap | Labels | Shape | Records |
|---|---|---|---|---|---|
| Full record | 10 seconds | None | SCP diagnostic codes | [N, 12, 1000] | 21,799 |

Pre-defined patient-level folds used directly from strat_fold column.
Folds 1–8: train (17,418), Fold 9: val (2,183), Fold 10: test (2,198).
0 records skipped — all 21,799 records processed successfully.

**Split strategy — ECG:**
PTB-XL provides 10 pre-defined stratified folds in the strat_fold column, where all records from the same patient are guaranteed to be in the same fold. This is the split strategy recommended by the dataset authors. Folds 9 and 10 are used for val and test respectively as they contain the highest proportion of human-validated labels. The remaining 8 folds serve as both the training set and cross-validation folds for downstream model selection.

**Cross-validation:** The 8 training folds (strat_fold 1–8) serve directly as pre-defined cross-validation folds — all records from the same patient are guaranteed to be in the same fold, preventing patient-level leakage.

---

## 4. Missing Values, Null Labels, and Signal Cleaning

### PAMAP2
- **NaN interpolation:** linear interpolation for gaps of 1 second or less; segments with longer gaps excluded
- **Activity ID 0:** explicitly discarded per dataset documentation — covers transient periods between activities
- **Outlier clipping:** accelerometer clipped to ±16g; gyroscope clipped to ±2000°/s
- **Edge case:** S104/running had only 1 sample after decimation — skipped and logged

### WISDM
- **Semicolon terminator:** stripped from each line during parsing
- **Missing values:** no explicit NaN indicator — verified during parsing; any malformed rows dropped
- **Label strategy:** all 18 activity codes retained for pretraining; only shared activities (walking, running, stairs, sitting, standing, computer_work, folding_laundry) used for supervised output

### EEGMMIDB
- **Bandpass filter:** 1–40 Hz (4th order Butterworth, zero-phase) — removes slow drift and high-frequency noise
- **Notch filter:** 50 Hz — removes UK mains electrical interference
- **Re-referencing:** average reference — standard for motor imagery EEG
- **Corrupted segments:** channels with flat signal or amplitude > 500 µV excluded
- **Runs used:** 4, 8, 12 only — consistent left/right fist imagery across all three repetitions

### ECG (PTB-XL)
- **Bandpass filter:** 0.5–40 Hz — removes baseline wander and high-frequency noise
- **Normalisation:** z-score per lead per record — removes amplitude scale differences between patients
- **No beat segmentation:** full 10-second records used — avoids introducing beat detection errors
- **QC flags:** signal quality metadata (static_noise, burst_noise, baseline_drift, electrodes_problems) preserved in output metadata for downstream filtering

---

## 5. Storage, RAM, and Compute Considerations

| Dataset | Raw Size | Processed Size | Peak RAM | Chunking Needed |
|---|---|---|---|---|
| PAMAP2 | 657 MB | 13.8 MB (npz) + 965 KB (csv) | ~2 GB | No |
| WISDM | 296 MB | 87 MB (npz) + 5.9 MB (csv) | ~4 GB | No |
| EEGMMIDB | ~500 MB (runs 4/8/12 only) | 1.2 GB (npz) + 3.2 MB (csv) | ~6 GB | Yes — processed per subject |
| PTB-XL | ~1.7 GB (100 Hz) | 928 MB (npz) + 3.9 MB (csv) | ~8 GB | No — processed record by record |

EEG preprocessing is performed subject-by-subject to keep peak RAM within acceptable bounds on a standard research laptop.

---

## 6. Main Engineering Risks

| Risk | Mitigation |
|---|---|
| Subject leakage across splits | Subject/patient IDs preserved in all metadata; splits enforced at subject level |
| PAMAP2 column misalignment | Column indices hardcoded and verified against known header |
| WISDM semicolon parsing errors | Stripped before CSV parsing; malformed rows logged and dropped |
| EEG annotation misalignment | MNE annotation onset verified against raw sample index before windowing |
| PTB-XL duplicate records | Removed in v1.0.3 — using latest version |
| Silent NaN propagation | Explicit NaN/Inf check run on every output array before saving |
| HAR label imbalance | Label distribution reported in validation report |
| EEGMMIDB mixed sampling rates | Auto-detected and resampled to 160 Hz before preprocessing |

---

## 7. Evaluation — What Could Be Improved

### Split strategy improvements
The current subject-level splits are functional but have limitations worth noting:

**PAMAP2** has only 9 subjects, meaning the val and test sets each contain exactly 1 subject. This makes evaluation highly sensitive to individual subject characteristics. A leave-one-subject-out (LOSO) cross-validation scheme would be more statistically robust, though computationally more expensive.

**WISDM** has 51 subjects with a roughly 80/10/10 split. While adequate, the split is deterministic rather than stratified by activity distribution across subjects. A stratified split ensuring each activity is proportionally represented across train/val/test would reduce evaluation variance.

**EEG** uses a consecutive subject split (1–87 train, 88–98 val, 99–109 test). This assumes subjects are exchangeable — if there is any systematic difference between early and late recordings (e.g. equipment calibration drift over time), this could introduce bias. A random subject-level split would be preferable.

### Class imbalance
All three modalities show class imbalance:

- **HAR:** WISDM contributes ~6x more windows than PAMAP2 in the combined output
- **EEG:** Rest (T0) windows are roughly twice as frequent as left/right fist imagery
- **ECG:** NORM accounts for ~42% of records

Potential mitigations include:
- **Oversampling:** techniques such as SMOTE (Synthetic Minority Oversampling Technique) could be applied to minority classes at the window level, though this should be done only within the training split to avoid leakage
- **Weighted loss functions:** downstream models could use class-weighted loss functions rather than modifying the data itself
- **Undersampling:** randomly removing majority class windows to balance distributions, at the cost of reduced training data

These improvements are left to downstream modelling rather than implemented at the preprocessing stage, as the brief specifies preparing datasets ready for analysis rather than model balanced datasets.