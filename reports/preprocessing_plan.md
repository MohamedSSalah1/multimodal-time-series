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
| PTB-XL | 100/500 Hz | 100 Hz | None | 100 Hz sufficient for 0.5–40 Hz bandpass; halves storage vs 500 Hz |

---

## 3. Window Definitions, Overlap, and Split Strategy

### HAR
### HAR
| Output | Window | Overlap | Labels | Shape | Windows Generated |
|---|---|---|---|---|---|
| Pretraining | 10 seconds | 0% | None | [N, 6, 200] | 19,542 combined (2,721 PAMAP2 + 16,821 WISDM) |
| Supervised | 5 seconds | 50% | Majority vote | [N, 6, 100] | 30,499 combined (4,478 PAMAP2 + 26,021 WISDM) |

Label assignment for supervised windows uses majority vote — the most frequent
label among all samples within a window is assigned as the window label.
This is robust to short label transitions at window boundaries.

Subject-level train/val/test split applied after windowing to prevent leakage.
PAMAP2 (9 subjects): 7 train, 1 val (S108), 1 test (S109).
WISDM (51 subjects): 41 train, 5 val (1646–1650), 5 test (1641–1645).

### EEG
| Output | Window | Overlap | Labels | Shape | Windows Generated |
|---|---|---|---|---|---|
| Event-aligned | 4 seconds | None | T0/T1/T2 event code | [N, 64, 640] | 8,254 total |

**T0 sampling strategy:** T0 (rest) windows are included alongside T1 and T2
imagery windows. Each T0 annotation is treated identically to T1/T2 —
a fixed 4-second window is extracted at T0 onset. T0 naturally alternates
with every imagery event in runs 4, 8, and 12, resulting in roughly twice
as many rest windows as imagery windows (rest: 4,007, left_fist: 2,138,
right_fist: 2,109). This class imbalance is documented and preserved in
metadata for downstream handling.

**Sampling rate note:** Three subjects (S088, S092, S100) had recordings at
128Hz rather than the expected 160Hz. These were automatically resampled
to 160Hz before preprocessing to ensure consistent output shape [N, 64, 640].
This is documented in the processing logs.

Subject-level split: train (7,357 windows, subjects 1-87), val (484,
subjects 88-98), test (413, subjects 99-109).

### ECG
| Output | Window | Overlap | Labels | Shape |
|---|---|---|---|---|
| Full record | 10 seconds | None | SCP diagnostic codes | [N, 12, 1000] |

Pre-defined patient-level folds used directly:
Folds 1–8 → training (further split into 8-fold cross-validation).
Fold 9 → validation. Fold 10 → test.

---

## 4. Missing Values, Null Labels, and Signal Cleaning

### PAMAP2
- **NaN interpolation:** linear interpolation for gaps of 1 second or less; segments with longer gaps excluded
- **Activity ID 0:** explicitly discarded per dataset documentation — covers transient periods between activities
- **Outlier clipping:** accelerometer clipped to ±16g; gyroscope clipped to ±2000°/s

- **Edge case — S104/running:** Subject 104 had only 1 sample of running
  after decimation — too few to process. Segment skipped and logged as a
  warning. This is expected given the realistic nature of the dataset.

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

---

## 5. Storage, RAM, and Compute Considerations

| Dataset | Raw Size | Processed Size | Peak RAM | Chunking Needed |
|---|---|---|---|---|
| PAMAP2 | 657MB | 13.8MB (npz) + 965KB (csv) | ~2 GB | No |
| WISDM | 296MB | 87MB (npz) + 5.9MB (csv) | ~4 GB | No |
| EEGMMIDB | ~500MB (runs 4/8/12 only) | 1.2GB (npz) + 3.2MB (csv) | ~6 GB | Yes — process per subject |
| PTB-XL | ~1.7GB (100 Hz) | TBC after processing | ~8 GB | Yes — process in batches |

EEG and ECG preprocessing is performed subject-by-subject and batch-by-batch respectively to keep peak RAM within acceptable bounds on a standard research laptop.

---

## 6. Main Engineering Risks

| Risk | Mitigation |
|---|---|
| Subject leakage across splits | Subject/patient IDs preserved in all metadata; splits enforced at subject level |
| PAMAP2 column misalignment | Column indices hardcoded and unit-tested against known header |
| WISDM semicolon parsing errors | Stripped before CSV parsing; malformed rows logged and dropped |
| EEG annotation misalignment | MNE annotation onset verified against raw sample index before windowing |
| PTB-XL duplicate records | Removed in v1.0.3 — using latest version |
| Silent NaN propagation | Explicit NaN/Inf check run on every output array before saving |
| HAR label imbalance | Label distribution reported in validation report |