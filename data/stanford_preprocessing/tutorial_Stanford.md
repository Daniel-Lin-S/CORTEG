# Tutorial: Preprocessing Stanford FingerFlex Dataset

This document provides a detailed tutorial for preprocessing the Stanford FingerFlex dataset and preparing it for a neural decoding prediction task.

## 1. Dataset Acquisition
The Stanford FingerFlex dataset consists of ECoG recordings from 9 subjects while they performed finger flexion tasks.
* **Download Link:** [Stanford-FingerFlex: zk881ps0522](https://searchworks.stanford.edu/view/zk881ps0522)
* **Storage:** Keep the raw subject folders in a directory of your choice and set `CORTEG_STANFORD_RAW_ROOT` to that directory, for example `/data/lyh/neural_data/ecog/fingerflex/data`. The preprocessing script reads that directory without modifying it.

## 2. Environment Setup
The pipeline requires two distinct environments:

### MATLAB Environment
* **Software:** MATLAB R2022b.
* **Toolbox:** FieldTrip (version fieldtrip-20230926).

### Python Environment
* **Toolbox:** MNE (v1.11.0).
* **Libraries:** `scipy`, `numpy`, `pickle`.

---

## 3. Step 1: MATLAB Signal Preprocessing
Run the script `data_preprocessing_Stanford.m` to clean the raw ECoG signals.

### Key Operations:
* **Scaling:** Data is scaled by a factor of 0.0298 to adjust units. 
* **Line Noise Removal:** Bandstop filters are applied at 60Hz, 120Hz, and 180Hz (bandwidth of 1Hz each) using a Butterworth filter.
* **Automatic Bad Channel Selection:** After notch filtering, the script computes variance per channel on the chronological training portion only. Channels with variance above `varianceHighMultiplier × median variance` or below `varianceLowMultiplier × median variance` are removed. The defaults are `5` and `0.2`; change them in `configs/stanford_preprocessing.json` using the `overrides.bad_channel` object. The resolved JSON is saved as `preprocessed_data/Stanford/resolved_config.json`. The median uses all channels. The removed labels and thresholds are saved inside the cleaned `.mat` file.
* **Common Average Referencing (CAR):** All remaining channels are re-referenced to the global average to reduce common noise.
* **Input and output:** Set `CORTEG_STANFORD_RAW_ROOT` to the raw downloaded `data/` directory containing `<subject>/<subject>_fingerflex.mat`. The raw files are read only. Cleaned `.mat` files are saved under the repository’s `preprocessed_data/Stanford/` directory. The script reuses existing cleaned files only when their saved provenance matches the resolved configuration. No separate electrode-location file is created: the cleaned `channelName` selects coordinates from raw `locs` when needed.

---

## 4. Step 2: Python Task Formatting & Feature Extraction
Run `prepare_taskFormatedData_Stanford.py` to prepare the data for machine learning models.

### Feature Engineering (`HGALFS_feature_extractor`):
For each epoch, the following features are extracted from 1-second ECoG windows:
1.  **High Gamma Activity (HGA):**
    * Filtered between 70Hz and 200Hz.
    * Envelope extracted via Hilbert transform.
    * Downsampled to 200 time points ($T=200$).
2.  **Low Frequency Signal (LFS):**
    * Raw ECoG signal downsampled to 200Hz. 
3.  **Temporal Alignment:**
    * A **40ms delay** is applied between the neural window and the target trajectory to account for delay.

### Dataset Splitting:
The script splits the data chronologically:
* At or above `split.duration_cutoff_seconds` (600): `split.at_or_above_cutoff.train_seconds` (400) is the training duration.
* Below the cutoff: `split.below_cutoff.train_parts` (2) of `split.below_cutoff.total_parts` (3) equal whole-sample parts are training.

---

## 5. Final Output Data Structure
The processing generates a `.pkl` file for each subject containing a list with the following components:

| Variable | Description | Dimensions |
| :--- | :--- | :--- |
| `ECoG_train` | Combined HGA and LFS features (Train) | `[nEpoch, nChannel, 200, 2]` |
| `trajectory_train` | Finger flexion ground truth (Train) | `[nEpoch, 5]` |
| `ECoG_test` | Combined HGA and LFS features (Test) | `[nEpoch, nChannel, 200, 2]` |
| `trajectory_test` | Finger flexion ground truth (Test) | `[nEpoch, 5]` |
| `ECoG_train_128` | ECoG downsampled to 128Hz (Train) | `[nEpoch, nChannel, 128]` |
| `ECoG_test_128` | ECoG downsampled to 128Hz (Test) | `[nEpoch, nChannel, 128]` |

---
**References:**
1. Miller, Kai J. "A library of human electrocorticographic data and analyses." Nature human behaviour 3.11 (2019): 1225-1235.
2. Miller, Kai J., et al. "Human motor cortical activity is selectively phase-entrained on underlying rhythms." (2012): e1002655.
