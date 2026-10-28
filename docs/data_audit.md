# Data Audit (Phase 1)

## Findings

### 1. Subject and Exercise Inventory
- **Total Subjects Found:** 26
- **Missing Subjects:** Subject 6 is missing from the dataset (s6.zip does not exist in the raw directory).
- **Exercises per Subject:** Each subject has exactly 3 exercises (E1, E2, E3).
- **Total `.mat` files:** 78

### 2. File Contents and Array Shapes
A typical file contains the following arrays:
- `emg`: shape (N, 10), where N is the number of samples. There are 10 EMG channels.
- `stimulus`, `restimulus`: shape (N, 1), representing movement labels (0 is rest, >0 are movement classes).
- `repetition`, `rerepetition`: shape (N, 1), representing the movement repetition number.
- `glove`: shape (N, 22), tracking 22 kinematic glove sensors. We are ignoring this per the problem scope.

### 3. Sampling Rate
- The `.mat` files do not explicitly store the sampling rate.
- According to official Ninapro DB1 documentation, the data is sampled at **100 Hz**. We must assume this value.

### 4. Label Structure
Labels across exercises have been remapped to a unified label space to avoid overlap:
- **Rest:** 0
- **Exercise 1 (12 movements):** Mapped to 1-12
- **Exercise 2 (17 movements):** Mapped to 13-29
- **Exercise 3 (23 movements):** Mapped to 30-52
Total functional classes = 52, plus Rest = 53 total classes.

### 5. Class Balance and Data Quality
- **Class Balance:** There is a heavy imbalance, with the `rest` class (0) massively dominating the sample count compared to active movements.
- **Data Quality:** No obvious NaNs or corrupted channels were found in the initial check.

### 6. Filtering Options
At a 100 Hz sampling rate, the signal is already heavily downsampled. Raw surface EMG typically contains frequency content up to 500 Hz. The Ninapro DB1 data (recorded with Otto Bock MyoBock hardware) outputs an envelope-detected signal (rectified and low-pass filtered natively by the sensors). 
- **Bandpass Filtering (e.g., 20-500 Hz):** Impossible, as the Nyquist frequency is 50 Hz.
- **Notch Filtering (50/60 Hz):** Impossible/meaningless near Nyquist.
- **Our Options:**
  - **Option A:** No filtering. Use the data as-is, as it's already an envelope. (Recommended)
  - **Option B:** Light smoothing (e.g., Moving Average or Savitzky-Golay) to reduce spike artifacts.

## Summary of Generated Figures
(Available in the `notebooks/01_data_audit.ipynb` and `figures/` directory)
- **`raw_emg_shaded.png`**: Multichannel EMG for one subject with movement labels shaded.
- **`closeup_repetition.png`**: Close-up of one repetition across channels.
- **`movement_across_subjects.png`**: The same movement (Class 1) across 3 different subjects.
- **`amplitude_distribution.png`**: Per-channel amplitude distribution.
- **`psd.png`**: Power spectrum (PSD) per channel group.
- **`class_counts.png`**: Class-count bar chart showing the rest class dominance.
- **`reps_heatmap.png`**: Heatmap of repetitions per subject/exercise.
