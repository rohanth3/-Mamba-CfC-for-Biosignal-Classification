import nbformat as nbf
from pathlib import Path

def generate_notebook():
    nb = nbf.v4.new_notebook()
    
    md_intro = """\
# Phase 1: Data Audit
This notebook imports the generated audit statistics and displays the key figures for the Mamba-CfC study.
"""
    
    code_summary = """\
import json
import pandas as pd

# Load Audit Summary
with open('../reports/data_audit_summary.json', 'r') as f:
    summary = json.load(f)

print(f"Total Subjects: {summary['num_subjects']}")
print(f"Missing Subjects: {summary['missing_subjects']}")
print(f"Total Files: {summary['total_files']}")
print(f"Channels: {summary['channels']}")
print(f"Sampling Rate: {summary['sampling_rate']['value']}")

# Display Class Counts
counts = summary['class_counts_samples']
df_counts = pd.DataFrame(list(counts.items()), columns=['Unified_Class', 'Samples'])
df_counts = df_counts.sort_values('Samples', ascending=False)
display(df_counts.head(10))
"""

    md_findings = """\
## Findings & Open Questions
- **Missing Subject:** Subject 6 is entirely missing (files s1 to s27 exist, minus s6). We found 26 subjects.
- **Sampling Rate Verification:** We could not automatically verify the sampling rate from the `.mat` file contents (no time vector or explicit `Fs` key exists). We assume 100 Hz based on the Ninapro DB1 documentation, but we need confirmation.
- **Data Quality:** The data contains no obvious NaNs. The `restimulus` (re-aligned labels) has a massive class imbalance dominated by class 0 (Rest).
- **Filtering Options:** DB1 signals are recorded at 100 Hz. Given this low sampling rate and the signal appearance, these are likely **envelope-detected (RMS or rectified low-pass)** signals, not raw 2kHz EMG. 
    - **Option A (No filtering):** Leave as is, as it's already pre-processed by the hardware.
    - **Option B (Smoothing):** Apply a light moving average or Savitzky-Golay filter to remove artifact spikes.
    - **NOT Recommended:** 50/60 Hz Notch or standard 20-500 Hz Bandpass (impossible at 100 Hz sampling rate).

## Figures

### Multichannel Raw EMG (with Movements Shaded)
![Raw EMG](../figures/raw_emg_shaded.png)

### Repetition Close-up
![Close-up](../figures/closeup_repetition.png)

### Movement Across Subjects (Class 1)
![Movement Subjects](../figures/movement_across_subjects.png)

### Amplitude Distribution
![Amp Dist](../figures/amplitude_distribution.png)

### Power Spectrum (PSD)
![PSD](../figures/psd.png)

### Class Counts
![Class Counts](../figures/class_counts.png)

### Repetitions per Subject/Exercise
![Repetitions Heatmap](../figures/reps_heatmap.png)
"""

    nb['cells'] = [
        nbf.v4.new_markdown_cell(md_intro),
        nbf.v4.new_code_cell(code_summary),
        nbf.v4.new_markdown_cell(md_findings)
    ]
    
    out_path = Path(r"C:\Users\Rohanth Ganesha C\Documents\Capstone\Project\notebooks\01_data_audit.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        nbf.write(nb, f)
        
    print(f"Notebook written to {out_path}")

if __name__ == '__main__':
    generate_notebook()
