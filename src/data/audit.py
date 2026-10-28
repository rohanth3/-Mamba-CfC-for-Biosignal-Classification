import numpy as np
import scipy.io as sio
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def get_db1_unified_label(exercise, label):
    """
    Map exercise-specific labels to a unified label space.
    E1: 1-12
    E2: 1-17 (mapped to 13-29)
    E3: 1-23 (mapped to 30-52)
    Rest: 0 -> 0
    """
    if label == 0:
        return 0
    if exercise == 1:
        return label
    elif exercise == 2:
        return label + 12
    elif exercise == 3:
        return label + 12 + 17
    else:
        raise ValueError(f"Unknown exercise {exercise}")

def run_audit(data_dir: str, figures_dir: str, reports_dir: str):
    data_path = Path(data_dir)
    fig_path = Path(figures_dir)
    rep_path = Path(reports_dir)
    
    mat_files = list(data_path.glob("*.mat"))
    subjects = sorted(list(set([int(f.name.split('_')[0][1:]) for f in mat_files])))
    
    summary = {
        "num_subjects": len(subjects),
        "missing_subjects": [s for s in range(1, 28) if s not in subjects],
        "total_files": len(mat_files),
        "channels": 10,
        "sampling_rate": {
            "value": "Unknown from files (assumed 100Hz from docs)",
            "verified": False
        },
        "class_counts": {},
        "class_counts_samples": {},
        "exercise_reps": {int(s): {1:0, 2:0, 3:0} for s in subjects}
    }
    
    # Track samples for plotting
    sample_signals = {}  # Store one S, one rep for closeups
    all_amplitudes = []
    psd_data = []
    
    print("Scanning files to compute statistics...")
    for i, f in enumerate(mat_files):
        print(f"Processing file {i+1}/{len(mat_files)}: {f.name}", flush=True)
        data = sio.loadmat(str(f))
        subj = int(data['subject'][0,0])
        exer = int(data['exercise'][0,0])
        emg = data['emg']
        restimulus = data['restimulus'].flatten()
        rerepetition = data['rerepetition'].flatten()
        
        # Repetition heatmap tracking
        reps = len(np.unique(rerepetition[rerepetition > 0]))
        summary["exercise_reps"][subj][exer] = reps
        
        # Sample extraction for plots
        if subj == 1 and exer == 1:
            sample_signals['raw_s1_e1'] = (emg, restimulus)
            
        # Extract a single movement repetition (e.g. class 1)
        if 'closeup' not in sample_signals and exer == 1:
            rep_mask = (rerepetition == 1) & (restimulus == 1)
            if np.any(rep_mask):
                sample_signals['closeup'] = emg[rep_mask]
                
        # Extract same movement across subjects (e.g. class 1, rep 1)
        if exer == 1 and 'movement_subj' not in sample_signals:
            sample_signals['movement_subj'] = {}
        if exer == 1 and subj in [1, 2, 3] and subj not in sample_signals.get('movement_subj', {}):
            rep_mask = (rerepetition == 1) & (restimulus == 1)
            if np.any(rep_mask):
                sample_signals['movement_subj'][subj] = emg[rep_mask]
                
        # Aggregate classes
        for label in np.unique(restimulus):
            unified_lbl = int(get_db1_unified_label(exer, label))
            count = np.sum(restimulus == label)
            summary["class_counts_samples"][unified_lbl] = summary["class_counts_samples"].get(unified_lbl, 0) + int(count)
            
        # Sample amplitudes (take a subset to save memory)
        all_amplitudes.append(emg[::10].flatten())
        
        # PSD subset (first 10 seconds if available)
        if len(psd_data) < 5:
            psd_data.append(emg[:1000]) # 10 seconds at 100Hz

    # Generate Figures
    print("Generating Figures...")
    
    # a) Multichannel raw EMG with labels shaded
    if 'raw_s1_e1' in sample_signals:
        plt.figure(figsize=(15, 6))
        emg_sig, lbl = sample_signals['raw_s1_e1']
        t = np.arange(len(emg_sig))
        for c in range(10):
            plt.plot(t, emg_sig[:, c] + c * 0.05, label=f"Ch {c+1}", alpha=0.7)
        # Shade movement regions
        active = (lbl > 0)
        from itertools import groupby
        regions = []
        idx = 0
        for k, g in groupby(active):
            length = len(list(g))
            if k:
                regions.append((idx, idx + length))
            idx += length
        for (start, end) in regions:
            plt.axvspan(start, end, color='red', alpha=0.1)
        plt.title("Subject 1, Exercise 1: Raw EMG with Movements Shaded")
        plt.xlabel("Samples")
        plt.ylabel("Channels (offset)")
        plt.savefig(fig_path / "raw_emg_shaded.png", dpi=300, bbox_inches='tight')
        plt.close()

    # b) Closeup of one repetition
    if 'closeup' in sample_signals:
        plt.figure(figsize=(10, 4))
        emg_sig = sample_signals['closeup']
        plt.plot(emg_sig)
        plt.title("Close-up of One Repetition (Class 1)")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.savefig(fig_path / "closeup_repetition.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    # c) Same movement across 3 subjects
    if 'movement_subj' in sample_signals:
        plt.figure(figsize=(12, 6))
        for i, (subj, emg_sig) in enumerate(sample_signals['movement_subj'].items()):
            plt.subplot(3, 1, i+1)
            plt.plot(emg_sig)
            plt.title(f"Subject {subj} - Class 1")
            plt.ylabel("Amp")
        plt.tight_layout()
        plt.savefig(fig_path / "movement_across_subjects.png", dpi=300, bbox_inches='tight')
        plt.close()

    # d) Amplitude distribution
    plt.figure(figsize=(10, 5))
    amps = np.concatenate(all_amplitudes)
    sns.histplot(amps[np.abs(amps) < 0.2], bins=100, kde=True) # Trim extreme outliers for plotting
    plt.title("Per-channel Amplitude Distribution (trimmed for visibility)")
    plt.xlabel("Amplitude")
    plt.savefig(fig_path / "amplitude_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # e) PSD
    plt.figure(figsize=(10, 5))
    fs = 100 # Assumed
    for c in range(10):
        # Average PSD across a few samples
        pxx = []
        for d in psd_data:
            f, p = plt.mlab.psd(d[:, c], NFFT=256, Fs=fs)
            pxx.append(p)
        plt.semilogy(f, np.mean(pxx, axis=0), label=f"Ch {c+1}")
    plt.axvline(fs/2, color='r', linestyle='--', label='Nyquist')
    plt.title("Power Spectrum (PSD) per channel")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency")
    plt.legend()
    plt.savefig(fig_path / "psd.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # f) Class-count bar chart
    plt.figure(figsize=(12, 5))
    lbls = list(summary["class_counts_samples"].keys())
    counts = list(summary["class_counts_samples"].values())
    plt.bar(lbls, counts)
    plt.title("Class Count (Samples)")
    plt.xlabel("Unified Class Label (0 = Rest)")
    plt.ylabel("Total Samples")
    plt.savefig(fig_path / "class_counts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # g) Repetitions Heatmap
    heatmap_data = np.zeros((len(subjects), 3))
    for i, subj in enumerate(subjects):
        for exer in [1, 2, 3]:
            heatmap_data[i, exer-1] = summary["exercise_reps"][subj][exer]
            
    plt.figure(figsize=(8, 10))
    sns.heatmap(heatmap_data, annot=True, xticklabels=["E1", "E2", "E3"], yticklabels=subjects, cmap="YlGnBu")
    plt.title("Repetitions per Subject/Exercise")
    plt.xlabel("Exercise")
    plt.ylabel("Subject ID")
    plt.savefig(fig_path / "reps_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save JSON summary
    with open(rep_path / "data_audit_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    print("Audit summary and figures generated.")

if __name__ == "__main__":
    run_audit(
        r"C:\Users\Rohanth Ganesha C\Documents\Capstone\Project\data\raw",
        r"C:\Users\Rohanth Ganesha C\Documents\Capstone\Project\figures",
        r"C:\Users\Rohanth Ganesha C\Documents\Capstone\Project\reports"
    )
