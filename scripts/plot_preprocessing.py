import os
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio

import matplotlib
matplotlib.use('Agg')

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SPLITS_DIR = "splits"
FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)

def plot_before_after():
    with open(os.path.join(SPLITS_DIR, "subject_splits.json"), "r") as f:
        subs = json.load(f)
    train_subj = subs["train"][0]
    
    raw_path = os.path.join(RAW_DIR, f"S{train_subj}_A1_E1.mat")
    data = sio.loadmat(raw_path)
    raw_data = data['emg']
    restimulus = data['restimulus'].squeeze()
    
    with open(os.path.join(PROCESSED_DIR, "norm_stats.json"), "r") as f:
        stats = json.load(f)
    mean_v = np.array(stats["mean"])
    std_v = np.array(stats["std"])
    
    prep_data = (raw_data - mean_v) / std_v
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    sns.lineplot(data=raw_data[1000:1500, :3], ax=axes[0], dashes=False)
    axes[0].set_title(f"Unnormalized vs z-score normalized (100 Hz envelope, filter: none)\nUnnormalized (Subject {train_subj}, E1, First 3 Channels)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(['Ch 1', 'Ch 2', 'Ch 3'], loc='upper right')
    
    for i in range(1000, 1500):
        if restimulus[i] > 0:
            axes[0].axvspan(i - 1000, i - 1000 + 1, color='red', alpha=0.1, lw=0)
            axes[1].axvspan(i - 1000, i - 1000 + 1, color='red', alpha=0.1, lw=0)
    
    sns.lineplot(data=prep_data[1000:1500, :3], ax=axes[1], dashes=False)
    axes[1].set_title(f"Z-Score Normalized EMG (Subject {train_subj}, E1, First 3 Channels)")
    axes[1].set_ylabel("Normalized Amplitude")
    axes[1].set_xlabel("Time (Samples, 100Hz)")
    axes[1].legend(['Ch 1', 'Ch 2', 'Ch 3'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "before_after_preprocessing.png"), dpi=150)
    plt.close()

def plot_windowing_diagram():
    subj = 9
    raw_path = os.path.join(RAW_DIR, f"S{subj}_A1_E1.mat")
    data = sio.loadmat(raw_path)
    emg = data['emg']
    restimulus = data['restimulus'].squeeze()
    
    onset = np.where(restimulus > 0)[0]
    start_idx = onset[0] - 35 if len(onset) > 0 else 1000
    
    sig = emg[start_idx:start_idx+60, 0]
    restim_slice = restimulus[start_idx:start_idx+60]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(np.arange(60), sig, color='gray', label='Continuous Signal (Ch 1)')
    
    wl = 30
    st = 10
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, c in enumerate(colors):
        w_start = i * st
        w_end = w_start + wl
        last_label = restim_slice[w_end-1]
        matches = np.sum(restim_slice[w_start:w_end] == last_label)
        purity = matches / wl
        
        ymin = 0.1 + i*0.2
        ymax = 0.25 + i*0.2
        ax1.axvspan(w_start, w_end-1, ymin=ymin, ymax=ymax, color=c, alpha=0.6, label=f'W{i+1}')
        ax1.plot(w_end-1, sig[w_end-1], 'ro', markersize=8)
        
        # Annotate each window with its label and label_purity
        ax1.text(w_start + 2, (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * (ymax + ymin)/2 + ax1.get_ylim()[0], 
                 f'Lbl: {last_label}, Pur: {purity:.2f}', color='black', fontsize=9, va='center')
        
    ax1.set_title("Window Extraction (30 samples length, 10 samples stride) & Last-Sample Labeling (Red Dot)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    ax2.plot(np.arange(60), restim_slice, 'r-', drawstyle='steps-post')
    ax2.set_ylabel("Restimulus Label")
    ax2.set_xlabel("Time (Samples at 100Hz)")
    ax2.set_ylim(-0.5, np.max(restim_slice)+1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "window_extraction_diagram.png"), dpi=150)
    plt.close()

def plot_subject_splits():
    with open(os.path.join(SPLITS_DIR, "subject_splits.json"), "r") as f:
        subs = json.load(f)
        
    split_map = {}
    for s in subs["train"]: split_map[s] = 0
    for s in subs["val"]: split_map[s] = 1
    for s in subs["test"]: split_map[s] = 2
    
    subject_ids = np.arange(1, 28)
    split_assignments = np.array([split_map[s] for s in subject_ids])
    
    fig, ax = plt.subplots(figsize=(12, 1.5))
    cmap = matplotlib.colors.ListedColormap(['#2ecc71', '#f1c40f', '#e74c3c'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(split_assignments.reshape(1, -1), cmap=cmap, norm=norm, aspect='auto')
    
    # White separators
    for i in range(26):
        ax.axvline(i + 0.5, color='white', linewidth=2)
    
    ax.set_yticks([])
    ax.set_xticks(np.arange(27))
    ax.set_xticklabels(subject_ids)
    
    import matplotlib.patches as mpatches
    train_patch = mpatches.Patch(color='#2ecc71', label=f'Train ({len(subs["train"])})')
    val_patch = mpatches.Patch(color='#f1c40f', label=f'Val ({len(subs["val"])})')
    test_patch = mpatches.Patch(color='#e74c3c', label=f'Test ({len(subs["test"])})')
    ax.legend(handles=[train_patch, val_patch, test_patch], loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "subject_split_diagram.png"), dpi=150)
    plt.close()

def plot_class_distribution():
    splits = ["train", "val", "test"]
    dist = {}
    rest_pct = {}
    
    for s in splits:
        with h5py.File(os.path.join(PROCESSED_DIR, f"{s}.h5"), "r") as hf:
            Y = hf["Y"][:]
            labels = Y[:, 3]
            counts = np.bincount(labels.astype(int), minlength=53)
            dist[s] = counts
            rest_pct[s] = (counts[0] / len(labels)) * 100
            
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [3, 1]})
    
    for i, s in enumerate(splits):
        # Full distribution
        axes[i, 0].bar(range(53), dist[s], color=sns.color_palette("muted")[i])
        axes[i, 0].set_title(f"{s.capitalize()} Split (Rest = {rest_pct[s]:.1f}%)")
        axes[i, 0].set_ylabel("Count")
        if i == 2:
            axes[i, 0].set_xlabel("Unified Class Label (0 = Rest, 1-52 = Movements)")
            
        # Movements only (1-52)
        axes[i, 1].bar(range(1, 53), dist[s][1:], color=sns.color_palette("muted")[i])
        axes[i, 1].set_title(f"Movements Only (1-52)")
        if i == 2:
            axes[i, 1].set_xlabel("Class Label")
            
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_distribution_per_split.png"), dpi=150)
    plt.close()

def plot_normalization_stats():
    with open(os.path.join(PROCESSED_DIR, "norm_stats.json"), "r") as f:
        stats = json.load(f)
        
    means = stats["mean"]
    stds = stats["std"]
    channels = [f"Ch {i}" for i in range(1, 11)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.bar(channels, means, color='#3498db')
    ax1.set_ylabel('Mean Amplitude')
    ax1.set_title("Normalization Statistics\n18 training subjects, raw samples")
    
    ax2.bar(channels, stds, color='#e74c3c')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_xlabel('EMG Channel')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "normalization_statistics.png"), dpi=150)
    plt.close()

def main():
    print("Generating slide-ready figures...")
    plot_before_after()
    plot_windowing_diagram()
    plot_subject_splits()
    plot_class_distribution()
    plot_normalization_stats()
    print("Done!")

if __name__ == "__main__":
    main()
