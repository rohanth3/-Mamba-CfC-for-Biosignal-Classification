import os
import glob
import json
import yaml
import h5py
import hashlib
import time
import numpy as np
import scipy.io as sio
import sys
from datetime import datetime

# Adjust Python path to allow absolute imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.utils import get_db1_unified_label, get_subject_splits, generate_grouped_kfold

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
SPLITS_DIR = "splits"
REPORTS_DIR = "reports"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def main():
    # Load config
    config_path = os.path.join("configs", "preprocess_db1.yaml")
    with open(config_path, "r") as f:
        config_str = f.read()
        CONFIG = yaml.safe_load(config_str)
        
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()

    # Discover subjects
    mat_files = glob.glob(os.path.join(RAW_DIR, "*.mat"))
    if not mat_files:
        print("No .mat files found in data/raw.")
        return

    subjects = sorted(list(set([int(os.path.basename(f).split('_')[0][1:]) for f in mat_files])))
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    # Check if exactly 27 subjects (1..27) are present
    
    train_subs, val_subs, test_subs = get_subject_splits(
        subjects, 
        CONFIG["train_size"], 
        CONFIG["val_size"], 
        CONFIG["test_size"], 
        CONFIG["seed"]
    )
    
    splits_dict = {
        "train": train_subs,
        "val": val_subs,
        "test": test_subs
    }
    with open(os.path.join(SPLITS_DIR, "subject_splits.json"), "w") as f:
        json.dump(splits_dict, f, indent=4)
        
def compute_norm_stats(train_subs, mat_files):
    train_emg_list = []
    for f_path in mat_files:
        subj = int(os.path.basename(f_path).split('_')[0][1:])
        if subj in train_subs:
            data = sio.loadmat(f_path)
            emg = data['emg']  # continuous samples
            train_emg_list.append(emg)
    
    all_train_emg = np.vstack(train_emg_list)
    mean_vals = np.mean(all_train_emg, axis=0)
    std_vals = np.std(all_train_emg, axis=0)
    std_vals[std_vals == 0] = 1.0 # avoid div zero
    
    return {
        "mean": mean_vals.tolist(),
        "std": std_vals.tolist()
    }

def main():

    # Process and Window Data
    def process_split(split_name, subs):
        print(f"Processing split: {split_name}...", flush=True)
        X_list = []
        Y_list = []
        
        split_files = [f for f in mat_files if int(os.path.basename(f).split('_')[0][1:]) in subs]
        
        for f_path in split_files:
            data = sio.loadmat(f_path)
            emg = data['emg']
            restimulus = data['restimulus'].squeeze()
            rerepetition = data['rerepetition'].squeeze()
            subj = int(data['subject'][0,0])
            exer = int(data['exercise'][0,0])
            
            # Apply normalization
            emg = (emg - mean_vals) / std_vals
            
            wl = CONFIG["window_length"]
            st = CONFIG["stride"]
            
            if emg.shape[0] < wl:
                continue
                
            # Vectorized sliding window
            emg_windows = sliding_window_view(emg, window_shape=wl, axis=0)[::st]
            emg_windows = np.swapaxes(emg_windows, 1, 2)
            
            restimulus_windows = sliding_window_view(restimulus, window_shape=wl, axis=0)[::st]
            rerepetition_windows = sliding_window_view(rerepetition, window_shape=wl, axis=0)[::st]
            
            num_windows = emg_windows.shape[0]
            
            # Extract last sample labels
            final_labels_raw = restimulus_windows[:, -1]
            final_reps = rerepetition_windows[:, -1]
            
            # Label purity
            matches = (restimulus_windows == final_labels_raw[:, None])
            purities = np.sum(matches, axis=1) / wl
            
            # Map labels
            mapped_labels = np.array([get_db1_unified_label(exer, l) for l in final_labels_raw])
            
            start_indices = np.arange(0, num_windows * st, st)
            subj_col = np.full(num_windows, subj)
            exer_col = np.full(num_windows, exer)
            
            # Meta: [subject, exercise, repetition, label, label_purity, start_idx]
            meta_array = np.column_stack([
                subj_col, exer_col, final_reps, mapped_labels, purities, start_indices
            ])
            
            X_list.append(emg_windows)
            Y_list.append(meta_array)
                
        if len(X_list) == 0:
            print(f"No windows found for {split_name}.")
            return 0
            
        X_all = np.vstack(X_list).astype(np.float32)
        Y_all = np.vstack(Y_list).astype(np.float32)
        
        h5_path = os.path.join(PROCESSED_DIR, f"{split_name}.h5")
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("X", data=X_all, compression="gzip")
            hf.create_dataset("Y", data=Y_all, compression="gzip")
        print(f"Saved {split_name}.h5 with {X_all.shape[0]} windows.", flush=True)
        return X_all.shape[0]

    n_train = process_split("train", train_subs)
    n_val = process_split("val", val_subs)
    n_test = process_split("test", test_subs)
    
    # Save run info
    run_info = {
        "timestamp": datetime.now().isoformat(),
        "config_hash": config_hash,
        "seed": CONFIG["seed"],
        "package_versions": {
            "numpy": np.__version__,
            "h5py": h5py.__version__
        },
        "subjects_processed": len(subjects),
        "files_processed": len(mat_files),
        "windows_generated": {
            "train": n_train,
            "val": n_val,
            "test": n_test
        }
    }
    with open(os.path.join(REPORTS_DIR, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=4)
        
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
