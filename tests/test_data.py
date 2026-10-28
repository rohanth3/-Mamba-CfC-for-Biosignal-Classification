import os
import json
import h5py
import yaml
import numpy as np
import scipy.io as sio
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.utils import get_subject_splits, generate_grouped_kfold
from src.data.preprocess import compute_norm_stats

PROCESSED_DIR = os.path.join("data", "processed")
SPLITS_DIR = "splits"
CONFIG_PATH = os.path.join("configs", "preprocess_db1.yaml")
RAW_DIR = os.path.join("data", "raw")

def test_split_generation():
    """Same seed gives same split, different gives different; union exactly 1..27."""
    subjects = list(range(1, 28))
    t1, v1, ts1 = get_subject_splits(subjects, 18, 4, 5, seed=42)
    t2, v2, ts2 = get_subject_splits(subjects, 18, 4, 5, seed=42)
    t3, v3, ts3 = get_subject_splits(subjects, 18, 4, 5, seed=999)
    assert t1 == t2, "Same seed must give same split"
    assert t1 != t3, "Different seed must give different split"

    assert len(t1) == 18
    assert len(v1) == 4
    assert len(ts1) == 5
    assert len(set(t1) & set(v1)) == 0, "Train and Val must be disjoint"
    assert len(set(t1) & set(ts1)) == 0, "Train and Test must be disjoint"
    assert len(set(v1) & set(ts1)) == 0, "Val and Test must be disjoint"
    assert set(t1 + v1 + ts1) == set(subjects), "Union must be exactly subjects 1..27"

def test_kfold_helper():
    """Grouped K-fold train/test split disjointness."""
    subjects = list(range(1, 28))
    folds = generate_grouped_kfold(subjects, k=5, seed=42)
    assert len(folds) == 5
    for fold in folds:
        train = fold["train"]
        test = fold["test"]
        assert len(set(train) & set(test)) == 0, "K-Fold train and test must be disjoint"
        assert set(train + test) == set(subjects), "K-Fold train + test must equal all subjects"

def test_shapes_labels_disjoint():
    """No window overlaps boundaries. Correct counts and starts."""
    assert os.path.exists(os.path.join(PROCESSED_DIR, "train.h5"))
    assert os.path.exists(os.path.join(PROCESSED_DIR, "val.h5"))
    assert os.path.exists(os.path.join(PROCESSED_DIR, "test.h5"))
    
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    wl = config["window_length"]
    stride = config["stride"]

    train_subs = set()
    val_subs = set()
    test_subs = set()

    for split, sub_set in [("train", train_subs), ("val", val_subs), ("test", test_subs)]:
        with h5py.File(os.path.join(PROCESSED_DIR, f"{split}.h5"), "r") as hf:
            Y = hf["Y"][:]
            X = hf["X"][:]
            
            assert X.shape[1] == wl, f"Window length mismatch in {split}"
            assert X.shape[2] == 10, f"Channel mismatch in {split}"
            assert not np.isnan(X).any(), f"NaNs found in {split}"
            assert not np.isinf(X).any(), f"Infs found in {split}"
            
            labels = Y[:, 3]
            assert labels.min() >= 0 and labels.max() <= 52, f"Labels out of bounds in {split}"
            
            unique_subs = set(Y[:, 0])
            sub_set.update(unique_subs)

            # Check starts against true recording length
            import glob
            # Get unique recordings in this split
            recs = list(set((row[0], row[1]) for row in Y))
            for sub, ex in recs:
                rec_rows = Y[(Y[:,0] == sub) & (Y[:,1] == ex)]
                starts = rec_rows[:, 5]
                assert np.all(starts == np.arange(0, len(starts) * stride, stride)), "Starts must be 0, stride, 2*stride..."
                
                # Fetch real length
                file_path = glob.glob(os.path.join(RAW_DIR, f"S{int(sub)}_A1_E{int(ex)}.mat"))[0]
                mat = sio.loadmat(file_path)
                L = mat['emg'].shape[0]
                
                assert len(starts) == (L - wl) // stride + 1, "Window count mismatch"
                assert starts[-1] + wl <= L, "Last window exceeds recording bounds"
            
    assert len(train_subs & val_subs) == 0, "Leakage: Train and Val have overlapping subjects at window level"
    assert len(train_subs & test_subs) == 0, "Leakage: Train and Test have overlapping subjects at window level"
    assert len(val_subs & test_subs) == 0, "Leakage: Val and Test have overlapping subjects at window level"

def test_data_integrity():
    """For 1000 random windows, assert X equals scaled raw data and labels are correct."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    wl = config["window_length"]
    
    with open(os.path.join(PROCESSED_DIR, "norm_stats.json"), "r") as f:
        stats = json.load(f)
    train_mean = np.array(stats["mean"])
    train_std = np.array(stats["std"])

    import glob
    for split in ["train", "val", "test"]:
        with h5py.File(os.path.join(PROCESSED_DIR, f"{split}.h5"), "r") as hf:
            Y = hf["Y"][:]
            X = hf["X"][:]
            
            # Select random indices
            np.random.seed(42)
            n_samples = min(333, len(Y))
            indices = np.random.choice(len(Y), n_samples, replace=False)
            
            for idx in indices:
                sub = int(Y[idx][0])
                ex = int(Y[idx][1])
                rep = int(Y[idx][2])
                label = int(Y[idx][3])
                purity = float(Y[idx][4])
                start = int(Y[idx][5])
                
                x_val = X[idx]
                
                # Find raw file
                file_path = glob.glob(os.path.join(RAW_DIR, f"S{sub}_A1_E{ex}.mat"))[0]
                mat = sio.loadmat(file_path)
                raw_emg = mat['emg']
                restim = mat['restimulus']
                
                # Verify scaling
                raw_window = raw_emg[start:start+wl]
                scaled_window = (raw_window - train_mean) / train_std
                assert np.allclose(x_val, scaled_window, atol=1e-4), "Data scaling integrity failed"
                
                # Verify label (at start+wl-1)
                expected_label = restim[start+wl-1, 0]
                # Label mapping: E1: 1-12. E2: 13-29 (if E2, expected is expected + 12). E3: 30-52 (expected + 29).
                # Actually, our mapping adds offsets for E2 and E3. Let's just do a naive check if it's > 0.
                if expected_label > 0:
                    if ex == 2: expected_label += 12
                    elif ex == 3: expected_label += 29
                assert expected_label == label, "Label mapping integrity failed"

def test_subject_ids_match_splits():
    """Test subject IDs in h5 match splits/subject_splits.json exactly."""
    with open(os.path.join(SPLITS_DIR, "subject_splits.json"), "r") as f:
        splits = json.load(f)
    
    total_recs = 0
    for split_name in ["train", "val", "test"]:
        expected_subs = set(splits[split_name])
        with h5py.File(os.path.join(PROCESSED_DIR, f"{split_name}.h5"), "r") as hf:
            Y = hf["Y"][:]
            actual_subs = set(Y[:, 0])
            assert expected_subs == actual_subs, f"Subject mismatch in {split_name}"
            # Count recordings
            recs = set((row[0], row[1]) for row in Y)
            total_recs += len(recs)
            
    assert total_recs == 81, "Exactly 81 (sub, ex) recordings must be present"

def test_compute_norm_stats_integration():
    """Recompute compute_norm_stats and assert matches saved json."""
    with open(os.path.join(SPLITS_DIR, "subject_splits.json"), "r") as f:
        splits = json.load(f)
    train_subs = splits["train"]
    
    import glob
    all_files = glob.glob(os.path.join(RAW_DIR, "*.mat"))
    stats = compute_norm_stats(train_subs, all_files)
    
    with open(os.path.join(PROCESSED_DIR, "norm_stats.json"), "r") as f:
        saved_stats = json.load(f)
        
    assert np.allclose(stats["mean"], saved_stats["mean"]), "Integration: mean changed"
    assert np.allclose(stats["std"], saved_stats["std"]), "Integration: std changed"

def test_train_normalization_sanity():
    """Check normalized train data has mean ~0 and std ~1."""
    train_path = os.path.join(PROCESSED_DIR, "train.h5")
    with h5py.File(train_path, "r") as hf:
        X_sample = hf["X"][::100]
        X_flat = X_sample.reshape(-1, 10)
        sample_mean = np.mean(X_flat, axis=0)
        sample_std = np.std(X_flat, axis=0)
        
        assert np.allclose(sample_mean, 0, atol=0.1), "Train mean not close to 0"
        assert np.allclose(sample_std, 1, atol=0.1), "Train std not close to 1"

def test_no_normalization_leakage():
    """Fits stats on train, perturbs val, refits, stats identical. Also perturb train changes stats."""
    temp_dir = tempfile.mkdtemp()
    try:
        train_subs = [1, 2]
        val_subs = [3]
        
        train_files = []
        for s in train_subs:
            f_path = os.path.join(temp_dir, f"S{s}_A1_E1.mat")
            sio.savemat(f_path, {'emg': np.random.randn(100, 10)})
            train_files.append(f_path)
            
        val_file = os.path.join(temp_dir, "S3_A1_E1.mat")
        sio.savemat(val_file, {'emg': np.random.randn(100, 10)})
        
        all_files = train_files + [val_file]
        
        loaded_files = []
        original_loadmat = sio.loadmat
        def mock_loadmat(f_path):
            loaded_files.append(f_path)
            return original_loadmat(f_path)
            
        import unittest.mock
        with unittest.mock.patch('scipy.io.loadmat', side_effect=mock_loadmat):
            stats_before = compute_norm_stats(train_subs, all_files)
            
        assert val_file not in loaded_files, "Leakage: Validation file was loaded by compute_norm_stats!"
        for tf in train_files:
            assert tf in loaded_files, "Train file was not loaded by compute_norm_stats!"
        
        # Perturb val heavily
        sio.savemat(val_file, {'emg': np.random.randn(100, 10) * 1000 + 500})
        stats_after_val_perturb = compute_norm_stats(train_subs, all_files)
        
        assert np.allclose(stats_before["mean"], stats_after_val_perturb["mean"]), "Leakage: mean changed!"
        assert np.allclose(stats_before["std"], stats_after_val_perturb["std"]), "Leakage: std changed!"
        
        # Positive control: Perturb train file
        sio.savemat(train_files[0], {'emg': np.random.randn(100, 10) * 1000 + 500})
        stats_after_train_perturb = compute_norm_stats(train_subs, all_files)
        
        assert not np.allclose(stats_before["mean"], stats_after_train_perturb["mean"]), "Positive control failed: mean identical after train perturb"
        
    finally:
        shutil.rmtree(temp_dir)
