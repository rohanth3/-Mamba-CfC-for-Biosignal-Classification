import os
from pathlib import Path
import scipy.io as sio
import numpy as np

def initial_inspect():
    raw_dir = Path(r"C:\Users\Rohanth Ganesha C\Documents\Capstone\Project\data\raw")
    mat_files = list(raw_dir.glob("*.mat"))
    
    if not mat_files:
        print("No .mat files found yet.")
        return
        
    print(f"Total .mat files found: {len(mat_files)}")
    
    # Check what exercises exist for Subject 1 (assuming S1 exists)
    s1_files = sorted(raw_dir.glob("S1_*.mat"))
    print("\nFiles for Subject 1:")
    for f in s1_files:
        print(f.name)
        
    if s1_files:
        print(f"\nInspecting {s1_files[0].name}:")
        data = sio.loadmat(str(s1_files[0]))
        
        # print keys ignoring meta keys
        keys = [k for k in data.keys() if not k.startswith("__")]
        print(f"Keys: {keys}")
        
        for k in keys:
            if isinstance(data[k], np.ndarray):
                print(f" - {k}: shape {data[k].shape}, dtype {data[k].dtype}")
            else:
                print(f" - {k}: {type(data[k])}")

if __name__ == "__main__":
    initial_inspect()
