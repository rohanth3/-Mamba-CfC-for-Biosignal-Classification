import os
import subprocess
from datetime import datetime

header = f"""Test Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Test Assertions:
1. test_split_generation: Asserts same seed gives same split, different seed gives different split. Asserts train/val/test are pairwise disjoint and union is exactly subjects 1..27.
2. test_kfold_helper: Asserts k-fold splits are pairwise disjoint between train and test per fold, and union is all subjects.
3. test_shapes_labels_disjoint: Asserts no NaN/inf, labels are in 0..52, array shapes match YAML config, no window spans two recordings, and no subject id appears in more than one split's metadata.
4. test_train_normalization_sanity: Asserts normalized training sample has mean ~0 and std ~1.
5. test_no_normalization_leakage: Asserts fit function correctly ignores val/test files, and perturbing val data does not alter the saved normalization stats.

======================================================================
"""

result = subprocess.run([r"C:\Users\Rohanth Ganesha C\Documents\Capstone\Project\.venv\Scripts\pytest", "tests/test_data.py", "-v"], capture_output=True, text=True)

with open(r"C:\Users\Rohanth Ganesha C\Documents\Capstone\Project\reports\test_results.txt", "w") as f:
    f.write(header)
    f.write(result.stdout)
    f.write(result.stderr)

print("Test results written.")
