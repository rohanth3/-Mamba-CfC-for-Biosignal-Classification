import numpy as np

def get_db1_unified_label(exercise, label):
    """
    Maps 3 distinct Ninapro DB1 exercises into a unified 53-class space.
    Rest is always 0.
    E1: 1-12
    E2: 13-29
    E3: 30-52
    """
    if label == 0:
        return 0
    if exercise == 1:
        return label
    elif exercise == 2:
        return label + 12
    elif exercise == 3:
        return label + 29
    return label

def get_subject_splits(subjects, train_size, val_size, test_size, seed):
    """
    Given a list of subjects, deterministically split them using a seed.
    """
    rng = np.random.RandomState(seed)
    shuffled_subjects = rng.permutation(sorted(subjects))
    train_subs = shuffled_subjects[:train_size].tolist()
    val_subs = shuffled_subjects[train_size:train_size+val_size].tolist()
    test_subs = shuffled_subjects[train_size+val_size:train_size+val_size+test_size].tolist()
    return train_subs, val_subs, test_subs

def generate_grouped_kfold(subjects, k=5, seed=42):
    """
    Generates a list of (train_subs, test_subs) for K-Fold cross validation
    ensuring subjects are completely disjoint between train and test.
    """
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(sorted(subjects)).tolist()
    
    folds = []
    chunk_size = len(shuffled) // k
    for i in range(k):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != k - 1 else len(shuffled)
        test_subs = shuffled[start:end]
        train_subs = [s for s in shuffled if s not in test_subs]
        folds.append({"train": train_subs, "test": test_subs})
    return folds
