import numpy as np
from collections import Counter


def fraction_db_ofir(db, fraction=0.5):  # val-train split
    np_labels = np.array(db.targets)
    counter = Counter(np_labels)
    db_subset_idx_a = []
    db_subset_idx_b = []
    for lbl in counter:
        db_subset_idx_a.extend(np.arange(0, len(np_labels))[np_labels == lbl][:int(fraction * counter[lbl])])
        db_subset_idx_b.extend(np.arange(0, len(np_labels))[np_labels == lbl][int(fraction * counter[lbl]):])

    return db_subset_idx_a, db_subset_idx_b

def fraction_db(db, fraction=0.5):  # val-train split
    perm_idx = np.random.permutation(np.arange(len(db)))
    cutoff = int(fraction)*len(db)
    return perm_idx[:cutoff],  perm_idx[cutoff:]


