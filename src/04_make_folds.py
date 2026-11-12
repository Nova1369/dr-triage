"""Step 4. Find copies of the same photo and split APTOS into 10 folds -> data/table.csv

Why copies matter: APTOS contains some photos more than once, sometimes with different grades. If one copy
were used for training and the other for testing, the test would be too easy. Copies must stay together.

How copies are found: two photos are copies when their features from step 3 are almost identical, that is,
when the cosine similarity of the two feature vectors is above 0.995. Different photos, even of similar
eyes, stay clearly below that. Photos linked by such pairs form a group (if A is a copy of B and B is a
copy of C, all three are one group). A photo without copies is a group of its own.

The folds: APTOS is split into 10 folds so that every fold has about the same mix of the five DR grades
and a group is never split across two folds. This is done like dealing cards: the groups are shuffled
(with a fixed seed, 42, so the split is the same on every run), sorted by grade, and dealt out one by one
to fold 0, 1, ..., 9, 0, 1, ... IDRiD and Messidor-2 get no folds (fold = -1): they play the new clinics.

The result, data/table.csv, is the one table that every later step reads: one row per photo, in the same
order as the rows of data/features.npy.

Run from the project folder:   python src/04_make_folds.py
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

COPY = 0.995     # cosine similarity above which two photos count as copies
FOLDS = 10
SEED = 42

photos = pd.read_csv("data/photos.csv")
stats = pd.read_csv("data/photo_stats.csv")
features = np.load("data/features.npy")
n = len(photos)
assert len(stats) == n and len(features) == n, "run steps 2 and 3 to the end first"

# cosine similarity of every pair of photos
unit = features / np.linalg.norm(features, axis=1, keepdims=True)
similarity = unit @ unit.T
first, second = np.where(np.triu(similarity > COPY, k=1))        # each pair of copies once

# groups: photos connected through pairs of copies
links = csr_matrix((np.ones(len(first)), (first, second)), shape=(n, n))
n_groups, group = connected_components(links, directed=False)

table = pd.concat([photos, stats], axis=1)
table["group"] = group

# 10 folds on APTOS: deal the groups out like cards, grade by grade
aptos = table[table["dataset"] == "aptos"]
grade_of_group = aptos.groupby("group")["grade"].max()                     # copies can differ: take the highest grade
shuffled = grade_of_group.iloc[np.random.RandomState(SEED).permutation(len(grade_of_group))]
dealt = shuffled.sort_values(kind="stable")                                # by grade; ties keep the shuffled order
fold_of_group = pd.Series(np.arange(len(dealt)) % FOLDS, index=dealt.index)
table["fold"] = table["group"].map(fold_of_group).fillna(-1).astype(int)   # -1 for IDRiD and Messidor-2

table.to_csv("data/table.csv", index=False)

# what was found
sizes = table["group"].map(table["group"].value_counts())
copies = table[sizes > 1]
per_group = copies.groupby("group")
print(f"Pairs of copies: {len(first)}; photos with at least one copy: {len(copies)}, in {per_group.ngroups} groups")
print("  by dataset:", copies["dataset"].value_counts().to_dict())
print(f"  groups whose copies carry different grades: {(per_group['grade'].nunique() > 1).sum()}")
print(f"  groups that span two datasets: {(per_group['dataset'].nunique() > 1).sum()}")
print("\nAPTOS folds (photos per DR grade):")
print(table[table.fold >= 0].groupby(["fold", "grade"]).size().unstack())
print(f"\n{n} photos in data/table.csv")
