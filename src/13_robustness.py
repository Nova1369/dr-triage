"""Step 13. Worse photos, a simpler pipeline, a stricter target
-> results/robustness.csv, results/ablations.csv and results/stricter_target.csv

Three questions about the rule C1 on APTOS, all tested as in step 6 (10 folds: train on eight, set the
thresholds on one, decide the last one).

1. Robustness. What if the photos to decide are worse than the photos the rule was built on?
   The model and the thresholds come from the original photos; the test photos are blurred, darkened or
   compressed (features of step 12). Then the same again with the thresholds set on calibration photos
   that are worse in the same way, which is what recalibration (step 8) would do at such a clinic.

2. Ablations. What does each part of the pipeline add? One part is changed at a time:
   no PCA, fewer or more PCA components, no standardising, weaker regularisation, half the training photos.

3. A stricter target. The rule lets through at most alpha = 10 % of each class. What do 20 %, 5 % and 2 % cost?

Run from the project folder:   python src/13_robustness.py        (about 3 minutes; needs step 12)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal import GRADER, NO_REFERRAL, ROUTINE, URGENT, class_thresholds, decide, prediction_sets

SEED = 42
CHANGES = {"blur": "blurred", "dark": "half brightness", "jpeg": "JPEG quality 20"}
MEASURES = ["AUROC referable", "AUROC urgent", "true class in the set %", "referable eyes sent home % (target <= 10)",
            "urgent eyes referred urgently % (target >= 90)", "healthy eyes sent home %", "to a grader %",
            "referrals per 1,000 eyes"]

table = pd.read_csv("data/table.csv")
aptos = (table["dataset"] == "aptos").to_numpy()
X = {"original": np.load("data/features.npy")[aptos]}
for name in CHANGES:
    X[name] = np.load(f"data/features_{name}.npy")
y, fold = table["label"][aptos].to_numpy(), table["fold"][aptos].to_numpy()


def pipeline(pca=100, standardise=True, C=0.01):
    steps = ([StandardScaler()] if standardise else []) + ([PCA(pca, random_state=SEED)] if pca else [])
    return make_pipeline(*steps, LogisticRegression(C=C, max_iter=5000))


def measures(p, sets):
    """The numbers of MEASURES from every photo's probabilities and prediction set."""
    d = decide(sets)
    return [f"{roc_auc_score(y >= 1, 1 - p[:, 0]):.3f}", f"{roc_auc_score(y == 2, p[:, 2]):.3f}",
            *[f"{value:.1f}" for value in (100 * sets[np.arange(len(y)), y].mean(), 100 * np.mean(d[y >= 1] == NO_REFERRAL),
                                           100 * np.mean(d[y == 2] == URGENT), 100 * np.mean(d[y == 0] == NO_REFERRAL),
                                           100 * np.mean(d == GRADER))],
            f"{1000 * np.mean((d == ROUTINE) | (d == URGENT)):.0f}"]


def out_of_fold(make_model, test_photos="original", calibration_photos="original", alpha=0.1, keep=1.0):
    """Probabilities and prediction sets for every photo. The model is always trained on original photos."""
    p, sets = np.zeros((len(y), 3)), np.zeros((len(y), 3), dtype=bool)
    rng = np.random.RandomState(SEED)
    for k in range(10):
        test, cal = fold == k, fold == (k + 1) % 10
        train = np.flatnonzero(~test & ~cal)
        train = rng.permutation(train)[: int(keep * len(train))]              # keep < 1: fewer training photos
        model = make_model().fit(X["original"][train], y[train])
        p[test] = model.predict_proba(X[test_photos][test])
        thresholds = class_thresholds(model.predict_proba(X[calibration_photos][cal]), y[cal], alpha)
        sets[test] = prediction_sets(p[test], thresholds)
    return p, sets


# ---- 1. robustness ----
rows = [["original", "original", *measures(*out_of_fold(pipeline))]]
for name, label in CHANGES.items():
    rows.append([label, "original", *measures(*out_of_fold(pipeline, test_photos=name))])
    rows.append([label, "changed in the same way", *measures(*out_of_fold(pipeline, test_photos=name, calibration_photos=name))])
robustness = pd.DataFrame(rows, columns=["test photos", "thresholds set on photos that are", *MEASURES])
robustness.to_csv("results/robustness.csv", index=False)
print("1. Worse photos\n" + robustness.T.to_string(header=False), flush=True)

# ---- 2. ablations ----
variants = {"the pipeline as it is (standardise, PCA 100, C = 0.01)": {},
            "PCA with 20 components": {"make_model": lambda: pipeline(pca=20)},
            "PCA with 300 components": {"make_model": lambda: pipeline(pca=300)},
            "no PCA (all 2,048 features)": {"make_model": lambda: pipeline(pca=None)},
            "no standardising": {"make_model": lambda: pipeline(standardise=False)},
            "weaker regularisation (C = 1)": {"make_model": lambda: pipeline(C=1)},
            "half of the training photos": {"keep": 0.5},
            "a tenth of the training photos": {"keep": 0.1}}
rows = []
for name, change in variants.items():
    rows.append([name, *measures(*out_of_fold(change.get("make_model", pipeline), keep=change.get("keep", 1.0)))])
    print(f"  done: {name}", flush=True)
ablations = pd.DataFrame(rows, columns=["pipeline", *MEASURES])
ablations.to_csv("results/ablations.csv", index=False)
print("\n2. One part changed at a time\n" + ablations.to_string(index=False))

# ---- 3. a stricter target ----
rows = [[f"{100 * alpha:.0f}", *measures(*out_of_fold(pipeline, alpha=alpha))[2:]] for alpha in [0.2, 0.1, 0.05, 0.02]]
target = pd.DataFrame(rows, columns=["allowed miss rate alpha %", *MEASURES[2:]])
target.to_csv("results/stricter_target.csv", index=False)
print("\n3. Other targets\n" + target.to_string(index=False))
