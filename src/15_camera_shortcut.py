"""Step 15. ADDITIONAL WORK (not in the proposal), part A. The camera shortcut, and a guarantee for each group
-> results/camera_shortcut.csv and results/group_thresholds.csv

Step 10 found that the rule C1 meets its target on average (4.4 % of referable eyes sent home) but not on
the cameras that photographed mostly healthy eyes (53 %). Conformal prediction promises an average over
all eyes of a class; it promises nothing for a group. Two questions.

1. Does the model use the camera as a shortcut?  Three tests:
     a. can the camera be read from the features (logistic regression, 10 folds)?
     b. does an eye of the same DR grade get a different probability on a different camera?
     c. how well does the model rank referable eyes inside one camera group, where the camera cannot help?

2. Can the promise be kept for each group?  Five rules, all with the same folds:
     C1         one threshold per class (the rule so far)
     camera     one threshold per class and per camera group. Needs the camera: the best case
     cluster    (ours) one threshold per class and per k-means cluster of the features (step 14).
                Needs no camera and no labels to form the groups
     reweight   the model is trained again with every (cluster, class) pair counting the same, so that
                "this camera has mostly healthy eyes" stops paying; then C1. A standard way to remove a shortcut
     no home    C1, but no eye of the cluster with the fewest referable eyes is sent home by the rule: a grader
                sees it. The simple safe rule
   A threshold needs at least 9 calibration photos of its class (src/conformal.py). The troubled cameras have
   only 55 referable eyes in all of APTOS, so here three folds are used for calibration instead of one
   (train on six, calibrate on three, decide one), and a class with fewer than 9 calibration photos in a
   group uses the threshold of all groups together.

As in step 10, the photo size stands for the camera, and cameras are grouped by what they photographed:
"mostly healthy" (under 10 % of the eyes need a referral), "mostly sick" (over 50 %), "mixed".
The camera groups are used to judge every rule; only the rule "camera" may use them to decide.

Run from the project folder:   python src/15_camera_shortcut.py        (about a minute)
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal import GRADER, NO_REFERRAL, ROUTINE, URGENT, class_thresholds, conformal_threshold, decide, prediction_sets

ALPHA = 0.1
SEED = 42
GROUPS = ["mostly healthy", "mixed", "mostly sick"]
RULES = {"C1": "C1 one threshold per class", "camera": "thresholds per camera group (needs the camera)",
         "cluster": "thresholds per k-means cluster (ours)", "reweight": "model reweighted by cluster and class, then C1",
         "no home": "C1, no eye of the suspicious cluster sent home"}

table = pd.read_csv("data/table.csv")
aptos = table[table["dataset"] == "aptos"].reset_index(drop=True)
X = np.load("data/features.npy")[(table["dataset"] == "aptos").to_numpy()]
y, fold, grade = aptos["label"].to_numpy(), aptos["fold"].to_numpy(), aptos["grade"].to_numpy()
camera = (aptos["width"].astype(str) + " x " + aptos["height"].astype(str)).to_numpy()
referable_share = (aptos["label"] >= 1).groupby(camera).transform("mean")
camera_group = np.select([referable_share < 0.1, referable_share > 0.5], [0, 2], 1)        # index into GROUPS


def wilson(k, n):
    """95 % interval for a share of k out of n, in percent."""
    z, share = 1.96, k / n
    middle = (share + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, 100 * (middle - half)), 100 * (middle + half)


def group_thresholds(p, labels, group, overall):
    """One row of thresholds per group; a class with fewer than 9 photos in a group keeps the overall threshold."""
    thresholds = np.tile(overall, (3, 1))
    for g in range(3):
        for c in range(3):
            scores = 1 - p[(group == g) & (labels == c), c]
            if np.isfinite(conformal_threshold(scores, ALPHA)):
                thresholds[g, c] = conformal_threshold(scores, ALPHA)
    return thresholds


def equal_weights(group, labels):
    """Weights that make every (group, class) pair count the same in training."""
    weight = np.ones(len(labels))
    for g in np.unique(group):
        for c in range(3):
            pair = (group == g) & (labels == c)
            if pair.any():
                weight[pair] = 1 / pair.sum()
    return weight / weight.mean()


# ---- test a: can the camera be read from the features? (cameras with at least 20 photos) ----
big = pd.Series(camera).map(pd.Series(camera).value_counts()).to_numpy() >= 20
guess = np.empty(len(y), dtype=object)
for k in range(10):
    train, test = big & (fold != k), big & (fold == k)
    model = make_pipeline(StandardScaler(), PCA(100, random_state=SEED), LogisticRegression(C=0.01, max_iter=3000))
    guess[test] = model.fit(X[train], camera[train]).predict(X[test])
tests = [["a. camera read from the features", f"{pd.Series(camera[big]).nunique()} cameras, {big.sum()} photos",
          f"accuracy {100 * np.mean(guess[big] == camera[big]):.1f} %",
          f"always guessing the most common camera: {100 * pd.Series(camera[big]).value_counts(normalize=True).iloc[0]:.1f} %"]]

# ---- the five rules ----
decision = {rule: np.zeros(len(y), dtype=int) for rule in RULES}
p_model = {"C1": np.zeros((len(y), 3)), "reweight": np.zeros((len(y), 3))}
agreement = []
for k in range(10):
    test, cal = fold == k, np.isin(fold, [(k + 1) % 10, (k + 2) % 10, (k + 3) % 10])
    train = ~test & ~cal
    prepare = make_pipeline(StandardScaler(), PCA(100, random_state=SEED)).fit(X[train])
    z_train, z_cal, z_test = prepare.transform(X[train]), prepare.transform(X[cal]), prepare.transform(X[test])
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(z_train)          # no labels, no camera
    c_train, c_cal, c_test = kmeans.labels_, kmeans.predict(z_cal), kmeans.predict(z_test)
    agreement.append(adjusted_rand_score(camera_group[test], c_test))

    model = LogisticRegression(C=0.01, max_iter=3000).fit(z_train, y[train])
    cal_p, test_p = model.predict_proba(z_cal), model.predict_proba(z_test)
    overall = class_thresholds(cal_p, y[cal], ALPHA)
    p_model["C1"][test] = test_p
    decision["C1"][test] = decide(prediction_sets(test_p, overall))
    decision["camera"][test] = decide(prediction_sets(test_p, group_thresholds(cal_p, y[cal], camera_group[cal], overall)[camera_group[test]]))
    decision["cluster"][test] = decide(prediction_sets(test_p, group_thresholds(cal_p, y[cal], c_cal, overall)[c_test]))

    suspicious = np.argmin([np.mean(y[cal][c_cal == g] >= 1) for g in range(3)])       # cluster with the fewest referable eyes
    d = decision["C1"][test].copy()
    d[(c_test == suspicious) & (d == NO_REFERRAL)] = GRADER
    decision["no home"][test] = d

    model = LogisticRegression(C=0.01, max_iter=3000).fit(z_train, y[train], sample_weight=equal_weights(c_train, y[train]))
    p_model["reweight"][test] = model.predict_proba(z_test)
    decision["reweight"][test] = decide(prediction_sets(p_model["reweight"][test], class_thresholds(model.predict_proba(z_cal), y[cal], ALPHA)))

# ---- tests b and c, from the model of C1 ----
p = p_model["C1"]
for g, name in enumerate(GROUPS):
    here = camera_group == g
    tests.append(["b. average P(no referral) of grade-2 eyes", f"cameras: {name}", f"{p[here & (grade == 2), 0].mean():.2f}",
                  f"{(here & (grade == 2)).sum()} eyes"])
tests.append(["c. AUROC referable", "all APTOS photos", f"{roc_auc_score(y >= 1, 1 - p[:, 0]):.3f}", f"{len(y)} photos"])
for g, name in enumerate(GROUPS):
    here = camera_group == g
    tests.append(["c. AUROC referable", f"inside cameras: {name}", f"{roc_auc_score(y[here] >= 1, 1 - p[here, 0]):.3f}",
                  f"{here.sum()} photos, {(y[here] >= 1).sum()} referable"])
tests.append(["k-means clusters against the camera groups", "test folds", f"ARI {np.mean(agreement):.2f}",
              f"lowest fold {min(agreement):.2f}, highest {max(agreement):.2f}"])
tests = pd.DataFrame(tests, columns=["test", "where", "value", "note"])
tests.to_csv("results/camera_shortcut.csv", index=False)
print("1. Is the camera a shortcut?\n" + tests.to_string(index=False))

# ---- the rules, judged group by group ----
rows = []
for rule, name in RULES.items():
    d = decision[rule]
    row, worst = [name], 0
    for g in range(3):
        eyes = (y >= 1) & (camera_group == g)
        errors = int(np.sum(d[eyes] == NO_REFERRAL))
        worst = max(worst, 100 * errors / eyes.sum())
        row.append(f"{100 * errors / eyes.sum():.1f} ({wilson(errors, eyes.sum())[0]:.1f}-{wilson(errors, eyes.sum())[1]:.1f})")
    healthy = y == 0
    row += [f"{worst:.1f}", f"{100 * np.mean(d[y >= 1] == NO_REFERRAL):.1f}", f"{100 * np.mean(d[y == 2] == URGENT):.1f}",
            f"{100 * np.mean(d[healthy] == NO_REFERRAL):.1f}", f"{100 * np.mean(d[healthy & (camera_group == 0)] == NO_REFERRAL):.1f}",
            f"{100 * np.mean(d == GRADER):.1f}", f"{1000 * np.mean((d == ROUTINE) | (d == URGENT)):.0f}"]
    rows.append(row)
result = pd.DataFrame(rows, columns=["rule", *[f"sent home %, cameras {name} (95 % interval)" for name in GROUPS],
                                     "sent home %, worst camera group", "sent home %, all referable eyes (target <= 10)",
                                     "urgent eyes referred urgently % (target >= 90)", "healthy eyes sent home %",
                                     "healthy eyes sent home %, cameras mostly healthy", "to a grader %", "referrals per 1,000 eyes"])
result.to_csv("results/group_thresholds.csv", index=False)
print("\n2. The rules, judged group by group\n" + result.T.to_string(header=False))
print(f"\nAUROC referable inside the mostly-healthy cameras: C1 model {roc_auc_score(y[camera_group == 0] >= 1, 1 - p[camera_group == 0, 0]):.3f}, "
      f"reweighted model {roc_auc_score(y[camera_group == 0] >= 1, 1 - p_model['reweight'][camera_group == 0, 0]):.3f}; "
      f"over all photos {roc_auc_score(y >= 1, 1 - p[:, 0]):.3f} and {roc_auc_score(y >= 1, 1 - p_model['reweight'][:, 0]):.3f}")
