"""Step 7 (experiment E0). The rule at two new clinics, with nothing changed -> results/new_clinics.csv

Conformal prediction promises 90 % only if the new photos are like the calibration photos. Here we check
what happens when they are not: the model and the thresholds come from APTOS, exactly as in step 6, and
are applied unchanged to IDRiD (another Indian clinic) and Messidor-2 (France). No photo of the new
clinics is used for training or for the thresholds.

How it is tested, for each of the 10 folds in turn (the same 10 models and thresholds as in step 6):
  * training photos     eight APTOS folds,
  * calibration photos  one APTOS fold: sets the thresholds,
  * test photos         the remaining APTOS fold, all 516 IDRiD photos, all 1,744 Messidor-2 photos.
For APTOS the 10 test folds are pooled (the numbers of step 6). For a new clinic every one of the 10
rounds decides all its photos, so we report the average of the 10 rounds and the lowest and highest round.

Rules:  B4  standard conformal prediction, one threshold for all classes
        C1  one threshold per class (ours)

Run from the project folder:   python src/07_new_clinics.py        (under a minute)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal import GRADER, NO_REFERRAL, ROUTINE, URGENT, class_thresholds, conformal_threshold, decide, prediction_sets

ALPHA = 0.1      # allowed miss rate: 10 %
SEED = 42
CLINICS = {"aptos": "APTOS (home)", "idrid": "IDRiD (new clinic)", "messidor2": "Messidor-2 (new clinic)"}
MEASURES = ["true class in the set %", "no referral in its set %", "routine in its set %", "urgent in its set %",
            "referable eyes sent home % (target <= 10)", "urgent eyes referred urgently % (target >= 90)",
            "to a grader %", "referrals per 1,000 eyes"]

table = pd.read_csv("data/table.csv")
features = np.load("data/features.npy")
dataset, label, fold = table["dataset"].to_numpy(), table["label"].to_numpy(), table["fold"].to_numpy()


def measures(sets, y):
    """The numbers of one rule on one group of photos (the order of MEASURES)."""
    inside = sets[np.arange(len(y)), y]                           # is the true class in the photo's set?
    d = decide(sets)
    return [100 * inside.mean(), 100 * inside[y == 0].mean(), 100 * inside[y == 1].mean(), 100 * inside[y == 2].mean(),
            100 * np.mean(d[y >= 1] == NO_REFERRAL), 100 * np.mean(d[y == 2] == URGENT),
            100 * np.mean(d == GRADER), 1000 * np.mean((d == ROUTINE) | (d == URGENT))]


aptos_sets = {"B4": np.zeros((len(table), 3), dtype=bool), "C1": np.zeros((len(table), 3), dtype=bool)}
rounds = {(clinic, rule): [] for clinic in ["idrid", "messidor2"] for rule in ["B4", "C1"]}
auroc = {clinic: [] for clinic in CLINICS}                        # how well the model ranks the eyes, per round

for k in range(10):
    test, cal = (dataset == "aptos") & (fold == k), (dataset == "aptos") & (fold == (k + 1) % 10)
    train = (dataset == "aptos") & ~test & ~cal
    model = make_pipeline(StandardScaler(), PCA(100, random_state=SEED), LogisticRegression(C=0.01, max_iter=2000))
    model.fit(features[train], label[train])
    cal_p = model.predict_proba(features[cal])
    thresholds = {"B4": np.full(3, conformal_threshold(1 - cal_p[np.arange(cal.sum()), label[cal]], ALPHA)),
                  "C1": class_thresholds(cal_p, label[cal], ALPHA)}

    for clinic in CLINICS:
        photos = test if clinic == "aptos" else dataset == clinic
        p = model.predict_proba(features[photos])
        auroc[clinic].append([roc_auc_score(label[photos] >= 1, 1 - p[:, 0]), roc_auc_score(label[photos] == 2, p[:, 2])])
        for rule in ["B4", "C1"]:
            sets = prediction_sets(p, thresholds[rule])
            if clinic == "aptos":
                aptos_sets[rule][photos] = sets
            else:
                rounds[(clinic, rule)].append(measures(sets, label[photos]))

rows = []
for clinic, name in CLINICS.items():
    for rule in ["B4", "C1"]:
        if clinic == "aptos":
            here = dataset == "aptos"
            cells = [f"{v:.1f}" for v in measures(aptos_sets[rule][here], label[here])]
        else:
            r = np.array(rounds[(clinic, rule)])
            cells = [f"{m:.1f} ({a:.1f}-{b:.1f})" for m, a, b in zip(r.mean(axis=0), r.min(axis=0), r.max(axis=0))]
        rows.append([name, {"B4": "B4 one conformal threshold", "C1": "C1 one threshold per class (ours)"}[rule], *cells])

result = pd.DataFrame(rows, columns=["clinic", "rule", *MEASURES])
result.to_csv("results/new_clinics.csv", index=False)
print("New clinics: average of the 10 rounds (lowest round - highest round). APTOS: the 10 test folds pooled.\n")
for column in MEASURES:
    print(column)
    for row in result.itertuples(index=False):
        print(f"  {row[0]:24s} {row[1][:2]}   {row[2 + MEASURES.index(column)]}")

print("\nThe clinics themselves")
for clinic, name in CLINICS.items():
    y = label[dataset == clinic]
    a = np.mean(auroc[clinic], axis=0)
    print(f"  {name:24s} {len(y):5d} photos: no referral {100 * np.mean(y == 0):.0f} %, routine {100 * np.mean(y == 1):.0f} %, "
          f"urgent {100 * np.mean(y == 2):.0f} %;   AUROC referable {a[0]:.3f}, urgent {a[1]:.3f}")
