"""Step 8 (experiment E2). A few local photos at the new clinic -> results/recalibration.csv

Step 7 showed that thresholds set at APTOS do not keep their promise at another clinic. Suppose the new
clinic can grade a small number n of its own photos. What is the best use of them?

  C1  ignore them: APTOS model, APTOS thresholds (step 7)
  C2  recalibrate (ours): keep the APTOS model, set the thresholds again on the n local photos
  B6  retrain: add the n local photos to the training photos, keep thresholds from APTOS calibration photos
  B7  retrain and recalibrate: add half of the local photos to the training photos,
      set the thresholds on the other half

Only C2 and B7 set thresholds on photos of the new clinic, so only they carry the 90 % guarantee there.

How it is tested, for n = 50, 100, 200, 300 and each new clinic (IDRiD, Messidor-2):
  * the 10 APTOS models and calibration folds are the ones of steps 6 and 7,
  * for each of them the n local photos are drawn at random 5 times (copies of a photo stay together),
  * every rule is tested on the photos of the clinic that were not drawn (the same photos for all rules).
That gives 50 runs per case. We report the average and the range that holds 90 % of the runs.

Run from the project folder:   python src/08_recalibration.py        (about 8 minutes)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal import GRADER, NO_REFERRAL, ROUTINE, URGENT, class_thresholds, decide, prediction_sets

ALPHA = 0.1      # allowed miss rate: 10 %
SEED = 42
SIZES = [50, 100, 200, 300]                                       # n: local photos the clinic grades
DRAWS = 5                                                         # random draws of the local photos per APTOS model
CLINICS = {"idrid": "IDRiD", "messidor2": "Messidor-2"}
RULES = {"C1": "C1 APTOS model, APTOS thresholds", "C2": "C2 recalibrated on local photos (ours)",
         "B6": "B6 retrained with local photos", "B7": "B7 retrained and recalibrated, half each"}
MEASURES = ["true class in the set %", "no referral in its set %", "routine in its set %", "urgent in its set %",
            "referable eyes sent home % (target <= 10)", "urgent eyes referred urgently % (target >= 90)",
            "to a grader %", "referrals per 1,000 eyes"]

table = pd.read_csv("data/table.csv")
features = np.load("data/features.npy")
dataset, label = table["dataset"].to_numpy(), table["label"].to_numpy()
fold, group = table["fold"].to_numpy(), table["group"].to_numpy()


def new_model():
    return make_pipeline(StandardScaler(), PCA(100, random_state=SEED), LogisticRegression(C=0.01, max_iter=2000))


def measures(sets, y):
    """The numbers of one rule on one group of photos (the order of MEASURES)."""
    inside = sets[np.arange(len(y)), y]                           # is the true class in the photo's set?
    d = decide(sets)
    return [100 * inside.mean(), 100 * inside[y == 0].mean(), 100 * inside[y == 1].mean(), 100 * inside[y == 2].mean(),
            100 * np.mean(d[y >= 1] == NO_REFERRAL), 100 * np.mean(d[y == 2] == URGENT),
            100 * np.mean(d == GRADER), 1000 * np.mean((d == ROUTINE) | (d == URGENT))]


def draw_local(clinic, n, rng):
    """n random photos of the clinic (row numbers, in random order) and the rest; copies are never split."""
    rows = np.flatnonzero(dataset == clinic)
    groups = np.unique(group[rows])
    position = pd.Series(rng.permutation(len(groups)), index=groups)   # every group of copies gets a random position
    rows = rows[np.argsort(position[group[rows]].to_numpy(), kind="stable")]
    local = rows[:n]
    rest = rows[n:][~np.isin(group[rows[n:]], group[local])]
    return local, rest


runs = {(clinic, n, rule): [] for clinic in CLINICS for n in SIZES for rule in RULES}
rng = np.random.RandomState(SEED)

for k in range(10):
    test, cal = (dataset == "aptos") & (fold == k), (dataset == "aptos") & (fold == (k + 1) % 10)
    train = np.flatnonzero((dataset == "aptos") & ~test & ~cal)
    aptos_model = new_model().fit(features[train], label[train])
    aptos_thresholds = class_thresholds(aptos_model.predict_proba(features[cal]), label[cal], ALPHA)
    p_aptos_model = aptos_model.predict_proba(features)           # the APTOS model on every photo, once

    for clinic in CLINICS:
        for n in SIZES:
            for _ in range(DRAWS):
                local, rest = draw_local(clinic, n, rng)
                y = label[rest]
                # C1: nothing changes.   C2: same model, thresholds from the local photos.
                runs[(clinic, n, "C1")].append(measures(prediction_sets(p_aptos_model[rest], aptos_thresholds), y))
                local_thresholds = class_thresholds(p_aptos_model[local], label[local], ALPHA)
                runs[(clinic, n, "C2")].append(measures(prediction_sets(p_aptos_model[rest], local_thresholds), y))
                # B6: all local photos join the training photos; thresholds from the APTOS calibration fold.
                both = np.concatenate([train, local])
                model = new_model().fit(features[both], label[both])
                thresholds = class_thresholds(model.predict_proba(features[cal]), label[cal], ALPHA)
                runs[(clinic, n, "B6")].append(measures(prediction_sets(model.predict_proba(features[rest]), thresholds), y))
                # B7: half of the local photos join the training photos; thresholds from the other half.
                half, other = local[: n // 2], local[n // 2:]
                both = np.concatenate([train, half])
                model = new_model().fit(features[both], label[both])
                thresholds = class_thresholds(model.predict_proba(features[other]), label[other], ALPHA)
                runs[(clinic, n, "B7")].append(measures(prediction_sets(model.predict_proba(features[rest]), thresholds), y))
    print(f"APTOS model {k + 1} of 10 done", flush=True)

rows = []
for (clinic, n, rule), values in runs.items():
    r = np.array(values)
    low, high = np.percentile(r, [5, 95], axis=0)
    rows.append([CLINICS[clinic], n, RULES[rule], *[f"{m:.1f} ({a:.1f}-{b:.1f})" for m, a, b in zip(r.mean(axis=0), low, high)]])
result = pd.DataFrame(rows, columns=["clinic", "local photos", "rule", *MEASURES])
result.to_csv("results/recalibration.csv", index=False)

print("\nAverage of 50 runs (the range that holds 90 % of the runs)")
for column in [MEASURES[0], MEASURES[4], MEASURES[5], MEASURES[7]]:
    print("\n" + column)
    for clinic in CLINICS.values():
        for n in SIZES:
            part = result[(result["clinic"] == clinic) & (result["local photos"] == n)]
            print(f"  {clinic:10s} n = {n:3d}   " + "   ".join(f"{rule[:2]} {cell:17s}" for rule, cell in zip(part["rule"], part[column])))
