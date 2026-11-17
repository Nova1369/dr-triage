"""Step 10. Which eyes does the rule get wrong? -> results/errors_by_group.csv and results/missed_urgent_eyes.csv

The rule (C1) meets its two targets on APTOS as a whole (step 6). A target met on average can still be
missed badly for one kind of eye. Here the two errors that matter are counted group by group:

  sent home       an eye that needs a referral (grade 2-4) gets "no referral"      (target: at most 10 %)
  missed urgent   an urgent eye (grade 3-4) is not referred urgently               (target: at most 10 %)

The groups:
  DR grade     2, 3, 4
  camera       APTOS does not name the camera, but each camera saves photos of one size, so the photo size
               (width x height) stands for it. Cameras are put in three groups by what they photographed:
               "mostly healthy" (under 10 % of its eyes need a referral), "mostly sick" (over 50 %), "mixed".
  sharpness    the blurriest, middle and sharpest third of the photos (sharpness from step 1)

Every rate comes with a 95 % interval (Wilson), because some groups are small.
Every missed urgent eye is listed with its probabilities, so the photos can be looked at one by one.

The decisions are the ones of step 6: every APTOS photo is decided by a model and thresholds that never saw it.

Run from the project folder:   python src/10_error_analysis.py        (under a minute)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal import DECISIONS, NO_REFERRAL, URGENT, class_thresholds, decide, prediction_sets

ALPHA = 0.1      # allowed miss rate: 10 %
SEED = 42

table = pd.read_csv("data/table.csv")
features = np.load("data/features.npy")
aptos = table[table["dataset"] == "aptos"].reset_index(drop=True)
X, y, fold = features[(table["dataset"] == "aptos").to_numpy()], aptos["label"].to_numpy(), aptos["fold"].to_numpy()


def new_model():
    return make_pipeline(StandardScaler(), PCA(100, random_state=SEED), LogisticRegression(C=0.01, max_iter=2000))


def wilson(k, n):
    """95 % interval for a share of k out of n, in percent. Unlike k/n +- 2 standard errors it works for small n."""
    z, share = 1.96, k / n
    middle = (share + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, 100 * (middle - half)), 100 * (middle + half)


# ---- the decisions of step 6 ----
p = np.zeros((len(y), 3))
for k in range(10):
    test, cal = fold == k, fold == (k + 1) % 10
    model = new_model().fit(X[~test & ~cal], y[~test & ~cal])
    p[test] = model.predict_proba(X[test])
    thresholds = class_thresholds(model.predict_proba(X[cal]), y[cal], ALPHA)
    aptos.loc[test, "decision"] = decide(prediction_sets(p[test], thresholds))
decision = aptos["decision"].to_numpy().astype(int)
sent_home = (y >= 1) & (decision == NO_REFERRAL)
missed_urgent = (y == 2) & (decision != URGENT)

# ---- the groups ----
aptos["camera"] = aptos["width"].astype(str) + " x " + aptos["height"].astype(str)
referable_share = (aptos["label"] >= 1).groupby(aptos["camera"]).transform("mean")
aptos["camera group"] = np.select([referable_share < 0.1, referable_share > 0.5], ["mostly healthy", "mostly sick"], "mixed")
aptos["sharpness third"] = pd.qcut(aptos["sharpness"], 3, labels=["blurriest", "middle", "sharpest"]).astype(str)
aptos["all"] = "all eyes"

rows = []
for kind, column in [("all", "all"), ("DR grade", "grade"), ("camera group", "camera group"),
                     ("sharpness", "sharpness third"), ("camera (photo size)", "camera")]:
    for name, part in aptos.groupby(column, sort=True):
        i = part.index.to_numpy()
        n_ref, n_urg = int((y[i] >= 1).sum()), int((y[i] == 2).sum())
        row = [kind, name, len(i), n_ref, n_urg]
        for errors, n in [(int(sent_home[i].sum()), n_ref), (int(missed_urgent[i].sum()), n_urg)]:
            row += [errors, f"{100 * errors / n:.1f} ({wilson(errors, n)[0]:.1f}-{wilson(errors, n)[1]:.1f})" if n else ""]
        if n_ref:                                                    # groups without referable eyes have no errors to count
            rows.append(row)
groups = pd.DataFrame(rows, columns=["grouped by", "group", "photos", "referable eyes", "urgent eyes", "sent home",
                                     "sent home % (95 % interval)", "missed urgent", "missed urgent % (95 % interval)"])
groups.to_csv("results/errors_by_group.csv", index=False)

# ---- every missed urgent eye ----
missed = aptos.loc[missed_urgent, ["photo", "path", "grade", "camera", "camera group", "sharpness"]].copy()
missed["sharpness"] = missed["sharpness"].round(1)
missed["decision"] = [DECISIONS[d] for d in decision[missed_urgent]]
for c, name in enumerate(["P(no referral)", "P(routine)", "P(urgent)"]):
    missed[name] = p[missed_urgent, c].round(3)
missed.sort_values("P(urgent)").to_csv("results/missed_urgent_eyes.csv", index=False)

print(groups[groups["grouped by"] != "camera (photo size)"].to_string(index=False))
print("\nBy camera (photo size), cameras with at least 20 referable eyes")
print(groups[(groups["grouped by"] == "camera (photo size)") & (groups["referable eyes"] >= 20)].drop(columns="grouped by").to_string(index=False))

print(f"\nThe {missed_urgent.sum()} missed urgent eyes of {(y == 2).sum()} got: "
      + ", ".join(f"{DECISIONS[d]} {n}" for d, n in missed["decision"].map(DECISIONS.index).value_counts().sort_index().items()))
print(f"Median P(urgent): {np.median(p[missed_urgent, 2]):.3f} for the missed urgent eyes, "
      f"{np.median(p[(y == 2) & ~missed_urgent, 2]):.3f} for the urgent eyes that were caught")

# ---- does the model lean on the camera? the same eyes, on different cameras ----
print("\nAverage P(no referral) the model gives to eyes of the same grade, by camera group")
lean = pd.DataFrame({"grade": aptos["grade"], "camera group": aptos["camera group"], "p": p[:, 0]})
print(lean.pivot_table(index="grade", columns="camera group", values="p", aggfunc="mean").round(2).to_string())
print("Photos per grade and camera group")
print(pd.crosstab(aptos["grade"], aptos["camera group"]).to_string())
