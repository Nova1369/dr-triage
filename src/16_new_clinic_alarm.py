"""Step 16. ADDITIONAL WORK (not in the proposal), part B. An alarm for a new clinic, and its camera groups
-> results/alarm.csv and results/groups_new_clinics.csv

Steps 7 and 13 showed that the rule fails silently when the photos to decide are unlike the calibration
photos: at Messidor-2 it sends home 87 % of the eyes that need a referral and nothing warns anyone.
Labels would reveal it, but a clinic has no labels when it starts. Two questions.

1. Can the failure be seen WITHOUT labels?  Two alarms, both given only m photos of the clinic:
     distance    (the obvious idea) how far is each photo from its 5 nearest APTOS training photos?
                 Alarm if the clinic's photos are farther than the calibration photos (Mann-Whitney test, p < 0.01).
     classifier  (ours) can a logistic regression tell the clinic's photos from the calibration photos?
                 It is tested by 5-fold cross-validation and scored by AUROC: 0.5 = it cannot tell them
                 apart, 1 = it always can. Alarm if the AUROC is at least 0.80.
   The alarms are tried on photos where the rule holds (fresh APTOS photos), on photos where only the
   share of sick eyes differs (the rule still holds there, step 9), and on photos where the rule fails:
   the two new clinics and the blurred, dark and compressed photos of step 12.
   For each of the 10 APTOS models, m photos are drawn 20 times.

2. After recalibration (step 8, 300 local photos), are the cameras of the new clinic treated alike?
   Step 15 found that APTOS's cameras are not. IDRiD has one camera; Messidor-2 has three photo sizes.

Run from the project folder:   python src/16_new_clinic_alarm.py        (about 4 minutes; needs step 12)
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal import GRADER, NO_REFERRAL, URGENT, class_thresholds, conformal_threshold, decide, prediction_sets

warnings.filterwarnings("ignore")
ALPHA = 0.1
SEED = 42
SIZES = [10, 20, 50, 100]      # m: photos of the clinic given to the alarm
DRAWS = 20
ALARM_AUROC = 0.80

table = pd.read_csv("data/table.csv")
features = np.load("data/features.npy")
dataset, label = table["dataset"].to_numpy(), table["label"].to_numpy()
fold, group, width = table["fold"].to_numpy(), table["group"].to_numpy(), table["width"].to_numpy()
aptos = dataset == "aptos"
changed = {"blurred": np.load("data/features_blur.npy"), "half brightness": np.load("data/features_dark.npy"),
           "JPEG quality 20": np.load("data/features_jpeg.npy")}                 # APTOS photos only, in table order
aptos_position = np.cumsum(aptos) - 1                                            # row of the table -> row of those files
rng = np.random.RandomState(SEED)


def classifier_auroc(calibration, clinic):
    """How well logistic regression tells the clinic's photos from the calibration photos (cross-validated AUROC)."""
    x = np.vstack([calibration, clinic])
    which = np.r_[np.zeros(len(calibration)), np.ones(len(clinic))]
    model = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
    scores = cross_val_predict(model, x, which, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="decision_function")
    return roc_auc_score(which, scores)


# ---- 1. the alarms ----
runs, harm = {}, {}
for k in range(10):
    test, cal = aptos & (fold == k), aptos & (fold == (k + 1) % 10)
    train = aptos & ~test & ~cal
    prepare = make_pipeline(StandardScaler(), PCA(100, random_state=SEED)).fit(features[train])
    model = LogisticRegression(C=0.01, max_iter=2000).fit(prepare.transform(features[train]), label[train])
    z_cal = prepare.transform(features[cal])
    thresholds = class_thresholds(model.predict_proba(z_cal), label[cal], ALPHA)
    neighbours = NearestNeighbors(n_neighbors=5).fit(prepare.transform(features[train]))
    distance_cal = neighbours.kneighbors(z_cal)[0][:, -1]

    rows = np.flatnonzero(test)
    healthy, sick = rows[label[rows] == 0], rows[label[rows] >= 1]
    few_sick = np.concatenate([healthy, rng.choice(sick, round(len(healthy) * 0.05 / 0.95), replace=False)])
    places = {"APTOS, fresh photos": (features[rows], label[rows]),
              "APTOS, 5 % referable eyes": (features[few_sick], label[few_sick]),
              "IDRiD": (features[dataset == "idrid"], label[dataset == "idrid"]),
              "Messidor-2": (features[dataset == "messidor2"], label[dataset == "messidor2"])}
    for name, x in changed.items():
        places["APTOS, " + name] = (x[aptos_position[rows]], label[rows])

    for name, (x, y) in places.items():
        z = prepare.transform(x)
        d = decide(prediction_sets(model.predict_proba(z), thresholds))            # what C1 does there (needs labels to judge)
        harm.setdefault(name, []).append([100 * np.mean(d[y >= 1] == NO_REFERRAL), 100 * np.mean(d[y == 2] == URGENT)])
        distance = neighbours.kneighbors(z)[0][:, -1]
        for m in SIZES:
            for _ in range(DRAWS):
                pick = rng.choice(len(z), m, replace=False)
                auroc = classifier_auroc(z_cal, z[pick])
                farther = mannwhitneyu(distance[pick], distance_cal, alternative="greater").pvalue < 0.01
                runs.setdefault((name, m), []).append([auroc, auroc >= ALARM_AUROC, farther])
    print(f"APTOS model {k + 1} of 10 done", flush=True)

rows_out = []
for name in places:
    h = np.mean(harm[name], axis=0)
    r = {m: np.mean(runs[(name, m)], axis=0) for m in SIZES}
    rows_out.append([name, f"{h[0]:.1f}", f"{h[1]:.1f}", "yes" if h[0] <= 10 and h[1] >= 90 else "no", f"{r[50][0]:.2f}",
                     *[f"{100 * r[m][1]:.0f}" for m in SIZES], *[f"{100 * r[m][2]:.0f}" for m in SIZES]])
alarm = pd.DataFrame(rows_out, columns=["photos to decide", "C1 referable eyes sent home %", "C1 urgent eyes referred urgently %",
                                        "C1 meets both targets", "classifier AUROC (m = 50)",
                                        *[f"classifier alarm rings %, m = {m}" for m in SIZES],
                                        *[f"distance alarm rings %, m = {m}" for m in SIZES]])
alarm.to_csv("results/alarm.csv", index=False)
print("\n1. The alarms (an alarm should ring exactly where C1 does not meet both targets)\n" + alarm.T.to_string(header=False))

# ---- 2. after recalibration, are the cameras of Messidor-2 treated alike? ----
clinic = np.flatnonzero(dataset == "messidor2")
sizes = np.unique(width[clinic])
camera = np.searchsorted(sizes, width[clinic])                                    # 0, 1, 2: the three photo sizes
names = [f"{w} px wide ({np.sum(camera == c)} photos, {100 * np.mean(label[clinic][camera == c] >= 1):.0f} % referable)"
         for c, w in enumerate(sizes)]
y = label[clinic]
runs = {"C2 one threshold per class": [], "C2 thresholds per camera": []}
for k in range(10):
    test, cal = aptos & (fold == k), aptos & (fold == (k + 1) % 10)
    model = make_pipeline(StandardScaler(), PCA(100, random_state=SEED), LogisticRegression(C=0.01, max_iter=2000))
    p = model.fit(features[aptos & ~test & ~cal], label[aptos & ~test & ~cal]).predict_proba(features[clinic])
    for _ in range(5):
        order = rng.permutation(len(clinic))
        local, rest = order[:300], order[300:]
        rest = rest[~np.isin(group[clinic][rest], group[clinic][local])]          # copies of a photo stay together
        overall = class_thresholds(p[local], y[local], ALPHA)
        per_camera = np.tile(overall, (3, 1))
        for c in range(3):
            for cls in range(3):
                t = conformal_threshold(1 - p[local][(camera[local] == c) & (y[local] == cls), cls], ALPHA)
                if np.isfinite(t):                                                # under 9 photos: keep the overall threshold
                    per_camera[c, cls] = t
        for rule, thresholds in [("C2 one threshold per class", np.tile(overall, (len(rest), 1))),
                                 ("C2 thresholds per camera", per_camera[camera[rest]])]:
            d, yr, cr = decide(prediction_sets(p[rest], thresholds)), y[rest], camera[rest]
            runs[rule].append([*[100 * np.mean(d[(yr >= 1) & (cr == c)] == NO_REFERRAL) for c in range(3)],
                               100 * np.mean(d[yr == 2] == URGENT), 100 * np.mean(d[yr == 0] == NO_REFERRAL), 100 * np.mean(d == GRADER)])
cameras = pd.DataFrame([[rule, *[f"{v:.1f}" for v in np.mean(values, axis=0)]] for rule, values in runs.items()],
                       columns=["rule at Messidor-2 (300 local photos)", *[f"referable eyes sent home %, {name}" for name in names],
                                "urgent eyes referred urgently %", "healthy eyes sent home %", "to a grader %"])
cameras.to_csv("results/groups_new_clinics.csv", index=False)
print("\n2. Messidor-2 after recalibration, camera by camera (average of 50 runs)\n" + cameras.T.to_string(header=False))
