"""Step 11. The other scores promised in the proposal
-> results/more_scores.csv, results/per_class.csv and results/calibration.png

Steps 5 to 7 scored the model by AUROC and accuracy and the rule by its two targets. Here are the rest:

  quadratic weighted kappa   agreement between the most likely class and the true class, for ordered
                             classes: calling an urgent eye "no referral" (2 classes away) costs 4 times
                             as much as calling it "routine" (1 away). 1 = perfect, 0 = no better than chance.
  Brier score                average of (probability - what happened) squared, summed over the 3 classes.
                             0 = perfect; the lower the better.
  ECE                        calibration error: can a probability be read as a risk? The photos are put in
                             10 bins by their probability; in each bin the average probability is compared
                             with the share of eyes that really were referable (or urgent). ECE is the
                             average gap, weighted by the size of the bins. 0 = perfect.
  per class                  sensitivity (share of the class's eyes that were given the class), specificity
                             (share of the other eyes that were not) and PPV (share of the eyes given the
                             class that really had it), for the most likely class and for the rule C1.

Where:  APTOS, every photo predicted by a model that never saw it (the 10 folds of step 6);
        IDRiD and Messidor-2, predicted by one model trained on all APTOS photos.
The intervals are 95 % bootstrap intervals (1,000 resamples of the photos).

Run from the project folder:   python src/11_more_scores.py        (under a minute)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal import class_thresholds, decide, prediction_sets

ALPHA = 0.1
SEED = 42
BINS = 10
CLASSES = ["no referral", "routine", "urgent"]

table = pd.read_csv("data/table.csv")
features = np.load("data/features.npy")
dataset, label, fold = table["dataset"].to_numpy(), table["label"].to_numpy(), table["fold"].to_numpy()
aptos = dataset == "aptos"


def new_model():
    return make_pipeline(StandardScaler(), PCA(100, random_state=SEED), LogisticRegression(C=0.01, max_iter=2000))


def bins_of(probability, happened):
    """For each of the 10 bins: number of photos, average probability, share of photos where it happened."""
    which = np.minimum((probability * BINS).astype(int), BINS - 1)
    count = np.bincount(which, minlength=BINS)
    filled = count > 0
    return (count[filled], np.bincount(which, probability, BINS)[filled] / count[filled],
            np.bincount(which, happened, BINS)[filled] / count[filled])


def ece(probability, happened):
    count, average, share = bins_of(probability, happened)
    return np.sum(count * np.abs(average - share)) / len(probability)


def scores(p, y):
    return [np.mean(p.argmax(axis=1) == y), cohen_kappa_score(y, p.argmax(axis=1), weights="quadratic"),
            np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)), ece(1 - p[:, 0], y >= 1), ece(p[:, 2], y == 2)]


def per_class(decision, y):
    """Sensitivity, specificity and PPV of each class, in percent."""
    rows = []
    for c, name in enumerate(CLASSES):
        given, has = decision == c, y == c
        rows.append([name, 100 * given[has].mean(), 100 * (~given)[~has].mean(), 100 * has[given].mean()])
    return rows


# ---- probabilities: APTOS out of fold (with the decision of C1), the two clinics from one all-APTOS model ----
p = np.zeros((len(label), 3))
c1 = np.zeros(len(label), dtype=int)
shares = np.zeros((len(label), 3))                 # the class shares of the training folds, for every photo
for k in range(10):
    test, cal = aptos & (fold == k), aptos & (fold == (k + 1) % 10)
    train = aptos & ~test & ~cal
    model = new_model().fit(features[train], label[train])
    p[test] = model.predict_proba(features[test])
    c1[test] = decide(prediction_sets(p[test], class_thresholds(model.predict_proba(features[cal]), label[cal], ALPHA)))
    shares[test] = np.bincount(label[train], minlength=3) / train.sum()
p[~aptos] = new_model().fit(features[aptos], label[aptos]).predict_proba(features[~aptos])

# ---- the scores ----
rng = np.random.RandomState(SEED)
rows, classes = [], []
places = [("APTOS, class shares only (no photo)", aptos, shares), ("APTOS", aptos, p),
          ("IDRiD", dataset == "idrid", p), ("Messidor-2", dataset == "messidor2", p)]
for name, here, probabilities in places:
    q, y = probabilities[here], label[here]
    resampled = np.array([scores(q[r], y[r]) for r in (rng.randint(0, len(y), len(y)) for _ in range(1000))])
    low, high = np.percentile(resampled, [2.5, 97.5], axis=0)
    rows.append([name, len(y), *[f"{v:.3f} ({a:.3f}-{b:.3f})" for v, a, b in zip(scores(q, y), low, high)]])
    if probabilities is p:
        classes += [[name, "most likely class", *row] for row in per_class(q.argmax(axis=1), y)]
classes += [["APTOS", "C1 (eyes sent to a grader count as not given any class)", *row] for row in per_class(c1[aptos], label[aptos])]

result = pd.DataFrame(rows, columns=["photos from", "photos", "accuracy", "quadratic weighted kappa", "Brier score",
                                     "ECE of P(referable)", "ECE of P(urgent)"])
classes = pd.DataFrame(classes, columns=["photos from", "decision by", "class", "sensitivity %", "specificity %", "PPV %"]).round(1)
result.to_csv("results/more_scores.csv", index=False)
classes.to_csv("results/per_class.csv", index=False)
print(result.to_string(index=False) + "\n\n" + classes.to_string(index=False))

# ---- the calibration chart: a calibrated model stays on the diagonal ----
figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for axis, title in zip(axes, ["P(referable)", "P(urgent)"]):
    for (name, here, _), colour in zip(places[1:], ["#1f77b4", "#2ca02c", "#d62728"]):
        q, y = p[here], label[here]
        probability, happened = (1 - q[:, 0], y >= 1) if title == "P(referable)" else (q[:, 2], y == 2)
        _, average, share = bins_of(probability, happened)
        axis.plot(average, share, "o-", color=colour, label=f"{name} (ECE {ece(probability, happened):.3f})")
    axis.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    axis.set(xlabel=f"{title} given by the model", ylabel="share of eyes that really were", title=title)
    axis.legend(fontsize=8)
figure.tight_layout()
figure.savefig("results/calibration.png", dpi=150)
