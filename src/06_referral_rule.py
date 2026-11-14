"""Step 6 (contribution C1). The referral rule on APTOS, against simpler rules -> results/referral_rule.csv

The two targets:   at most 10 % of the eyes that need a referral are sent home, and
                   at least 90 % of the urgent eyes are referred urgently.

How every rule is tested, for each of the 10 folds in turn:
  * test photos         the fold itself,
  * calibration photos  the next fold (fold 9 uses fold 0): used only to set thresholds,
  * training photos     the other eight folds: logistic regression is trained on them.
So every APTOS photo gets one decision from a model and thresholds that never saw it.

The rules:
  Most likely class   the class with the highest probability
  B0                  the two trivial rules: send everyone home, refer everyone urgently
  B3 tuned            two cut-offs set on the calibration photos so that 90 % of their referable eyes and
                      90 % of their urgent eyes are caught
  B4 one threshold    standard conformal prediction: a single threshold for all classes
  B5 unsure band      to a grader if the highest probability is below 0.6, else the most likely class
  C1 (ours)           one conformal threshold per class (src/conformal.py)

Run from the project folder:   python src/06_referral_rule.py        (under a minute)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal import GRADER, NO_REFERRAL, ROUTINE, URGENT, class_thresholds, conformal_threshold, decide, prediction_sets

ALPHA = 0.1      # allowed miss rate: 10 %
SEED = 42

table = pd.read_csv("data/table.csv")
features = np.load("data/features.npy")
aptos = (table["dataset"] == "aptos").to_numpy()
X, y, fold = features[aptos], table["label"][aptos].to_numpy(), table["fold"][aptos].to_numpy()


def new_model():
    """The model chosen in step 5: standardise, 100 principal components, logistic regression (C = 0.01)."""
    return make_pipeline(StandardScaler(), PCA(100, random_state=SEED), LogisticRegression(C=0.01, max_iter=2000))


def most_likely(p):
    return p.argmax(axis=1)


def tuned_cutoffs(cal_p, cal_y):
    """B3: the cut-offs that 90 % of the calibration photos' referable eyes / urgent eyes reach."""
    referable = np.quantile(1 - cal_p[cal_y >= 1, 0], ALPHA)      # on P(needs referral) = 1 - P(no referral)
    urgent = np.quantile(cal_p[cal_y == 2, 2], ALPHA)             # on P(urgent)
    return referable, urgent


def tuned_decide(p, cutoffs):
    referable, urgent = cutoffs
    decision = np.full(len(p), NO_REFERRAL)
    decision[1 - p[:, 0] >= referable] = ROUTINE
    decision[p[:, 2] >= urgent] = URGENT
    return decision


def unsure_band(p):
    decision = most_likely(p)
    decision[p.max(axis=1) < 0.6] = GRADER
    return decision


RULES = ["Most likely class", "B0 send everyone home", "B0 refer everyone urgently", "B3 tuned cut-offs",
         "B4 one conformal threshold", "B5 unsure band", "C1 one threshold per class (ours)"]
decisions = {rule: np.zeros(len(y), dtype=int) for rule in RULES}
in_set = {"B4": np.zeros(len(y), dtype=bool), "C1": np.zeros(len(y), dtype=bool)}   # is the true class in the set?

for k in range(10):
    test, cal = fold == k, fold == (k + 1) % 10
    train = ~test & ~cal
    model = new_model().fit(X[train], y[train])
    cal_p, test_p = model.predict_proba(X[cal]), model.predict_proba(X[test])
    rows = np.arange(test.sum())

    decisions["Most likely class"][test] = most_likely(test_p)
    decisions["B0 send everyone home"][test] = NO_REFERRAL
    decisions["B0 refer everyone urgently"][test] = URGENT
    decisions["B3 tuned cut-offs"][test] = tuned_decide(test_p, tuned_cutoffs(cal_p, y[cal]))
    decisions["B5 unsure band"][test] = unsure_band(test_p)

    one = conformal_threshold(1 - cal_p[np.arange(cal.sum()), y[cal]], ALPHA)      # B4: every photo's own class
    sets = prediction_sets(test_p, np.full(3, one))
    decisions["B4 one conformal threshold"][test] = decide(sets)
    in_set["B4"][test] = sets[rows, y[test]]

    sets = prediction_sets(test_p, class_thresholds(cal_p, y[cal], ALPHA))         # C1: one threshold per class
    decisions["C1 one threshold per class (ours)"][test] = decide(sets)
    in_set["C1"][test] = sets[rows, y[test]]


def summary(d, rows):
    """The numbers of one rule on the given photos, in percent (referrals: per 1,000 eyes)."""
    d, t = d[rows], y[rows]
    return [100 * np.mean(d[t >= 1] == NO_REFERRAL), 100 * np.mean(d[t == 2] == URGENT),
            100 * np.mean(d[t == 0] == NO_REFERRAL), 100 * np.mean(d == GRADER),
            1000 * np.mean((d == ROUTINE) | (d == URGENT))]


rng = np.random.RandomState(SEED)
resamples = [rng.randint(0, len(y), len(y)) for _ in range(1000)]
rows_out = []
for rule in RULES:
    value = summary(decisions[rule], np.arange(len(y)))
    low, high = np.percentile([summary(decisions[rule], r)[:2] for r in resamples], [2.5, 97.5], axis=0)
    meets = "yes" if value[0] <= 10 and value[1] >= 90 else "no"
    rows_out.append([rule, f"{value[0]:.1f} ({low[0]:.1f}-{high[0]:.1f})", f"{value[1]:.1f} ({low[1]:.1f}-{high[1]:.1f})",
                     f"{value[2]:.1f}", f"{value[3]:.1f}", f"{value[4]:.0f}", meets])

result = pd.DataFrame(rows_out, columns=["rule", "referable eyes sent home % (target <= 10)",
                                         "urgent eyes referred urgently % (target >= 90)", "healthy eyes sent home %",
                                         "to a grader %", "referrals per 1,000 eyes", "meets both targets"])
result.to_csv("results/referral_rule.csv", index=False)
print(result.to_string(index=False))

print("\nTrue class inside the prediction set, per class (target: 90 %)")
for name in ["B4", "C1"]:
    print(f"  {name}: " + ", ".join(f"{label} {100 * in_set[name][y == c].mean():.1f} %"
                                    for c, label in enumerate(["no referral", "routine", "urgent"])))

print("\nC1's decisions by DR grade (% of the grade's eyes)")
grade = table["grade"][aptos].to_numpy()
by_grade = pd.crosstab(pd.Series(grade, name="grade"), decisions["C1 one threshold per class (ours)"], normalize="index") * 100
by_grade.columns = [["no referral", "routine", "urgent", "grader"][c] for c in by_grade.columns]
print(by_grade.round(1).to_string())
