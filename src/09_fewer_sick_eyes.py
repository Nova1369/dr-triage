"""Step 9 (contribution C3, experiment E3). A clinic with fewer sick eyes
-> results/fewer_sick_eyes.csv and results/workload_forecast.csv

In APTOS 41 % of the eyes need a referral. In a screening programme it is nearer 5 to 25 %. Two questions:

1. Does the rule keep its promise when sick eyes are rarer?  It should: every class has its own threshold,
   and a threshold only looks at the eyes of its class, so it does not matter how many of them there are.
   Standard conformal prediction (B4, one threshold for all classes) has no such protection.

2. Can the clinic know its workload in advance (referrals per 1,000 eyes)?
       forecast = 1000 * sum over the classes of (share of the class) * (how often the rule refers that class)
   How often the rule refers each class is measured on APTOS. The shares of the classes at the clinic come from:
     B8  APTOS's shares: nothing is corrected
     C3  (ours) an estimate from the clinic's photos WITHOUT labels (estimate_shares below)
     B9  the true shares, which nobody knows in practice: the best a forecast could do

How it is tested. Every APTOS photo gets probabilities and a decision as in step 6. Then, 200 times for each
share of referable eyes: APTOS is cut in two halves (copies of a photo stay together); the first half is
what we know; from the second half a clinic of 1,000 eyes with the wanted share is drawn at random.
Last, the same forecast is made for the two real clinics (IDRiD, Messidor-2), with the rule of step 7.

Run from the project folder:   python src/09_fewer_sick_eyes.py        (under a minute)
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
SHARES = [0.05, 0.125, 0.25, None]        # share of referable eyes at the clinic; None = as in APTOS
REPEATS = 200
CLINIC_SIZE = 1000
CLINICS = {"idrid": "IDRiD", "messidor2": "Messidor-2"}

table = pd.read_csv("data/table.csv")
features = np.load("data/features.npy")
dataset, label = table["dataset"].to_numpy(), table["label"].to_numpy()
fold, group = table["fold"].to_numpy(), table["group"].to_numpy()
aptos = dataset == "aptos"


def new_model():
    return make_pipeline(StandardScaler(), PCA(100, random_state=SEED), LogisticRegression(C=0.01, max_iter=2000))


def class_shares(y):
    return np.bincount(y, minlength=3) / len(y)


def estimate_shares(p, old_shares):
    """The shares of the three classes among photos without labels (Saerens et al., 2002).

    p: the model's probabilities for these photos. The model learnt them where the shares were old_shares.
    Start from the old shares and repeat until nothing changes:
      1. correct every probability for the current guess of the shares:  p * guess / old_shares, rescaled to sum 1;
      2. the new guess is the average of the corrected probabilities.
    """
    guess = old_shares
    for _ in range(10000):
        corrected = p * guess / old_shares
        corrected /= corrected.sum(axis=1, keepdims=True)
        new = corrected.mean(axis=0)
        if np.abs(new - guess).max() < 1e-9:
            break
        guess = new
    return new


def referral_rates(d, y):
    """How often the rule refers (routine or urgent) the eyes of each class."""
    referred = (d == ROUTINE) | (d == URGENT)
    return np.array([referred[y == c].mean() for c in range(3)])


def referrals(d):
    return 1000 * np.mean((d == ROUTINE) | (d == URGENT))


# ---- every photo gets probabilities and decisions from a model and thresholds that never saw it (step 6) ----
p = np.zeros((len(label), 3))                    # probabilities (APTOS photos only)
c1 = np.zeros(len(label), dtype=int)             # decision of C1, one threshold per class (APTOS photos only)
b4_in_set = np.zeros(len(label), dtype=bool)     # B4, one threshold: is the true class in the set?
at_clinic = {clinic: [] for clinic in CLINICS}   # per APTOS model: the clinic's estimated shares and C1's referrals

for k in range(10):
    test, cal = aptos & (fold == k), aptos & (fold == (k + 1) % 10)
    train = aptos & ~test & ~cal
    model = new_model().fit(features[train], label[train])
    cal_p, p[test] = model.predict_proba(features[cal]), model.predict_proba(features[test])
    thresholds = class_thresholds(cal_p, label[cal], ALPHA)
    c1[test] = decide(prediction_sets(p[test], thresholds))
    one = conformal_threshold(1 - cal_p[np.arange(cal.sum()), label[cal]], ALPHA)
    b4_in_set[test] = prediction_sets(p[test], np.full(3, one))[np.arange(test.sum()), label[test]]
    for clinic in CLINICS:
        here = dataset == clinic
        clinic_p = model.predict_proba(features[here])
        at_clinic[clinic].append([*estimate_shares(clinic_p, class_shares(label[train])),
                                  referrals(decide(prediction_sets(clinic_p, thresholds)))])

# ---- clinics with fewer sick eyes, drawn from APTOS ----
rng = np.random.RandomState(SEED)
aptos_rows, groups = np.flatnonzero(aptos), np.unique(group[aptos])
kept, forecasts = [], []
for share in SHARES:
    runs = []
    for _ in range(REPEATS):
        first = np.isin(group[aptos_rows], rng.permutation(groups)[: len(groups) // 2])
        known, second = aptos_rows[first], aptos_rows[~first]
        referable = class_shares(label[known])[1:].sum() if share is None else share
        n = round(CLINIC_SIZE * referable)
        eyes = np.concatenate([rng.choice(second[label[second] >= 1], n, replace=False),
                               rng.choice(second[label[second] == 0], CLINIC_SIZE - n, replace=False)])
        y, d = label[eyes], c1[eyes]
        rates, old = referral_rates(c1[known], label[known]), class_shares(label[known])
        estimate = estimate_shares(p[eyes], old)
        referred = (d == ROUTINE) | (d == URGENT)
        runs.append([100 * np.mean(d[y >= 1] == NO_REFERRAL), 100 * np.mean(d[y == 2] == URGENT),
                     100 * b4_in_set[eyes].mean(), 100 * b4_in_set[eyes][y == 2].mean(),
                     100 * np.mean(d == GRADER), referrals(d), 100 * np.mean(y[referred] >= 1),
                     100 * estimate[1:].sum(), 1000 * old @ rates, 1000 * estimate @ rates, 1000 * class_shares(y) @ rates])
    r = np.array(runs)
    m, (low, high) = r.mean(axis=0), np.percentile(r, [5, 95], axis=0)
    miss = np.abs(r[:, 8:] - r[:, [5]]).mean(axis=0)              # how far each forecast is from what happened
    name = f"{100 * (0.406 if share is None else share):.1f}" + (" (APTOS)" if share is None else "")
    kept.append([name, *[f"{m[i]:.1f} ({low[i]:.1f}-{high[i]:.1f})" for i in range(5)], f"{m[5]:.0f}", f"{m[6]:.1f}"])
    forecasts.append([f"APTOS with {name.replace(' (APTOS)', '')} % referable", name.split()[0], f"{m[7]:.1f}", f"{m[5]:.0f}",
                      *[f"{m[i]:.0f} ({miss[i - 8]:.0f})" for i in (8, 9, 10)]])

# ---- the two real clinics: the APTOS rule unchanged (step 7); referral rates from all APTOS decisions ----
rates, old = referral_rates(c1[aptos], label[aptos]), class_shares(label[aptos])
for clinic, name in CLINICS.items():
    r = np.array(at_clinic[clinic])
    estimates, happened, true = r[:, :3], r[:, 3], class_shares(label[dataset == clinic])
    forecast = np.column_stack([np.full(10, 1000 * old @ rates), 1000 * estimates @ rates, np.full(10, 1000 * true @ rates)])
    miss = np.abs(forecast - happened[:, None]).mean(axis=0)
    forecasts.append([name, f"{100 * true[1:].sum():.1f}", f"{100 * estimates[:, 1:].sum(axis=1).mean():.1f}", f"{happened.mean():.0f}",
                      *[f"{forecast[:, i].mean():.0f} ({miss[i]:.0f})" for i in range(3)]])

kept = pd.DataFrame(kept, columns=["referable eyes at the clinic %", "C1 referable eyes sent home % (target <= 10)",
                                   "C1 urgent eyes referred urgently % (target >= 90)", "B4 true class in the set %",
                                   "B4 urgent in its set %", "C1 to a grader %", "C1 referrals per 1,000 eyes",
                                   "C1 referrals that were needed %"])
forecasts = pd.DataFrame(forecasts, columns=["clinic", "referable eyes, true %", "referable eyes, estimated without labels %",
                                             "referrals per 1,000, observed", "forecast B8 APTOS shares (average miss)",
                                             "forecast C3 estimated shares (average miss)", "forecast B9 true shares (average miss)"])
kept.to_csv("results/fewer_sick_eyes.csv", index=False)
forecasts.to_csv("results/workload_forecast.csv", index=False)

print(f"1. Does the promise hold with fewer sick eyes?  Average of {REPEATS} clinics of {CLINIC_SIZE:,} eyes "
      "(the range that holds 90 % of them)\n")
print(kept.T.to_string(header=False))
print("\n2. Can the workload be forecast?  Referrals per 1,000 eyes; in brackets how far the forecast is off on average\n")
print(forecasts.to_string(index=False))
