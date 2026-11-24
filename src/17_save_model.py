"""Step 17. Save the final referral rule -> models/referral_rule.npz

Steps 5 to 16 trained many models to test the rule. This step trains the one that would be used:

  training photos       APTOS folds 0 to 8: standardise, 100 principal components, logistic regression (C = 0.01)
  calibration photos    APTOS fold 9: the three thresholds of the rule (src/conformal.py), allowed miss rate 10 %

The model is saved as plain arrays of numbers, not as a Python object, so that it can be read with any
version of numpy and used without scikit-learn (src/decide.py):

  scaler_mean, scaler_scale     standardise:            z = (features - scaler_mean) / scaler_scale
  pca_mean, pca_components      principal components:   c = (z - pca_mean) x pca_components'
  coefficients, intercepts      logistic regression:    p = softmax(c x coefficients' + intercepts)
  thresholds                    the rule:               class k is in the set if 1 - p[k] <= thresholds[k]

Before saving, the script checks that these formulas give the same probabilities as scikit-learn.
The thresholds were set at APTOS. At another clinic they must be set again (steps 7, 8 and 16).

Run from the project folder:   python src/17_save_model.py        (a few seconds)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal import class_thresholds
from decide import probabilities

ALPHA = 0.1
SEED = 42
MODEL_FILE = Path("models/referral_rule.npz")

table = pd.read_csv("data/table.csv")
features = np.load("data/features.npy")
aptos = (table["dataset"] == "aptos").to_numpy()
train, cal = aptos & (table["fold"] != 9).to_numpy(), aptos & (table["fold"] == 9).to_numpy()
label = table["label"].to_numpy()

pipeline = make_pipeline(StandardScaler(), PCA(100, random_state=SEED), LogisticRegression(C=0.01, max_iter=2000))
pipeline.fit(features[train], label[train])
scaler, pca, regression = pipeline.steps[0][1], pipeline.steps[1][1], pipeline.steps[2][1]
cal_p = pipeline.predict_proba(features[cal])
thresholds = class_thresholds(cal_p, label[cal], ALPHA)

model = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "pca_mean": pca.mean_, "pca_components": pca.components_,
         "coefficients": regression.coef_, "intercepts": regression.intercept_, "thresholds": thresholds}
difference = np.abs(probabilities(features[cal], model) - cal_p).max()
assert difference < 1e-5, f"the saved formulas and scikit-learn disagree by {difference}"

MODEL_FILE.parent.mkdir(exist_ok=True)
np.savez_compressed(MODEL_FILE, **model)
print(f"Trained on {train.sum()} photos, thresholds set on {cal.sum()} photos "
      f"({', '.join(str(int((label[cal] == c).sum())) for c in range(3))} of the three classes)")
print("Thresholds (no referral, routine, urgent):", np.round(thresholds, 3))
print(f"Largest difference between the saved formulas and scikit-learn: {difference:.1e}")
print(f"Saved to {MODEL_FILE} ({MODEL_FILE.stat().st_size / 1e6:.1f} MB)")
