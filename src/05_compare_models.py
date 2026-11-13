"""Step 5 (experiment E1). Which of the course's models reads the features best? -> results/models.csv

Every model is tested in the same way on the 3,662 APTOS photos:

  * 10-fold cross-validation with the folds of step 4: train on nine folds, predict the tenth, so every
    photo is predicted once by a model that never saw it (or a copy of it).
  * Inside the nine training folds, a small grid search (3 folds, copies kept together) picks the model's
    main setting. Testing the settings on the training folds only is what "nested" cross-validation means:
    the test fold plays no part in choosing them.
  * Before each model, the 2,048 features are standardised and reduced to 100 principal components (PCA).
    Both are learned on the training folds only.

Two baselines show what a score is worth:
  B1  predicts the class shares of the training folds for every photo (knows nothing about the photo);
  B2  gradient boosting on the photo's size, brightness, sharpness and retina fill only: it never looks
      at the retina, so whatever it gets right comes from the camera and the image quality.

Scores, from the pooled out-of-fold predictions:
  AUROC referable  how well the model ranks eyes that need a referral (classes 1, 2) above the others
  AUROC urgent     how well it ranks urgent eyes (class 2) above the others
  accuracy         share of photos whose most likely class is the true class
The 95 % intervals come from 1,000 bootstrap resamples of the photos.

Run from the project folder:   python src/05_compare_models.py        (about 15 minutes on a laptop)
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
PHOTO_NOTES = ["width", "height", "brightness", "sharpness", "retina_fill"]

# name: (model, the settings tried in the inner grid search)
MODELS = {
    "Logistic regression": (LogisticRegression(max_iter=2000), {"C": [0.01, 0.1, 1]}),
    "Naive Bayes": (GaussianNB(), {"var_smoothing": [1e-9, 1e-2]}),
    "LDA": (LinearDiscriminantAnalysis(), {}),
    "QDA": (QuadraticDiscriminantAnalysis(), {"reg_param": [0.1, 0.5]}),
    "k-nearest neighbours": (KNeighborsClassifier(), {"n_neighbors": [5, 15, 45]}),
    "Decision tree": (DecisionTreeClassifier(random_state=SEED), {"max_depth": [3, 6, 10]}),
    "Random forest": (RandomForestClassifier(n_estimators=200, random_state=SEED), {"min_samples_leaf": [1, 5]}),
    "AdaBoost": (AdaBoostClassifier(random_state=SEED), {"n_estimators": [50, 200]}),
    "Gradient boosting": (HistGradientBoostingClassifier(random_state=SEED), {"learning_rate": [0.05, 0.1]}),
    "SVM (RBF)": (SVC(probability=True, random_state=SEED), {"C": [1, 10]}),
    "MLP": (MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=SEED), {"alpha": [0.001, 0.1]}),
}

table = pd.read_csv("data/table.csv")
features = np.load("data/features.npy")
aptos = (table["dataset"] == "aptos").to_numpy()
X, y = features[aptos], table["label"][aptos].to_numpy()
fold, group = table["fold"][aptos].to_numpy(), table["group"][aptos].to_numpy()
notes = table.loc[aptos, PHOTO_NOTES].to_numpy()


def out_of_fold(make_model, inputs):
    """Class probabilities for every APTOS photo from the model trained on the other nine folds."""
    probabilities, chosen = np.zeros((len(y), 3)), []
    for k in range(10):
        train, test = fold != k, fold == k
        model = make_model()
        if isinstance(model, GridSearchCV):
            model.fit(inputs[train], y[train], groups=group[train])     # groups: copies stay in one inner fold
        else:
            model.fit(inputs[train], y[train])
        probabilities[test] = model.predict_proba(inputs[test])
        if isinstance(model, GridSearchCV) and model.best_params_:
            chosen.append(str(list(model.best_params_.values())[0]))
    return probabilities, (pd.Series(chosen).mode()[0] if chosen else "")


def with_grid_search(model, grid):
    """Standardise -> PCA -> model, with the model's setting picked by a 3-fold grid search."""
    steps = Pipeline([("scale", StandardScaler()), ("pca", PCA(100, random_state=SEED)), ("model", model)])
    grid = {"model__" + name: values for name, values in grid.items()}
    return GridSearchCV(steps, grid, cv=GroupKFold(3), scoring="roc_auc_ovr")


def scores(p):
    """AUROC referable, AUROC urgent, accuracy; each with its 95 % bootstrap interval."""
    def all_three(rows):
        return (roc_auc_score(y[rows] >= 1, 1 - p[rows, 0]), roc_auc_score(y[rows] == 2, p[rows, 2]),
                np.mean(p[rows].argmax(axis=1) == y[rows]))
    rng = np.random.RandomState(SEED)
    resampled = np.array([all_three(rng.randint(0, len(y), len(y))) for _ in range(1000)])
    low, high = np.percentile(resampled, [2.5, 97.5], axis=0)
    return [f"{value:.3f} ({a:.3f}-{b:.3f})" for value, a, b in zip(all_three(np.arange(len(y))), low, high)]


rows = []
for name, (model, grid) in MODELS.items():
    start = time.time()
    p, setting = out_of_fold(lambda: with_grid_search(model, grid), X)
    rows.append([name, *scores(p), f"{list(grid)[0]} = {setting}" if grid else ""])
    print(f"{name:22s} {rows[-1][1]}   {time.time() - start:4.0f} s", flush=True)

p, _ = out_of_fold(lambda: DummyClassifier(strategy="prior"), X)
rows.append(["B1 class shares", *scores(p), ""])
p, _ = out_of_fold(lambda: HistGradientBoostingClassifier(random_state=SEED), notes)
rows.append(["B2 photo notes only (no retina)", *scores(p), ""])

result = pd.DataFrame(rows, columns=["model", "AUROC referable", "AUROC urgent", "accuracy", "setting chosen most often"])
result = result.sort_values("AUROC referable", ascending=False)
Path("results").mkdir(exist_ok=True)
result.to_csv("results/models.csv", index=False)
print("\n" + result.to_string(index=False))
