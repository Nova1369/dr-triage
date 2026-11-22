"""Step 14. Do the photos group by DR grade, or by camera and clinic? -> results/clusters.csv and results/clusters.png

k-means is given the features of the photos and nothing else: no grade, no camera, no clinic. It puts
the photos in k groups (clusters) so that photos in one cluster have similar features. Afterwards we ask
what the clusters agree with:

  the referral class or DR grade  ->  the features are mostly about the disease, which is what we want
  the camera or the clinic        ->  the features are mostly about how the photo was taken

Agreement is measured with the adjusted Rand index (ARI): 1 if the clusters are exactly the groups we
compare them with, 0 if they agree no more than random groups would.

Two questions:
  1. All photos of the three datasets: do the clusters follow the clinic?
  2. APTOS only: do the clusters follow the referral class, the grade, or the camera? APTOS does not name
     the camera; as in step 10, the photo size (width x height) stands for it.

The features are prepared as for the model: standardised, then 100 principal components.
k-means starts from random centres, so it is run 10 times for every k and the best run is kept.

Run from the project folder:   python src/14_clusters.py        (under a minute)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

SEED = 42
KS = [2, 3, 5, 8, 15]          # 3 = clinics or referral classes, 5 = DR grades, 15 = cameras in APTOS

table = pd.read_csv("data/table.csv")
features = np.load("data/features.npy")
table["camera"] = table["width"].astype(str) + " x " + table["height"].astype(str)
referable_share = (table["label"] >= 1).groupby([table["dataset"], table["camera"]]).transform("mean")
table["camera group"] = np.select([referable_share < 0.1, referable_share > 0.5], ["mostly healthy", "mostly sick"], "mixed")


def components(rows):
    """Standardised features of the given photos, reduced to 100 principal components."""
    return PCA(100, random_state=SEED).fit_transform(StandardScaler().fit_transform(features[rows]))


def clusters(x, k):
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(x)


rows_out = []
everything, aptos = np.ones(len(table), dtype=bool), (table["dataset"] == "aptos").to_numpy()
x_all, x_aptos = components(everything), components(aptos)
for name, rows, x, compare in [("all three datasets", everything, x_all, ["dataset", "label", "grade"]),
                               ("APTOS", aptos, x_aptos, ["label", "grade", "camera", "camera group"])]:
    for k in KS:
        c = clusters(x, k)
        scores = {column: adjusted_rand_score(table.loc[rows, column], c) for column in compare}
        rows_out.append([name, k, *[f"{scores[column]:.3f}" if column in scores else "" for column in
                                    ["dataset", "label", "grade", "camera", "camera group"]]])
result = pd.DataFrame(rows_out, columns=["photos", "k", "ARI with the clinic", "ARI with the referral class",
                                         "ARI with the DR grade", "ARI with the camera", "ARI with the camera group"])
result.to_csv("results/clusters.csv", index=False)
print(result.to_string(index=False))

# ---- what is in each cluster (k = 3) ----
print("\nAll three datasets, k = 3: photos of each clinic in each cluster")
print(pd.crosstab(table["dataset"], pd.Series(clusters(x_all, 3), name="cluster")).to_string())
c = pd.Series(clusters(x_aptos, 3), name="cluster")
a = table[aptos].reset_index(drop=True)
print("\nAPTOS, k = 3: photos of each referral class, and of each camera group, in each cluster")
print(pd.crosstab(a["label"].map({0: "no referral", 1: "routine", 2: "urgent"}), c).to_string())
print(pd.crosstab(a["camera group"], c).to_string())
print("\nAPTOS, k = 3: the five largest cameras in each cluster")
print(pd.crosstab(a["camera"], c).loc[a["camera"].value_counts().index[:5]].to_string())

# ---- the chart: the first two principal components, coloured in four ways ----
figure, axes = plt.subplots(1, 4, figsize=(17, 4.4))
panels = [(x_all, table["dataset"].map({"aptos": "APTOS", "idrid": "IDRiD", "messidor2": "Messidor-2"}), "All photos, by clinic"),
          (x_aptos, c.map(lambda i: f"cluster {i}"), "APTOS, by k-means cluster (k = 3)"),
          (x_aptos, a["label"].map({0: "no referral", 1: "routine", 2: "urgent"}), "APTOS, by referral class"),
          (x_aptos, a["camera group"].map(lambda g: "cameras: " + g), "APTOS, by camera group")]
for axis, (x, groups, title) in zip(axes, panels):
    for colour, group in zip(["#1f77b4", "#ff7f0e", "#2ca02c"], sorted(groups.unique())):
        here = (groups == group).to_numpy()
        axis.scatter(x[here, 0], x[here, 1], s=3, alpha=0.35, color=colour, label=f"{group} ({here.sum():,})")
    axis.set(title=title, xlabel="principal component 1", ylabel="principal component 2")
    legend = axis.legend(fontsize=7, markerscale=4)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
figure.tight_layout()
figure.savefig("results/clusters.png", dpi=130)
