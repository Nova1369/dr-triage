"""Step 1. List every photo with its DR grade and referral class -> data/photos.csv

The three datasets store their labels differently, so this step puts them into one table:

  APTOS 2019   datasets/aptos/train.csv                     columns: id_code, diagnosis
  IDRiD        datasets/idrid/2. Groundtruths/*.csv         columns: Image name, Retinopathy grade
  Messidor-2   datasets/messidor2/messidor_data.csv         columns: image_id, adjudicated_dr_grade,
                                                                     adjudicated_gradable

The DR grade (0-4) of each photo is grouped into the three referral classes of the project:

  label 0 = no referral       (grade 0 no DR, grade 1 mild)
  label 1 = routine referral  (grade 2 moderate)
  label 2 = urgent referral   (grade 3 severe, grade 4 proliferative)

Run from the project folder:   python src/01_list_photos.py
"""
from pathlib import Path

import pandas as pd

DATASETS = Path("datasets")
OUT = Path("data")
OUT.mkdir(exist_ok=True)


def referral_class(grade):
    if grade <= 1:
        return 0
    if grade == 2:
        return 1
    return 2


rows = []   # one (dataset, photo name, file path, grade) per photo

# APTOS 2019: the training set (the labels of its test set were never published)
aptos = pd.read_csv(DATASETS / "aptos/train.csv")
for name, grade in zip(aptos["id_code"], aptos["diagnosis"]):
    rows.append(("aptos", name, f"aptos/train_images/{name}.png", grade))

# IDRiD: one label file for its training photos and one for its test photos; we use all of them
for labels, folder in [("a. IDRiD_Disease Grading_Training Labels.csv", "a. Training Set"),
                       ("b. IDRiD_Disease Grading_Testing Labels.csv", "b. Testing Set")]:
    table = pd.read_csv(DATASETS / "idrid/2. Groundtruths" / labels)
    table.columns = [c.strip() for c in table.columns]          # the headers have stray spaces
    for name, grade in zip(table["Image name"], table["Retinopathy grade"]):
        rows.append(("idrid", name, f"idrid/1. Original Images/{folder}/{name}.jpg", grade))

# Messidor-2: grades agreed by a panel of specialists; photos they could not grade are left out
messidor = pd.read_csv(DATASETS / "messidor2/messidor_data.csv")
messidor = messidor[messidor["adjudicated_gradable"] == 1]
files = {f.name.lower(): f.name for f in (DATASETS / "messidor2/IMAGES").iterdir()}   # .jpg on paper, .JPG on disk
for name, grade in zip(messidor["image_id"], messidor["adjudicated_dr_grade"]):
    rows.append(("messidor2", Path(name).stem, f"messidor2/IMAGES/{files[name.lower()]}", grade))

photos = pd.DataFrame(rows, columns=["dataset", "photo", "path", "grade"])
photos["grade"] = photos["grade"].astype(int)
photos["label"] = photos["grade"].map(referral_class)

missing = [p for p in photos["path"] if not (DATASETS / p).exists()]
assert not missing, f"{len(missing)} photos in the label files are not on disk, for example {missing[:3]}"

photos.to_csv(OUT / "photos.csv", index=False)

counts = photos.groupby(["dataset", "label"]).size().unstack()
counts.columns = ["no referral", "routine", "urgent"]
counts["all"] = counts.sum(axis=1)
print(counts)
print(f"\n{len(photos)} photos listed in {OUT / 'photos.csv'}")
