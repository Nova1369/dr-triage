# Diabetic retinopathy referral triage with a sensitivity guarantee

Machine Learning course project, M.Tech (Data and Computational Sciences), IIT Jodhpur.
Swayam Sahu, Sahil Sihag, Nitesh Kumar.

## What the project does

People with diabetes need regular eye checks, because diabetic retinopathy (DR) can blind them before
they notice anything. From one photo of the retina (a fundus photo), the system decides what should
happen next:

* **no referral** (DR grade 0 or 1),
* **routine referral** to an eye doctor (grade 2),
* **urgent referral** (grade 3 or 4).

A plain classifier gives no promise about how many sick eyes it sends home. Our aim is a referral rule
with a stated guarantee:

* at most 10 % of the eyes that need a referral (grades 2-4) are sent home, and
* at least 90 % of the urgent eyes (grades 3-4) are referred urgently.

## Method in three lines

1. A pretrained network (ResNet-50) turns each photo into 2,048 numbers. We do not train it.
2. A simple classifier from the course is trained on those numbers and gives a probability for each class.
3. Conformal prediction turns the probabilities into a referral decision, with one threshold per class
   chosen on photos that were kept aside for this purpose (calibration photos).

## Data

Three public datasets. None of them is in this repository; download them and place them like this:

| Dataset | Role | Download | Folder |
|---|---|---|---|
| APTOS 2019 (3,662 photos, India) | training and testing | https://www.kaggle.com/competitions/aptos2019-blindness-detection/data | `datasets/aptos/` (`train.csv`, `train_images/`) |
| IDRiD, part B "Disease Grading" (516 photos, India) | a new clinic | https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid | `datasets/idrid/` (`1. Original Images/`, `2. Groundtruths/`) |
| Messidor-2 (1,748 photos, France) | a second new clinic | https://www.adcis.net/en/third-party/messidor2/ | `datasets/messidor2/IMAGES/` |
| Messidor-2 grades (Krause et al.) | labels for Messidor-2 | https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades | `datasets/messidor2/messidor_data.csv` |

Messidor-2 images: Kindly provided by the Messidor program partners (see https://www.adcis.net/en/third-party/messidor/).
Messidor-2 grades: Krause et al., Ophthalmology 125(8), 2018.

The Messidor-2 licence does not allow passing the photos on, in any form, so no photo, crop or feature
file is stored here (`datasets/` and `data/` are ignored by git).

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The pretrained ResNet-50 weights (102 MB) go to `weights/resnet50_ram-a26f946b.pth`:
https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth
(SHA-256 `a26f946bdcb2e618263d03d8b60b66a26f7788dbadc1940c315b2bf62020a3e7`).

## Steps

Run every script from the project folder, in this order. Each one prints a short summary of what it did.

| Step | Command | What it does |
|---|---|---|
| 1 | `python src/01_list_photos.py` | lists every photo with its DR grade and referral class in one table |
| 2 | `python src/02_crop_photos.py` | crops each photo to its retina and resizes it to 224 x 224 |
| 3 | `python src/03_extract_features.py` | computes the 2,048 ResNet-50 features of each crop |
| 4 | `python src/04_make_folds.py` | finds copies of the same photo and splits APTOS into 10 folds |
| 5 | `python src/05_compare_models.py` | compares the course's models on the features with nested cross-validation (about 15 minutes) |

Steps 2 and 3 take a few minutes each on a laptop and can be stopped and restarted.

## Results so far

**Step 5: which model reads the features best?** 3,662 APTOS photos, 10-fold nested cross-validation;
full table with 95 % intervals in [`results/models.csv`](results/models.csv).

| Model | AUROC referable | AUROC urgent | Accuracy |
|---|---|---|---|
| Logistic regression | 0.972 | 0.917 | 0.825 |
| LDA | 0.970 | 0.919 | 0.816 |
| SVM (RBF) | 0.970 | 0.910 | 0.824 |
| MLP | 0.966 | 0.875 | 0.790 |
| Gradient boosting | 0.963 | 0.898 | 0.799 |
| k-nearest neighbours | 0.957 | 0.878 | 0.774 |
| Random forest | 0.953 | 0.891 | 0.777 |
| AdaBoost | 0.944 | 0.873 | 0.763 |
| QDA | 0.941 | 0.881 | 0.775 |
| B2: photo size, brightness, sharpness only | 0.921 | 0.782 | 0.725 |
| Decision tree | 0.908 | 0.821 | 0.721 |
| Naive Bayes | 0.839 | 0.790 | 0.645 |
| B1: class shares | 0.494 | 0.492 | 0.594 |

Logistic regression, LDA and the SVM are within each other's intervals; we carry logistic regression
forward because it is the simplest. B2 never sees the retina and still reaches 0.921: in APTOS, the
camera and the image quality already say a lot about the grade, which we look at again in the error
analysis.

## Still to come

The referral rule and its baselines, recalibration at a new clinic, the effect of fewer sick eyes on the
workload, and the error and robustness analyses.
