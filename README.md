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
| 6 | `python src/06_referral_rule.py` | tests the referral rule (`src/conformal.py`) on APTOS against simpler rules |
| 7 | `python src/07_new_clinics.py` | applies the APTOS model and thresholds, unchanged, to IDRiD and Messidor-2 |

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

**Step 6: does the referral rule keep its promise?** Every APTOS photo is decided once, by a model and
thresholds that never saw it; full table in [`results/referral_rule.csv`](results/referral_rule.csv).

| Rule | Referable eyes sent home (target at most 10 %) | Urgent eyes referred urgently (target at least 90 %) | Healthy eyes sent home | To a grader | Referrals per 1,000 |
|---|---|---|---|---|---|
| Most likely class | 11.7 % | 51.0 % | 93.4 % | 0 % | 398 |
| B3 tuned cut-offs | 9.0 % | 87.9 % | 91.0 % | 0 % | 423 |
| B4 one conformal threshold | 6.6 % | 69.7 % | 89.2 % | 8.5 % | 358 |
| B5 unsure band | 7.9 % | 39.3 % | 90.8 % | 14.3 % | 286 |
| **C1 one threshold per class (ours)** | **4.4 %** | **91.4 %** | 87.2 % | 1.8 % | 446 |

Only C1 meets both targets (apart from the trivial rule of referring everyone urgently). With one
threshold for all classes (B4), the true class is in the set for 95.9 % of healthy eyes but only 69.7 %
of urgent eyes; with one threshold per class the three classes get 90.3 %, 90.9 % and 91.4 %. The price
of C1 is more referrals: 446 per 1,000 eyes, because 68 % of grade-2 eyes are referred urgently.

**Step 7: does the promise hold at a new clinic?** The same 10 models and thresholds as in step 6,
applied unchanged to all photos of IDRiD and Messidor-2 (average of the 10 rounds); full table with the
lowest and highest round in [`results/new_clinics.csv`](results/new_clinics.csv).

| Clinic | Rule | True class in the set (promised: 90 %) | Referable eyes sent home (target at most 10 %) | Urgent eyes referred urgently (target at least 90 %) | Referrals per 1,000 |
|---|---|---|---|---|---|
| APTOS (home) | B4 one conformal threshold | 90.3 % | 6.6 % | 69.7 % | 358 |
| APTOS (home) | C1 one threshold per class | 90.6 % | 4.4 % | 91.4 % | 446 |
| IDRiD | B4 one conformal threshold | 76.8 % | 16.8 % | 58.2 % | 391 |
| IDRiD | C1 one threshold per class | 82.2 % | 13.6 % | 87.0 % | 578 |
| Messidor-2 | B4 one conformal threshold | 75.3 % | 91.8 % | 19.2 % | 17 |
| Messidor-2 | C1 one threshold per class | 76.3 % | 87.1 % | 38.8 % | 44 |

No. The 90 % of conformal prediction holds only when the new photos are like the calibration photos, and
at a new clinic they are not. At IDRiD both rules slip below their targets. At Messidor-2 the rule
fails completely: the model gives almost every photo a high probability of "no referral" (0.96 on
average even for grade-2 eyes), so 87 % of the eyes that need a referral are sent home. The model's
ranking of referable eyes drops from an AUROC of 0.972 at APTOS to 0.910 at IDRiD and 0.684 at
Messidor-2. Thresholds set at one clinic cannot be carried to another; the next step sets them again
with a few photos of the new clinic.

## Still to come

Recalibration at a new clinic with a few local photos, the effect of fewer sick eyes on the workload,
and the error and robustness analyses.
