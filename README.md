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
| 8 | `python src/08_recalibration.py` | sets the thresholds again with a few photos of the new clinic, and compares that with retraining (about 8 minutes) |
| 9 | `python src/09_fewer_sick_eyes.py` | tests the rule at clinics with fewer sick eyes, and forecasts the number of referrals from photos without labels |

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

**Step 8: can a few local photos repair it?** The new clinic grades n of its own photos. We compare four
uses of them, each tested on the clinic's other photos (average of 50 runs); all sizes n = 50 to 300 and
the spread of the runs are in [`results/recalibration.csv`](results/recalibration.csv). With n = 300:

| Clinic | Rule | True class in the set (promised: 90 %) | Referable eyes sent home (target at most 10 %) | Urgent eyes referred urgently (target at least 90 %) | To a grader | Referrals per 1,000 |
|---|---|---|---|---|---|---|
| IDRiD | C1 nothing changed | 82.1 % | 13.6 % | 87.1 % | 5.5 % | 578 |
| IDRiD | **C2 thresholds set again on local photos (ours)** | **90.5 %** | **5.7 %** | **90.3 %** | 20.8 % | 553 |
| IDRiD | B6 model retrained with the local photos | 82.4 % | 6.8 % | 96.6 % | 3.3 % | 690 |
| IDRiD | B7 half to retrain, half to set thresholds | 91.4 % | 4.8 % | 91.6 % | 20.7 % | 566 |
| Messidor-2 | C1 nothing changed | 76.2 % | 87.1 % | 38.4 % | 0.2 % | 44 |
| Messidor-2 | **C2 thresholds set again on local photos (ours)** | **90.2 %** | **6.4 %** | **92.2 %** | 45.4 % | 411 |
| Messidor-2 | B6 model retrained with the local photos | 80.1 % | 46.8 % | 71.6 % | 8.2 % | 202 |
| Messidor-2 | B7 half to retrain, half to set thresholds | 90.8 % | 2.8 % | 95.6 % | 19.3 % | 740 |

Yes. Setting the thresholds again on local photos brings the 90 % back at both clinics and for every n
we tried, and both targets are met on average, without touching the model. Retraining alone (B6) does
not: 300 local photos among 2,900 APTOS photos barely move the model, and at Messidor-2 it still sends
home 47 % of the eyes that need a referral. B7 also keeps the promise, because it too sets its
thresholds on local photos, but it needs a new model for every clinic.

The guarantee does not make the model better, and the cost shows in the workload. At Messidor-2, where
the model ranks eyes poorly, the rule stays safe by being unsure: 45 % of the eyes go to a human grader.
With very few local photos it is safe but useless: among 50 Messidor-2 photos there are about 3 urgent
eyes, too few to set the urgent threshold, so every eye is referred urgently. About 200 to 300 local
photos are needed there before the rule starts to save work. The numbers are averages: a single clinic's
draw can land a few points on either side of the targets (see the ranges in the CSV).

**Step 9: what if the clinic has fewer sick eyes?** In APTOS 41 % of the eyes need a referral; in a
screening programme it is nearer 5 to 25 %. We draw clinics of 1,000 eyes with fewer referable eyes from
APTOS (average of 200 clinics); the spread is in [`results/fewer_sick_eyes.csv`](results/fewer_sick_eyes.csv).

| Referable eyes at the clinic | C1: referable eyes sent home (target at most 10 %) | C1: urgent eyes referred urgently (target at least 90 %) | B4: true class in the set | B4: urgent eyes with "urgent" in the set | C1: referrals per 1,000 | C1: referrals that were needed |
|---|---|---|---|---|---|---|
| 5 % | 4.0 % | 91.5 % | 95.2 % | 70.1 % | 151 | 31.1 % |
| 12.5 % | 4.5 % | 91.2 % | 94.2 % | 68.4 % | 214 | 55.0 % |
| 25 % | 4.6 % | 91.4 % | 92.5 % | 69.7 % | 316 | 74.2 % |
| 40.6 % (APTOS) | 4.4 % | 91.6 % | 90.3 % | 69.7 % | 446 | 85.4 % |

The promise does not move. Each class has its own threshold, set only on eyes of that class, so it does
not matter how many of them a clinic has. Standard conformal prediction (B4) looks better and better on
paper as sick eyes get rarer (95.2 % at 5 %), while it still covers only 70 % of the urgent eyes: one
promise for all eyes together is mostly a promise about the healthy ones. What does move is the
workload. At 5 % referable eyes C1 makes 151 referrals per 1,000 eyes instead of 446, and only 31 % of
them were needed.

A clinic should know that workload in advance. The forecast is 1000 x the sum over the classes of
(share of the class) x (how often the rule refers that class, measured on APTOS). The shares come from
APTOS (B8), from an estimate on the clinic's photos without labels (C3, ours; Saerens et al., 2002), or
from the truth (B9, not known in practice). Referrals per 1,000 eyes, and in brackets how far the forecast
is off on average; full table in [`results/workload_forecast.csv`](results/workload_forecast.csv).

| Clinic | Referable eyes: true | Referable eyes: estimated without labels | Referrals observed | Forecast B8 (APTOS shares) | Forecast C3 (estimated shares, ours) | Forecast B9 (true shares) |
|---|---|---|---|---|---|---|
| APTOS with 5 % referable | 5.0 % | 5.8 % | 151 | 445 (294) | 157 (11) | 151 (11) |
| APTOS with 12.5 % referable | 12.5 % | 13.0 % | 214 | 446 (232) | 218 (9) | 213 (11) |
| APTOS with 25 % referable | 25.0 % | 25.1 % | 316 | 447 (132) | 319 (7) | 318 (10) |
| IDRiD | 62.6 % | 60.9 % | 578 | 446 (131) | 612 (44) | 632 (55) |
| Messidor-2 | 26.2 % | 0.3 % | 44 | 446 (402) | 113 (69) | 326 (282) |

When only the share of sick eyes changes, the forecast from photos without labels is as good as with
the true shares (off by about 10 referrals per 1,000), and without a correction it is off by up to 294.
At the real clinics more than the shares has changed. At IDRiD the estimate is still close. At Messidor-2
it says 0.3 % referable eyes when the truth is 26.2 %, because the model calls almost every photo
healthy (step 7); even the true shares give a wrong forecast there, because the rule no longer refers
each class as often as it did at APTOS. The estimate without labels is safe to use only after the
model has been checked on some labelled local photos, and those photos (step 8) then give the shares
directly.

## Still to come

The error and robustness analyses.
