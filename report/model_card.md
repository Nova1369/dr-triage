# Model card: diabetic retinopathy referral triage with a sensitivity guarantee

A research prototype from a Machine Learning course project (M.Tech, Data and Computational Sciences, IIT Jodhpur) by Swayam Sahu, Sahil Sihag and Nitesh Kumar. It is not a medical device and has not been validated in a clinic.

## What it is

From one retinal (fundus) photo the system decides what should happen next: **no referral** (DR grade 0-1), **routine referral** (grade 2), **urgent referral** (grade 3-4), or **to a grader** (a person decides).

1. A pretrained ResNet-50, not trained by us, turns the photo (cropped to 224 x 224) into 2,048 numbers.
2. Standardising, PCA with 100 components and logistic regression (C = 0.01) give a probability for each of the three classes.
3. Conformal prediction (`src/conformal.py`) turns the probabilities into a decision with one threshold per class, set on calibration photos with known grades, at alpha = 0.1.

The aim is a stated promise: at most 10 % of the eyes that need a referral (grades 2-4) are sent home, and at least 90 % of the urgent eyes (grades 3-4) are referred urgently. On APTOS, with a model and thresholds that never saw the photo, 4.4 % of referable eyes are sent home and about 91 % of urgent eyes are referred urgently. The AUROC for referable eyes is 0.972.

## Intended use

- **For:** screening triage, which sorts photos so that a person can look first at the eyes most likely to need care. The promise is meant for a screening programme that sets the thresholds again on a few hundred graded photos of its own (step 8).
- **Not for:** diagnosis, treatment decisions, or replacing a grader or an eye doctor. Not for use at a new clinic, camera or photo quality without setting the thresholds again. Not for other eye diseases: the labels are DR grades only, and macular edema is not part of the decision.
- **Not tested on:** any population, camera or setting outside the three datasets below. We did not analyse results by age, sex or ethnicity.

## Data and licences

No photo, crop or feature file is stored in the repository (`datasets/` and `data/` are ignored by git).

| Dataset | Used for | Size | Terms |
|---|---|---|---|
| APTOS 2019 (Kaggle; photos from Aravind Eye Hospital, India) | training, calibration and testing | 3,662 photos | Kaggle competition rules, section 7 (competition data). Read the rules page before reuse. |
| IDRiD, part B "Disease Grading" (India) | a new clinic | 516 photos (413 training, 103 test) | Creative Commons Attribution 4.0 (dataset page and Porwal et al. 2018) |
| Messidor-2 (France) | a second new clinic | 1,748 photos of 874 examinations | Free for research and educational use; copying, redistribution and unauthorised commercial use are prohibited (ADCIS). Credit: "Kindly provided by the Messidor program partners". |
| Messidor-2 grades (Krause et al. 2018) | labels for Messidor-2 | 1,744 gradable photos | Grades adjudicated by three retinal specialists, from a Kaggle dataset. Check its page for the licence. |

Grades come from different groups of experts in the three datasets.

## When the guarantee fails

The promise of conformal prediction holds only when the new photos are like the calibration photos. We found three cases where they are not.

1. **A new clinic (step 7).** With the APTOS thresholds, the rule sends home 13.6 % of referable eyes at IDRiD, with 87.0 % of urgent eyes referred urgently, and 87.1 % of referable eyes at Messidor-2, with only 38.8 % of urgent eyes referred urgently. The AUROC for referable eyes falls from 0.972 (APTOS) to 0.910 (IDRiD) and 0.684 (Messidor-2). *What repairs it:* setting the thresholds again on local graded photos (step 8). With 300 photos, 5.7 % (IDRiD) and 6.4 % (Messidor-2) of referable eyes are sent home. At Messidor-2 the rule is safe but not useful with 50 photos, because it refers every eye urgently, and it starts to save work at about 200 to 300 photos. Even then 45 % of its eyes go to a grader.
2. **Worse photos (step 13).** Blurred photos send home 11.8 % of referable eyes, and photos saved as JPEG at quality 20 about 15 %. Darker photos keep the sent-home rate within the target but refer only about 87 % of urgent eyes urgently. *What repairs it:* thresholds set on photos that are worse in the same way, at a price. With compressed photos only about 39 % of healthy eyes are sent home and about 21 % of all eyes go to a grader.
3. **The camera shortcut (step 15).** The camera can be read from the features (94.3 % correct), and the model uses it. On the APTOS cameras that photographed mostly healthy eyes, 52.7 % (95 % interval 39.8-65.3) of referable eyes are sent home, hidden in the overall 4.4 %. *What repairs it:* thresholds per k-means cluster bring every camera group below 10 % without knowing the camera, but about 60 % of healthy eyes are sent home instead of 87 %, and about 19 % of eyes go to a grader instead of 2 %.

When only the share of sick eyes at a clinic changes, the promise holds (step 9), but the workload changes a lot: at 5 % referable eyes the rule makes 151 referrals per 1,000 eyes instead of 446, and only 31 % of them are needed.

## Known limits

- The promise is an average over the eyes of a class. It says nothing about one camera, clinic, age group or sex unless the thresholds are set for that group. We did not analyse results by age, sex or ethnicity.
- Each class needs enough calibration photos. A threshold is infinite below 9 photos of a class. For 95 % of clinics to get at least 85 % coverage, a class needs about 110 calibration photos (68 is the first size where this happens; `extra/calibration_size.py`). The troubled cameras hold only 15 urgent eyes in all of APTOS, too few to set an urgent threshold for them (step 15).
- The promise is paid for with workload. At APTOS the rule makes 446 referrals per 1,000 eyes, and only 34.7 % of its urgent referrals are urgent eyes.
- It still misses 8.6 % of urgent eyes (42 of 488) on APTOS. These are confident mistakes (median P(urgent) 0.043), not near misses.
- The frozen network describes how a photo was taken before it describes the disease (step 14). The model ranks eyes poorly at Messidor-2, and the rule then stays safe by being unsure.
- Only three public datasets, from India and France, were used. There was no prospective study and no clinical or regulatory approval.
- There is no alarm yet that warns when a clinic's photos are unlike the calibration photos.

## References

APTOS 2019 Blindness Detection, Kaggle, 2019. Porwal P, et al. Indian Diabetic Retinopathy Image Dataset (IDRiD). *Data* 3(3):25, 2018. Decencière E, et al. Feedback on a publicly distributed image database: the Messidor database. *Image Anal Stereol* 33(3):231-234, 2014. Krause J, et al. *Ophthalmology* 125(8), 2018.
