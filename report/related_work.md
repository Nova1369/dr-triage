# Related work

Every number below was checked against the paper itself (abstract or full text). "Referable" means moderate or worse DR, as in our project.

## Automated DR screening

Abràmoff et al. (2013) tested the Iowa Detection Program (IDP) on 874 people with diabetes photographed in primary care clinics in France (the Messidor-2 cohort), with consensus grading by three retinal specialists as the standard. Referable DR (more than mild NPDR and/or macular edema) had a prevalence of 21.7 %. At a predetermined, fixed set point the IDP had sensitivity 96.8 % and specificity 59.4 % (6/874 false negatives, none meeting treatment criteria) and an AUC of 0.937. Before consensus, the three specialists had sensitivity of 0.71 to 0.91 and specificity of 0.95 to 1.00.

Gulshan et al. (2016) trained a deep network on 128,175 fundus photos and tested it for referable DR on EyePACS-1 (9,963 photos) and Messidor-2 (1,748 photos). The AUC was 0.991 and 0.990. At the operating point chosen for high sensitivity on the development data, sensitivity was 97.5 % and 96.1 % and specificity 93.4 % and 93.9 %.

Gulshan et al. (2019) tested a similar system prospectively at two sites in India and compared it with manual grading. The system reached sensitivity of 88.9 % and 92.1 % and specificity of 92.2 % and 95.2 % at the two sites, equal to or better than individual manual graders (sensitivity 73.4 to 89.8 %, specificity 83.5 to 98.7 %).

Abràmoff et al. (2016) added deep learning to the Iowa Detection Program (IDx-DR) and tested it on Messidor-2: AUC 0.980, sensitivity 96.8 %, specificity 87.0 %, 6/874 false negatives, and no severe NPDR, PDR or macular edema missed.

How we differ. These papers measure the sensitivity and specificity of a fixed algorithm at an operating point. We do not compete on accuracy: our model is a frozen ResNet-50 plus logistic regression trained on 3,662 APTOS photos. Our contribution is the rule on top of it, which turns two targets (at most 10 % of referable eyes sent home, at least 90 % of urgent eyes referred urgently) into a guarantee that we then test at new clinics. Messidor-2, the test set of Gulshan 2016 and of both Abràmoff papers, is our second new clinic.

## Conformal prediction

Vovk (2012) notes that inductive conformal predictors were known to control only unconditional coverage, and studies several kinds of conditional validity. A conditional version of the method gives label-conditional validity, which controls the set-prediction counterparts of false-positive and false-negative rates. Our one threshold per class is of this kind, and it is what lets us state a guarantee for each referral class.

Hasan et al. (2026) is the closest work. They train EfficientNet-B0, ResNet-50 and ViT-Base on APTOS, calibrate split conformal predictors (LAC, APS, RAPS and an ordinal score) on held-out APTOS photos, and test on IDRiD and, with a ViT, Messidor-2. Coverage was about nominal on APTOS but fell on both external sets. At the 90 % target, coverage on IDRiD was 0.657 (LAC), 0.669 (RAPS), 0.710 (APS) and 0.766 (their ordinal score) for EfficientNet-B0, and 0.684 (LAC), 0.723 (RAPS), 0.814 (APS) and 0.858 (ordinal) for ResNet-50. The loss was largest for severe DR (grade 3): 0.302 to 0.431 (EfficientNet-B0) and 0.452 to 0.636 (ResNet-50) for the three standard scores. On Messidor-2 (ViT, 90 %) coverage was 0.632 (APS) and 0.676 (ordinal). They conclude that ordinal sets help but do not restore the guarantee. They name label-prior adjustment and shift-aware calibration as future work.
How we differ. (1) Our guarantee is per referral class, not marginal, and is tied to clinical targets. (2) We do not stop at showing the failure: setting the thresholds again with 50 to 300 labelled local photos brings the coverage back to about 90 % at IDRiD and Messidor-2 in our experiments, without retraining. This needs enough photos: at Messidor-2, 50 local photos hold too few urgent eyes to set the urgent threshold, so the rule is safe but refers every eye urgently, and it starts to save work only at about 200 to 300 photos. (3) We forecast the workload at a clinic from unlabeled photos. Our one-threshold baseline (the LAC score with a single threshold for all three referral classes, on frozen ResNet-50 features with logistic regression) puts the true class in the set for 76.8 % of the 516 IDRiD photos, against 68.4 % for their ResNet-50 trained on APTOS with LAC on 455 photos and five DR grades. The two numbers are indicative only, because the label sets and photo sets differ.

Ahamed et al. (2025) (MICCAI 2025 workshop) study how five training augmentations (none, geometric, CLAHE, Mixup, CutMix) change conformal prediction for DR grading, with ResNet-50 and a CoaT transformer on the DDR dataset. They measure coverage, average set size and a correct-efficiency metric, and find that Mixup and CutMix improve accuracy and give more reliable and efficient uncertainty, while CLAHE can hurt.
How we differ. They vary the training recipe on one dataset. We keep the model fixed and vary the clinic and the way thresholds are set, and our guarantee is per class.

## Label shift

Podkopaev and Ramdas (2021) argue that label shift, a change in class shares, hurts uncertainty quantification as well as prediction. They adapt conformal prediction and calibration by reweighting when some unlabeled data from the target are available, and they also consider label-conditional conformal prediction as another route.
How we differ. Their setting is label shift. At our new clinics the photos themselves also change (the model's AUROC for referable eyes falls from 0.972 at APTOS to 0.910 at IDRiD and 0.684 at Messidor-2), which is a different kind of shift. Per-class thresholds make the guarantee independent of the class shares by construction, and labelled local photos handle the rest.

Saerens et al. (2002) give an EM procedure that estimates the new class probabilities of a data set from unlabeled data and adjusts a classifier's outputs accordingly, without refitting. They also give a test of whether the class probabilities changed, and report that the adjusted classifier is more accurate when they did.
How we differ. We use the same estimate for a different purpose: the forecast of referrals per 1,000 eyes at a clinic. When only the class shares change, the forecast is off by 7 to 11 referrals per 1,000, against up to 294 when APTOS shares are used. At Messidor-2 it fails: the estimate says 0.3 % referable eyes, the truth is 26.2 %.

## References

- Abràmoff MD, Folk JC, Han DP, et al. Automated analysis of retinal images for detection of referable diabetic retinopathy. *JAMA Ophthalmol* 131(3):351-357, 2013.
- Gulshan V, et al. Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. *JAMA* 316(22), 2016.
- Gulshan V, et al. Performance of a deep-learning algorithm vs manual grading for detecting diabetic retinopathy in India. *JAMA Ophthalmol* 137(9), 2019.
- Abràmoff MD, et al. Improved automated detection of diabetic retinopathy on a publicly available dataset through integration of deep learning. *Invest Ophthalmol Vis Sci* 57(13), 2016.
- Vovk V. Conditional validity of inductive conformal predictors. *ACML 2012* (PMLR 25).
- Hasan U, Nayeem MA, Alghamdi TG. Uncertainty-aware AI-assisted diabetic retinopathy grading from fundus images with ordinal conformal prediction. *Diagnostics* 16(16):2645, 2026.
- Ahamed R, et al. Effect of data augmentation on conformal prediction for diabetic retinopathy. MICCAI 2025 workshop, arXiv:2508.14266.
- Podkopaev A, Ramdas A. Distribution-free uncertainty quantification for classification under label shift. *UAI 2021* (PMLR 161).
- Saerens M, Latinne P, Decaestecker C. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Computation* 14(1), 2002.
