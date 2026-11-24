#!/bin/bash
# Runs every step of the project in order, from the raw datasets to the saved model.
#
#   bash run_all.sh          all steps (about 50 minutes on a laptop)
#   bash run_all.sh 5        start again from step 5 (steps 1-4 only need to run once)
#
# Before the first run: datasets in datasets/, ResNet-50 weights in weights/ (see the README).
# The script stops at the first step that fails.
set -e
first=${1:-1}

steps=(
  "01_list_photos.py"       # the table of all photos
  "02_crop_photos.py"       # crops (a few minutes; can be stopped and restarted)
  "03_extract_features.py"  # ResNet-50 features (a few minutes; can be stopped and restarted)
  "04_make_folds.py"        # copies of a photo, 10 folds
  "05_compare_models.py"    # the course's models (about 15 minutes)
  "06_referral_rule.py"     # C1 against simpler rules
  "07_new_clinics.py"       # the rule at IDRiD and Messidor-2
  "08_recalibration.py"     # C2: thresholds set again on local photos (about 8 minutes)
  "09_fewer_sick_eyes.py"   # C3: fewer sick eyes, workload forecast
  "10_error_analysis.py"    # errors by grade, camera, sharpness
  "11_more_scores.py"       # kappa, Brier score, calibration, per-class scores
  "12_change_photos.py"     # features of blurred, dark, compressed photos (about 10 minutes)
  "13_robustness.py"        # robustness, ablations, other targets (about 3 minutes)
  "14_clusters.py"          # k-means
  "15_camera_shortcut.py"   # additional work, part A
  "16_new_clinic_alarm.py"  # additional work, part B (about 4 minutes)
  "17_save_model.py"        # the final model
)

for step in "${steps[@]}"; do
  number=$((10#${step:0:2}))
  if [ "$number" -ge "$first" ]; then
    echo
    echo "================ step $number: python src/$step ================"
    python "src/$step"
  fi
done
echo
echo "All steps finished. Tables and charts are in results/, the model in models/."
