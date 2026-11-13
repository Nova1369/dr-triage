"""The referral rule: one conformal threshold per class. The later steps import these functions.

A model gives every photo three probabilities: no referral (class 0), routine (1), urgent (2).
The rule turns them into a decision in four small steps.

1. Score.  For a class c, the score of a photo is  s = 1 - p(c).  Low score = the model finds c likely.

2. Threshold (class_thresholds).  Take calibration photos: photos with known labels that the model was
   not trained on. For class c, look only at the calibration photos whose true class is c, and at their
   scores for c. Sort these n scores. The threshold q_c is the score at position ceil((n + 1) * 0.9):
   about 90 % of the photos of class c score below it. With fewer than 9 photos of a class the position
   does not exist; the threshold is then infinite and the class is always kept.

3. Prediction set (prediction_sets).  For a new photo, keep every class whose score is at or below its
   threshold. The set can hold one class, several, or none. The guarantee of conformal prediction: if the
   new photos are like the calibration photos, the true class is in the set at least 90 % of the time,
   and this holds for each class separately.

4. Decision (decide).  urgent if "urgent" is in the set; routine if the set is exactly {routine};
   no referral if the set is exactly {no referral}; otherwise a human grader decides.

Why this gives the two targets of the project. An eye that needs a referral is sent home only when its
set is exactly {no referral}, so its true class was left out, which happens at most 10 % of the time.
An urgent eye is referred urgently whenever "urgent" is in its set, which happens at least 90 % of the time.

Run this file to see the guarantee on made-up numbers:   python src/conformal.py
"""
import math

import numpy as np

NO_REFERRAL, ROUTINE, URGENT, GRADER = 0, 1, 2, 3
DECISIONS = ["no referral", "routine referral", "urgent referral", "to a grader"]


def conformal_threshold(scores, alpha=0.1):
    """The ceil((n + 1)(1 - alpha))-th smallest of n scores; infinite if there are too few scores."""
    n = len(scores)
    position = math.ceil((n + 1) * (1 - alpha))
    if position > n:
        return np.inf
    return float(np.sort(scores)[position - 1])


def class_thresholds(probabilities, labels, alpha=0.1):
    """One threshold per class, each from the calibration photos of that class only."""
    scores = 1 - probabilities
    return np.array([conformal_threshold(scores[labels == c, c], alpha) for c in range(probabilities.shape[1])])


def prediction_sets(probabilities, thresholds):
    """True/False table (photos x classes): is the class in the photo's prediction set?"""
    return (1 - probabilities) <= thresholds


def decide(sets):
    """Referral decision for every photo from its prediction set."""
    urgent = sets[:, URGENT]
    only_routine = sets[:, ROUTINE] & ~sets[:, NO_REFERRAL] & ~urgent
    only_no_referral = sets[:, NO_REFERRAL] & ~sets[:, ROUTINE] & ~urgent
    decision = np.full(len(sets), GRADER)
    decision[only_no_referral] = NO_REFERRAL
    decision[only_routine] = ROUTINE
    decision[urgent] = URGENT
    return decision


if __name__ == "__main__":
    # Self-check on made-up data: a noisy three-class "model", 300 calibration and 20,000 new photos.
    rng = np.random.RandomState(0)

    def fake_photos(n):
        labels = rng.choice(3, size=n, p=[0.6, 0.27, 0.13])
        signal = rng.normal(size=(n, 3)) + 2.0 * np.eye(3)[labels]       # the true class gets a higher value
        probabilities = np.exp(signal) / np.exp(signal).sum(axis=1, keepdims=True)
        return probabilities, labels

    coverage = []
    for _ in range(200):                                                  # 200 different calibration draws
        cal_p, cal_y = fake_photos(300)
        new_p, new_y = fake_photos(20000)
        sets = prediction_sets(new_p, class_thresholds(cal_p, cal_y))
        coverage.append([sets[new_y == c, c].mean() for c in range(3)])
    print("True class inside the set, average over 200 calibration draws (target: at least 0.90)")
    for c, value in enumerate(np.mean(coverage, axis=0)):
        print(f"  {DECISIONS[c]:17s} {value:.3f}")
    decisions = decide(sets)
    print("Decisions in the last draw:", {DECISIONS[d]: int((decisions == d).sum()) for d in range(4)})
