import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import conformal as cf  # noqa: E402

CLASS_SHARES = [0.6, 0.27, 0.13]


def photos_from_labels(labels, rng):
    """Made-up probabilities: the true class gets a higher value, then softmax."""
    signal = rng.normal(size=(len(labels), 3)) + 2.0 * np.eye(3)[labels]
    return np.exp(signal) / np.exp(signal).sum(axis=1, keepdims=True)


def fake_photos(n, rng):
    labels = rng.choice(3, size=n, p=CLASS_SHARES)
    return photos_from_labels(labels, rng), labels


def photos_with_counts(counts, rng):
    """Exactly counts[c] photos of class c."""
    labels = np.repeat(np.arange(3), counts)
    return photos_from_labels(labels, rng), labels


def one_set(no_referral, routine, urgent):
    """A prediction set for one photo, as the True/False row that decide() expects."""
    return np.array([[no_referral, routine, urgent]], dtype=bool)


# 1. Infinite threshold with fewer than 9 calibration photos ------------------

@pytest.mark.parametrize("n", range(0, 9))
def test_threshold_is_infinite_with_fewer_than_9_scores(n):
    scores = np.random.default_rng(n).uniform(size=n)
    assert cf.conformal_threshold(scores) == np.inf


def test_threshold_is_finite_with_9_scores():
    scores = np.random.default_rng(0).uniform(size=9)
    assert cf.conformal_threshold(scores) == pytest.approx(scores.max())


def test_threshold_takes_the_right_position():
    # n = 19: position ceil(20 * 0.9) = 18; n = 100: position ceil(101 * 0.9) = 91
    assert cf.conformal_threshold(np.arange(1, 20)) == 18
    assert cf.conformal_threshold(np.arange(1, 101)) == 91


def test_class_with_8_photos_gets_infinite_threshold_and_is_always_kept():
    rng = np.random.default_rng(0)
    cal_p, cal_y = photos_with_counts([100, 100, 8], rng)
    thresholds = cf.class_thresholds(cal_p, cal_y)
    assert np.isinf(thresholds[2])
    assert np.isfinite(thresholds[:2]).all()
    new_p, _ = fake_photos(500, rng)
    assert cf.prediction_sets(new_p, thresholds)[:, 2].all()


def test_class_with_9_photos_gets_finite_threshold():
    cal_p, cal_y = photos_with_counts([100, 100, 9], np.random.default_rng(0))
    assert np.isfinite(cf.class_thresholds(cal_p, cal_y)).all()


# 2. True class in the set about 90 % of the time, for each class ------------

@pytest.fixture(scope="module")
def coverage():
    """Average over 20 calibration draws of: share of photos of class c whose set holds c."""
    rng = np.random.default_rng(1)
    runs = []
    for _ in range(20):
        cal_p, cal_y = fake_photos(5000, rng)
        new_p, new_y = fake_photos(20000, rng)
        sets = cf.prediction_sets(new_p, cf.class_thresholds(cal_p, cal_y))
        runs.append([sets[new_y == c, c].mean() for c in range(3)])
    return np.mean(runs, axis=0)


@pytest.mark.parametrize("c", [cf.NO_REFERRAL, cf.ROUTINE, cf.URGENT])
def test_true_class_is_in_the_set_about_90_percent(coverage, c):
    assert coverage[c] == pytest.approx(0.90, abs=0.01)


# 3. decide ---------------------------------------------------------------------

@pytest.mark.parametrize("no_referral,routine", list(itertools.product([False, True], repeat=2)))
def test_urgent_in_the_set_always_gives_urgent(no_referral, routine):
    assert cf.decide(one_set(no_referral, routine, True))[0] == cf.URGENT


def test_empty_set_goes_to_a_grader():
    assert cf.decide(one_set(False, False, False))[0] == cf.GRADER


def test_single_class_sets():
    assert cf.decide(one_set(True, False, False))[0] == cf.NO_REFERRAL
    assert cf.decide(one_set(False, True, False))[0] == cf.ROUTINE


def test_no_referral_and_routine_together_go_to_a_grader():
    assert cf.decide(one_set(True, True, False))[0] == cf.GRADER


def test_decide_handles_many_photos_at_once():
    sets = np.vstack([one_set(*s) for s in itertools.product([False, True], repeat=3)])
    expected = [cf.GRADER, cf.URGENT, cf.ROUTINE, cf.URGENT,
                cf.NO_REFERRAL, cf.URGENT, cf.GRADER, cf.URGENT]
    assert cf.decide(sets).tolist() == expected


# 4. Shuffling the calibration photos does not change the thresholds ---------

@pytest.mark.parametrize("counts", [[300, 150, 60], [300, 150, 8]])  # second case has an infinite threshold
def test_shuffling_calibration_photos_does_not_change_thresholds(counts):
    cal_p, cal_y = photos_with_counts(counts, np.random.default_rng(3))
    before = cf.class_thresholds(cal_p, cal_y)
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(len(cal_y))
        after = cf.class_thresholds(cal_p[order], cal_y[order])
        np.testing.assert_array_equal(before, after)