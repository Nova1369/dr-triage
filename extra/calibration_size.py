"""How many calibration photos of a class does the 90 % promise need?   -> figures/calibration_size.png

The promise of conformal prediction holds on average over all possible sets of calibration photos. A single
clinic has one set. If it happens to hold unlucky photos of a class, its threshold is off and its coverage
for that class is below 90 %.

With n calibration photos of one class and scores without ties, the coverage of ONE clinic (the chance that
a new photo of the class lands in its set, given that clinic's calibration photos) follows a Beta distribution:

        coverage ~ Beta(l, n + 1 - l),     l = ceil((n + 1) * (1 - alpha))

Its mean is l / (n + 1), at least 1 - alpha. Its spread shrinks as n grows. This script
  1. computes the coverage of the unluckiest 5 % of clinics for n = 9 to 1,000 with scipy.stats.beta,
  2. checks the formula by simulation, with class_thresholds from src/conformal.py on made-up photos,
  3. finds the n at which 95 % of clinics get at least 85 % coverage,
  4. marks the class sizes of our own experiments.

Run from anywhere:   python extra/calibration_size.py        (about 20 seconds)
Needs numpy, scipy and matplotlib. Writes only figures/calibration_size.png.
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conformal import URGENT, class_thresholds  # noqa: E402

ALPHA = 0.1                 # allowed miss rate of each class
UNLUCKY = 0.05              # "the unluckiest 5 % of clinics"
GOOD_ENOUGH = 0.85          # coverage we still accept for 95 % of the clinics
N_MIN, N_MAX = 9, 1000      # below 9 photos the threshold is infinite

# Class sizes of our own experiments (one point each on the plot).
REAL_SIZES = [
    (15, "15 urgent eyes on the troubled cameras, all folds together (step 15)"),
    (49, "about 49 urgent eyes per APTOS calibration fold (488 / 10)"),
    (50, "50 local photos at Messidor-2 (step 8; only about 3 of them are urgent)"),
]

SIM_SIZES = [9, 15, 20, 30, 49, 50, 100, 200, 500, 1000]
SIM_CLINICS = 2000          # clinics drawn for each size
POOL = 200_000              # new urgent photos that measure each clinic's coverage
SEED = 0


def beta_parameters(n, alpha=ALPHA):
    """l and n + 1 - l; the same ceil(...) as conformal_threshold in src/conformal.py."""
    l = math.ceil((n + 1) * (1 - alpha))
    return l, n + 1 - l


def unlucky_coverage(n, q=UNLUCKY, alpha=ALPHA):
    """Coverage that 1 - q of the clinics reach or beat, with n calibration photos of the class."""
    l, rest = beta_parameters(n, alpha)
    return float(beta.ppf(q, l, rest))


def mean_coverage(n, alpha=ALPHA):
    l, rest = beta_parameters(n, alpha)
    return l / (l + rest)


# 1. The formula ------------------------------------------------------------------------------------
sizes = np.arange(N_MIN, N_MAX + 1)
unlucky = np.array([unlucky_coverage(n) for n in sizes])
means = np.array([mean_coverage(n) for n in sizes])

# 3. Smallest n at which 95 % of clinics get at least 85 % (first time, and from then on for good)
reached = unlucky >= GOOD_ENOUGH
n_first = int(sizes[reached.argmax()]) if reached.any() else None
n_stays = int(sizes[np.nonzero(~reached)[0].max() + 1]) if (~reached).any() and reached[-1] else n_first


# 2. Simulation with src/conformal.py -----------------------------------------------------------------
def fake_urgent_photos(count, rng):
    """Made-up class probabilities of urgent eyes (the same noisy model as in src/conformal.py's self-check)."""
    signal = rng.normal(size=(count, 3)) + 2.0 * np.eye(3)[URGENT]
    return np.exp(signal) / np.exp(signal).sum(axis=1, keepdims=True)


def simulate(n, rng, pool_scores):
    """Coverage of SIM_CLINICS clinics, each with its own n calibration photos of the urgent class."""
    coverage = np.empty(SIM_CLINICS)
    labels = np.full(n, URGENT)
    for c in range(SIM_CLINICS):
        threshold = class_thresholds(fake_urgent_photos(n, rng), labels, ALPHA)[URGENT]
        coverage[c] = np.searchsorted(pool_scores, threshold, side="right") / len(pool_scores)
    return coverage


rng = np.random.RandomState(SEED)
pool_scores = np.sort(1 - fake_urgent_photos(POOL, rng)[:, URGENT])
simulated = {n: simulate(n, rng, pool_scores) for n in SIM_SIZES}

print(f"alpha = {ALPHA}: coverage of one clinic ~ Beta(l, n + 1 - l), l = ceil((n + 1) * {1 - ALPHA:g})\n")
print(f"{'n':>6} {'mean (formula)':>15} {'mean (simulated)':>17} {'unluckiest 5 % (formula)':>26} {'unluckiest 5 % (simulated)':>28}")
for n in SIM_SIZES:
    print(f"{n:>6} {mean_coverage(n):>15.3f} {simulated[n].mean():>17.3f} "
          f"{unlucky_coverage(n):>26.3f} {np.quantile(simulated[n], UNLUCKY):>28.3f}")
print(f"\nSmallest n with 95 % of clinics at {100 * GOOD_ENOUGH:.0f} % coverage or better: {n_first}"
      f" (and for every larger n up to {N_MAX}: {n_stays})")
print("Our class sizes:")
for n, label in REAL_SIZES:
    print(f"  n = {n:>3}: unluckiest 5 % of clinics get {100 * unlucky_coverage(n):.1f} % or less   ({label})")


# 4. The plot -------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sizes, unlucky, color="C0", linewidth=2,
        label=f"Coverage that 95 % of clinics reach or beat (Beta formula)")
ax.plot(sizes, means, color="grey", linestyle=":", label="Average coverage over all clinics")
ax.plot(SIM_SIZES, [np.quantile(simulated[n], UNLUCKY) for n in SIM_SIZES], "o", color="C3", markersize=5,
        label=f"Simulation with src/conformal.py ({SIM_CLINICS:,} clinics for each n)")
ax.axhline(1 - ALPHA, color="grey", linewidth=0.8)
ax.axhline(GOOD_ENOUGH, color="C2", linestyle="--", linewidth=1)
ax.text(N_MAX, GOOD_ENOUGH - 0.008, f"{100 * GOOD_ENOUGH:.0f} %", color="C2", ha="right", va="top", fontsize=9)
if n_first is not None:
    # The curve is a sawtooth (the position l jumps up by one every ten photos), so there are two answers.
    ax.axvline(n_first, color="C2", linestyle=":", linewidth=1.2)
    ax.text(n_first * 0.97, 0.63, f"n = {n_first}: first time\n95 % of clinics reach {100 * GOOD_ENOUGH:.0f} %",
            color="C2", ha="right", fontsize=8)
if n_stays is not None:
    ax.axvline(n_stays, color="C2", linestyle="--", linewidth=1.2)
    ax.text(n_stays * 1.03, 0.63, f"n = {n_stays}: from here on, always", color="C2", ha="left", fontsize=8)
tag_offsets = [(-16, 4), (8, -14), (-4, 9)]
notes = []
for tag, (n, label), offset in zip("ABC", REAL_SIZES, tag_offsets):
    y = unlucky_coverage(n)
    ax.plot([n], [y], "s", color="black", markersize=6, zorder=5)
    ax.annotate(tag, (n, y), textcoords="offset points", xytext=offset, fontsize=10, fontweight="bold")
    notes.append(f"{tag}: {label}\n     unluckiest 5 % of clinics: {100 * y:.1f} % coverage or less")
ax.text(0.015, 0.03, "\n".join(notes), transform=ax.transAxes, fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="grey"))
ax.set_xscale("log")
ax.set_xlim(N_MIN, N_MAX)
ax.set_ylim(0.5, 1.0)
ax.set_xticks([9, 15, 20, 30, 50, 100, 200, 500, 1000])
ax.set_xticklabels(["9", "15", "20", "30", "50", "100", "200", "500", "1,000"])
ax.set_xlabel("Calibration photos of the class at one clinic (n)")
ax.set_ylabel("Coverage of the class at that clinic")
ax.set_title(f"How lucky must a clinic be? Coverage of the class for the unluckiest 5 % of clinics (alpha = {ALPHA:g})")
ax.legend(loc="center right", fontsize=8)
fig.tight_layout()
out = ROOT / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "calibration_size.png", dpi=200, bbox_inches="tight")
print("\nsaved", out / "calibration_size.png")