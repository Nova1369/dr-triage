"""Draws the report figures from the CSVs in results/ and saves them to figures/.

Run from anywhere:   python extra/make_figures.py
Reads only results/*.csv. Writes only figures/*.png.

Figures
  1 sent_home_by_group.png            sent-home rate per camera group, with intervals   (errors_by_group.csv)
  2 recalibration_by_clinic.png       the four recalibration rules, one column per clinic (recalibration.csv)
  3 referrals_vs_referable_share.png  referrals per 1,000 and the forecasts              (fewer_sick_eyes.csv, workload_forecast.csv)
  4 cost_of_alpha.png                 what each alpha costs                              (stricter_target.csv)
  5 robustness.png                    worse photos, original vs recalibrated thresholds  (robustness.csv)
  6 ablations.png                     one part of the pipeline changed at a time         (ablations.csv)
  7 rules_by_camera_group.png         the five rules of step 15 per camera group         (group_thresholds.csv)
"""
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

NUMBER = re.compile(r"\d+(?:\.\d+)?")

SENT = "referable eyes sent home % (target <= 10)"
URGENT = "urgent eyes referred urgently % (target >= 90)"
HEALTHY = "healthy eyes sent home %"
GRADER = "to a grader %"


def first(cell):
    """First number in a cell: '446' -> 446, '4.4 (3.5-5.6)' -> 4.4, '40.6 (APTOS)' -> 40.6."""
    found = NUMBER.search(str(cell))
    return float(found.group()) if found else float("nan")


def with_interval(cell):
    """'4.4 (3.5-5.6)' -> (4.4, 3.5, 5.6). A cell without an interval gives (value, nan, nan)."""
    numbers = [float(x) for x in NUMBER.findall(str(cell))]
    if len(numbers) >= 3:
        return numbers[0], numbers[1], numbers[2]
    if len(numbers) >= 1:
        return numbers[0], float("nan"), float("nan")
    return float("nan"), float("nan"), float("nan")


def read(name, columns):
    df = pd.read_csv(RESULTS / f"{name}.csv", encoding="utf-8-sig")
    for column in columns:
        if column not in df.columns:
            raise SystemExit(f"{name}.csv has no column '{column}'.\nColumns found: {list(df.columns)}")
    return df


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


# Figure 1: sent-home rate per camera group, with 95 % intervals -------------------------------
def fig_groups():
    rate_col = "sent home % (95 % interval)"
    df = read("errors_by_group", ["grouped by", "group", "referable eyes", rate_col])
    d = df[df["grouped by"] == "camera group"]
    if d.empty:
        raise SystemExit(f"No 'camera group' rows. Values of 'grouped by': {sorted(df['grouped by'].unique())}")
    order = ["mostly healthy", "mixed", "mostly sick"]
    if set(order) == set(d["group"]):
        d = d.set_index("group").loc[order].reset_index()
    parsed = [with_interval(x) for x in d[rate_col]]
    value = pd.Series([p[0] for p in parsed], index=d.index)
    low = pd.Series([p[1] for p in parsed], index=d.index)
    high = pd.Series([p[2] for p in parsed], index=d.index)
    labels = [f"{g}\n({int(r)} referable eyes)" for g, r in zip(d["group"], d["referable eyes"])]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(range(len(d)), value, yerr=[value - low, high - value], fmt="o", capsize=5, color="C0")
    ax.axhline(10, linestyle="--", color="grey", label="target: at most 10 %")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(d) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Camera group")
    ax.set_ylabel("Referable eyes sent home (%)")
    ax.set_title("Sent-home rate by camera group (95 % intervals)")
    ax.legend()
    save(fig, "sent_home_by_group")


# Figure 2: the four recalibration rules against the number of local photos, one panel per clinic
def fig_recalibration():
    df = read("recalibration", ["clinic", "local photos", "rule", SENT, URGENT])
    clinics = list(dict.fromkeys(df["clinic"]))
    rules = list(dict.fromkeys(df["rule"]))
    metrics = [(SENT, "Referable eyes sent home (%)", 10, "target: at most 10 %"),
               (URGENT, "Urgent eyes referred urgently (%)", 90, "target: at least 90 %")]

    fig, axes = plt.subplots(2, len(clinics), figsize=(4.6 * len(clinics), 7), sharex=True, squeeze=False)
    for j, clinic in enumerate(clinics):
        for i, (column, label, target, target_label) in enumerate(metrics):
            ax = axes[i][j]
            for k, rule in enumerate(rules):
                g = df[(df["clinic"] == clinic) & (df["rule"] == rule)].sort_values("local photos")
                parsed = [with_interval(x) for x in g[column]]
                value = [p[0] for p in parsed]
                low = [p[1] for p in parsed]
                high = [p[2] for p in parsed]
                ax.plot(g["local photos"], value, marker="o", color=f"C{k}", label=rule)
                ax.fill_between(g["local photos"], low, high, color=f"C{k}", alpha=0.12)
            ax.axhline(target, linestyle="--", color="grey", linewidth=1)
            if i == 0:
                ax.set_title(clinic)
            if j == 0:
                ax.set_ylabel(label)
            if i == 1:
                ax.set_xlabel("Local photos used")
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, fontsize=8, title="Rule (shaded: range over the runs; dashed: target)")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUT / "recalibration_by_clinic.png", dpi=200)
    plt.close(fig)
    print("saved", OUT / "recalibration_by_clinic.png")


# Figure 3: referrals per 1,000 against the share of referable eyes, with the forecasts -------------
def fig_workload():
    share_col = "referable eyes at the clinic %"
    ref_col = "C1 referrals per 1,000 eyes"
    fewer = read("fewer_sick_eyes", [share_col, ref_col])
    forecast_cols = [("forecast B8 APTOS shares (average miss)", "B8 forecast, APTOS shares", "x", "C3"),
                     ("forecast C3 estimated shares (average miss)", "C3 forecast, estimated shares (ours)", "o", "C0"),
                     ("forecast B9 true shares (average miss)", "B9 forecast, true shares", "s", "C2")]
    wf = read("workload_forecast", ["clinic", "referable eyes, true %", "referrals per 1,000, observed"]
              + [c[0] for c in forecast_cols])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot([first(x) for x in fewer[share_col]], [first(x) for x in fewer[ref_col]],
            "-", color="black", marker="D", label="Observed (APTOS clinics with fewer sick eyes)")

    share = [first(x) for x in wf["referable eyes, true %"]]
    for column, label, marker, color in forecast_cols:
        ax.scatter(share, [first(x) for x in wf[column]], marker=marker, color=color, s=45, label=label, zorder=3)

    real = ~wf["clinic"].str.startswith("APTOS")
    ax.scatter([s for s, r in zip(share, real) if r],
               [first(x) for x, r in zip(wf["referrals per 1,000, observed"], real) if r],
               marker="D", facecolors="none", edgecolors="black", s=70, label="Observed (real new clinic)", zorder=4)
    for name, s, y, r in zip(wf["clinic"], share, wf["referrals per 1,000, observed"], real):
        if r:
            ax.annotate(name, (s, first(y)), textcoords="offset points", xytext=(6, -12), fontsize=8)

    ax.set_xlabel("Referable eyes at the clinic (%)")
    ax.set_ylabel("Referrals per 1,000 eyes")
    ax.set_title("Workload of the referral rule and its forecasts")
    ax.legend(fontsize=8)
    save(fig, "referrals_vs_referable_share")


# Figure 4: what each alpha costs -------------------------------------------------------------------
def fig_alpha():
    a = "allowed miss rate alpha %"
    refs = "referrals per 1,000 eyes"
    df = read("stricter_target", [a, SENT, URGENT, HEALTHY, GRADER, refs])
    df = df.assign(_a=[first(x) for x in df[a]]).sort_values("_a")
    x = df["_a"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    ax.plot(x, df[SENT], "o-", label="Referable eyes sent home")
    ax.plot(x, df[URGENT], "s-", label="Urgent eyes referred urgently")
    ax.axhline(10, linestyle="--", color="C0", linewidth=1)
    ax.axhline(90, linestyle="--", color="C1", linewidth=1)
    ax.set_ylabel("%")
    ax.set_title("Safety (dashed: the 10 % and 90 % targets)")
    ax = axes[1]
    ax.plot(x, df[HEALTHY], "o-", label="Healthy eyes sent home")
    ax.plot(x, df[GRADER], "s-", label="Eyes sent to a grader")
    ax.set_ylabel("%")
    ax.set_title("Usefulness")
    ax = axes[2]
    ax.plot(x, df[refs], "o-", color="C3", label="Referrals per 1,000 eyes")
    ax.set_ylabel("Referrals per 1,000 eyes")
    ax.set_title("Workload")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xlabel("alpha: allowed miss rate of each class (%)")
        ax.axvline(10, color="grey", alpha=0.3, linewidth=6)
        ax.legend(fontsize=8)
    save(fig, "cost_of_alpha")


# Figure 5: worse photos, with the thresholds from the original photos and set again --------------------
def fig_robustness():
    key = "thresholds set on photos that are"
    df = read("robustness", ["test photos", key, SENT, URGENT, HEALTHY, GRADER])
    conditions = list(dict.fromkeys(df["test photos"]))
    styles = [("original", "Thresholds from the original photos", "C0"),
              ("changed in the same way", "Thresholds set again on photos changed in the same way", "C1")]
    panels = [(SENT, "Referable eyes sent home (%)", 10, "target: at most 10 %"),
              (URGENT, "Urgent eyes referred urgently (%)", 90, "target: at least 90 %"),
              (HEALTHY, "Healthy eyes sent home (%)", None, None),
              (GRADER, "Eyes sent to a grader (%)", None, None)]
    width = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (column, title, target, target_label) in zip(axes.flat, panels):
        for j, (thresholds, label, color) in enumerate(styles):
            xs, ys = [], []
            for i, condition in enumerate(conditions):
                row = df[(df["test photos"] == condition) & (df[key] == thresholds)]
                if row.empty:           # the original photos have no 'set again' row
                    continue
                single = (df["test photos"] == condition).sum() == 1
                xs.append(i if single else i + (j - 0.5) * width)      # a lone bar is centred
                ys.append(float(row[column].iloc[0]))
            container = ax.bar(xs, ys, width, color=color, label=label)
            ax.bar_label(container, fmt="%.1f", fontsize=7)
        if target is not None:
            ax.axhline(target, linestyle="--", color="grey", linewidth=1, label=target_label)
        if column in (URGENT, HEALTHY):
            ax.set_ylim(0, 105)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([c[0].upper() + c[1:] for c in conditions])
        ax.set_title(title)
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Worse photos: the promise breaks, and setting the thresholds again repairs it at a price")
    save(fig, "robustness")


# Figure 6: one part of the pipeline changed at a time ---------------------------------------------------
def fig_ablations():
    df = read("ablations", ["pipeline", SENT, URGENT, HEALTHY, GRADER])
    labels = ["\n".join(textwrap.wrap(p, 30)) for p in df["pipeline"]]
    positions = list(range(len(df)))
    panels = [(SENT, "Referable eyes sent home (%)", 10),
              (URGENT, "Urgent eyes referred urgently (%)", 90),
              (HEALTHY, "Healthy eyes sent home (%)", None),
              (GRADER, "Eyes sent to a grader (%)", None)]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), sharey=True)
    for ax, (column, title, target) in zip(axes, panels):
        colors = ["C1"] + ["C0"] * (len(df) - 1)          # the first row is the pipeline as it is
        container = ax.barh(positions, df[column], color=colors)
        ax.bar_label(container, fmt="%.1f", fontsize=7, padding=2)
        if target is not None:
            ax.axvline(target, linestyle="--", color="grey", linewidth=1)
        if column in (URGENT, HEALTHY):
            ax.set_xlim(0, 105)
        else:
            ax.set_xlim(0, max(df[column].max() * 1.2, (target or 0) * 1.2))
        ax.set_title(title, fontsize=10)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("One part of the pipeline changed at a time (orange: the pipeline as it is; dashed: target)")
    save(fig, "ablations")


# Figure 7: the five rules of step 15, sent-home rate per camera group, with intervals ------------------------
def fig_rules_by_group():
    groups = [("mostly healthy", "sent home %, cameras mostly healthy (95 % interval)"),
              ("mixed", "sent home %, cameras mixed (95 % interval)"),
              ("mostly sick", "sent home %, cameras mostly sick (95 % interval)")]
    df = read("group_thresholds", ["rule"] + [c for _, c in groups])

    counts = {}                                           # referable eyes per camera group, for the axis labels
    try:
        eg = pd.read_csv(RESULTS / "errors_by_group.csv", encoding="utf-8-sig")
        eg = eg[eg["grouped by"] == "camera group"]
        counts = {g: int(n) for g, n in zip(eg["group"], eg["referable eyes"])}
    except (FileNotFoundError, KeyError, ValueError):
        pass

    rules = list(df["rule"])
    width = 0.8 / len(rules)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for k, rule in enumerate(rules):
        for g, (name, column) in enumerate(groups):
            value, low, high = with_interval(df[column].iloc[k])
            x = g + (k - (len(rules) - 1) / 2) * width
            yerr = None if low != low else [[value - low], [high - value]]       # low != low: no interval
            ax.errorbar(x, value, yerr=yerr, fmt="o", color=f"C{k}", capsize=3, label=rule if g == 0 else None)
    ax.axhline(10, linestyle="--", color="grey", linewidth=1)
    ax.text(len(groups) - 0.55, 10.8, "target: at most 10 %", ha="right", fontsize=8, color="grey")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{name}\n({counts[name]} referable eyes)" if name in counts else name for name, _ in groups])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Camera group")
    ax.set_ylabel("Referable eyes sent home (%)")
    ax.set_title("Sent-home rate by camera group for the five rules (95 % intervals)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    save(fig, "rules_by_camera_group")


if __name__ == "__main__":
    for figure in (fig_groups, fig_recalibration, fig_workload, fig_alpha,
                   fig_robustness, fig_ablations, fig_rules_by_group):
        figure()