"""Checks that the numbers in the README tables match the CSV files in results/.

Run from the project folder:   python -m pytest tests/test_results.py -v

How it works. Each test finds one table of README.md (by its header), pairs every row with a row of the CSV,
and compares the numbers cell by cell. A cell passes when the two numbers differ by no more than one unit of the
last digit the README shows (0.1 for "4.4 %", 0.001 for "0.972"), because the same average can be rounded either
way. A whole number in the README ("578") may differ from the CSV (578.4) by up to 0.5. Where a README cell holds
an interval ("4.4 % (3.5-5.6)") and the CSV cell holds one too, the interval ends are compared as well. A failing
test lists every cell that disagrees, so the README can be corrected (or the CSV rerun).

Covered tables: steps 6, 8, 9 (two tables), 10, 13 (three tables) and 15. A test is skipped when its CSV is missing.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
NUMBER = re.compile(r"\d[\d,]*\.?\d*")

SENT = "referable eyes sent home % (target <= 10)"
URGENT = "urgent eyes referred urgently % (target >= 90)"
HEALTHY = "healthy eyes sent home %"
GRADER = "to a grader %"
TRUE_CLASS = "true class in the set %"
REFERRALS = "referrals per 1,000 eyes"


# ---------------------------------------------------------------------------------------------------
def numbers(text):
    """All numbers in a cell: '1,487' -> [1487.0], '4.4 % (3.5-5.6)' -> [4.4, 3.5, 5.6]."""
    return [float(x.replace(",", "")) for x in NUMBER.findall(str(text))]


def find_table(*must_contain):
    """The rows (lists of cells) of the README table whose header line contains all the given strings."""
    for i, line in enumerate(README):
        if line.startswith("|") and all(text in line for text in must_contain):
            rows = []
            for row in README[i + 2:]:                    # i + 1 is the |---|---| line
                if not row.startswith("|"):
                    break
                rows.append([cell.strip().replace("**", "").strip() for cell in row.strip().strip("|").split("|")])
            return rows
    raise AssertionError(f"No README table with a header containing {must_contain}")


def load(name):
    path = ROOT / "results" / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"results/{name}.csv not found")
    return pd.read_csv(path, encoding="utf-8-sig")


def tolerance(token):
    """How far a CSV number may be from a README number: one unit of the README's last digit
    (0.1 for '4.4', 0.001 for '0.972'), or 0.5 for a whole number ('578' against 578.4)."""
    decimals = len(token.split(".")[1]) if "." in token else 0
    return 0.5 if decimals == 0 else 10 ** -decimals + 1e-9


def compare(table, row_label, column, readme_cell, csv_cell, problems):
    """Add a line to problems when the numbers of a README cell and a CSV cell differ."""
    tokens = NUMBER.findall(str(readme_cell))
    a, b = numbers(readme_cell), numbers(csv_cell)
    if not a:                                              # an empty README cell says nothing
        return
    if not b:
        problems.append(f"{table} | {row_label} | {column}: README says {readme_cell!r}, the CSV cell is empty")
        return
    for token, x, y in zip(tokens, a, b):
        if abs(x - y) > tolerance(token):
            problems.append(f"{table} | {row_label} | {column}: README says {readme_cell!r}, CSV says {csv_cell}")
            return


def check(problems):
    assert not problems, "README and CSV disagree in %d cell(s):\n  " % len(problems) + "\n  ".join(problems)


def by_row_order(table, readme_rows, df, label_cell, pairs):
    """README row i against CSV row i; pairs = [(README cell index, CSV column), ...]."""
    problems = []
    if len(readme_rows) != len(df):
        problems.append(f"{table}: README has {len(readme_rows)} rows, the CSV has {len(df)}")
    for row, (_, csv_row) in zip(readme_rows, df.iterrows()):
        for index, column in pairs:
            compare(table, row[label_cell], column, row[index], csv_row[column], problems)
    check(problems)


# ---------------------------------------------------------------------------------------------------
def test_step6_referral_rule():
    df = load("referral_rule").set_index("rule")
    columns = [SENT, URGENT, HEALTHY, GRADER, REFERRALS]
    problems = []
    for row in find_table("| Rule | Referable eyes sent home (target at most 10 %) |"):
        if row[0] not in df.index:
            problems.append(f"step 6 | {row[0]}: no such rule in referral_rule.csv")
            continue
        for cell, column in zip(row[1:], columns):
            compare("step 6", row[0], column, cell, df.loc[row[0], column], problems)
    check(problems)


def test_step8_recalibration():
    df = load("recalibration")
    columns = [TRUE_CLASS, SENT, URGENT, GRADER, REFERRALS]
    problems = []
    for row in find_table("| Clinic | Rule |", "| To a grader |"):
        code = row[1].split()[0]                           # C1, C2, B6 or B7
        match = df[(df["clinic"] == row[0]) & (df["local photos"] == 300) & df["rule"].str.startswith(code)]
        if len(match) != 1:
            problems.append(f"step 8 | {row[0]} {code}: found {len(match)} rows with 300 local photos")
            continue
        for cell, column in zip(row[2:], columns):
            compare("step 8", f"{row[0]} {code}", column, cell, match.iloc[0][column], problems)
    check(problems)


def test_step9_fewer_sick_eyes():
    df = load("fewer_sick_eyes")
    pairs = [(1, "C1 referable eyes sent home % (target <= 10)"), (2, "C1 urgent eyes referred urgently % (target >= 90)"),
             (3, "B4 true class in the set %"), (4, "B4 urgent in its set %"),
             (5, "C1 referrals per 1,000 eyes"), (6, "C1 referrals that were needed %")]
    problems = []
    for row in find_table("| Referable eyes at the clinic | C1: referable eyes sent home"):
        share = numbers(row[0])[0]
        match = df[[abs(numbers(v)[0] - share) < 0.01 for v in df["referable eyes at the clinic %"]]]
        if len(match) != 1:
            problems.append(f"step 9 | {row[0]}: found {len(match)} rows in fewer_sick_eyes.csv")
            continue
        for index, column in pairs:
            compare("step 9", row[0], column, row[index], match.iloc[0][column], problems)
    check(problems)


def test_step9_workload_forecast():
    df = load("workload_forecast")
    pairs = [(1, "referable eyes, true %"), (2, "referable eyes, estimated without labels %"),
             (3, "referrals per 1,000, observed"), (4, "forecast B8 APTOS shares (average miss)"),
             (5, "forecast C3 estimated shares (average miss)"), (6, "forecast B9 true shares (average miss)")]
    problems = []
    for row in find_table("| Clinic | Referable eyes: true |"):
        share = numbers(row[1])[0]
        match = df[[abs(numbers(v)[0] - share) < 0.01 for v in df["referable eyes, true %"]]]
        if len(match) != 1:
            problems.append(f"step 9 forecast | {row[0]}: found {len(match)} rows with {share} % referable eyes")
            continue
        for index, column in pairs:
            compare("step 9 forecast", row[0], column, row[index], match.iloc[0][column], problems)
    check(problems)


def test_step10_errors_by_group():
    df = load("errors_by_group")
    df["group"] = df["group"].astype(str)
    rows = {"All eyes": ("all", "all eyes"), "Grade 2": ("DR grade", "2"), "Grade 3": ("DR grade", "3"),
            "Grade 4": ("DR grade", "4"), "Cameras with mostly healthy eyes": ("camera group", "mostly healthy"),
            "Cameras with mixed eyes": ("camera group", "mixed"), "Cameras with mostly sick eyes": ("camera group", "mostly sick")}
    pairs = [(1, "referable eyes"), (2, "sent home % (95 % interval)"), (3, "urgent eyes"), (4, "missed urgent % (95 % interval)")]
    problems = []
    for row in find_table("| Group | Referable eyes | Sent home"):
        if row[0] not in rows:                             # the sharpness rows are not checked
            continue
        kind, group = rows[row[0]]
        match = df[(df["grouped by"] == kind) & (df["group"] == group)]
        if len(match) != 1:
            problems.append(f"step 10 | {row[0]}: found {len(match)} rows in errors_by_group.csv")
            continue
        for index, column in pairs:
            compare("step 10", row[0], column, row[index], match.iloc[0][column], problems)
    check(problems)


def test_step13_robustness():
    df = load("robustness")
    photos = {"Original": "original", "Blurred": "blurred", "Half brightness": "half brightness",
              "JPEG quality 20": "JPEG quality 20"}
    thresholds = {"original": "original", "recalibrated": "changed in the same way"}
    pairs = [(2, "AUROC referable"), (3, TRUE_CLASS), (4, SENT), (5, URGENT), (6, HEALTHY), (7, GRADER)]
    problems = []
    for row in find_table("| Photos to decide | Thresholds |"):
        match = df[(df["test photos"] == photos[row[0]]) &
                   (df["thresholds set on photos that are"] == thresholds[row[1]])]
        label = f"{row[0]} / {row[1]}"
        if len(match) != 1:
            problems.append(f"step 13 worse photos | {label}: found {len(match)} rows in robustness.csv")
            continue
        for index, column in pairs:
            compare("step 13 worse photos", label, column, row[index], match.iloc[0][column], problems)
    check(problems)


def test_step13_ablations():
    df = load("ablations")
    pairs = [(1, "AUROC referable"), (2, "AUROC urgent"), (3, SENT), (4, URGENT), (5, HEALTHY), (6, REFERRALS)]
    by_row_order("step 13 pipeline", find_table("| Pipeline | AUROC referable |"), df, 0, pairs)


def test_step13_stricter_target():
    df = load("stricter_target")
    pairs = [(0, "allowed miss rate alpha %"), (1, TRUE_CLASS), (2, SENT), (3, URGENT), (4, HEALTHY), (5, GRADER),
             (6, REFERRALS)]
    by_row_order("step 13 alpha", find_table("| alpha | True class in the set |"), df, 0, pairs)


def test_step15_group_thresholds():
    df = load("group_thresholds")
    pairs = [(1, "sent home %, cameras mostly healthy (95 % interval)"), (2, "sent home %, cameras mixed (95 % interval)"),
             (3, "sent home %, cameras mostly sick (95 % interval)"), (4, URGENT), (5, HEALTHY), (6, GRADER)]
    by_row_order("step 15", find_table("| Rule | Referable eyes sent home: mostly-healthy cameras |"), df, 0, pairs)