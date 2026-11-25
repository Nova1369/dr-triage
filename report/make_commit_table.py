"""Writes report/commit_table.tex from the git history.   Run from the project folder:  python report/make_commit_table.py

One row per commit that changes at least one file, oldest first: short hash, author, date and time, message.
Commits that change no file are counted in the caption and not listed.
"""
import subprocess
from pathlib import Path

log = subprocess.run(["git", "log", "--reverse", "--date=format:%d %b %H:%M", "--format=@%h|%an|%ad|%s", "--shortstat"],
                     capture_output=True, text=True, check=True).stdout
commits, empty = [], 0
for block in log.split("@")[1:]:
    lines = [line for line in block.strip().split("\n") if line.strip()]
    if len(lines) < 2:                                   # no "files changed" line: the commit changes nothing
        empty += 1
        continue
    short, author, date, message = lines[0].split("|", 3)
    author = {"Swayam Sahu": "Swayam", "Sahil Sihag": "Sahil", "Sahilsihag4849": "Sahil", "Nitesh Kumar": "Nitesh"}.get(author, author)
    for a, b in [("\\", ""), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")]:
        message = message.replace(a, b)
    commits.append((short, author, date, message))

half = (len(commits) + 1) // 2
parts = ["% Written by report/make_commit_table.py from git log. Do not edit by hand.", "\\begin{table*}[t]\\centering"]
for start in (0, half):
    parts += ["\\begin{minipage}[t]{0.495\\textwidth}\\centering\\scriptsize",
              "\\begin{tabular}{@{}r@{\\hspace{3pt}}l@{\\hspace{3pt}}l@{\\hspace{3pt}}l@{\\hspace{3pt}}p{0.53\\linewidth}@{}}\\toprule",
              "\\# & Commit & By & Date & Message\\\\\\midrule"]
    parts += [f"{i + 1} & \\texttt{{{c[0]}}} & {c[1]} & {c[2]} & {c[3]}\\\\" for i, c in enumerate(commits) if start <= i < start + half]
    parts += ["\\bottomrule", "\\end{tabular}", "\\end{minipage}" + ("\\hfill" if start == 0 else "")]
parts += [f"\\caption{{Commit history of the repository up to the writing of this report (branch \\texttt{{main}}, 2026, times in IST): "
          f"the {len(commits)} commits that change files. {empty} further commits change no file and are not listed.}}\\label{{tab:commits}}",
          "\\end{table*}", ""]
Path("report/commit_table.tex").write_text("\n".join(parts), encoding="utf-8")
print(f"{len(commits)} commits listed, {empty} without file changes left out -> report/commit_table.tex")
