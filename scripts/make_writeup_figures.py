"""Figures for docs/WRITEUP.md. Every number is copied from docs/RESULTS.md or from the gate logs it cites.

    python scripts/make_writeup_figures.py      # writes docs/figures/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Reference palette (dataviz skill), fixed order; gray for context.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, INK2, SURF = "#a9a8a2", "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": "#d6d5d0", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlelocation": "left",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180)
    plt.close(fig)
    print("wrote", OUT / name)


# 1. What moved the score: the base system, not the tweaks.
eras = [("Graph explorer\n(no LLM)", 0.25, 1), ("Duck harness\n+ Qwen 27B", 1.59, 2),
        ("+ Qwen3.8-Flash-Next\nNVFP4 / vLLM", 3.18, 11), ("Franzen M2 stack\n(SGLang INT4)", 25.3, 8),
        ("+ our final stack\n(S1 + M1 + C1)", 25.2, 5)]
fig, ax = plt.subplots(figsize=(8.2, 3.9))
x = np.arange(len(eras))
colors = [GRAY, GRAY, GRAY, BLUE, ORANGE]
ax.bar(x, [e[1] for e in eras], width=0.55, color=colors, edgecolor=SURF, linewidth=2)
for i, (lab, v, n) in enumerate(eras):
    ax.text(i, v + 0.6, f"{v:.2f}".rstrip("0").rstrip(".") + f"\n(n = {n})", ha="center", va="bottom", fontsize=9.5, color=INK)
ax.set_xticks(x, [e[0] for e in eras], fontsize=9)
ax.set_ylim(0, 32)
ax.set_ylabel("hidden-set score (mean)")
ax.set_title("Changing the base system moved the score 8×; tweaks on top did not")
save(fig, "fig1_history.png")

# 2. Every hidden sample by arm, against the control's band.
control = [25.01, 25.17, 23.39, 27.15, 27.63, 25.97, 24.30, 24.00]
arms = [("Control (Franzen M2)", control), ("H1 sandbox fixes", [24.17]), ("E1 effort = medium", [20.98]),
        ("D′ scheduler priority", [27.30, 25.30]), ("S1 solved-level memory", [26.20]),
        ("X1 = S1 + M1 helpers", [26.91, 25.64]), ("X2 = X1 + C1 reset", [27.27, 24.32, 23.34, 27.30, 23.69])]
m, s = np.mean(control), np.std(control, ddof=1)
fig, ax = plt.subplots(figsize=(8.2, 3.9))
ax.axvspan(m - s, m + s, color=BLUE, alpha=0.10, lw=0)
ax.axvline(m, color=BLUE, lw=2)
ax.text(m, -0.95, f"control mean {m:.1f} ± 1 sd", color=INK2, fontsize=9, ha="center")
ax.set_ylim(-1.25, len(arms) - 0.5)
for i, (name, vals) in enumerate(arms[::-1]):
    c = BLUE if name.startswith("Control") else (ORANGE if name.startswith("X") else INK2)
    ax.scatter(vals, [i] * len(vals), s=60, color=c, edgecolor=SURF, linewidth=1.5, zorder=3)
    if len(vals) > 1:
        ax.plot([np.mean(vals)] * 2, [i - 0.28, i + 0.28], color=c, lw=2)
ax.set_yticks(range(len(arms)), [a[0] for a in arms[::-1]])
ax.set_xlim(19.5, 29)
ax.set_xlabel("hidden-set score (one dot per daily submission; tick = arm mean)")
ax.set_title("No arm left the control's noise band, except one that hurt")
ax.grid(axis="y", visible=False)
save(fig, "fig2_arms.png")

# 3. Local gates do not rank: gate mean vs hidden mean for the same kernels.
pts = [("Control", 42.15, 25.33), ("S1", 44.91, 26.20), ("X1", 40.09, 26.28), ("X2", 43.47, 25.18),
       ("C1", 50.68, None), ("M1", 45.48, None)]
fig, ax = plt.subplots(figsize=(6.2, 3.9))
for name, g, h in pts:
    if h is None:
        continue
    ax.scatter(g, h, s=70, color=BLUE if name == "Control" else ORANGE, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.annotate(name, (g, h), xytext=(7, 4), textcoords="offset points", fontsize=9.5, color=INK)
ax.errorbar([42.15], [25.33], xerr=[[4], [4]], yerr=[[1.6], [1.6]], fmt="none", ecolor=GRAY, elinewidth=1.2, capsize=3)
ax.text(38.2, 27.2, "gray bars: one sd of a single\nrun (gate ≈ 4, hidden ≈ 1.6)", fontsize=8.5, color=INK2)
ax.set_xlim(37, 48)
ax.set_ylim(23, 28)
ax.set_xlabel("local gate score (10 public games, 25 min)")
ax.set_ylabel("hidden-set score (mean)")
ax.set_title("A gate run is too noisy to rank changes")
save(fig, "fig3_gates.png")

# 4. Where the metric is lost (50 gate games, weights = level index).
parts = [("earned", 43.0, BLUE), ("lost: level never solved", 54.8, GRAY), ("lost: extra actions", 2.3, ORANGE)]
fig, ax = plt.subplots(figsize=(8.2, 1.9))
left = 0
for lab, v, c in parts:
    ax.barh(0, v, left=left, color=c, edgecolor=SURF, linewidth=2, height=0.5)
    if v > 8:
        ax.text(left + v / 2, 0, f"{lab}\n{v:.1f} %", ha="center", va="center", fontsize=9.5,
                color="white" if c == BLUE else INK)
    left += v
ax.annotate(f"{parts[2][0]} {parts[2][1]:.1f} %", (99, 0.25), xytext=(80, 0.55), fontsize=9, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlim(0, 100)
ax.set_ylim(-0.4, 0.75)
ax.axis("off")
ax.set_title("Share of the level-weighted metric, 50 local game runs")
save(fig, "fig4_metric_loss.png")

# 5. Concurrency: 12 streams decode faster at peak but generate fewer tokens.
run10 = {6: 568, 7: 639, 8: 672, 9: 681, 10: 687}
run12 = {10: 637, 11: 652, 12: 759}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
a1.plot(list(run10), list(run10.values()), color=BLUE, lw=2, marker="o", ms=7, mec=SURF, mew=1.5, label="10 streams (X2)")
a1.plot(list(run12), list(run12.values()), color=ORANGE, lw=2, marker="o", ms=7, mec=SURF, mew=1.5, label="12 streams (T2)")
a1.set_xlabel("requests decoding at once")
a1.set_ylabel("decode tokens / s")
a1.set_title("Peak speed rises…")
a1.legend(frameon=False, fontsize=9, loc="lower right")
a2.bar([0, 1], [842, 662], color=[BLUE, ORANGE], width=0.55, edgecolor=SURF, linewidth=2)
for i, v in enumerate([842, 662]):
    a2.text(i, v + 15, f"{v} K", ha="center", fontsize=9.5)
a2.set_xticks([0, 1], ["10 streams", "12 streams"])
a2.set_ylim(0, 950)
a2.set_ylabel("tokens generated in 25 min")
a2.set_title("…but output falls 21 %")
a2.grid(axis="x", visible=False)
save(fig, "fig5_concurrency.png")
