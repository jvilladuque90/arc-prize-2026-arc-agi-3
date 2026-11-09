"""Cover image for the Kaggle write-up (2:1, 1120x560 px).

    python scripts/make_cover.py      # writes docs/figures/cover.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "cover.png"
BLUE, ORANGE, GRAY, INK, INK2, BG = "#2a78d6", "#eb6834", "#a9a8a2", "#f4f3ef", "#b9b8b2", "#14161a"

fig = plt.figure(figsize=(11.2, 5.6), dpi=100, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 112)
ax.set_ylim(0, 56)
ax.axis("off")

# A small ARC-style grid as the visual motif.
grid = ["..BB....", ".BBB..O.", "..B..OOO", "......O.", "GG......", "GG..BB..", "....BB..", "........"]
cmap = {"B": BLUE, "O": ORANGE, "G": GRAY}
s, x0, y0 = 3.5, 80, 13
for r, row in enumerate(grid):
    for c, ch in enumerate(row):
        ax.add_patch(Rectangle((x0 + c * s, y0 + (7 - r) * s), s - 0.5, s - 0.5,
                               color=cmap.get(ch, "#22252b"), lw=0))

ax.text(6, 44, "What Moves an ARC-AGI-3 Agent", color=INK, fontsize=26, weight="bold", va="center")
ax.text(6, 37.5, "the base system, measured honestly", color=INK2, fontsize=21, va="center")
stats = [("0.25 → 25.3", "hidden score from changing\nthe base system"),
         ("~25 changes", "on top of it: none left\nthe noise band"),
         ("55 %", "of the metric lost on\nlevels never solved")]
for i, (big, small) in enumerate(stats):
    y = 25 - i * 8.2
    ax.text(6, y, big, color=ORANGE if i == 0 else BLUE, fontsize=19, weight="bold", va="center")
    ax.text(36, y, small, color=INK2, fontsize=11.5, va="center", linespacing=1.2)
ax.text(6, 3.5, "ARC Prize 2026 · Paper Track · Julian Camilo Villaduque", color=INK2, fontsize=11)
fig.savefig(OUT, facecolor=BG)
print("wrote", OUT)
