from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

HERE = Path(__file__).parent
DATA_FILE = HERE / "Fibrinogen Analyse Daten.txt"

GROUPS = {
    "Dried fibrinogen":  ["A1", "A4", "A5"],
    "Wet fibrinogen":    ["B2", "B4", "B5"],
    "Uncoated COC":      ["C1", "C2", "C4"],
    "Reference (well)":  ["D1", "D2", "D3"],
}

FOCUS_ARTEFACT = {"A1": [60.0, 72.0]}
T_TOL = 0.5

with open(DATA_FILE, encoding="utf-8") as f:
    lines = f.readlines()
header_i = next(i for i, l in enumerate(lines) if "Elapsed" in l)
header = lines[header_i].rstrip("\n").split("\t")

def column_of(well):
    hits = [i for i, c in enumerate(header) if c.strip().endswith(f"({well})")]
    if len(hits) != 1:
        raise KeyError(f"well {well}: {len(hits)} matching columns in the export")
    return hits[0]

COLUMNS = {w: column_of(w) for wells in GROUPS.values() for w in wells}
WELL_INDEX = {w: i for i, w in enumerate(COLUMNS)}

t, rows = [], []
for line in lines[header_i + 1:]:
    p = line.rstrip("\n").split("\t")
    if len(p) < len(header):
        continue
    try:
        tt = float(p[1].replace(",", "."))
    except ValueError:
        continue
    t.append(tt)
    rows.append([float(p[COLUMNS[w]].replace(",", ".")) for w in COLUMNS])
t = np.array(t)
M = np.array(rows)

M[M <= 0] = np.nan
for _well, _hours in FOCUS_ARTEFACT.items():
    for _h in _hours:
        M[np.abs(t - _h) <= T_TOL, WELL_INDEX[_well]] = np.nan

TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE, TICK_SIZE = 20, 16, 14, 12
plt.rcParams.update({
    "font.size": LABEL_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.linewidth": 1.5,
})

fig, ax = plt.subplots(figsize=(13, 8))
colors = cm.viridis(np.linspace(0.0, 0.95, len(GROUPS)))

for (label, wells), color in zip(GROUPS.items(), colors):
    sub = M[:, [WELL_INDEX[w] for w in wells]]
    mean = np.nanmean(sub, axis=1)
    sd = np.nanstd(sub, axis=1, ddof=1)
    ax.plot(
        t, mean,
        color=color, lw=2.2,
        marker="o", markersize=5, markerfacecolor=color, markeredgecolor=color,
        label=f"{label}  (n=3)",
    )
    band = ~np.isnan(mean) & ~np.isnan(sd)
    ax.fill_between(t[band], (mean - sd)[band], (mean + sd)[band],
                    color=color, alpha=0.18, linewidth=0)

ax.set_xlabel("Elapsed time (h)")
ax.set_ylabel("Phase Object Confluence (%)")
ax.set_title("HUVEC Cell Confluence on Fibrinogen-Coated COC Surface")

for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(1.5)
    sp.set_color("black")

ax.tick_params(axis="both", which="both", direction="in",
               length=6, width=1.4, top=False, right=False)
ax.grid(False)
ax.set_xlim(t.min(), t.max())
ax.set_ylim(0, None)

leg = ax.legend(loc="center left", frameon=True, fancybox=False,
                edgecolor="black", handlelength=2.2,
                borderpad=0.6, labelspacing=0.5)
leg.get_frame().set_linewidth(1.2)

fig.tight_layout()
out_png = HERE / "Fibrinogentest 1.png"
out_pdf = HERE / "Fibrinogentest 1.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
