"""
Cytotox Green dead-cell signal (Total Green Object Area) for the cisplatin assay.

Reads the three per-plate green exports, averages over the plates and builds
- a house-style time-course figure  -> Bilder/cisplatin_green_timecourse.png (+pdf)
- a LaTeX table at 24/48/72 h       -> cisplatin_green_table.tex

Error = standard deviation over the three plates, expressed in 10^3 µm^2/image.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

HERE = Path(__file__).parent
# figure and LaTeX table are written next to this script
LATEX = HERE
BILDER = HERE
FILES = [HERE / f"Cisplatin Plate {i} green total area new analysis.txt" for i in (1, 2, 3)]

CIS = [("Cisplatin 6.25 6.25 µM", 6.25), ("Cisplatin 12.5 µM 12.5 µM", 12.5),
       ("Cisplatin 25 µM 25 µM", 25), ("Cisplatin 50 µM 50 µM", 50),
       ("Cisplatin 100 µM 100 µM", 100), ("Cisplatin 200 µM 200 µM", 200)]
VEHICLE = "Reference Cytotox Green"
NODYE = "Reference Without Cytotox Green"


def load(p):
    df = pd.read_csv(p, sep="\t", skiprows=7, decimal=",", encoding="utf-8")
    df = df.dropna(axis=1, how="all")
    df.columns = [c.strip() for c in df.columns]
    return df


plates = [load(p) for p in FILES]
t = plates[0]["Elapsed"].to_numpy(float)


def stack(col):
    M = np.vstack([pl[col].to_numpy(float) for pl in plates]) / 1e3  # 3 x T, 10^3 µm^2
    return M.mean(0), M.std(0, ddof=1)


def at(arr, hour):
    return arr[int(np.argmin(np.abs(t - hour)))]


# ---------------------------- Figure --------------------------------------
LAB, LEG, TICK = 16, 14, 12
plt.rcParams.update({"font.size": LAB, "axes.labelsize": LAB,
                     "xtick.labelsize": TICK, "ytick.labelsize": TICK,
                     "legend.fontsize": LEG, "axes.linewidth": 1.5})

fig, ax = plt.subplots(figsize=(11, 7))
colors = cm.viridis(np.linspace(0.12, 0.9, len(CIS)))
for (col, conc), c in zip(CIS, colors):
    m, s = stack(col)
    ax.plot(t, m, color=c, lw=2.2, marker="o", ms=4, label=f"{conc:g} µM", zorder=3)
    ax.fill_between(t, m - s, m + s, color=c, alpha=0.12, zorder=1)
m, s = stack(VEHICLE)
ax.plot(t, m, color="black", lw=2.0, ls="--", label="Vehicle (dye, no cisplatin)", zorder=3)
m, s = stack(NODYE)
ax.plot(t, m, color="grey", lw=1.6, ls=":", label="No-dye reference", zorder=2)

ax.set_xlabel("Time (h)")
ax.set_ylabel("Total green object area (10$^3$ µm²/image)")
ax.set_xlim(0, t.max())
for sp in ax.spines.values():
    sp.set_visible(True); sp.set_linewidth(1.5); sp.set_color("black")
ax.tick_params(axis="both", which="both", direction="in", length=6, width=1.4, top=False, right=False)
ax.grid(False)
leg = ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="black",
                title="Cisplatin", title_fontsize=LEG, ncol=2, borderpad=0.6, labelspacing=0.4)
leg.get_frame().set_linewidth(1.2)
fig.tight_layout()
for d in (HERE, BILDER):
    fig.savefig(d / "cisplatin_green_timecourse.png", dpi=300, bbox_inches="tight")
fig.savefig(BILDER / "cisplatin_green_timecourse.pdf", bbox_inches="tight")
print("figure ->", BILDER / "cisplatin_green_timecourse.png")

# ---------------------------- Table ---------------------------------------
def cell(col, hour):
    m, s = stack(col)
    return f"{at(m, hour):.0f}\\,$\\pm$\\,{at(s, hour):.0f}"

L = [r"\begin{table}[H]", r"\centering",
     r"\caption{Cytotox Green dead-cell signal (total green object area) in the "
     r"cisplatin assay at 24, 48 and 72\,h (mean\,$\pm$\,standard deviation over the "
     r"three plates, in $10^3$\,\textmu m$^2$/image). The signal rises with the "
     r"cisplatin dose in the toxic window. The no-dye reference stayed near zero "
     r"($<1$) and is omitted.}",
     r"\label{tab:cisplatin-green}", r"\small",
     r"\begin{tabular}{l ccc}", r"\hline",
     r"\textbf{Condition} & \textbf{24\,h} & \textbf{48\,h} & \textbf{72\,h} \\", r"\hline"]
for col, conc in CIS:
    L.append(f"Cisplatin {conc:g}\\,\\textmu M & {cell(col,24)} & {cell(col,48)} & {cell(col,72)} \\\\")
L.append(f"Vehicle (dye, no cisplatin) & {cell(VEHICLE,24)} & {cell(VEHICLE,48)} & {cell(VEHICLE,72)} \\\\")
L += [r"\hline", r"\end{tabular}", r"\end{table}"]
(LATEX / "cisplatin_green_table.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
print("table  ->", LATEX / "cisplatin_green_table.tex")

for th in (24, 48, 72):
    print(f"{th}h:", ", ".join(f"{c:g}={at(stack(col)[0],th):.0f}" for col, c in CIS),
          f"| vehicle={at(stack(VEHICLE)[0],th):.0f}")
