from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

HERE = Path(__file__).parent
FILES = [HERE / f"Plate {i}.txt" for i in (1, 2, 3)]

CONDITIONS = [
    "Reference Without Cytotox Green",
    "Reference Cytotox Green",
    "Cisplatin 6.25 6.25 µM",
    "Cisplatin 12.5 µM 12.5 µM",
    "Cisplatin 25 µM 25 µM",
    "Cisplatin 50 µM 50 µM",
    "Cisplatin 100 µM 100 µM",
    "Cisplatin 200 µM 200 µM",
]

LABELS = [
    "Reference (no CytotoxGreen Reference)",
    "Reference (CytotoxGreen Vehicle Control)",
    "Cisplatin 6.25 µM",
    "Cisplatin 12.5 µM",
    "Cisplatin 25 µM",
    "Cisplatin 50 µM",
    "Cisplatin 100 µM",
    "Cisplatin 200 µM",
]

def load_plate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=7,
        decimal=",",
        encoding="utf-8",
    )
    df = df.dropna(axis=1, how="all")
    df.columns = [c.strip() for c in df.columns]
    return df

plates = [load_plate(p) for p in FILES]

elapsed = plates[0]["Elapsed"].to_numpy(dtype=float)

mean_norm = {}
tech_norm = {}

for cond in CONDITIONS:
    se_col = f"{cond} (Std Err Img)"
    mat_mean = np.vstack([p[cond].to_numpy(float) for p in plates])
    mat_se   = np.vstack([p[se_col].to_numpy(float) for p in plates])
    start = mat_mean[:, [0]]
    mean_norm[cond] = mat_mean / start
    tech_norm[cond] = mat_se / start

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

colors = cm.viridis(np.linspace(0.0, 0.95, len(CONDITIONS)))

for cond, label, color in zip(CONDITIONS, LABELS, colors):
    m = mean_norm[cond]
    t = tech_norm[cond]
    mean_t = m.mean(axis=0)
    sigma_bio  = m.std(axis=0, ddof=1)
    sigma_tech = t.mean(axis=0)
    sigma_tot  = np.sqrt(sigma_bio**2 + sigma_tech**2)

    ax.errorbar(
        elapsed, mean_t, yerr=sigma_tot,
        color=color, lw=2.2,
        marker="o", markersize=5, markerfacecolor=color, markeredgecolor=color,
        elinewidth=1.2, capsize=3, capthick=1.2,
        label=label,
    )

ax.set_xlabel("Elapsed time (h)")
ax.set_ylabel("Normalised confluence (a.u.)")
ax.set_title("Static Reference – Time Dependent Confluence of Cisplatin Treated HUVEC")

for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(1.5)
    sp.set_color("black")

ax.tick_params(axis="both", which="both", direction="in",
               length=6, width=1.4, top=False, right=False)
ax.grid(False)

ax.set_xlim(elapsed.min(), elapsed.max())
leg = ax.legend(
    loc="upper left",
    frameon=True, fancybox=False, edgecolor="black",
    fontsize=LEGEND_SIZE, handlelength=2.2, borderpad=0.6, labelspacing=0.5,
)
leg.get_frame().set_linewidth(1.2)

fig.tight_layout()
out_png = HERE / "cisplatin_plates_combined.png"
out_pdf = HERE / "cisplatin_plates_combined.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
