from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

HERE = Path(__file__).parent
FILES = [HERE / f"Plate {i}.txt" for i in (1, 2, 3)]

VEHICLE = "Reference Cytotox Green"
CISPLATIN = [
    ("Cisplatin 6.25 6.25 µM",   6.25),
    ("Cisplatin 12.5 µM 12.5 µM", 12.5),
    ("Cisplatin 25 µM 25 µM",    25.0),
    ("Cisplatin 50 µM 50 µM",    50.0),
    ("Cisplatin 100 µM 100 µM", 100.0),
    ("Cisplatin 200 µM 200 µM", 200.0),
]
TIMEPOINTS_H = [24, 48, 72]

def load_plate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", skiprows=7, decimal=",", encoding="utf-8")
    df = df.dropna(axis=1, how="all")
    df.columns = [c.strip() for c in df.columns]
    return df

plates = [load_plate(p) for p in FILES]
elapsed = plates[0]["Elapsed"].to_numpy(float)

def value_at(plate: pd.DataFrame, condition: str, t_hours: float) -> float:
    idx = int(np.argmin(np.abs(elapsed - t_hours)))
    return float(plate[condition].iloc[idx])

concentrations = np.array([c for _, c in CISPLATIN])
responses = {}
for th in TIMEPOINTS_H:
    R = np.zeros((len(plates), len(CISPLATIN)))
    for i, plate in enumerate(plates):
        veh_growth = value_at(plate, VEHICLE, th) / value_at(plate, VEHICLE, 0)
        for j, (cond, _) in enumerate(CISPLATIN):
            tr_growth = value_at(plate, cond, th) / value_at(plate, cond, 0)
            R[i, j] = (tr_growth / veh_growth) * 100.0
    responses[th] = R

def hill(c, ic50, hill_slope, bottom):
    return bottom + (100.0 - bottom) / (1.0 + (c / ic50) ** hill_slope)

def fit_hill(c, y, p0=(25.0, 1.0, 0.0)):
    popt, pcov = curve_fit(
        hill, c, y, p0=p0,
        bounds=([1e-3, 0.1, -20.0], [1e4, 10.0, 49.9]),
        maxfev=20000,
    )
    return popt, pcov

def abs_ic50(ic50, hill_slope, bottom):
    return ic50 * (50.0 / (50.0 - bottom)) ** (1.0 / hill_slope)

per_plate_rel = {}
per_plate_abs = {}
for th in TIMEPOINTS_H:
    rels, abss = [], []
    for i in range(len(plates)):
        try:
            popt, _ = fit_hill(concentrations, responses[th][i])
            rels.append(popt[0])
            abss.append(abs_ic50(*popt))
        except RuntimeError:
            rels.append(np.nan)
            abss.append(np.nan)
    per_plate_rel[th] = np.array(rels)
    per_plate_abs[th] = np.array(abss)

pooled = {}
for th in TIMEPOINTS_H:
    R = responses[th]
    c_long = np.tile(concentrations, R.shape[0])
    y_long = R.flatten()
    popt, pcov = fit_hill(c_long, y_long)
    pooled[th] = dict(popt=popt, pcov=pcov, n=len(y_long))

TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE, TICK_SIZE = 20, 16, 14, 12
plt.rcParams.update({
    "font.size":        LABEL_SIZE,
    "axes.labelsize":   LABEL_SIZE,
    "axes.titlesize":   TITLE_SIZE,
    "xtick.labelsize":  TICK_SIZE,
    "ytick.labelsize":  TICK_SIZE,
    "legend.fontsize":  LEGEND_SIZE,
    "axes.linewidth":   1.5,
})

fig, ax = plt.subplots(figsize=(11, 7))

tp_colors = cm.viridis(np.linspace(0.15, 0.80, len(TIMEPOINTS_H)))

c_grid = np.logspace(np.log10(2.0), np.log10(400.0), 200)

for th, color in zip(TIMEPOINTS_H, tp_colors):
    R       = responses[th]
    mean_y  = R.mean(axis=0)
    sd_y    = R.std(axis=0, ddof=1)

    ax.errorbar(
        concentrations, mean_y, yerr=sd_y,
        color=color, lw=0,
        marker="o", markersize=7,
        markerfacecolor=color, markeredgecolor=color,
        elinewidth=1.2, capsize=3, capthick=1.2,
        zorder=3,
    )

    popt = pooled[th]["popt"]
    ic50_a = abs_ic50(*popt)
    y_fit = hill(c_grid, *popt)
    ax.plot(c_grid, y_fit, color=color, lw=2.4, zorder=2)

    ax.plot([ic50_a, ic50_a], [-5, 50], color=color, lw=1.5, ls=":", zorder=1)

    rel_m       = np.nanmean(per_plate_rel[th])
    abs_m, abs_s = np.nanmean(per_plate_abs[th]), np.nanstd(per_plate_abs[th], ddof=1)
    label = (f"{th} h:  IC$_{{50}}$ = {abs_m:.1f} ± {abs_s:.1f} µM"
             f"   (rel. {rel_m:.1f} µM)")
    ax.plot([], [], color=color, lw=2.4, marker="o", markersize=7, label=label)

ax.axhline(50, color="grey", lw=1, ls="--", alpha=0.6, zorder=1)

ic50_marked = abs_ic50(*pooled[TIMEPOINTS_H[0]]["popt"])
ax.annotate(
    "absolute IC$_{50}$",
    xy=(ic50_marked, 50), xytext=(ic50_marked, 75),
    textcoords="data", fontsize=LEGEND_SIZE,
    ha="center", va="bottom", color="black", zorder=4,
    arrowprops=dict(arrowstyle="->", lw=1.6, color="black",
                    shrinkA=2, shrinkB=2),
)

ax.set_xscale("log")
ax.set_xlabel("Cisplatin concentration (µM)")
ax.set_ylabel("Response (% of vehicle control)")
ax.set_title("Cisplatin dose–response of HUVEC – time dependence")

ax.set_xlim(4, 320)
ax.set_ylim(-5, 130)
ax.set_xticks([6.25, 12.5, 25, 50, 100, 200])
ax.set_xticklabels(["6.25", "12.5", "25", "50", "100", "200"])

for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(1.5)
    sp.set_color("black")
ax.tick_params(axis="both", which="both", direction="in",
               length=6, width=1.4, top=False, right=False)
ax.grid(False)

leg = ax.legend(
    loc="upper right", title="IC$_{50}$ = absolute (50 % of control)",
    frameon=True, fancybox=False, edgecolor="black",
    fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE, handlelength=2.2, borderpad=0.6, labelspacing=0.5,
)
leg.get_frame().set_linewidth(1.2)

fig.tight_layout()
out_png = HERE / "cisplatin_ic50_timecourse.png"
out_pdf = HERE / "cisplatin_ic50_timecourse.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")

print("\n=== IC50 per plate (relativ / absolut) ===")
for th in TIMEPOINTS_H:
    r, a = per_plate_rel[th], per_plate_abs[th]
    print(f"  {th:>3} h:  rel = {np.nanmean(r):6.2f} +/- {np.nanstd(r, ddof=1):4.2f} uM   "
          f"abs = {np.nanmean(a):6.2f} +/- {np.nanstd(a, ddof=1):4.2f} uM")

print("\n=== Pooled fits ===")
for th in TIMEPOINTS_H:
    popt = pooled[th]["popt"]
    perr = np.sqrt(np.diag(pooled[th]["pcov"]))
    n    = pooled[th]["n"]
    dof  = n - 3
    tval = t_dist.ppf(0.975, dof)
    ic50, hill_slope, bottom = popt
    ic50_se = perr[0]
    ci_low  = ic50 - tval * ic50_se
    ci_high = ic50 + tval * ic50_se
    print(f"  {th:>3} h:  rel IC50 = {ic50:6.2f} uM   95%-CI [{ci_low:5.2f}, {ci_high:5.2f}]   "
          f"abs IC50 = {abs_ic50(*popt):6.2f} uM   hill = {hill_slope:.2f}   bottom = {bottom:5.1f} %")
