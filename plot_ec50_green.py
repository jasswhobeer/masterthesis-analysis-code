"""
EC50 of the Cytotox Green dead-cell signal at 24 h.

The confluence readout falls with the dose and therefore yields an IC50.
The green dead-cell signal rises with the dose, so the corresponding parameter
is an EC50. Both are the midpoint of a dose-response curve, they differ only in
the direction of the response.

Normalisation: the green signal of the vehicle control is subtracted at the
same time point, so that only the cisplatin-induced dead-cell signal remains
and the vehicle sits at zero.

The signal is deliberately NOT divided by the confluence. In this assay the
confluence is the effect itself and falls from about 50 % to 10 % across the
dose range. Dividing a rising dead-cell signal by a falling confluence counts
the same effect twice, the ratio then cannot reach a plateau within the tested
range, and the EC50 and the plateau become jointly unidentifiable. (The LNP
readout does divide by the confluence, but there the confluence differences are
an incidental nuisance and not the measured effect.)

Because the vehicle is zero by construction and there is no full-kill control,
the upper end of the scale is not anchored. Only a relative EC50 can be
obtained, that is the concentration at half of the fitted plateau. Its
structural counterpart on the confluence side is the relative IC50, not the
absolute one.

Expects the IncuCyte S3 exports next to this file:
  Plate 1-3.txt                                     (Phase Object Confluence)
  Cisplatin Plate 1-3 green total area new analysis.txt   (Total Green Object Area)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

HERE = Path(__file__).parent
CONF_FILES = [HERE / f"Plate {i}.txt" for i in (1, 2, 3)]
GREEN_FILES = [HERE / f"Cisplatin Plate {i} green total area new analysis.txt"
               for i in (1, 2, 3)]

VEHICLE = "Reference Cytotox Green"
CISPLATIN = [
    ("Cisplatin 6.25 6.25 µM",   6.25),
    ("Cisplatin 12.5 µM 12.5 µM", 12.5),
    ("Cisplatin 25 µM 25 µM",    25.0),
    ("Cisplatin 50 µM 50 µM",    50.0),
    ("Cisplatin 100 µM 100 µM", 100.0),
    ("Cisplatin 200 µM 200 µM", 200.0),
]
TIMEPOINT_H = 24

# confluence readout at 24 h, for the comparison line
CONF_IC50_REL = 24.9
CONF_IC50_ABS = 32.7


def load_plate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", skiprows=7, decimal=",", encoding="utf-8")
    df = df.dropna(axis=1, how="all")
    df.columns = [c.strip() for c in df.columns]
    return df


conf_plates = [load_plate(p) for p in CONF_FILES]
green_plates = [load_plate(p) for p in GREEN_FILES]


def value_at(plate: pd.DataFrame, condition: str, t_hours: float) -> float:
    elapsed = plate["Elapsed"].to_numpy(float)
    idx = int(np.argmin(np.abs(elapsed - t_hours)))
    return float(plate[condition].iloc[idx])


concentrations = np.array([c for _, c in CISPLATIN])

# dead-cell signal per cell area, vehicle subtracted, one row per plate
R = np.zeros((len(green_plates), len(CISPLATIN)))
for i, (gp, cp) in enumerate(zip(green_plates, conf_plates)):
    g_veh = value_at(gp, VEHICLE, TIMEPOINT_H)
    for j, (cond, _) in enumerate(CISPLATIN):
        R[i, j] = value_at(gp, cond, TIMEPOINT_H) - g_veh


def hill_rising(c, ec50, hill_slope, top):
    """Rising Hill curve anchored at zero, since the vehicle is subtracted."""
    return top * c ** hill_slope / (ec50 ** hill_slope + c ** hill_slope)


def fit_hill(c, y, p0=(35.0, 2.0, None)):
    if p0[2] is None:
        p0 = (p0[0], p0[1], float(np.nanmax(y)) * 1.2)
    popt, pcov = curve_fit(
        hill_rising, c, y, p0=p0,
        bounds=([1e-3, 0.1, 0.0], [1e4, 10.0, np.inf]),
        maxfev=20000,
    )
    return popt, pcov


per_plate = []
for i in range(len(green_plates)):
    try:
        popt, _ = fit_hill(concentrations, R[i])
        per_plate.append(popt[0])
    except RuntimeError:
        per_plate.append(np.nan)
per_plate = np.array(per_plate)

c_long = np.tile(concentrations, R.shape[0])
y_long = R.flatten()
popt, pcov = fit_hill(c_long, y_long)
perr = np.sqrt(np.diag(pcov))
dof = len(y_long) - len(popt)
tval = t_dist.ppf(0.975, dof)
ec50, hill_slope, top = popt
ci_low = ec50 - tval * perr[0]
ci_high = ec50 + tval * perr[0]

ss_res = np.sum((y_long - hill_rising(c_long, *popt)) ** 2)
ss_tot = np.sum((y_long - y_long.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot

TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE, TICK_SIZE = 20, 16, 14, 12
plt.rcParams.update({
    "font.size":       LABEL_SIZE,
    "axes.labelsize":  LABEL_SIZE,
    "axes.titlesize":  TITLE_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.linewidth":  1.5,
})

fig, ax = plt.subplots(figsize=(11, 7))
color = cm.viridis(0.15)
conf_color = cm.viridis(0.80)

mean_y = R.mean(axis=0)
sd_y = R.std(axis=0, ddof=1)

ax.errorbar(
    concentrations, mean_y, yerr=sd_y,
    color=color, lw=0, marker="o", markersize=7,
    markerfacecolor=color, markeredgecolor=color,
    elinewidth=1.2, capsize=3, capthick=1.2, zorder=3,
)

c_grid = np.logspace(np.log10(2.0), np.log10(400.0), 200)
ax.plot(c_grid, hill_rising(c_grid, *popt), color=color, lw=2.4, zorder=2)

ax.axhline(top / 2.0, color="grey", lw=1, ls="--", alpha=0.6, zorder=1)
ax.plot([ec50, ec50], [0, top / 2.0], color=color, lw=1.5, ls=":", zorder=1)
ax.plot([CONF_IC50_REL, CONF_IC50_REL], [0, top * 1.05],
        color=conf_color, lw=1.5, ls="-.", zorder=1)

ax.plot([], [], color=color, lw=2.4, marker="o", markersize=7,
        label=(f"Cytotox Green:  EC$_{{50}}$ = {np.nanmean(per_plate):.1f} "
               f"± {np.nanstd(per_plate, ddof=1):.1f} µM"))
ax.plot([], [], color=conf_color, lw=1.5, ls="-.",
        label=f"Confluence:  rel. IC$_{{50}}$ = {CONF_IC50_REL:.1f} µM")

ax.set_xscale("log")
ax.set_xlabel("Cisplatin concentration (µM)")
ax.set_ylabel("Vehicle-corrected green object area (µm²/image)")
ax.set_title("Cytotox Green dose–response of HUVEC at 24 h")

ax.set_xlim(4, 320)
ax.set_ylim(0, top * 1.15)
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
    loc="upper left", title="EC$_{50}$ = half of the fitted plateau",
    frameon=True, fancybox=False, edgecolor="black",
    fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE,
    handlelength=2.2, borderpad=0.6, labelspacing=0.5,
)
leg.get_frame().set_linewidth(1.2)

fig.tight_layout()
out_png = HERE / "cisplatin_ec50_green.png"
out_pdf = HERE / "cisplatin_ec50_green.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")

print("\n=== Cytotox Green EC50 at 24 h ===")
print(f"  per plate : {np.array2string(per_plate, precision=2)}")
print(f"  mean +/- SD: {np.nanmean(per_plate):.2f} +/- "
      f"{np.nanstd(per_plate, ddof=1):.2f} uM")
print(f"  pooled     : {ec50:.2f} uM   95%-CI [{ci_low:.2f}, {ci_high:.2f}]")
print(f"  hill = {hill_slope:.2f}   plateau = {top:.1f} a.u.   R2 = {r2:.3f}")
print(f"\n  confluence absolute IC50 at 24 h: {CONF_IC50_ABS:.1f} uM")
