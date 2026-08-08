from pathlib import Path

import numpy as np
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, ticker

HERE = Path(__file__).resolve().parent
XLSX = HERE / "Klassifizierung Pumpe neu.xlsx"
OUT_DIR = HERE
OUT_STEM = "pump_characterization"

LABEL_SIZE, TICK_SIZE = 16, 12
plt.rcParams.update({
    "font.size":      LABEL_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "axes.linewidth": 1.5,
})

def apply_axes_style(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.5)
        sp.set_color("black")
    ax.tick_params(axis="both", which="both", direction="in",
                   length=6, width=1.4, top=False, right=False)
    ax.tick_params(which="minor", length=3.5, width=1.0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.grid(False)

def read_data():
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["Tabelle1"]
    volts, means, stds = [], [], []
    for row in ws.iter_rows(min_row=4, values_only=True):
        v = row[0]
        if v is None:
            continue
        ist = [row[4], row[7], row[10]]
        if any(x is None or not isinstance(x, (int, float)) for x in ist):
            continue
        arr = np.array(ist, dtype=float)
        volts.append(float(v))
        means.append(arr.mean())
        stds.append(arr.std(ddof=1))
    order = np.argsort(volts)
    return (np.array(volts)[order], np.array(means)[order], np.array(stds)[order])

def main():
    volts, means, stds = read_data()
    color = cm.viridis(0.25)
    fit_color = cm.viridis(0.65)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    slope, intercept = np.polyfit(volts, means, 1)
    r2 = np.corrcoef(volts, means)[0, 1] ** 2
    v_fit = np.linspace(volts.min(), volts.max(), 200)
    ax.plot(v_fit, slope * v_fit + intercept, color=fit_color, linewidth=2.0,
            zorder=2)

    ax.errorbar(volts, means, yerr=stds, marker="o", markersize=6,
                linestyle="none", elinewidth=1.4, capsize=3, capthick=1.4,
                color=color, markeredgecolor="black", markeredgewidth=0.6,
                zorder=3)

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Average flow rate (mL/min)")
    ax.set_xlim(0.4, 3.2)
    ax.set_ylim(bottom=0)
    apply_axes_style(ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{OUT_STEM}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"wrote {png}")
    print(f"  fit: {slope:.4f} mL/min per V, intercept {intercept:+.4f} mL/min, "
          f"R^2 = {r2:.4f}")
    for v, m, s in zip(volts, means, stds):
        print(f"  {v:>4.1f} V  ->  {m:6.3f} +/- {s:5.3f} mL/min")

if __name__ == "__main__":
    main()
