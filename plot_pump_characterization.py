"""Pump characterisation: average flow rate vs. supply voltage.

Reads the three replicate flow measurements per voltage from
`Klassifizierung Pumpe neu.xlsx`, computes mean and standard deviation over the
replicates, and renders a house-style plot (English axes) into the thesis
folder next to this script. Style matches plot_incucyte.py (closed box,
inner ticks, no grid, viridis, 300 dpi).
"""

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

# ── house style (mirrors plot_incucyte.py) ───────────────────────────
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
    """Return (volts, means, stds) sorted by voltage, excluding 0.3 V (no flow)."""
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["Tabelle1"]
    # header row 3: Volt | Ampere | D1(v,n,ist) | D2(...) | D3(...) | Average/min | ...
    # replicate "ist" values are columns E, H, K (0-based 4, 7, 10)
    volts, means, stds = [], [], []
    for row in ws.iter_rows(min_row=4, values_only=True):
        v = row[0]
        if v is None:
            continue
        ist = [row[4], row[7], row[10]]
        if any(x is None or not isinstance(x, (int, float)) for x in ist):
            continue  # e.g. the 0.3 V "keine Pumpenbewegung" row
        arr = np.array(ist, dtype=float)
        volts.append(float(v))
        means.append(arr.mean())
        stds.append(arr.std(ddof=1))
    order = np.argsort(volts)
    return (np.array(volts)[order], np.array(means)[order], np.array(stds)[order])


def main():
    volts, means, stds = read_data()
    color = cm.viridis(0.25)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # dashed connector: only the individual voltages were measured, the line is
    # a guide to the eye and not a measured continuous characteristic
    ax.errorbar(volts, means, yerr=stds, marker="o", markersize=6,
                linewidth=1.8, linestyle=(0, (6, 2)), capsize=3, color=color,
                markeredgecolor="black", markeredgewidth=0.6, zorder=3)

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
    for v, m, s in zip(volts, means, stds):
        print(f"  {v:>4.1f} V  ->  {m:6.3f} +/- {s:5.3f} mL/min")


if __name__ == "__main__":
    main()
