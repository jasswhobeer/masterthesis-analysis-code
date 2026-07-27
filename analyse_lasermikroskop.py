"""
Laser scanning microscopy (Keyence VK-X4000) - channel geometry and surface texture
===================================================================================
Reads the Keyence raw files (.vk4/.vk6) DIRECTLY via the `surfalize` package, so
no CSV export from the MultiFileAnalyzer is required. The MultiFileAnalyzer only
exports the parameter table, not the profile points.

Part A - channel geometry
    File: Oberflaeche_XY_10x_Kanal.vk4
    For every cross-section of the height map the plateau and floor levels are
    taken as the median of the upper and lower point population (robust against
    outliers), the channel depth is their difference. The channel width follows
    from the two flank positions at half depth (50 % criterion). Reported as
    mean +/- SD and median over all cross-sections.

Part B - surface texture of the reservoir SIDE WALL, within a layer and across the layers
    Files: Oberfkache_Z_20x_Reservoir.vk4                  -> "normal processed"
           Oberfkache_Z_20x_Reservoir_2x autoklaviert.vk6  -> "recycled"

    The measured surface is the SIDE WALL of the reservoir, that is the surface
    built up layer by layer in the z direction of the printer. It touches neither
    the adhesive foil nor the thread, but it is the surface the medium inside the
    reservoir is in contact with.

    The layer lines therefore run along the x axis of the field of view: profiles
    taken along y cross the stacked layers and carry three to four times the
    amplitude of profiles taken along x. The y axis of the field is consequently
    the BUILD DIRECTION and x lies WITHIN a layer. The measured texture direction
    Std = 90 deg and the isotropy Str = 0.02 confirm this from the data alone.

    The figures show many individual traces per direction so that the real
    surface character is visible, the parameters are then averaged over ALL
    lines of the field separately for each direction.

    WHY NO CUT-OFF FILTER: the grooves have periods of roughly 85 to 350 um. A
    Gaussian filter that keeps them in the roughness profile needs lambda_c of
    about 0.8 mm and a matching evaluation length. The 20x field is only 705 um
    wide in x, so that length does not exist there, and any cut-off short enough
    to fit would remove exactly the groove structure the comparison is about.
    Evaluation is therefore done on the primary profile after removing a
    least-squares straight line, giving Pa, Pq, Pz and PSm per ISO 21920.

Run:  python analyse_lasermikroskop.py
Creates in the same folder:
    channel_cross_section.png     cross-section with width, depth and histograms
    channel_cross_section_simple.png  reduced variant, the one used in the thesis
    surface_topography.png        height maps of both surfaces, same colour scale
    surface_profiles.png          stacked real traces, both directions
    texture_parameters.png        Pa, Pq, Pz averaged over all lines
    lasermikroskopie_ergebnisse.txt   all numbers
    tab_texture.tex               LaTeX table for the thesis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import cm
from scipy import ndimage
from scipy.stats import kurtosis, skew
from surfalize import Surface

HERE = Path(__file__).parent

DATEI_KANAL = HERE / "Oberfläche_XY_10x_Kanal.vk4"
DATEI_NORMAL = HERE / "Oberfkache_Z_20x_Reservoir.vk4"
DATEI_RECYCELT = HERE / "Oberfkache_Z_20x_Reservoir_2x autoklaviert.vk6"

NAME_NORMAL = "normal processed"
NAME_RECYCELT = "recycled"

# the surface is the reservoir side wall: x lies within a printed layer,
# y runs along the build direction and therefore crosses the stacked layers
RICHTUNGSNAME = {"x": "within a printed layer",
                 "y": "along the build direction, across the layers"}

L_EVAL = 700.0        # evaluation length in um for the parameters, both directions
N_ABSCHNITTE = 5      # sections used for Pz (ISO 21920)
N_TRACES = 7          # individual traces shown per panel

BREITE_SOLL = 2000.0  # design width of the perfusion channel in um
TIEFE_SOLL = 200.0    # design depth of the perfusion channel in um

# Thesis house style (identical to plot_incucyte.py)
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


# ─────────────────────────────────────────────────────────────
#  PART A  -  CHANNEL GEOMETRY
# ─────────────────────────────────────────────────────────────
def kanal_kennwerte(profil, schritt):
    """Plateau, floor, depth and width of a single cross-section.

    Returns None if the profile does not contain both flanks.
    """
    mitte = 0.5 * (np.nanmin(profil) + np.nanmax(profil))
    oben = profil[profil > mitte]
    unten = profil[profil < mitte]
    if oben.size < 20 or unten.size < 20:
        return None

    plateau = float(np.median(oben))
    boden = float(np.median(unten))
    tiefe = plateau - boden
    schwelle = boden + 0.5 * tiefe

    ueber = profil > schwelle
    wechsel = np.flatnonzero(np.diff(ueber.astype(int)) != 0)
    if wechsel.size < 2:
        return None

    def kreuzung(i):
        z0, z1 = profil[i], profil[i + 1]
        if z1 == z0:
            return float(i)
        return i + (schwelle - z0) / (z1 - z0)

    fallend = [k for k in wechsel if profil[k] > schwelle]
    steigend = [k for k in wechsel if profil[k] <= schwelle]
    if not fallend or not steigend:
        return None

    x_links = kreuzung(fallend[0]) * schritt
    x_rechts = kreuzung(steigend[-1]) * schritt
    breite = x_rechts - x_links
    if breite <= 0:
        return None

    return {"plateau": plateau, "boden": boden, "tiefe": tiefe,
            "breite": breite, "x_links": x_links, "x_rechts": x_rechts}


def flankenwinkel(profil, schritt, ref):
    """Steepness of the two channel flanks, from the 10 % to the 90 % level.

    The horizontal run needed to cross the middle 80 % of the step is a robust
    measure of how sharply the printer reproduces a vertical wall.
    """
    boden, tiefe = ref["boden"], ref["tiefe"]
    winkel = []
    for x_flanke in (ref["x_links"], ref["x_rechts"]):
        i = int(round(x_flanke / schritt))
        fenster = slice(max(0, i - 120), min(profil.size, i + 120))
        stueck = profil[fenster]
        hoch = np.flatnonzero(stueck > boden + 0.90 * tiefe)
        tief = np.flatnonzero(stueck < boden + 0.10 * tiefe)
        if hoch.size and tief.size:
            lauf = abs(hoch[np.argmin(np.abs(hoch - (i - fenster.start)))]
                       - tief[np.argmin(np.abs(tief - (i - fenster.start)))]) * schritt
            if lauf > 0:
                winkel.append(np.degrees(np.arctan(0.8 * tiefe / lauf)))
    return {"winkel_mw": float(np.mean(winkel)) if winkel else float("nan"),
            "winkel": winkel}


def auswertung_kanal():
    flaeche = Surface.load(str(DATEI_KANAL))
    z = flaeche.data.astype(float)
    schritt = flaeche.step_y          # the cross-section runs along the long axis

    ergebnisse = [w for w in (kanal_kennwerte(z[:, s], schritt)
                              for s in range(z.shape[1])) if w is not None]

    tiefen = np.array([e["tiefe"] for e in ergebnisse])
    breiten = np.array([e["breite"] for e in ergebnisse])
    medianprofil = np.median(z, axis=1)
    # full envelope over ALL cross-sections, so the band shows the same spread
    # the reported ranges do and hides nothing
    unten, oben = z.min(axis=1), z.max(axis=1)
    referenz = kanal_kennwerte(medianprofil, schritt)

    return {
        "n": len(ergebnisse),
        "tiefen": tiefen, "breiten": breiten,
        "tiefe_mw": tiefen.mean(), "tiefe_sd": tiefen.std(ddof=1),
        "tiefe_med": float(np.median(tiefen)),
        "breite_mw": breiten.mean(), "breite_sd": breiten.std(ddof=1),
        "breite_med": float(np.median(breiten)),
        "profil": medianprofil, "band_unten": unten, "band_oben": oben,
        "x": np.arange(medianprofil.size) * schritt,
        "referenz": referenz,
        "flanke": flankenwinkel(medianprofil, schritt, referenz),
        "schritt": schritt,
    }


def speichern(fig, ziel):
    """House style: PNG at 300 dpi and a vector PDF next to it."""
    fig.savefig(ziel, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ziel.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_kanal(res, ziel, detail=True):
    """Cross-section of the channel with the scatter of all cross-sections.

    detail=True additionally shows the distribution of width and depth and the
    flank angle. detail=False keeps only what is needed to justify the width and
    the depth that enter the shear stress estimate.
    """
    farben = cm.viridis(np.linspace(0.05, 0.85, 3))
    if detail:
        fig = plt.figure(figsize=(16, 6.6))
        gitter = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], hspace=0.55, wspace=0.22)
        ax = fig.add_subplot(gitter[:, 0])
    else:
        fig, ax = plt.subplots(figsize=(11, 6.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ref = res["referenz"]
    tiefe, boden, plateau = ref["tiefe"], ref["boden"], ref["plateau"]

    # scatter of ALL cross-sections, so the uniformity is visible, not just a median
    ax.fill_between(res["x"], res["band_unten"], res["band_oben"],
                    color=farben[0], alpha=0.25, linewidth=0, zorder=2,
                    label=f"full range of all {res['n']} profiles")
    ax.plot(res["x"], res["profil"], color=farben[0], lw=2.4, zorder=3,
            label="median profile")

    ax.axhline(plateau, color=farben[2], ls=(0, (6, 2)), lw=1.4, zorder=1)
    ax.axhline(boden, color=farben[2], ls=(0, (6, 2)), lw=1.4, zorder=1)
    ax.set_ylim(boden - 0.30 * tiefe, plateau + 0.26 * tiefe)

    x_pfeil = ref["x_links"] + 0.28 * (ref["x_rechts"] - ref["x_links"])
    ax.annotate("", xy=(x_pfeil, plateau), xytext=(x_pfeil, boden),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.6), zorder=4)
    if detail:
        tiefe_text = (f"depth\n{res['tiefe_mw']:.1f} ± {res['tiefe_sd']:.1f} µm\n"
                      f"(design {TIEFE_SOLL:.0f} µm, "
                      f"{100 * (res['tiefe_mw'] - TIEFE_SOLL) / TIEFE_SOLL:+.1f} %)")
    else:
        # ranges, not the SD: they answer directly how far the print deviates
        tiefe_text = (f"depth {res['tiefen'].min():.0f}–{res['tiefen'].max():.0f} µm\n"
                      f"(design {TIEFE_SOLL:.0f} µm)")
    ax.text(x_pfeil + 70, boden + 0.42 * tiefe, tiefe_text,
            va="center", ha="left", fontsize=LEGEND_SIZE, zorder=4)

    y_pfeil = boden + 0.80 * tiefe
    ax.annotate("", xy=(ref["x_links"], y_pfeil), xytext=(ref["x_rechts"], y_pfeil),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.6), zorder=4)
    if detail:
        breite_text = (f"width {res['breite_mw']:.1f} ± {res['breite_sd']:.1f} µm  "
                       f"(design {BREITE_SOLL:.0f} µm, "
                       f"{100 * (res['breite_mw'] - BREITE_SOLL) / BREITE_SOLL:+.1f} %)")
    else:
        breite_text = (f"width {res['breiten'].min():.0f}–{res['breiten'].max():.0f} µm  "
                       f"(design {BREITE_SOLL:.0f} µm)")
    ax.text(0.5 * (ref["x_links"] + ref["x_rechts"]), y_pfeil - 0.05 * tiefe,
            breite_text, va="top", ha="center", fontsize=LEGEND_SIZE, zorder=4)

    if detail:
        ax.text(0.015, 0.05,
                f"flank angle {res['flanke']['winkel_mw']:.0f}°  (10–90 % of the step)\n"
                f"± values are the SD over all {res['n']} cross-sections along the channel",
                transform=ax.transAxes, fontsize=TICK_SIZE, va="bottom", ha="left")

    ax.set_xlabel("Position / µm")
    ax.set_ylabel("Height / µm")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    apply_axes_style(ax)
    leg = ax.legend(loc="upper right", frameon=True, fancybox=False,
                    edgecolor="black", framealpha=1.0)
    leg.get_frame().set_linewidth(1.2)

    if not detail:
        speichern(fig, ziel)
        return

    # distribution of width and depth over all cross-sections
    for zeile, (schluessel, soll, name, einheit) in enumerate(
            [("breiten", BREITE_SOLL, "Channel width", "µm"),
             ("tiefen", TIEFE_SOLL, "Channel depth", "µm")]):
        axh = fig.add_subplot(gitter[zeile, 1])
        axh.set_facecolor("white")
        werte = res[schluessel]
        axh.hist(werte, bins=45, color=farben[1], edgecolor="none", zorder=2)
        axh.axvline(werte.mean(), color=farben[0], lw=2.4, zorder=3,
                    label=f"mean {werte.mean():.1f} {einheit}")
        axh.axvline(soll, color="black", lw=1.6, ls=(0, (6, 2)), zorder=3,
                    label=f"design {soll:.0f} {einheit}")
        axh.set_xlabel(f"{name} / {einheit}")
        axh.set_ylabel("Cross-sections")
        apply_axes_style(axh)
        leg = axh.legend(loc="upper left", frameon=True, fancybox=False,
                         edgecolor="black", framealpha=1.0, fontsize=TICK_SIZE)
        leg.get_frame().set_linewidth(1.2)

    speichern(fig, ziel)


# ─────────────────────────────────────────────────────────────
#  PART B  -  SURFACE TEXTURE
# ─────────────────────────────────────────────────────────────
def ebene_abziehen(z, schritt_x, schritt_y):
    """Remove a least-squares plane (tilt of the sample under the microscope)."""
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    a = np.column_stack([xx.ravel() * schritt_x, yy.ravel() * schritt_y,
                         np.ones(z.size)])
    koeff, *_ = np.linalg.lstsq(a, z.ravel(), rcond=None)
    return z - (a @ koeff).reshape(z.shape)


def profilkennwerte(profil, schritt, laenge=L_EVAL, n_abschnitte=N_ABSCHNITTE):
    """Pa, Pq, Pz and mean groove spacing of a primary profile (ISO 21920)."""
    n_punkte = int(round(laenge / schritt))
    if profil.size < n_punkte:
        raise ValueError("profile shorter than the evaluation length")
    start = (profil.size - n_punkte) // 2
    ausschnitt = profil[start:start + n_punkte]

    x = np.arange(n_punkte) * schritt
    steigung, achsenabschnitt = np.polyfit(x, ausschnitt, 1)
    p = ausschnitt - (steigung * x + achsenabschnitt)

    pa = float(np.mean(np.abs(p)))
    pq = float(np.sqrt(np.mean(p ** 2)))

    je_abschnitt = n_punkte // n_abschnitte
    abschnitte = p[:je_abschnitt * n_abschnitte].reshape(n_abschnitte, je_abschnitt)
    pz = float(np.mean(abschnitte.max(axis=1) - abschnitte.min(axis=1)))

    # mean spacing of profile elements: upward crossings of the mean line, with
    # a hysteresis of 0.2 * Pq so that noise does not create extra counts
    schwelle = 0.2 * pq
    zustand, kreuzungen = 0, 0
    for wert in p:
        if zustand <= 0 and wert > schwelle:
            zustand, kreuzungen = 1, kreuzungen + 1
        elif zustand >= 0 and wert < -schwelle:
            zustand = -1
    psm = float(laenge / kreuzungen) if kreuzungen else float("nan")

    return {"Pa": pa, "Pq": pq, "Pz": pz, "PSm": psm}


def auswertung_oberflaeche(datei, bezeichnung):
    flaeche = Surface.load(str(datei))
    z = ebene_abziehen(flaeche.data.astype(float), flaeche.step_x, flaeche.step_y)

    richtungen = {}
    for achse in ("x", "y"):
        # x -> trace runs along the image width, y -> along the image height
        daten = z if achse == "y" else z.T
        schritt = flaeche.step_y if achse == "y" else flaeche.step_x

        # average over EVERY line of the field, separately per direction
        alle = [profilkennwerte(daten[:, s], schritt) for s in range(daten.shape[1])]
        eintrag = {"schritt": schritt, "n": len(alle),
                   "laenge_gesamt": daten.shape[0] * schritt}
        for kennwert in ("Pa", "Pq", "Pz", "PSm"):
            werte = np.array([a[kennwert] for a in alle], dtype=float)
            eintrag[kennwert + "_mw"] = float(np.nanmean(werte))
            eintrag[kennwert + "_sd"] = float(np.nanstd(werte, ddof=1))

        # traces for the figure: evenly spaced over the field, full length
        stellen = np.linspace(0.08, 0.92, N_TRACES)
        eintrag["traces"] = [
            {"pos_um": anteil * (daten.shape[1] - 1) * (
                flaeche.step_x if achse == "y" else flaeche.step_y),
             "x": np.arange(daten.shape[0]) * schritt,
             "z": daten[:, int(round(anteil * (daten.shape[1] - 1)))]}
            for anteil in stellen
        ]
        richtungen[achse] = eintrag

    return {"name": bezeichnung, "richtungen": richtungen,
            "karte": z, "step_x": flaeche.step_x, "step_y": flaeche.step_y,
            "dichtung": dichtungskennwerte(z, flaeche.step_x, flaeche.step_y),
            "abbott": abbott_kennwerte(flaeche)}


# ─────────────────────────────────────────────────────────────
#  MATERIAL RATIO (ABBOTT-FIRESTONE) CURVE
# ─────────────────────────────────────────────────────────────
def materialanteilkurve(z, n=2000):
    """Height against material ratio: for every height the share of the area
    lying at or above it. This is the height distribution summed from the top
    down, nothing more."""
    werte = np.sort(z[np.isfinite(z)])[::-1]
    anteil = np.linspace(0.0, 100.0, n)
    index = np.clip((anteil / 100.0 * (werte.size - 1)).astype(int), 0, werte.size - 1)
    return anteil, werte[index]


def sekantenkonstruktion(anteil, hoehe, fenster=40.0):
    """ISO 25178-2: the 40 % window with the smallest height drop defines the
    core, its secant extrapolated to 0 % and 100 % bounds the core zone."""
    breite = int(round(fenster / (anteil[1] - anteil[0])))
    abfall = hoehe[:-breite] - hoehe[breite:]
    i = int(np.argmin(abfall))
    steigung = (hoehe[i + breite] - hoehe[i]) / (anteil[i + breite] - anteil[i])
    oben = hoehe[i] - steigung * anteil[i]
    unten = hoehe[i] + steigung * (100.0 - anteil[i])
    return oben, unten


def abbott_kennwerte(flaeche):
    """Material ratio curve and the functional parameters derived from it."""
    eben = flaeche.level()
    z = eben.data.astype(float)
    z = z - float(np.nanmedian(z))
    anteil, hoehe = materialanteilkurve(z)
    oben, unten = sekantenkonstruktion(anteil, hoehe)
    return {"anteil": anteil, "hoehe": hoehe, "kern_oben": oben, "kern_unten": unten,
            "Sk": eben.Sk(), "Spk": eben.Spk(), "Svk": eben.Svk(),
            "Smr1": eben.Smr1(), "Smr2": eben.Smr2(),
            "Sa": eben.Sa(), "Sq": eben.Sq(), "Sz": eben.Sz(),
            "Ssk": eben.Ssk(), "Sku": eben.Sku(),
            "Sdr": eben.Sdr(), "Sdq": eben.Sdq(),
            "Vvv": eben.Vvv(), "Vmp": eben.Vmp(),
            "Std": eben.Std(), "Str": eben.Str()}


def planarer_ausschnitt(kante=700.0):
    """A clean square patch of the channel FLOOR, that is a surface the printer
    built in the build plane. Serves as the planar counterpart to the side wall."""
    flaeche = Surface.load(str(DATEI_KANAL))
    z = flaeche.data.astype(float)
    ref = kanal_kennwerte(np.median(z, axis=1), flaeche.step_y)

    mitte = 0.5 * (ref["x_links"] + ref["x_rechts"]) / flaeche.step_y
    halb = 0.5 * kante / flaeche.step_y
    zeilen = slice(int(mitte - halb), int(mitte + halb))
    spalten = slice(0, int(kante / flaeche.step_x))
    ausschnitt = z[zeilen, spalten]

    return {"name": "channel floor, printed in the build plane",
            "kurz": "planar", "karte": ebene_abziehen(ausschnitt, flaeche.step_x,
                                                      flaeche.step_y),
            "step_x": flaeche.step_x, "step_y": flaeche.step_y}


def plot_orientierung(planar, wand, ziel, kante=700.0):
    """The core statement about the printer: a surface built in the build plane
    is smooth, the same material built up in z is not. Both maps share one
    colour scale, otherwise the comparison would be meaningless."""
    # the same patch size for both, so the two maps are directly comparable
    schnitt = wand["karte"][:int(kante / wand["step_y"]),
                            :int(kante / wand["step_x"])]
    felder = [planar, {"name": "reservoir side wall, built up in z",
                       "kurz": "vertical", "karte": schnitt,
                       "step_x": wand["step_x"], "step_y": wand["step_y"]}]

    grenze = max(np.percentile(np.abs(f["karte"]), 99.5) for f in felder)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4))
    fig.patch.set_facecolor("white")
    for ax, feld in zip(axes, felder):
        z = feld["karte"]
        bild = ax.imshow(z, cmap="viridis", origin="lower", aspect="equal",
                         extent=(0, z.shape[1] * feld["step_x"],
                                 0, z.shape[0] * feld["step_y"]),
                         vmin=-grenze, vmax=grenze)
        # the numbers go into the thesis text and caption, not into the figure
        p1, p99 = np.percentile(z, [1, 99])
        print(f"  {feld['name']}: height variation +/- {0.5 * (p99 - p1):.0f} um, "
              f"Sa = {np.mean(np.abs(z)):.1f} um "
              f"(matched patch {kante:.0f} x {kante:.0f} um)")
        ax.set_title(feld["name"], fontsize=LABEL_SIZE)
        ax.set_xlabel("x / µm")
        ax.tick_params(axis="both", which="both", direction="out",
                       length=5, width=1.2, colors="black")
        for sp in ax.spines.values():
            sp.set_linewidth(1.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    axes[0].set_ylabel("y / µm")
    axes[1].set_yticklabels([])

    leiste = fig.colorbar(bild, ax=axes, orientation="vertical",
                          fraction=0.045, pad=0.03)
    leiste.set_label("Height / µm")
    leiste.outline.set_linewidth(1.5)

    # no suptitle, the thesis figure carries that information in its caption
    speichern(fig, ziel)


def plot_hoehenverteilung(flaechen, ziel, tief=20.0):
    """How the heights of the side wall are distributed, as printed and after
    reprocessing. A histogram needs no explanation, and the difference between
    the two surfaces sits entirely in the deep tail."""
    farben = cm.viridis(np.linspace(0.05, 0.85, len(flaechen)))
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    karten = [f["karte"] - np.median(f["karte"]) for f in flaechen]
    grenze = max(np.percentile(np.abs(k), 99.7) for k in karten)
    kanten = np.linspace(-grenze, grenze, 160)

    for karte, flaeche, farbe in zip(karten, flaechen, farben):
        anteil = np.histogram(karte, bins=kanten)[0] / karte.size * 100.0
        mitten = 0.5 * (kanten[:-1] + kanten[1:])
        ax.plot(mitten, anteil, color=farbe, lw=2.4, zorder=3,
                label=f"{flaeche['name']}  ({100.0 * np.mean(karte < -tief):.1f} % "
                      f"of the area below −{tief:.0f} µm)")
        ax.fill_between(mitten, anteil, where=mitten < -tief, color=farbe,
                        alpha=0.30, linewidth=0, zorder=2)

    ax.axvline(-tief, color="0.6", lw=1.0, ls=(0, (6, 2)), zorder=1)
    ax.text(-tief - 0.04 * grenze, ax.get_ylim()[1] * 0.55,
            f"deep craters\nbelow −{tief:.0f} µm", fontsize=TICK_SIZE,
            ha="right", va="center")

    ax.set_xlim(-grenze, grenze)
    # headroom so the legend never sits on top of a peak
    ax.set_ylim(0, 1.45 * max(np.histogram(k, bins=kanten)[0].max() / k.size * 100.0
                              for k in karten))
    ax.set_xlabel("Height relative to the median / µm")
    ax.set_ylabel("Share of the measured area / %")
    ax.set_title("Height distribution of the reservoir side wall",
                 fontsize=TITLE_SIZE)
    apply_axes_style(ax)
    leg = ax.legend(loc="upper left", frameon=True, fancybox=False,
                    edgecolor="black", framealpha=1.0, fontsize=TICK_SIZE)
    leg.get_frame().set_linewidth(1.2)
    speichern(fig, ziel)


def plot_abbott(flaechen, ziel):
    """Material ratio curves of both surfaces, with the core zone marked, and a
    zoom on the valley region where the two surfaces actually differ."""
    farben = cm.viridis(np.linspace(0.05, 0.85, len(flaechen)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6))
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("white")

    # ── left: full curves with the core zone ──────────────────
    # the core zones of the two surfaces overlap, so they are marked by their
    # bounding lines and a labelled arrow instead of by a filled band
    ax = axes[0]
    for i, (flaeche, farbe) in enumerate(zip(flaechen, farben)):
        a = flaeche["abbott"]
        ax.plot(a["anteil"], a["hoehe"], color=farbe, lw=2.4, zorder=3,
                label=flaeche["name"])
        for grenze in (a["kern_oben"], a["kern_unten"]):
            ax.axhline(grenze, color=farbe, ls=(0, (6, 2)), lw=1.2, zorder=2)
        x_pfeil = 21.0 + 21.0 * i
        ax.annotate("", xy=(x_pfeil, a["kern_oben"]), xytext=(x_pfeil, a["kern_unten"]),
                    arrowprops=dict(arrowstyle="<->", color=farbe, lw=1.6), zorder=4)
        ax.text(x_pfeil, a["kern_oben"] + 0.06 * a["Sk"],
                f"$Sk$ {a['Sk']:.1f} µm", color=farbe, fontsize=TICK_SIZE,
                va="bottom", ha="center", zorder=4)

    ax.axhline(0.0, color="0.6", lw=1.0, zorder=1)
    text = "\n".join(
        f"{f['name']}:  $Spk$ {f['abbott']['Spk']:.1f} µm   "
        f"$Svk$ {f['abbott']['Svk']:.1f} µm   $Sdr$ {f['abbott']['Sdr']:.0f} %"
        for f in flaechen)
    ax.text(0.03, 0.05, text, transform=ax.transAxes, fontsize=TICK_SIZE,
            va="bottom", ha="left")

    # clip to the bulk of the data, the deepest craters are single pixels
    def spanne(unteres, oberes):
        lo = min(np.interp(unteres, f["abbott"]["anteil"], f["abbott"]["hoehe"])
                 for f in flaechen)
        hi = max(np.interp(oberes, f["abbott"]["anteil"], f["abbott"]["hoehe"])
                 for f in flaechen)
        return lo, hi

    lo, hi = spanne(99.5, 0.5)
    ax.set_ylim(lo - 0.55 * (hi - lo), hi + 0.20 * (hi - lo))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Material ratio / %")
    ax.set_ylabel("Height / µm")
    ax.set_title("Material ratio curve, dashed lines bound the core zone",
                 fontsize=LABEL_SIZE)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    apply_axes_style(ax)
    leg = ax.legend(loc="upper right", frameon=True, fancybox=False,
                    edgecolor="black", framealpha=1.0)
    leg.get_frame().set_linewidth(1.2)

    # ── right: the valley tail, where the surfaces differ ─────
    ax = axes[1]
    for flaeche, farbe in zip(flaechen, farben):
        a = flaeche["abbott"]
        maske = a["anteil"] >= 70.0
        ax.plot(a["anteil"][maske], a["hoehe"][maske], color=farbe, lw=2.4,
                zorder=3, label=flaeche["name"])
        ax.axhline(a["kern_unten"], color=farbe, ls=(0, (6, 2)), lw=1.2, zorder=2)
        ax.fill_between(a["anteil"][maske], a["hoehe"][maske], a["kern_unten"],
                        where=a["hoehe"][maske] < a["kern_unten"],
                        color=farbe, alpha=0.25, linewidth=0, zorder=1)

    ax.set_xlim(70, 100)
    lo, hi = spanne(99.9, 70.0)
    ax.set_ylim(lo - 0.16 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.set_xlabel("Material ratio / %")
    ax.set_ylabel("Height / µm")
    ax.set_title("Valley region below the core, the area sets $Svk$",
                 fontsize=LABEL_SIZE)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    apply_axes_style(ax)
    leg = ax.legend(loc="lower left", frameon=True, fancybox=False,
                    edgecolor="black", framealpha=1.0)
    leg.get_frame().set_linewidth(1.2)

    speichern(fig, ziel)


def dichtungskennwerte(z, schritt_x, schritt_y, lambda_c=250.0):
    """Quantities that govern sealing and wetting, not the roughness amplitude.

    An adhesive film bridges micrometre roughness without trouble. What creates
    a leak path is long-wave undulation (waviness), out-of-flatness, and single
    continuous grooves that cross the sealing line. Those are computed here,
    together with skewness and kurtosis, which say whether the profile is made
    of deep narrow valleys or of rounded, even structures.
    """
    alpha = np.sqrt(np.log(2.0) / np.pi)
    ergebnis = {}

    for achse, schritt in (("x", schritt_x), ("y", schritt_y)):
        daten = z.T if achse == "x" else z
        sigma = alpha * lambda_c / np.sqrt(2.0 * np.pi) / schritt
        welligkeit = ndimage.gaussian_filter1d(daten, sigma, axis=0, mode="nearest")
        rauheit = daten - welligkeit

        rand = int(round(0.5 * lambda_c / schritt))
        w = welligkeit[rand:welligkeit.shape[0] - rand]
        r = rauheit[rand:rauheit.shape[0] - rand]

        stichprobe = r[:, ::max(1, r.shape[1] // 30)]
        ergebnis[achse] = {
            "Wa": float(np.mean(np.abs(w - w.mean(axis=0)))),
            "Wz": float(np.mean(w.max(axis=0) - w.min(axis=0))),
            "Ra": float(np.mean(np.abs(r))),
            "Rz": float(np.mean(r.max(axis=0) - r.min(axis=0))),
            "Rsk": float(np.mean(skew(stichprobe, axis=0))),
            "Rku": float(np.mean(kurtosis(stichprobe, axis=0, fisher=False))),
        }

    # out-of-flatness: long-wave form over the whole field
    form = ndimage.gaussian_filter(z, sigma=(200.0 / schritt_y, 200.0 / schritt_x),
                                   mode="nearest")
    ergebnis["ebenheit_pv"] = float(form.max() - form.min())
    ergebnis["feld_pv"] = float(z.max() - z.min())

    # depressions: a leak needs ONE continuous groove crossing the sealing line,
    # which no mean roughness parameter can show
    ergebnis["vertiefungen"] = {}
    for tiefe in (20.0, 30.0, 40.0):
        maske = z < (np.median(z) - tiefe)
        ergebnis["vertiefungen"][tiefe] = {
            "flaechenanteil": float(100.0 * maske.mean()),
            "n": int(ndimage.label(maske)[1]),
        }
    maske = z < (np.median(z) - 25.0)
    marken, anzahl = ndimage.label(maske)
    laengste = 0.0
    if anzahl:
        laengste = max((o[1].stop - o[1].start) * schritt_x
                       for o in ndimage.find_objects(marken))
    ergebnis["laengste_rille_x"] = float(laengste)
    ergebnis["feldbreite_x"] = float(z.shape[1] * schritt_x)
    return ergebnis


def plot_topographie(flaechen, ziel):
    """Height maps of both surfaces on one common colour scale."""
    grenze = max(np.percentile(np.abs(f["karte"]), 99.5) for f in flaechen)

    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    for ax, flaeche in zip(axes, flaechen):
        z = flaeche["karte"]
        breite = z.shape[1] * flaeche["step_x"]
        hoehe = z.shape[0] * flaeche["step_y"]
        bild = ax.imshow(z, cmap="viridis", origin="lower", aspect="equal",
                         extent=(0, breite, 0, hoehe), vmin=-grenze, vmax=grenze)
        ax.set_title(flaeche["name"], pad=12)
        ax.set_xlabel("x / µm")
        ax.tick_params(axis="both", which="both", direction="out",
                       length=5, width=1.2, colors="black")
        for sp in ax.spines.values():
            sp.set_linewidth(1.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    axes[0].set_ylabel("y / µm")
    axes[1].set_yticklabels([])

    leiste = fig.colorbar(bild, ax=axes, orientation="vertical",
                          fraction=0.045, pad=0.03)
    leiste.set_label("Height / µm")
    leiste.outline.set_linewidth(1.5)

    fig.suptitle("Reservoir side wall after plane levelling, "
                 "the printed layers run along x", fontsize=TITLE_SIZE, y=1.02)
    speichern(fig, ziel)


def plot_profile(flaechen, ziel):
    """Several real traces per direction, stacked with a constant offset."""
    # one offset per direction, shared by both surfaces so the two panels of a
    # row stay directly comparable
    versaetze = {
        a: 1.10 * max(t["z"].max() - t["z"].min()
                      for f in flaechen for t in f["richtungen"][a]["traces"])
        for a in ("x", "y")
    }
    farben = cm.viridis(np.linspace(0.05, 0.85, N_TRACES))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for zeile, achse in enumerate(("x", "y")):   # within a layer on top
        versatz = versaetze[achse]
        balken = max(10.0, round(versatz / 2 / 10) * 10)
        for spalte, flaeche in enumerate(flaechen):
            ax = axes[zeile][spalte]
            eintrag = flaeche["richtungen"][achse]
            laenge = eintrag["laenge_gesamt"]
            for i, (farbe, trace) in enumerate(zip(farben, eintrag["traces"])):
                ax.plot(trace["x"], trace["z"] + i * versatz, color=farbe, lw=1.0)
                ax.text(-0.015 * laenge, i * versatz, f"{trace['pos_um']:.0f}",
                        color=farbe, ha="right", va="center",
                        fontsize=TICK_SIZE - 1)

            ax.set_ylim(-0.9 * versatz, (N_TRACES - 0.1) * versatz)
            ax.set_xlim(-0.12 * laenge, 1.09 * laenge)
            ax.set_title(f"{flaeche['name']}\n{RICHTUNGSNAME[achse]} ({achse})",
                         fontsize=LABEL_SIZE + 2)
            ax.set_xlabel("Position / µm")
            ax.set_yticks([])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
            apply_axes_style(ax)
            ax.yaxis.set_minor_locator(ticker.NullLocator())
            if spalte == 0:
                ax.set_ylabel(f"Height, traces offset by {versatz:.0f} µm\n"
                              f"(labels = position of the trace / µm)",
                              fontsize=LEGEND_SIZE)

            # scale bar in the free margin on the right, so the real amplitude
            # stays readable despite the vertical offset
            x0 = 1.045 * laenge
            ax.plot([x0, x0], [0, balken], color="black", lw=1.6,
                    solid_capstyle="butt", clip_on=False)
            for y0 in (0, balken):
                ax.plot([x0 - 0.012 * laenge, x0 + 0.012 * laenge], [y0, y0],
                        color="black", lw=1.6, clip_on=False)
            ax.text(x0 + 0.022 * laenge, balken / 2, f"{balken:.0f} µm",
                    rotation=90, ha="left", va="center", fontsize=TICK_SIZE)

    fig.suptitle("Individual height traces, plane levelled and vertically offset",
                 fontsize=TITLE_SIZE, y=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    speichern(fig, ziel)



# ─────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────
def schreibe_bericht(kanal, flaechen, ziel):
    z = []
    z.append("Laser scanning microscopy (Keyence VK-X4000) - results")
    z.append("=" * 62)
    z.append("")
    z.append("PART A - channel geometry")
    z.append(f"  file             : {DATEI_KANAL.name}")
    z.append(f"  cross-sections   : n = {kanal['n']}")
    z.append(f"  lateral sampling : {kanal['schritt']:.3f} um/point")
    z.append(f"  width            : mean {kanal['breite_mw']:.1f} +/- "
             f"{kanal['breite_sd']:.1f} um   (median {kanal['breite_med']:.1f} um)")
    z.append(f"  depth            : mean {kanal['tiefe_mw']:.1f} +/- "
             f"{kanal['tiefe_sd']:.1f} um   (median {kanal['tiefe_med']:.1f} um)")
    z.append(f"  plateau level    : {kanal['referenz']['plateau']:.1f} um")
    z.append(f"  floor level      : {kanal['referenz']['boden']:.1f} um")
    z.append(f"  design width    : {BREITE_SOLL:.0f} um  -> deviation "
             f"{kanal['breite_mw'] - BREITE_SOLL:+.1f} um "
             f"({100 * (kanal['breite_mw'] - BREITE_SOLL) / BREITE_SOLL:+.1f} %)")
    z.append(f"  design depth    : {TIEFE_SOLL:.0f} um  -> deviation "
             f"{kanal['tiefe_mw'] - TIEFE_SOLL:+.1f} um "
             f"({100 * (kanal['tiefe_mw'] - TIEFE_SOLL) / TIEFE_SOLL:+.1f} %)")
    z.append(f"  flank angle     : {kanal['flanke']['winkel_mw']:.1f} deg "
             f"(10-90 % of the step)")
    z.append("  Plateau and floor are the median of the upper and lower point")
    z.append("  population PER cross-section, width and depth are then the mean")
    z.append("  +/- SD over all cross-sections. The SD describes the uniformity")
    z.append("  along the channel, not the scatter between chips.")
    z.append("")
    z.append("PART B - surface texture, per direction")
    z.append("  surface          : reservoir SIDE WALL, built up layer by layer")
    z.append("                     in the z direction of the printer")
    z.append("  direction        : x lies within a printed layer, y runs along")
    z.append("                     the build direction and crosses the stacked")
    z.append("                     layers (y carries 3-4x the amplitude, and the")
    z.append("                     measured Std = 90 deg confirms the lay)")
    z.append(f"  evaluation length: {L_EVAL:.0f} um in BOTH directions "
             f"(limited by the x extent of the 20x field)")
    z.append("  form removal     : plane over the field, then a least-squares")
    z.append("                     straight line per profile")
    z.append("  filter           : none (primary profile), a cut-off short enough")
    z.append("                     to fit into the 705 um x field would remove the")
    z.append("                     print grooves themselves (periods 85-350 um)")
    z.append("  parameters       : ISO 21920, averaged over ALL lines of the field")
    z.append("")
    for flaeche in flaechen:
        z.append(f"  {flaeche['name']}")
        for achse in ("y", "x"):
            e = flaeche["richtungen"][achse]
            z.append(f"    {achse} - {RICHTUNGSNAME[achse]}  "
                     f"(field length {e['laenge_gesamt']:.0f} um, n = {e['n']} lines)")
            for k in ("Pa", "Pq", "Pz", "PSm"):
                z.append(f"      {k:<3} = {e[k + '_mw']:6.2f} +/- "
                         f"{e[k + '_sd']:5.2f} um")
        z.append("")

    normal, recycelt = flaechen
    z.append("  Anisotropy (across the layers relative to within a layer)")
    for flaeche in flaechen:
        rx, ry = flaeche["richtungen"]["x"], flaeche["richtungen"]["y"]
        z.append(f"    {flaeche['name']:<18}: Pa(y)/Pa(x) = "
                 f"{ry['Pa_mw'] / rx['Pa_mw']:.2f}")
    z.append("")
    z.append("  Effect of reprocessing (recycled relative to normal processed)")
    for achse in ("y", "x"):
        for k in ("Pa", "Pq", "Pz"):
            a = normal["richtungen"][achse][k + "_mw"]
            b = recycelt["richtungen"][achse][k + "_mw"]
            z.append(f"    {achse} ({RICHTUNGSNAME[achse]}), d{k} = "
                     f"{b - a:+.2f} um ({100.0 * (b - a) / a:+.1f} %)")
    z.append("")
    z.append("PART C - quantities relevant for sealing and wetting")
    z.append("  Roughness amplitude is NOT what decides whether a lid seals or how")
    z.append("  a meniscus climbs. An adhesive film bridges micrometre roughness")
    z.append("  easily, a leak needs long-wave undulation, out-of-flatness or one")
    z.append("  continuous groove crossing the sealing line. Those are listed here.")
    z.append("")
    kopf = f"    {'':<34}" + "".join(f"{f['name']:>18}" for f in flaechen)
    z.append(kopf)

    def zeile(text, werte, fmt="{:>18.2f}"):
        z.append(f"    {text:<34}" + "".join(fmt.format(w) for w in werte))

    for achse in ("x", "y"):
        z.append(f"    -- {achse} ({RICHTUNGSNAME[achse]})")
        for k in ("Wa", "Wz", "Ra", "Rz", "Rsk", "Rku"):
            zeile(f"      {k} / um", [f["dichtung"][achse][k] for f in flaechen])
    zeile("    out-of-flatness P-V / um",
          [f["dichtung"]["ebenheit_pv"] for f in flaechen])
    zeile("    field P-V / um", [f["dichtung"]["feld_pv"] for f in flaechen])
    for tiefe in (20.0, 30.0, 40.0):
        zeile(f"    area deeper than {tiefe:.0f} um / %",
              [f["dichtung"]["vertiefungen"][tiefe]["flaechenanteil"]
               for f in flaechen])
    zeile("    longest groove along x / um",
          [f["dichtung"]["laengste_rille_x"] for f in flaechen], "{:>18.0f}")
    zeile("    field width in x / um",
          [f["dichtung"]["feldbreite_x"] for f in flaechen], "{:>18.0f}")
    z.append("")
    z.append("  READING: in these two fields the recycled surface is smoother,")
    z.append("  flatter and has fewer deep depressions than the as-processed one,")
    z.append("  and it no longer carries a groove running through the whole field.")
    z.append("  The measured texture therefore does NOT explain a worse seal or a")
    z.append("  higher meniscus. Whatever causes that must sit outside what this")
    z.append("  field of view can see: millimetre-scale warping of the part, a")
    z.append("  change of surface chemistry (contact angle), or defects in regions")
    z.append("  that were not measured.")
    z.append("")
    z.append("PART D - areal parameters, ISO 25178 (whole field, plane levelled)")
    z.append("  Sk, Spk and Svk split the material ratio curve into a peak, a core")
    z.append("  and a valley zone. Sa alone cannot do that. Svk and Vvv are the")
    z.append("  retention volume, Sdr is the true surface area in excess of the")
    z.append("  projected one, which is the quantity behind 'enlarged surface'.")
    z.append("")
    z.append(f"    {'':<26}" + "".join(f"{f['name']:>20}" for f in flaechen)
             + f"{'change':>12}")
    for k, e in (("Sa", "um"), ("Sq", "um"), ("Sz", "um"), ("Ssk", "-"),
                 ("Sku", "-"), ("Sk", "um"), ("Spk", "um"), ("Svk", "um"),
                 ("Smr1", "%"), ("Smr2", "%"), ("Sdr", "%"), ("Sdq", "-"),
                 ("Vvv", "mL/m2"), ("Vmp", "mL/m2"), ("Std", "deg"), ("Str", "-")):
        werte = [f["abbott"][k] for f in flaechen]
        wandel = (100.0 * (werte[1] - werte[0]) / werte[0]) if werte[0] else float("nan")
        z.append(f"    {k + ' / ' + e:<26}" + "".join(f"{w:>20.2f}" for w in werte)
                 + f"{wandel:>11.1f} %")
    z.append("")
    z.append("  Std gives the texture direction measured from the data and Str the")
    z.append("  isotropy, Str well below 0.5 means a strongly directional surface.")
    z.append("")
    z.append("  NOTE: only ONE field of view per condition is available. The SD is")
    z.append("  the scatter between the lines within that field, it is not the")
    z.append("  scatter between parts. Several fields per condition would be")
    z.append("  needed before the difference can be called reproducible.")

    text = "\n".join(z) + "\n"
    ziel.write_text(text, encoding="utf-8")
    return text


def schreibe_latex_tabelle(flaechen, ziel):
    z = [
        "% generated by analyse_lasermikroskop.py",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Primary profile parameters of the reservoir surface of the "
        "as-processed and the reprocessed (twice autoclaved) chip, evaluated "
        "separately across the printed layers and within a layer. Plane "
        "levelled, form removed by a least-squares straight line, no cut-off "
        f"filter, evaluation length \\SI{{{L_EVAL:.0f}}}{{\\micro\\meter}} in both "
        "directions. Parameters according to ISO 21920, given as mean and "
        "standard deviation over all profile lines of the field of view.}",
        "  \\label{tab:surface_texture}",
        "  \\begin{tabular}{llcccc}",
        "    \\toprule",
        "    Surface & Direction & $Pa$ / \\si{\\micro\\meter} "
        "& $Pq$ / \\si{\\micro\\meter} & $Pz$ / \\si{\\micro\\meter} "
        "& $PSm$ / \\si{\\micro\\meter} \\\\",
        "    \\midrule",
    ]
    for flaeche in flaechen:
        for i, achse in enumerate(("y", "x")):
            e = flaeche["richtungen"][achse]
            name = flaeche["name"] if i == 0 else ""
            richtung = "across layers" if achse == "y" else "within layer"
            z.append(
                f"    {name} & {richtung} & "
                f"${e['Pa_mw']:.2f} \\pm {e['Pa_sd']:.2f}$ & "
                f"${e['Pq_mw']:.2f} \\pm {e['Pq_sd']:.2f}$ & "
                f"${e['Pz_mw']:.2f} \\pm {e['Pz_sd']:.2f}$ & "
                f"${e['PSm_mw']:.0f} \\pm {e['PSm_sd']:.0f}$ \\\\"
            )
        if flaeche is flaechen[0]:
            z.append("    \\midrule")
    z += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    ziel.write_text("\n".join(z) + "\n", encoding="utf-8")


def main():
    kanal = auswertung_kanal()
    plot_kanal(kanal, HERE / "channel_cross_section.png")
    # The thesis embeds the reduced variant (no histograms, ranges instead of
    # SD). It has to be produced here as well, otherwise the figure that is
    # actually printed cannot be regenerated from this script.
    plot_kanal(kanal, HERE / "channel_cross_section_simple.png", detail=False)

    flaechen = [
        auswertung_oberflaeche(DATEI_NORMAL, NAME_NORMAL),
        auswertung_oberflaeche(DATEI_RECYCELT, NAME_RECYCELT),
    ]
    plot_topographie(flaechen, HERE / "surface_topography.png")
    plot_profile(flaechen, HERE / "surface_profiles.png")
    # statement 1: build orientation decides the surface quality
    plot_orientierung(planarer_ausschnitt(), flaechen[0],
                      HERE / "orientation_planar_vs_wall.png")
    # statement 2: what reprocessing does to the side wall
    plot_hoehenverteilung(flaechen, HERE / "height_distribution.png")

    print(schreibe_bericht(kanal, flaechen,
                           HERE / "lasermikroskopie_ergebnisse.txt"))
    schreibe_latex_tabelle(flaechen, HERE / "tab_texture.tex")


if __name__ == "__main__":
    main()
