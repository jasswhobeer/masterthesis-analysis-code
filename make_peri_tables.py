"""
Erzeugt die LaTeX-Kurztabellen fuer den 2. LNP-Stabilitaetstest auf der
Peristaltikpumpe (dynamisch vs. statisch vs. frisch inkubiert, Medium-Referenz).

Ausgabe (in den Masterarbeit-LaTeX-Ordner):
  peri_tables_short.tex  -> Konfluenz (absolut + normiert zu t1) + GFP bg-korr.
  peri_tables_norm.tex   -> GFP pro Konfluenz (auf 100 % hochgerechnet, inkl. Peak)

Analog zu make_lnp_tables.py (1. Versuch), mit zwei Unterschieden:
  * GFP-Hintergrund = nur die Medium-Referenz (kein empty-LNP in diesem Lauf).
  * Konfluenz-Normierung auf t1 statt t0 (die Referenz hat bei t0 einen
    Fokus-/Aussaat-Ausreisser, der eine t0-Normierung kuenstlich aufblaeht).
Fehler: Spalte "Std Dev Well" (Standardabweichung ueber die Wells).
"""

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
# the generated LaTeX tables are written next to this script
LATEX_DIR = HERE

FILES = {
    "hek_conf":   HERE / "HEK293 Confluence.txt",
    "huvec_conf": HERE / "HUVEC Confluence.txt",
    "hek_gfp":    HERE / "HEK293 Green Total Area.txt",
    "huvec_gfp":  HERE / "HUVEC Green Total Area.txt",
}

# (Original-Spaltenname, Anzeigename) — exakt wie in der Plot-Legende.
COND = [
    ("HEK 293 (LNP dynamic)", "HEK 293 (LNP dynamic)"),
    ("HEK 293 (LNP static)",  "HEK 293 (LNP static)"),
    ("HEK 293 (LNP fresh)",   "HEK 293 (LNP fresh)"),
    ("HEK 293",               "HEK 293 (reference)"),
    ("HUVEC (LNP dynamic)",   "HUVEC (LNP dynamic)"),
    ("HUVEC (LNP static)",    "HUVEC (LNP static)"),
    ("HUVEC (LNP fresh)",     "HUVEC (LNP fresh)"),
    ("HUVEC",                 "HUVEC (reference)"),
]
GROUPS = [("HEK293", COND[:4]), ("HUVEC", COND[4:])]
COND_GROUP = {orig: gname for gname, gconds in GROUPS for orig, _ in gconds}

# Nicht-leuchtende Kontrolle je Zellart -> Hintergrund-Basislinie fuer GFP.
GFP_BASELINE = {
    "HEK293": ["HEK 293"],
    "HUVEC":  ["HUVEC"],
}
ERR_SUFFIXES = [" (Std Err Img)", " (Std Dev Well)", " (Std Err Well)"]


# Aufnahme-Artefakt bei 44-45 h ausblenden (siehe plot_peristaltik.py): an diesen
# zwei Scans springen alle Bedingungen inkl. der zellfreien Referenz gleichzeitig
# hoch -> systematisches Imaging-Ereignis, keine Biologie. 46-48 h bleiben erhalten.
ART_WINDOW = (43.5, 47.5)   # Elapsed = 44, 45, 46, 47 raus; 48 h bleibt


def load(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    h = next(i for i, l in enumerate(lines) if "Elapsed" in l)
    df = pd.read_csv(path, sep="\t", decimal=",", skiprows=h, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df["Elapsed"] = pd.to_numeric(df["Elapsed"], errors="coerce")
    df = df.dropna(subset=["Elapsed"])
    lo, hi = ART_WINDOW
    return df[~((df["Elapsed"] > lo) & (df["Elapsed"] < hi))].reset_index(drop=True)


DF = {k: load(v) for k, v in FILES.items()}


def df_for(kind, cond):
    cell = "hek" if COND_GROUP[cond] == "HEK293" else "huvec"
    return DF[f"{cell}_{kind}"]


def err_col(df, cond):
    for s in ERR_SUFFIXES:
        if cond + s in df.columns:
            return cond + s
    return None


def series(kind, cond):
    df = df_for(kind, cond)
    tv = df["Elapsed"].to_numpy(float)
    ec = err_col(df, cond)
    e = df[ec].to_numpy(float) if ec else np.full(len(df), np.nan)
    return df[cond].to_numpy(float), e, tv


def at(arr, tv, hour):
    return arr[int(np.argmin(np.abs(tv - hour)))]


def gfp_baseline(group, tv):
    bs = []
    for bc in GFP_BASELINE[group]:
        y, _, _ = series("gfp", bc)
        bs.append(y - y[int(np.argmin(np.abs(tv)))])
    return np.mean(bs, axis=0)


def gfp_corrected(cond):
    """GFP-Signal, null-normiert (t0 abgezogen) und hintergrund-korrigiert."""
    y, e, tv = series("gfp", cond)
    corr = (y - y[int(np.argmin(np.abs(tv)))]) - gfp_baseline(COND_GROUP[cond], tv)
    return corr, e, tv


def gfp_norm(cond):
    """Hintergrund-korrigiertes GFP je Zeitpunkt durch Konfluenz geteilt und auf
    100 % Konfluenz hochgerechnet (Pro-Zell-Vergleich)."""
    corr, e, tv = gfp_corrected(cond)
    conf = df_for("conf", cond)[cond].to_numpy(float)
    scale = np.divide(100.0, conf, out=np.zeros_like(conf), where=conf > 1e-6)
    return corr * scale, e * scale, tv


def norm_with_error(y, e, tv, h, base_hour):
    """Konfluenz relativ zum Bezugszeitpunkt, mit fortgepflanztem Fehler.

    Der Nenner ist selbst eine Messung mit eigener Streuung. Wird nur der
    Zaehlerfehler durch den Nenner geteilt, faellt die Unsicherheit der
    normierten Werte je nach Bedingung um bis zu einen Faktor 10 zu klein aus
    (geprueft an den Rohdaten, am staerksten bei den Referenzen, deren Streuung
    am Bezugszeitpunkt gross ist).

    Zaehler und Nenner werden als unabhaengig behandelt. Da beide auf denselben
    Wells beruhen und positiv korreliert sind, ist das Ergebnis eine
    konservative obere Schranke der wahren Unsicherheit.
    """
    yb, eb = at(y, tv, base_hour), at(e, tv, base_hour)
    yt, et = at(y, tv, h), at(e, tv, h)
    if yb == 0:
        return float("nan"), float("nan")
    val = yt / yb * 100.0
    if yt == 0:
        return val, float("nan")
    rel = np.sqrt((et / yt) ** 2 + (eb / yb) ** 2)
    return val, abs(val) * rel


def pm(mean, sem, dec=1):
    return f"${mean:.{dec}f}\\pm{sem:.{dec}f}$"


# ─────────────────────────────────────────────────────────────
#  KURZE TABELLEN (Results)
# ─────────────────────────────────────────────────────────────
def short_confluence():
    L = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering")
    L.append(r"\caption{HEK\,293 and HUVEC confluence in the peristaltic-pump "
             r"incubation test at selected time points ($t_0$, 24\,h, 48\,h; "
             r"mean\,$\pm$\,standard deviation over the wells per condition). "
             r"Absolute phase-object confluence and the same data normalised to the "
             r"first stable time point $t_1$ (the medium-only reference has a "
             r"focus/seeding outlier at $t_0$, which a $t_0$ normalisation would "
             r"inflate). The uncertainty of the normalised columns propagates the "
             r"standard deviation of $t_1$ as well, numerator and denominator treated "
             r"as independent, which makes it a conservative upper bound.}")
    L.append(r"\label{tab:peri-confluence-short}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{l ccc cc}")
    L.append(r"\hline")
    L.append(r" & \multicolumn{3}{c}{\textbf{Absolute confluence (\%)}} & "
             r"\multicolumn{2}{c}{\textbf{Normalised (\% of $t_1$)}}\\")
    L.append(r"\textbf{Condition} & \textbf{$t_0$} & \textbf{24\,h} & \textbf{48\,h} "
             r"& \textbf{24\,h} & \textbf{48\,h}\\")
    L.append(r"\hline")
    for gname, gconds in GROUPS:
        L.append(r"\multicolumn{6}{l}{\emph{" + gname + r"}}\\")
        for orig, disp in gconds:
            y, e, tv = series("conf", orig)
            raw = [pm(at(y, tv, h), at(e, tv, h)) for h in (0, 24, 48)]
            norm = [pm(*norm_with_error(y, e, tv, h, 1))
                    for h in (24, 48)]
            L.append(f"{disp} & " + " & ".join(raw + norm) + r" \\")
    L.append(r"\hline")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def short_gfp():
    L = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering")
    L.append(r"\caption{Background-corrected GFP signal (total green-object area, in "
             r"$10^3$\,\textmu m$^2$/image) in the peristaltic-pump incubation test at "
             r"12, 24 and 48\,h (mean\,$\pm$\,standard deviation over the wells per "
             r"condition). For each cell type the per-timepoint value of the "
             r"medium-only reference was subtracted, so the reference scatters around "
             r"zero.}")
    L.append(r"\label{tab:peri-gfp-short}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{l ccc}")
    L.append(r"\hline")
    L.append(r"\textbf{Condition} & \textbf{12\,h} & \textbf{24\,h} & \textbf{48\,h}\\")
    L.append(r"\hline")
    for gname, gconds in GROUPS:
        L.append(r"\multicolumn{4}{l}{\emph{" + gname + r"}}\\")
        for orig, disp in gconds:
            corr, e, tv = gfp_corrected(orig)
            vals = [pm(at(corr, tv, h) / 1e3, at(e, tv, h) / 1e3) for h in (12, 24, 48)]
            L.append(f"{disp} & " + " & ".join(vals) + r" \\")
    L.append(r"\hline")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def short_gfp_norm():
    """GFP auf 100 % Konfluenz hochgerechnet (Pro-Zell-Vergleich), inkl. Peak."""
    L = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering")
    L.append(r"\caption{Confluence-normalised GFP signal in the peristaltic-pump "
             r"incubation test, i.e.\ the background-corrected total green-object area "
             r"divided by the phase-object confluence at each time point and expressed "
             r"per 100\,\% confluence (in $10^3$\,\textmu m$^2$/image; "
             r"mean\,$\pm$\,standard deviation over the wells). The peak column gives "
             r"the maximum value and the time at which it occurs. Only the GFP-loaded "
             r"conditions are listed; the medium-only reference remains at zero.}")
    L.append(r"\label{tab:peri-gfp-norm}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{l ccc c}")
    L.append(r"\hline")
    L.append(r"\textbf{Condition} & \textbf{12\,h} & \textbf{24\,h} & \textbf{48\,h} "
             r"& \textbf{Peak (@\,h)}\\")
    L.append(r"\hline")
    for gname, gconds in GROUPS:
        L.append(r"\multicolumn{5}{l}{\emph{" + gname + r"}}\\")
        for orig, disp in gconds:
            if "reference" in disp.lower():
                continue
            y, e, tv = gfp_norm(orig)
            vals = [pm(at(y, tv, h) / 1e3, at(e, tv, h) / 1e3) for h in (12, 24, 48)]
            k = int(np.nanargmax(y))
            peak = f"${y[k]/1e3:.0f}$ (@\\,{int(round(tv[k]))})"
            L.append(f"{disp} & " + " & ".join(vals) + f" & {peak}" + r" \\")
    L.append(r"\hline")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────
#  SCHREIBEN
# ─────────────────────────────────────────────────────────────
short = short_confluence() + "\n\n" + short_gfp() + "\n"
(LATEX_DIR / "peri_tables_short.tex").write_text(short, encoding="utf-8")
(LATEX_DIR / "peri_tables_norm.tex").write_text(short_gfp_norm() + "\n", encoding="utf-8")

print("Geschrieben:", LATEX_DIR / "peri_tables_short.tex")
print("           ", LATEX_DIR / "peri_tables_norm.tex")
print("\nKurz-Kontrolle (GFP hintergrund-korrigiert, 10^3 µm²/Image):")
for orig, disp in COND:
    corr, e, tv = gfp_corrected(orig)
    print(f"  {disp:<22} 12h={at(corr,tv,12)/1e3:8.1f}  24h={at(corr,tv,24)/1e3:8.1f}  "
          f"48h={at(corr,tv,48)/1e3:8.1f}")
