from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
LATEX_DIR = BASE / "Masterarbeit LaTeX"
DATA = BASE / "LNP" / "LNP 1.Stabilitätstest"

FILES = {
    "hek_conf":   DATA / "Hek Confluence new analysis.txt",
    "huvec_conf": DATA / "HUVEC confluence.txt",
    "hek_gfp":    DATA / "Hek green total area new analysis.txt",
    "huvec_gfp":  DATA / "HUVEC green fluorescent total area.txt",
}

COND = [
    ("HEK, 24h LNP",    "HEK 293 (LNP GFP 24h)"),
    ("HEK fresh LNP",   "HEK 293 (LNP GFP)"),
    ("HEK LNP EMPTY",   "HEK 293 (LNP empty)"),
    ("HEK REFERENCE",   "HEK 293"),
    ("HUVEC 24h LNP",   "HUVEC (LNP GFP 24h)"),
    ("HUVEC fresh LNP", "HUVEC (LNP GFP)"),
    ("HUVEC LNP empty", "HUVEC (LNP empty)"),
    ("HUVEC REFERENCE", "HUVEC"),
]
GROUPS = [("HEK293", COND[:4]), ("HUVEC", COND[4:])]
COND_GROUP = {orig: gname for gname, gconds in GROUPS for orig, _ in gconds}

GFP_BASELINE = {
    "HEK293": ["HEK LNP EMPTY", "HEK REFERENCE"],
    "HUVEC":  ["HUVEC LNP empty", "HUVEC REFERENCE"],
}
ERR_SUFFIXES = [" (Std Err Img)", " (Std Dev Well)", " (Std Err Well)"]

def load(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    h = next(i for i, l in enumerate(lines) if "Elapsed" in l)
    df = pd.read_csv(path, sep="\t", decimal=",", skiprows=h, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df["Elapsed"] = pd.to_numeric(df["Elapsed"], errors="coerce")
    return df.dropna(subset=["Elapsed"])

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
    y, e, tv = series("gfp", cond)
    corr = (y - y[int(np.argmin(np.abs(tv)))]) - gfp_baseline(COND_GROUP[cond], tv)
    return corr, e, tv

def gfp_norm(cond):
    corr, e, tv = gfp_corrected(cond)
    conf = df_for("conf", cond)[cond].to_numpy(float)
    scale = np.divide(100.0, conf, out=np.zeros_like(conf), where=conf > 1e-6)
    return corr * scale, e * scale, tv

def norm_with_error(y, e, tv, h, base_hour):
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

def short_confluence():
    L = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering")
    L.append(r"\caption{HEK293 and HUVEC confluence in the LNP stability test at "
             r"selected time points ($t_0$, 24\,h, 48\,h; mean\,$\pm$\,standard "
             r"deviation over the wells per condition). Absolute phase-object "
             r"confluence and the same "
             r"data normalised to the starting value $t_0$. The uncertainty of the "
             r"normalised columns propagates the standard deviation of $t_0$ as well, "
             r"numerator and denominator treated as independent, which makes it a "
             r"conservative upper bound.}")
    L.append(r"\label{tab:lnp-confluence-short}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{l ccc cc}")
    L.append(r"\hline")
    L.append(r" & \multicolumn{3}{c}{\textbf{Absolute confluence (\%)}} & "
             r"\multicolumn{2}{c}{\textbf{Normalised (\% of $t_0$)}}\\")
    L.append(r"\textbf{Condition} & \textbf{$t_0$} & \textbf{24\,h} & \textbf{48\,h} "
             r"& \textbf{24\,h} & \textbf{48\,h}\\")
    L.append(r"\hline")
    for gname, gconds in GROUPS:
        L.append(r"\multicolumn{6}{l}{\emph{" + gname + r"}}\\")
        for orig, disp in gconds:
            y, e, tv = series("conf", orig)
            raw = [pm(at(y, tv, h), at(e, tv, h)) for h in (0, 24, 48)]
            norm = [pm(*norm_with_error(y, e, tv, h, 0))
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
    L.append(r"\caption{Background-corrected GFP signal (total green-object area) in "
             r"the LNP stability test at 12, 24 and 48\,h (mean\,$\pm$\,standard "
             r"deviation over wells per condition).}")
    L.append(r"\label{tab:lnp-gfp-short}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{l ccc}")
    L.append(r"\hline")
    L.append(r"\textbf{Condition} & \multicolumn{3}{c}{\textbf{GFP signal "
             r"($10^3$\,\textmu m$^2$/image)}}\\")
    L.append(r" & \textbf{12\,h} & \textbf{24\,h} & \textbf{48\,h}\\")
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
    L = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering")
    L.append(r"\caption{GFP signal in the LNP stability test normalised to 100\,\% "
             r"confluence (in $10^3$\,\textmu m$^2$/image; mean\,$\pm$\,standard "
             r"deviation over the "
             r"wells). The peak column gives the maximum value and the time at which "
             r"it occurs. Only the GFP-loaded conditions are listed; the "
             r"non-fluorescent controls remain at zero.}")
    L.append(r"\label{tab:lnp-gfp-norm}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{l ccc c}")
    L.append(r"\hline")
    L.append(r"\textbf{Condition} & \textbf{12\,h} & \textbf{24\,h} & \textbf{48\,h} "
             r"& \textbf{Peak (@\,h)}\\")
    L.append(r"\hline")
    for gname, gconds in GROUPS:
        L.append(r"\multicolumn{5}{l}{\emph{" + gname + r"}}\\")
        for orig, disp in gconds:
            if "empty" in orig.lower() or "reference" in orig.lower():
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

short = short_confluence() + "\n\n" + short_gfp() + "\n"
(LATEX_DIR / "lnp_tables_short.tex").write_text(short, encoding="utf-8")
(LATEX_DIR / "lnp_tables_norm.tex").write_text(short_gfp_norm() + "\n", encoding="utf-8")

print("Geschrieben:", LATEX_DIR / "lnp_tables_short.tex")
print("           ", LATEX_DIR / "lnp_tables_norm.tex")
print("\nKurz-Kontrolle (GFP hintergrund-korrigiert, 10^3 µm²/Image):")
for orig, disp in COND:
    corr, e, tv = gfp_corrected(orig)
    print(f"  {disp:<18} 12h={at(corr,tv,12)/1e3:7.1f}  24h={at(corr,tv,24)/1e3:7.1f}  "
          f"48h={at(corr,tv,48)/1e3:7.1f}")
