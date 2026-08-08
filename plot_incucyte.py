from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path.cwd()

INPUT_DIR = HERE
FILE_PATTERN = "*.txt"

CONDITION_RENAME = {
    "HEK, 24h LNP":    "HEK 293 (LNP GFP 24h)",
    "HEK fresh LNP":   "HEK 293 (LNP GFP)",
    "HEK LNP EMPTY":   "HEK 293 (LNP empty)",
    "HEK REFERENCE":   "HEK 293",
    "HUVEC 24h LNP":   "HUVEC (LNP GFP 24h)",
    "HUVEC fresh LNP": "HUVEC (LNP GFP)",
    "HUVEC LNP empty": "HUVEC (LNP empty)",
    "HUVEC REFERENCE": "HUVEC",
}
CONDITION_ORDER = []

NORMALIZE = {
    "Confluence.txt":                    "relativ",
    "green fluorescent total area.txt":  "null",
    "HEK293 confluence.txt":                     "relativ",
    "HUVEC confluence.txt":                      "relativ",
    "HEK293 green fluorescent total area.txt":   "null",
    "HUVEC green fluorescent total area.txt":    "null",
    "Confluence new analysis.txt":               "relativ",
    "Green Image Mean new analysis.txt":         "null",
}
NORMALIZE_DEFAULT = "keine"

TITLES = {
    "Confluence.txt": "LNP Stability Test – Confluence",
    "green fluorescent total area.txt": "LNP Stability Test – GFP Expression",
    "HEK293 confluence.txt":                   "LNP Stability Test – Confluence (HEK293)",
    "HUVEC confluence.txt":                    "LNP Stability Test – Confluence (HUVEC)",
    "HEK293 green fluorescent total area.txt": "LNP Stability Test – GFP Expression (HEK293)",
    "HUVEC green fluorescent total area.txt":  "LNP Stability Test – GFP Expression (HUVEC)",
    "Green Image Mean new analysis.txt":       "LNP Stability Test – GFP Expression",
}
TITLE_DEFAULT = "vessel"

SPLIT_PLOTS = {
    "Confluence.txt": [
        {"suffix": "HEK",   "match": ["HEK"],   "label": "HEK293"},
        {"suffix": "HUVEC", "match": ["HUVEC"], "label": "HUVEC"},
    ],
}

ERROR_STYLE = {
    "relativ": "ci",
    "null":    "errorbar",
    "keine":   "errorbar",
}
CI_MULT = 1.96
CI_ALPHA = 0.09
ERRORBAR_EVERY = 4

LINESTYLE_BY_KEYWORD = {
    "reference": (0, (6, 2)),
}

ERR_SUFFIXES = [" (Std Err Img)", " (Std Dev Well)", " (Std Err Well)"]
ERR_SUFFIX = ERR_SUFFIXES[0]

def is_err_col(col):
    return any(col.endswith(suf) for suf in ERR_SUFFIXES)

def err_col_for(col, columns):
    for suf in ERR_SUFFIXES:
        if col + suf in columns:
            return col + suf
    return None

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

def read_meta(path):
    meta = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "Elapsed" in line:
                break
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta

def load(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    try:
        h = next(i for i, l in enumerate(lines) if "Elapsed" in l)
    except StopIteration:
        return None
    df = pd.read_csv(path, sep="\t", decimal=",", skiprows=h, encoding="utf-8")
    df = df.dropna(axis=1, how="all")
    df.columns = [c.strip() for c in df.columns]
    if "Elapsed" not in df.columns:
        return None
    df["Elapsed"] = pd.to_numeric(df["Elapsed"], errors="coerce")
    return df.dropna(subset=["Elapsed"])

def find_conditions(df):
    skip = {"Date Time", "Elapsed"}
    means = [c for c in df.columns if c not in skip and not is_err_col(c)]
    return [(c, err_col_for(c, df.columns)) for c in means]

def resolve_title(path, vessel, metric):
    if path.name in TITLES:
        return TITLES[path.name]
    if TITLE_DEFAULT == "vessel":
        return vessel
    if TITLE_DEFAULT == "metric":
        return metric
    if TITLE_DEFAULT == "filename":
        return path.stem
    return TITLE_DEFAULT

def decide_log(metric_label, values):
    if "area" in metric_label.lower():
        return True
    pos = values[values > 0]
    if pos.size and pos.max() / pos.min() > 1e3:
        return True
    return False

def ylabel_for(metric, mode):
    if mode == "relativ":
        base = metric.split("(")[0].strip()
        return f"{base} (% of t = 0)"
    if mode == "null":
        return f"Normalized {metric}"
    return metric

def linestyle_for(name):
    low = name.lower()
    for key, ls in LINESTYLE_BY_KEYWORD.items():
        if key in low:
            return ls
    return "-"

def build_series(df, pairs, mode):
    raw = []
    for cond, err in pairs:
        y = df[cond].to_numpy(float)
        e = df[err].to_numpy(float) if err is not None else None
        raw.append((cond, y, e))

    if mode == "relativ":
        out = []
        for cond, y, e in raw:
            base = y[0] if (np.isfinite(y[0]) and y[0] != 0) else np.nan
            f = 100.0 / base
            out.append((cond, y * f, None if e is None else e * f))
        return out

    if mode == "null":
        out = []
        for cond, y, e in raw:
            base = y[0] if np.isfinite(y[0]) else 0.0
            out.append((cond, y - base, e))
        return out

    return raw

def apply_axes_style(ax, log_y):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.5)
        sp.set_color("black")
    ax.tick_params(axis="both", which="both", direction="in",
                   length=6, width=1.4, top=False, right=False)
    ax.tick_params(which="minor", length=3.5, width=1.0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    if not log_y:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.grid(False)

def plot_file(path, title=None, renames=None, mode=None,
              conditions=None, error_style=None, out_suffix=""):
    path = Path(path)
    df = load(path)
    if df is None:
        return None
    pairs = find_conditions(df)
    if not pairs:
        return None

    meta = read_meta(path)
    metric = meta.get("Metric", "Value")
    vessel = meta.get("Vessel Name", path.stem)
    if title is None:
        title = resolve_title(path, vessel, metric)
    if mode is None:
        mode = NORMALIZE.get(path.name, NORMALIZE_DEFAULT)
    if renames is None:
        renames = CONDITION_RENAME

    order = CONDITION_ORDER if CONDITION_ORDER else [c for c, _ in pairs]
    pairs = [(c, e) for c, e in pairs if c in order]
    pairs.sort(key=lambda ce: order.index(ce[0]))
    if conditions is not None:
        pairs = [(c, e) for c, e in pairs if c in conditions]
    if not pairs:
        return None

    t = df["Elapsed"].to_numpy(float)
    series = build_series(df, pairs, mode)

    all_vals = np.concatenate([y for _, y, _ in series])
    finite = all_vals[np.isfinite(all_vals)]
    log_y = (mode == "keine") and decide_log(metric, finite)
    err_style = error_style or ERROR_STYLE.get(mode, "errorbar")

    colors = cm.viridis(np.linspace(0.05, 0.85, len(series)))

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if log_y:
        pos = all_vals[(all_vals > 0) & np.isfinite(all_vals)]
        floor = pos.min() * 0.5 if pos.size else 1e-2
    else:
        floor = None

    for i, ((cond, y, e), col) in enumerate(zip(series, colors)):
        label = renames.get(cond, cond)
        ls = linestyle_for(cond)
        ax.plot(t, y, color=col, lw=2.4, ls=ls, zorder=3, label=label)
        if e is None or err_style == "keine":
            continue
        if err_style == "ci":
            lo, hi = y - CI_MULT * e, y + CI_MULT * e
            if log_y:
                lo = np.maximum(lo, floor)
            ax.fill_between(t, lo, hi, color=col, alpha=CI_ALPHA,
                            linewidth=0, zorder=2)
        else:
            idx = np.arange(i % ERRORBAR_EVERY, len(t), ERRORBAR_EVERY)
            if log_y:
                lower = y - np.maximum(y - e, floor)
                yerr = np.vstack([lower, e])[:, idx]
            else:
                yerr = e[idx]
            ax.errorbar(t[idx], y[idx], yerr=yerr, fmt="none", ecolor=col,
                        elinewidth=1.1, capsize=2.5, capthick=1.1,
                        alpha=0.9, zorder=2)

    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=min(0.0, float(np.nanmin(finite))))

    ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("Elapsed Time (h)")
    ax.set_ylabel(ylabel_for(metric, mode))
    if title:
        ax.set_title(title)
    apply_axes_style(ax, log_y)

    leg = ax.legend(loc="best", frameon=True, fancybox=False,
                    edgecolor="black", framealpha=1.0,
                    handlelength=2.2, borderpad=0.6, labelspacing=0.5)
    leg.get_frame().set_linewidth(1.2)

    fig.tight_layout()
    stem = path.stem + (f"_{out_suffix}" if out_suffix else "") + "_plot"
    out_png = path.with_name(stem + ".png")
    out_pdf = path.with_name(stem + ".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.name}  ->  {out_png.name}  (Normierung: {mode}, Fehler: {err_style})")
    return out_png

def plot_target(path, title=None, renames=None, mode=None):
    path = Path(path)
    specs = SPLIT_PLOTS.get(path.name)
    if not specs:
        out = plot_file(path, title=title, renames=renames, mode=mode)
        return [out] if out else []

    df = load(path)
    if df is None:
        return []
    conds = [c for c, _ in find_conditions(df)]
    if title is None:
        meta = read_meta(path)
        title = resolve_title(path, meta.get("Vessel Name", path.stem),
                              meta.get("Metric", "Value"))
    outs = []
    for spec in specs:
        match = [m.lower() for m in spec["match"]]
        sub = [c for c in conds if any(m in c.lower() for m in match)]
        if not sub:
            continue
        stitle = f"{title} ({spec['label']})" if title else None
        out = plot_file(path, title=stitle, renames=renames, mode=mode,
                        conditions=set(sub), error_style="errorbar",
                        out_suffix=spec["suffix"])
        if out:
            outs.append(out)
    return outs

def plot_raw_norm_pair(path, conditions, title, renames=None, out_suffix=""):
    path = Path(path)
    df = load(path)
    if df is None:
        return None
    pairs = [(c, e) for c, e in find_conditions(df) if c in conditions]
    if not pairs:
        return None
    if renames is None:
        renames = CONDITION_RENAME
    t = df["Elapsed"].to_numpy(float)
    colors = cm.viridis(np.linspace(0.05, 0.85, len(pairs)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6))
    panels = [
        (axes[0], "keine",   "Absolute",             "Phase Object Confluence (%)"),
        (axes[1], "relativ", "Normalised to $t_0$",  "Phase Object Confluence (% of $t_0$)"),
    ]
    handles = None
    for ax, mode, sub, ylab in panels:
        series = build_series(df, pairs, mode)
        hs = []
        for i, ((cond, y, e), col) in enumerate(zip(series, colors)):
            ln, = ax.plot(t, y, color=col, lw=2.4, ls=linestyle_for(cond),
                          zorder=3, label=renames.get(cond, cond))
            hs.append(ln)
            if e is not None:
                idx = np.arange(i % ERRORBAR_EVERY, len(t), ERRORBAR_EVERY)
                ax.errorbar(t[idx], y[idx], yerr=e[idx], fmt="none", ecolor=col,
                            elinewidth=1.1, capsize=2.5, capthick=1.1,
                            alpha=0.9, zorder=2)
        ax.set_ylim(bottom=0)
        ax.set_xlim(t.min(), t.max())
        ax.set_xlabel("Elapsed Time (h)")
        ax.set_ylabel(ylab)
        ax.set_title(sub, fontsize=15)
        apply_axes_style(ax, False)
        if handles is None:
            handles = hs

    if title:
        fig.suptitle(title, fontsize=TITLE_SIZE, y=0.99)
    fig.legend(handles=handles, labels=[renames.get(c, c) for c, _ in pairs],
               loc="lower center", ncol=len(pairs), frameon=True,
               edgecolor="black", framealpha=1.0)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    stem = path.stem + (f"_{out_suffix}" if out_suffix else "") + "_pair"
    out_png = path.with_name(stem + ".png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_name(stem + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.name} -> {out_png.name}")
    return out_png

def plot_gfp_combined(paths, out_path, title=None, renames=None, order=None,
                      conditions=None, baseline=None, conf_path=None):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(p) for p in paths]
    out_path = Path(out_path)
    conf_df = load(Path(conf_path)) if conf_path is not None else None
    if renames is None:
        renames = CONDITION_RENAME
    if title is None:
        title = "LNP Stability Test – GFP Expression"

    frames, all_pairs = {}, []
    for p in paths:
        df = load(p)
        if df is None:
            raise ValueError(f"Ungueltiges Format: {p.name}")
        frames[p] = df
        all_pairs.append((p, find_conditions(df)))

    cond_to_file = {}
    ordered = []
    for p, pairs in all_pairs:
        for cond, err in pairs:
            cond_to_file[cond] = (p, cond, err)
    if order:
        ordered = [cond_to_file[c] for c in order if c in cond_to_file]
    else:
        for p, pairs in all_pairs:
            ordered += [cond_to_file[c] for c, _ in pairs]
    if conditions is not None:
        ordered = [t for t in ordered if t[1] in conditions]

    metric = read_meta(paths[0]).get("Metric", "GFP signal")
    colors = cm.viridis(np.linspace(0.05, 0.85, len(ordered)))

    baseline_cache = {}

    def baseline_series(bcols):
        bcols = (bcols,) if isinstance(bcols, str) else tuple(bcols)
        if bcols not in baseline_cache:
            series = []
            for bc in bcols:
                if bc in cond_to_file:
                    bp, bcc, be = cond_to_file[bc]
                    (_, by, _), = build_series(frames[bp], [(bcc, be)], "null")
                    series.append(by)
            baseline_cache[bcols] = np.mean(series, axis=0) if series else None
        return baseline_cache[bcols]

    def baseline_for(cond):
        b = baseline.get(cond) if isinstance(baseline, dict) else baseline
        return baseline_series(b) if b else None

    any_baseline = baseline is not None

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axhline(0.0, color="0.6", lw=1.0, ls="-", zorder=1)

    t_ref = None
    ymin, ymax = 0.0, 0.0
    for (p, cond, err), col in zip(ordered, colors):
        df = frames[p]
        t = df["Elapsed"].to_numpy(float)
        if t_ref is None:
            t_ref = t
        (_, y, e), = build_series(df, [(cond, err)], "null")
        by = baseline_for(cond)
        if by is not None:
            y = y - by
        if conf_df is not None and cond in conf_df.columns:
            conf = conf_df[cond].to_numpy(float)
            scale = np.divide(100.0, conf, out=np.zeros_like(conf),
                              where=conf > 1e-6)
            y = y * scale
            if e is not None:
                e = e * scale
        ls = linestyle_for(cond)
        ax.plot(t, y, color=col, lw=2.4, ls=ls, zorder=3,
                label=renames.get(cond, cond))
        band = (e if e is not None else 0)
        ymax = max(ymax, float(np.nanmax(y + band)))
        ymin = min(ymin, float(np.nanmin(y - band)))
        if e is not None:
            idx = np.arange(len(t)) % ERRORBAR_EVERY == 0
            ax.errorbar(t[idx], y[idx], yerr=e[idx], fmt="none", ecolor=col,
                        elinewidth=1.1, capsize=2.5, capthick=1.1,
                        alpha=0.9, zorder=2)

    pad = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(t_ref.min(), t_ref.max())
    ax.set_xlabel("Elapsed Time (h)")
    if conf_df is not None:
        if metric.endswith(")"):
            i = metric.rfind("(")
            ylab = metric[:i].strip() + " at 100 % confluence " + metric[i:]
        else:
            ylab = metric + " at 100 % confluence"
    elif any_baseline:
        ylab = (metric[:-1] + ", background-corrected)" if metric.endswith(")")
                else metric + " (background-corrected)")
    else:
        ylab = ylabel_for(metric, "null")
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    apply_axes_style(ax, False)

    leg = ax.legend(loc="upper left", frameon=True, fancybox=False,
                    edgecolor="black", framealpha=1.0,
                    handlelength=2.2, borderpad=0.6, labelspacing=0.5)
    leg.get_frame().set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  GFP -> {out_path.name}")
    return out_path

GFP_ORDER = [
    "HEK, 24h LNP", "HEK fresh LNP", "HEK LNP EMPTY",
    "HUVEC 24h LNP", "HUVEC fresh LNP", "HUVEC LNP empty",
    "HUVEC REFERENCE", "HEK REFERENCE",
]

BILDER_DIR = (HERE / ".." / ".." / "Masterarbeit LaTeX" / "Bilder").resolve()

def build_thesis_figures(copy_to_bilder=True, titled=False):
    import shutil

    hek_conf   = HERE / "Hek Confluence new analysis.txt"
    hek_gfp    = HERE / "Hek green total area new analysis.txt"
    huvec_conf = HERE / "HUVEC confluence.txt"
    huvec_gfp  = HERE / "HUVEC green fluorescent total area.txt"

    def T(text):
        return text if titled else ""

    outputs = {}

    hek_pair = plot_raw_norm_pair(
        hek_conf, conditions={"HEK, 24h LNP", "HEK fresh LNP",
                              "HEK LNP EMPTY", "HEK REFERENCE"},
        title=T("Time-dependent Confluence under LNP Exposure (HEK293)"),
        out_suffix="HEK")
    outputs["Confluence_HEK_pair.png"] = hek_pair

    huvec_pair = plot_raw_norm_pair(
        huvec_conf, conditions={"HUVEC 24h LNP", "HUVEC fresh LNP",
                                "HUVEC LNP empty", "HUVEC REFERENCE"},
        title=T("Time-dependent Confluence under LNP Exposure (HUVEC)"),
        out_suffix="HUVEC")
    outputs["Confluence_HUVEC_pair.png"] = huvec_pair

    gfp_hek = plot_gfp_combined(
        hek_gfp, out_path=HERE / "lnp_gfp_area_HEK.png",
        title=T("LNP Stability Test – GFP Expression (HEK293)"), order=GFP_ORDER,
        conditions={"HEK, 24h LNP", "HEK fresh LNP",
                    "HEK LNP EMPTY", "HEK REFERENCE"},
        baseline=["HEK LNP EMPTY", "HEK REFERENCE"])
    outputs["lnp_gfp_area_HEK.png"] = gfp_hek

    gfp_huvec = plot_gfp_combined(
        huvec_gfp, out_path=HERE / "lnp_gfp_area_HUVEC.png",
        title=T("LNP Stability Test – GFP Expression (HUVEC)"), order=GFP_ORDER,
        conditions={"HUVEC 24h LNP", "HUVEC fresh LNP",
                    "HUVEC LNP empty", "HUVEC REFERENCE"},
        baseline=["HUVEC LNP empty", "HUVEC REFERENCE"])
    outputs["lnp_gfp_area_HUVEC.png"] = gfp_huvec

    gfp_norm_hek = plot_gfp_combined(
        hek_gfp, out_path=HERE / "lnp_gfp_norm_HEK.png",
        title=T("GFP per Confluence (HEK293)"), order=GFP_ORDER,
        conditions={"HEK, 24h LNP", "HEK fresh LNP",
                    "HEK LNP EMPTY", "HEK REFERENCE"},
        baseline=["HEK LNP EMPTY", "HEK REFERENCE"], conf_path=hek_conf)
    outputs["lnp_gfp_norm_HEK.png"] = gfp_norm_hek

    gfp_norm_huvec = plot_gfp_combined(
        huvec_gfp, out_path=HERE / "lnp_gfp_norm_HUVEC.png",
        title=T("GFP per Confluence (HUVEC)"), order=GFP_ORDER,
        conditions={"HUVEC 24h LNP", "HUVEC fresh LNP",
                    "HUVEC LNP empty", "HUVEC REFERENCE"},
        baseline=["HUVEC LNP empty", "HUVEC REFERENCE"], conf_path=huvec_conf)
    outputs["lnp_gfp_norm_HUVEC.png"] = gfp_norm_huvec

    if copy_to_bilder and BILDER_DIR.is_dir():
        for target_name, src_png in outputs.items():
            if not src_png:
                continue
            for src in (src_png, src_png.with_suffix(".pdf")):
                if src.exists():
                    dst = BILDER_DIR / (Path(target_name).stem + src.suffix)
                    if src.resolve() == dst.resolve():
                        continue
                    shutil.copyfile(src, dst)
                    print(f"  kopiert -> Bilder/{dst.name}")
    return outputs

def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("Incucyte Plots")
    root.geometry("960x620")

    keys = []
    info = {}
    current = {"key": None}
    cond_entries = []

    def valid_conditions(path):
        df = load(path)
        if df is None:
            return None
        conds = [c for c, _ in find_conditions(df)]
        return conds or None

    def add_paths(paths):
        added = 0
        for p in paths:
            p = Path(p)
            key = str(p.resolve())
            if key in info:
                continue
            conds = valid_conditions(p)
            if not conds:
                continue
            meta = read_meta(p)
            metric = meta.get("Metric", "Value")
            vessel = meta.get("Vessel Name", p.stem)
            info[key] = {
                "path": p,
                "conditions": conds,
                "title": resolve_title(p, vessel, metric),
                "mode": NORMALIZE.get(p.name, NORMALIZE_DEFAULT),
                "renames": {c: CONDITION_RENAME.get(c, c) for c in conds},
            }
            keys.append(key)
            filelist.insert("end", p.name)
            added += 1
        if added and current["key"] is None:
            filelist.selection_clear(0, "end")
            filelist.selection_set(0)
            show_file(0)

    def choose_files():
        paths = filedialog.askopenfilenames(
            title="Incucyte-Textdateien wählen", initialdir=str(INPUT_DIR),
            filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")])
        add_paths(paths)

    def choose_folder():
        d = filedialog.askdirectory(title="Ordner wählen", initialdir=str(INPUT_DIR))
        if d:
            add_paths(sorted(Path(d).glob(FILE_PATTERN)))

    def remove_selected():
        sel = list(filelist.curselection())
        for i in reversed(sel):
            info.pop(keys.pop(i), None)
            filelist.delete(i)
        current["key"] = None
        clear_panel()

    def save_panel():
        key = current["key"]
        if key is None or key not in info:
            return
        info[key]["title"] = title_var.get()
        info[key]["mode"] = mode_var.get()
        for cond, ent in cond_entries:
            info[key]["renames"][cond] = ent.get()

    def clear_panel():
        title_var.set("")
        for w in cond_frame.winfo_children():
            w.destroy()
        cond_entries.clear()

    def show_file(index):
        save_panel()
        key = keys[index]
        current["key"] = key
        d = info[key]
        title_var.set(d["title"])
        mode_var.set(d["mode"])
        for w in cond_frame.winfo_children():
            w.destroy()
        cond_entries.clear()
        for r, cond in enumerate(d["conditions"]):
            tk.Label(cond_frame, text=cond, anchor="w", width=22,
                     font=("Consolas", 9)).grid(row=r, column=0, sticky="w", pady=1)
            ent = tk.Entry(cond_frame, width=30)
            ent.insert(0, d["renames"].get(cond, cond))
            ent.grid(row=r, column=1, padx=4, pady=1)
            cond_entries.append((cond, ent))

    def on_select(_evt):
        sel = filelist.curselection()
        if sel:
            show_file(sel[0])

    def do_plot(which):
        save_panel()
        target = [current["key"]] if which == "one" else list(keys)
        target = [k for k in target if k]
        if not target:
            messagebox.showinfo("Nichts zu plotten", "Bitte zuerst eine Datei wählen.")
            return
        done, errors = [], []
        for key in target:
            d = info[key]
            try:
                outs = plot_target(d["path"], title=d["title"],
                                   renames=d["renames"], mode=d["mode"])
                if outs:
                    done.extend(outs)
                else:
                    errors.append(f"{d['path'].name}: ungültiges Format")
            except Exception as e:
                errors.append(f"{d['path'].name}: {e}")
        if done and open_var.get():
            for o in done:
                try:
                    os.startfile(str(o))
                except Exception:
                    pass
        msg = f"{len(done)} Plot(s) erstellt:\n" + "\n".join(o.name for o in done)
        if errors:
            msg += "\n\nProbleme:\n" + "\n".join(errors)
        messagebox.showinfo("Fertig", msg)

    tk.Label(root, text="Incucyte-Plots", font=("Segoe UI", 13, "bold")).pack(pady=(10, 2))
    main = tk.Frame(root)
    main.pack(fill="both", expand=True, padx=10, pady=6)

    left = tk.Frame(main)
    left.pack(side="left", fill="y")
    tk.Label(left, text="Dateien", font=("Segoe UI", 10, "bold")).pack(anchor="w")
    filelist = tk.Listbox(left, width=30, height=16, exportselection=False,
                          font=("Consolas", 9))
    filelist.pack(fill="y", expand=True)
    filelist.bind("<<ListboxSelect>>", on_select)
    fb = tk.Frame(left)
    fb.pack(fill="x", pady=4)
    tk.Button(fb, text="Dateien …", command=choose_files).grid(row=0, column=0, padx=2)
    tk.Button(fb, text="Ordner …", command=choose_folder).grid(row=0, column=1, padx=2)
    tk.Button(fb, text="Entfernen", command=remove_selected).grid(row=0, column=2, padx=2)

    right = tk.Frame(main)
    right.pack(side="left", fill="both", expand=True, padx=(12, 0))
    tk.Label(right, text="Einstellungen für die gewählte Datei",
             font=("Segoe UI", 10, "bold")).pack(anchor="w")

    trow = tk.Frame(right)
    trow.pack(fill="x", pady=(6, 2))
    tk.Label(trow, text="Titel:", width=11, anchor="w").pack(side="left")
    title_var = tk.StringVar()
    tk.Entry(trow, textvariable=title_var).pack(side="left", fill="x", expand=True)

    mrow = tk.Frame(right)
    mrow.pack(fill="x", pady=2)
    tk.Label(mrow, text="Normierung:", width=11, anchor="w").pack(side="left")
    mode_var = tk.StringVar(value=NORMALIZE_DEFAULT)
    tk.OptionMenu(mrow, mode_var, "keine", "relativ", "null").pack(side="left")
    tk.Label(mrow, text="  relativ = % von t=0   ·   null = Start bei 0",
             fg="gray").pack(side="left")

    tk.Label(right, text="Bedingungen  →  Anzeige-Name in der Legende:",
             anchor="w").pack(anchor="w", pady=(8, 2))
    cond_frame = tk.Frame(right)
    cond_frame.pack(fill="both", expand=True)

    open_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="Plots nach dem Erstellen automatisch öffnen",
                   variable=open_var).pack()
    bb = tk.Frame(root)
    bb.pack(pady=(2, 10))
    tk.Button(bb, text="Gewählte Datei plotten", width=22,
              command=lambda: do_plot("one")).grid(row=0, column=0, padx=6)
    tk.Button(bb, text="Alle plotten", width=16, font=("Segoe UI", 10, "bold"),
              command=lambda: do_plot("all")).grid(row=0, column=1, padx=6)

    add_paths(sorted(INPUT_DIR.glob(FILE_PATTERN)))

    root.update_idletasks()
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    root.focus_force()
    print("Incucyte-Plots: Fenster geoeffnet.")
    root.mainloop()

def main():
    if "--thesis" in sys.argv[1:]:
        print("Baue Thesis-Figuren aus den getrennten Datensaetzen …")
        build_thesis_figures()
        return
    file_args = [a for a in sys.argv[1:]
                 if Path(a).is_file() and Path(a).suffix.lower() == ".txt"]
    if file_args:
        for a in file_args:
            plot_target(Path(a))
        return
    try:
        run_gui()
    except Exception as e:
        print(f"Oberflaeche nicht verfuegbar ({e}); plotte Ordner {INPUT_DIR}")
        for p in sorted(INPUT_DIR.glob(FILE_PATTERN)):
            plot_target(p)

if __name__ == "__main__":
    main()
