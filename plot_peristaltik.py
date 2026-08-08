from pathlib import Path
import importlib.util
import shutil
import numpy as np

HERE = Path(__file__).parent
BILDER_DIR = HERE

SRC = HERE / "plot_incucyte.py"
spec = importlib.util.spec_from_file_location("plot_incucyte", SRC)
pi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pi)

HEK = {
    "dynamic":   "HEK 293 (LNP dynamic)",
    "static":    "HEK 293 (LNP static)",
    "fresh":     "HEK 293 (LNP fresh)",
    "reference": "HEK 293",
}
HUVEC = {
    "dynamic":   "HUVEC (LNP dynamic)",
    "static":    "HUVEC (LNP static)",
    "fresh":     "HUVEC (LNP fresh)",
    "reference": "HUVEC",
}

REF_COLS = {HEK["reference"], HUVEC["reference"]}

def _linestyle_ref(name):
    return (0, (6, 2)) if name in REF_COLS else "-"

pi.linestyle_for = _linestyle_ref

ART_WINDOW = (43.5, 47.5)
_orig_load = pi.load

def _load_trunc(path):
    df = _orig_load(path)
    if df is None:
        return df
    lo, hi = ART_WINDOW
    keep = ~((df["Elapsed"] > lo) & (df["Elapsed"] < hi))
    return df[keep].reset_index(drop=True)

pi.load = _load_trunc

RENAMES = {
    HEK["reference"]:   "HEK 293 (reference)",
    HUVEC["reference"]: "HUVEC (reference)",
}

HEK_ORDER   = [HEK["dynamic"],   HEK["static"],   HEK["fresh"],   HEK["reference"]]
HUVEC_ORDER = [HUVEC["dynamic"], HUVEC["static"], HUVEC["fresh"], HUVEC["reference"]]

def _base_index(t):
    idx = np.where(t >= 1)[0]
    return int(idx[0]) if idx.size else 0

def plot_conf_pair_t1(path, conditions, title, out_suffix):
    plt, cm = pi.plt, pi.cm
    df = pi.load(Path(path))
    pairs = [(c, e) for c, e in pi.find_conditions(df) if c in conditions]
    order = [c for c in (conditions if isinstance(conditions, list) else pairs)]
    pairs.sort(key=lambda ce: list(conditions).index(ce[0])
               if ce[0] in conditions else 99)
    t_full = df["Elapsed"].to_numpy(float)
    bi = _base_index(t_full)
    disp = t_full >= 1.0
    t = t_full[disp]
    colors = cm.viridis(np.linspace(0.05, 0.85, len(pairs)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6))
    panels = [
        (axes[0], "abs",  "Absolute",            "Phase Object Confluence (%)"),
        (axes[1], "norm", "Normalised to $t_1$", "Phase Object Confluence (% of $t_1$)"),
    ]
    handles = None
    for ax, mode, sub, ylab in panels:
        hs = []
        lo_v, hi_v = np.inf, -np.inf
        for i, ((cond, err), col) in enumerate(zip(pairs, colors)):
            y = df[cond].to_numpy(float)
            e = df[err].to_numpy(float) if err is not None else None
            if mode == "norm":
                base = y[bi] if (np.isfinite(y[bi]) and y[bi] != 0) else np.nan
                f = 100.0 / base
                y = y * f
                e = None if e is None else e * f
            yv = y[disp]
            ev = e[disp] if e is not None else None
            ln, = ax.plot(t, yv, color=col, lw=2.4, ls=pi.linestyle_for(cond),
                          zorder=3, label=RENAMES.get(cond, cond))
            hs.append(ln)
            band = ev if ev is not None else np.zeros_like(yv)
            lo_v = min(lo_v, float(np.nanmin(yv - band)))
            hi_v = max(hi_v, float(np.nanmax(yv + band)))
            if ev is not None:
                idx = np.arange(i % pi.ERRORBAR_EVERY, len(t), pi.ERRORBAR_EVERY)
                ax.errorbar(t[idx], yv[idx], yerr=ev[idx], fmt="none", ecolor=col,
                            elinewidth=1.1, capsize=2.5, capthick=1.1,
                            alpha=0.9, zorder=2)
        top = max(110.0, hi_v + 3.0)
        ax.set_ylim(0.0, top)
        ax.set_xlim(0, t_full.max())
        ax.set_xlabel("Elapsed Time (h)")
        ax.set_ylabel(ylab)
        ax.set_title(sub, fontsize=15)
        pi.apply_axes_style(ax, False)
        if handles is None:
            handles = hs

    if title:
        fig.suptitle(title, fontsize=pi.TITLE_SIZE, y=0.99)
    fig.legend(handles=handles, labels=[RENAMES.get(c, c) for c, _ in pairs],
               loc="lower center", ncol=len(pairs), frameon=True,
               edgecolor="black", framealpha=1.0)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    stem = Path(path).stem + f"_{out_suffix}" + "_pair"
    out_png = Path(path).with_name(stem + ".png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(path).with_name(stem + ".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  {Path(path).name} -> {out_png.name}")
    return out_png

def build(titled=True, copy_to_bilder=False):
    hek_conf   = HERE / "HEK293 Confluence.txt"
    hek_gfp    = HERE / "HEK293 Green Total Area.txt"
    huvec_conf = HERE / "HUVEC Confluence.txt"
    huvec_gfp  = HERE / "HUVEC Green Total Area.txt"

    def T(text):
        return text if titled else ""

    outputs = {}

    outputs["Confluence_peri_HEK_pair.png"] = plot_conf_pair_t1(
        hek_conf, conditions=HEK_ORDER,
        title=T("LNP Incubation on Peristaltic Pump – Confluence (HEK 293)"),
        out_suffix="peri_HEK")
    outputs["Confluence_peri_HUVEC_pair.png"] = plot_conf_pair_t1(
        huvec_conf, conditions=HUVEC_ORDER,
        title=T("LNP Incubation on Peristaltic Pump – Confluence (HUVEC)"),
        out_suffix="peri_HUVEC")

    outputs["peri_gfp_area_HEK.png"] = pi.plot_gfp_combined(
        hek_gfp, out_path=HERE / "peri_gfp_area_HEK.png",
        title=T("LNP Incubation on Peristaltic Pump – GFP (HEK 293)"),
        order=HEK_ORDER, conditions=set(HEK_ORDER), renames=RENAMES,
        baseline=[HEK["reference"]])
    outputs["peri_gfp_area_HUVEC.png"] = pi.plot_gfp_combined(
        huvec_gfp, out_path=HERE / "peri_gfp_area_HUVEC.png",
        title=T("LNP Incubation on Peristaltic Pump – GFP (HUVEC)"),
        order=HUVEC_ORDER, conditions=set(HUVEC_ORDER), renames=RENAMES,
        baseline=[HUVEC["reference"]])

    outputs["peri_gfp_norm_HEK.png"] = pi.plot_gfp_combined(
        hek_gfp, out_path=HERE / "peri_gfp_norm_HEK.png",
        title=T("GFP per Confluence (HEK 293)"),
        order=HEK_ORDER, conditions=set(HEK_ORDER), renames=RENAMES,
        baseline=[HEK["reference"]], conf_path=hek_conf)
    outputs["peri_gfp_norm_HUVEC.png"] = pi.plot_gfp_combined(
        huvec_gfp, out_path=HERE / "peri_gfp_norm_HUVEC.png",
        title=T("GFP per Confluence (HUVEC)"),
        order=HUVEC_ORDER, conditions=set(HUVEC_ORDER), renames=RENAMES,
        baseline=[HUVEC["reference"]], conf_path=huvec_conf)

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

if __name__ == "__main__":
    import sys
    titled = "--titled" in sys.argv[1:]
    build(titled=titled, copy_to_bilder=True)
    print("Fertig.")
