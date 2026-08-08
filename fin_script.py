import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib.patches import Patch
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR = Path('.')

KDE_BW         = 50.0
FOCUS_TOL      = 40.0
CELL_FOCUS_MIN = 3000.0

MIN_VALID_PTS  = 8
MIN_TIME_SPAN  = 40.0

T_GRID_N       = 500

EXPLICIT_EXCL = {
    ('P1', 'Fibrinogen Coating (Dried) 100 mg/ml (A5)'),
    ('P1', 'Fibrinogen Coating 100 mg/ml (B4)'),
}

WELLS = {
    'Dried Fibrinogen': {
        'P1': ['Fibrinogen Coating (Dried) 100 mg/ml (A1)',
               'Fibrinogen Coating (Dried) 100 mg/ml (A2)',
               'Fibrinogen Coating (Dried) 100 mg/ml (A3)',
               'Fibrinogen Coating (Dried) 100 mg/ml (A4)',
               'Fibrinogen Coating (Dried) 100 mg/ml (A5)'],
        'P2': ['Fibrinogen Coating (Dried) 100 mg/ml (A1)',
               'Fibrinogen Coating (Dried) 100 mg/ml (A2)',
               'Fibrinogen Coating (Dried) 100 mg/ml (A3)']},
    'Wet Fibrinogen': {
        'P1': ['Fibrinogen Coating 100 mg/ml (B1)',
               'Fibrinogen Coating 100 mg/ml (B2)',
               'Fibrinogen Coating 100 mg/ml (B3)',
               'Fibrinogen Coating 100 mg/ml (B4)',
               'Fibrinogen Coating 100 mg/ml (B5)'],
        'P2': ['Fibrinogen Coating 100 mg/ml (B1)',
               'Fibrinogen Coating 100 mg/ml (B2)',
               'Fibrinogen Coating 100 mg/ml (B3)']},
    'COC Untreated': {
        'P1': ['COC untreated (C1)', 'COC untreated (C2)',
               'COC untreated (C3)', 'COC untreated (C4)',
               'COC untreated (C5)'],
        'P2': ['COC untreated (C1)', 'COC untreated (C2)',
               'COC untreated (C3)']},
    'Well Only (TC plastic)': {
        'P1': ['Well only Reference (A6)', 'Well only Reference (B6)',
               'Well only Reference (C6)', 'Well only Reference (D6)'],
        'P2': ['Well only Reference (D1)', 'Well only Reference (D2)',
               'Well only Reference (D3)']},
    'Inlet Only (no COC)': {
        'P1': ['Inlet only Reference (D1)', 'Inlet only Reference (D2)',
               'Inlet only Reference (D3)', 'Inlet only Reference (D4)',
               'Inlet only Reference (D5)'],
        'P2': []},
}
COND_LIST = list(WELLS.keys())
COLORS = {c: cm.viridis(v)
          for c, v in zip(COND_LIST, [0.05, 0.28, 0.52, 0.75, 0.92])}

def load(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    h = next(i for i, l in enumerate(lines) if 'Elapsed' in l)
    df = pd.read_csv(path, sep='\t', decimal=',',
                     skiprows=h, encoding='utf-8')
    df.columns = df.columns.str.strip()
    df['Elapsed'] = pd.to_numeric(df['Elapsed'], errors='coerce')
    return df.dropna(subset=['Elapsed'])

def find_cell_plane(f_values):
    fv = f_values[(~np.isnan(f_values)) & (f_values >= 0)]
    if len(fv) < 5:
        return None, 0.0
    if fv.std() < 1.0:
        pos = float(fv.mean())
        return (pos, 1.0) if pos >= CELL_FOCUS_MIN else (None, 0.0)
    kde = gaussian_kde(fv, bw_method=KDE_BW / fv.std())
    x = np.linspace(fv.min() - 50, fv.max() + 50, 2000)
    d = kde(x)
    peaks = [x[i] for i in range(1, len(d) - 1)
             if d[i] > d[i-1] and d[i] > d[i+1]]
    high = [(p, np.sum(np.abs(fv - p) <= FOCUS_TOL) / len(fv))
            for p in peaks if p >= CELL_FOCUS_MIN]
    return max(high, key=lambda x: x[1]) if high else (None, 0.0)

def gompertz(t, C0, A, mu, lag):
    return C0 + A * np.exp(-np.exp((mu * np.e / A) * (lag - t) + 1.0))

def gompertz_decline(t, C0, A, mu, lag, kd, td):
    g = gompertz(t, C0, A, mu, lag)
    s = 1.0 / (1.0 + np.exp(-(t - td) / 2.0))
    return g * (1 - s) + g * s * np.exp(-kd * np.clip(t - td, 0, None))

def logistic4(t, C0, A, k, tmid):
    return C0 + (A - C0) / (1.0 + np.exp(-k * (t - tmid)))

def linear(t, a, b):
    return a * t + b

def _fit(fn, t, y, p0, bounds, k):
    try:
        popt, _ = curve_fit(fn, t, y, p0=p0, bounds=bounds, maxfev=40000)
        ss_res = np.sum((y - fn(t, *popt))**2)
        aic = len(t) * np.log(max(ss_res / len(t), 1e-12)) + 2 * k
        return {'popt': popt, 'aic': aic, 'fn': fn, 'k': k}
    except Exception:
        return None

def fit_well(t, y):
    C0 = max(float(y[0]), 0.5)
    A0 = max(float(y.max()) - C0, 1.0)
    pk = int(np.argmax(y)); n = len(t)
    td0 = t[pk] if pk > n//3 else t[n//2]
    kd0 = max(-np.log(max(y[-1]/max(y[pk], 0.1), 0.01)) /
              max(t[-1]-t[pk], 1.0), 0.001) if pk < n-3 else 0.005
    Acap = min(max(float(y.max()) * 1.5, 5.0), 100.0)

    cands = {}
    r = _fit(gompertz, t, y,
             p0=[C0, A0, 0.5, max(t[0], 1.0)],
             bounds=([0, 0.1, 0.001, 0], [100, 100, 20, t.max()*0.8]), k=4)
    if r: cands['gompertz'] = r

    r = _fit(gompertz_decline, t, y,
             p0=[C0, max(float(y[pk])-C0, 1), 0.5, max(t[0], 1), kd0, td0],
             bounds=([0, 0.1, 0.001, 0, 0, t[min(3, n-1)]],
                     [100, Acap, 20, t.max()*0.7, 0.3, t.max()]), k=6)
    if r: cands['gompertz_decline'] = r

    A0l = min(max(float(y.max())*1.1, C0+1), 100)
    r = _fit(logistic4, t, y,
             p0=[C0, A0l, 0.05, float(np.median(t))],
             bounds=([0, C0, 0.001, t.min()-20], [100, 100, 5, t.max()+20]), k=4)
    if r: cands['logistic4'] = r

    r = _fit(linear, t, y, p0=[0.0, float(y.mean())],
             bounds=([-1, -10], [1, 110]), k=2)
    if r: cands['linear'] = r

    if not cands:
        return None

    best = min(v['aic'] for v in cands.values())
    chosen = sorted([(n, v) for n, v in cands.items()
                     if v['aic'] - best < 4.0],
                    key=lambda x: x[1]['k'])[0][1]
    return chosen

def assess(well, plate, conf_df, focus_df):
    r = {'well': well, 'plate': plate, 'pass': False, 'reason': '',
         't_all': None, 'c_all': None, 't_valid': None, 'c_valid': None,
         'cell_plane': np.nan, 'fit': None}

    if well not in conf_df.columns:
        r['reason'] = 'column missing'; return r

    t = conf_df['Elapsed'].values.astype(float)
    c = conf_df[well].values.astype(float)
    f = focus_df[well].values.astype(float)
    r['t_all'] = t; r['c_all'] = c

    if (plate, well) in EXPLICIT_EXCL:
        r['reason'] = 'pre-registered exclusion'; return r

    valid = (~np.isnan(c)) & (~np.isnan(f)) & (f >= 0)
    pos, _ = find_cell_plane(f[valid])
    if pos is None:
        r['reason'] = 'no cell-plane mode ≥ 3000 µm'; return r
    r['cell_plane'] = pos

    in_band = valid & (np.abs(f - pos) <= FOCUS_TOL)
    if in_band.sum() < MIN_VALID_PTS:
        r['reason'] = f'only {in_band.sum()} valid points'; return r

    idx = np.argsort(t[in_band])
    tv = t[in_band][idx]; cv = c[in_band][idx]

    if tv[-1] - tv[0] < MIN_TIME_SPAN:
        r['reason'] = f'time span {tv[-1]-tv[0]:.0f} h < {MIN_TIME_SPAN} h'
        return r

    r['t_valid'] = tv; r['c_valid'] = cv
    fit = fit_well(tv, cv)
    if fit is None:
        r['reason'] = 'no model converged'; return r

    r['fit'] = fit
    r['pass'] = True
    r['reason'] = 'passed'
    return r

DATA = {
    'P1': (load(INPUT_DIR / 'Fibrinogen_Analyse_Plate_1_Confluence.txt'),
           load(INPUT_DIR / 'Fibrinogen_Analyse_Plate_1_Focus_Position.txt')),
    'P2': (load(INPUT_DIR / 'Fibrinogen_Analyse_Plate_2_Confluence.txt'),
           load(INPUT_DIR / 'Fibrinogen_Analyse_Plate_2_Focus_Position.txt')),
}

all_wells = []
for cond, plates in WELLS.items():
    for plate, well_list in plates.items():
        cdf, fdf = DATA[plate]
        for well in well_list:
            r = assess(well, plate, cdf, fdf)
            r['condition'] = cond
            all_wells.append(r)

print('Wells per condition:')
for cond in COND_LIST:
    in_c = [r for r in all_wells if r['condition'] == cond]
    n_pass = sum(1 for r in in_c if r['pass'])
    print(f'  {cond:<28} {n_pass}/{len(in_c)} passed')
print('\nExcluded:')
for r in all_wells:
    if not r['pass']:
        print(f'  [{r["plate"]}] {r["well"][-10:]}  →  {r["reason"]}')

def plot_qc(all_wells, fname='fibrinogen_qc.png'):
    n = len(all_wells)
    nc = 6; nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(nc*2.8, nr*2.4),
                             constrained_layout=True)
    axes = np.atleast_2d(axes).flatten()
    tg = np.linspace(0, 140, 400)

    for ax, r in zip(axes, all_wells):
        if r['t_all'] is not None:
            ax.scatter(r['t_all'], r['c_all'], s=5, c='lightgray',
                       alpha=0.5, linewidths=0, zorder=1)
        if r['pass']:
            ax.scatter(r['t_valid'], r['c_valid'], s=7, c='#1a4a2e',
                       alpha=0.85, linewidths=0, zorder=3)
            yf = np.clip(r['fit']['fn'](tg, *r['fit']['popt']), 0, 100)
            ax.plot(tg, yf, '-', c='#e54f1f', lw=1.5, zorder=4)

        col = '#1a4a2e' if r['pass'] else '#a92020'
        label = 'PASS' if r['pass'] else 'EXCL'
        short = (r['well']
                 .replace('Fibrinogen Coating', 'FC')
                 .replace('100 mg/ml', '')
                 .replace('only Reference', 'ref').strip())[:22]
        ax.set_title(f'{r["plate"]} {short}\n[{label}]',
                     fontsize=7, color=col)
        ax.set_xlabel('t (h)', fontsize=6)
        ax.set_ylabel('Conf %', fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, 142); ax.set_ylim(-2, 105)
        if not r['pass']:
            ax.text(0.5, 0.5, 'EXCL', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12,
                    color='#a92020', alpha=0.28, fontweight='bold')

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle('Per-well QC  —  grey: all raw data  |  '
                 'green: focus-valid points  |  orange: fit from t = 0',
                 fontsize=10)
    fig.savefig(fname, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {fname}')

def plot_growth(all_wells, fname='fibrinogen_growth.png'):
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    tg = np.linspace(0, 140, T_GRID_N)
    handles = []

    for cond in COND_LIST:
        col = COLORS[cond]
        passed = [r for r in all_wells
                  if r['condition'] == cond and r['pass']]
        n_p = len(passed)

        for r in passed:
            yf = np.clip(r['fit']['fn'](tg, *r['fit']['popt']), 0, 100)
            ax.plot(tg, yf, '-', color=col, alpha=0.22, lw=1.0, zorder=2)

        if n_p == 0:
            handles.append(Patch(facecolor=col, edgecolor='black',
                                 label=f'{cond}  (n={n_p})'))
            continue

        mat = np.array([
            np.clip(r['fit']['fn'](tg, *r['fit']['popt']), 0, 100)
            for r in passed
        ])
        mean = mat.mean(axis=0)
        if n_p > 1:
            se = mat.std(axis=0, ddof=1) / np.sqrt(n_p)
            ci_lo, ci_hi = mean - 1.96*se, mean + 1.96*se
        else:
            ci_lo = ci_hi = mean

        ax.plot(tg, mean, '-', color=col, lw=2.8, zorder=5)
        ax.fill_between(tg, ci_lo, ci_hi, color=col, alpha=0.20, zorder=4)
        handles.append(Patch(facecolor=col, edgecolor='black',
                             label=f'{cond}  (n={n_p})'))

    for sp in ax.spines.values():
        sp.set_linewidth(1.4); sp.set_color('black')
    ax.tick_params(direction='in', length=7, width=1.2,
                   top=True, right=True, labelsize=13)
    ax.tick_params(which='minor', direction='in', length=3.5,
                   width=0.8, top=True, right=True)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_xlim(0, 142); ax.set_ylim(0)
    ax.set_xlabel('Elapsed Time (h)', fontsize=14, labelpad=10)
    ax.set_ylabel('Phase Object Confluence (%)', fontsize=14, labelpad=10)
    ax.set_title('HUVEC Cell Confluence on Fibrinogen-Coated COC Surfaces',
                 fontsize=14, pad=12, fontweight='bold')
    ax.legend(handles=handles, loc='upper left', fontsize=11,
              frameon=True, edgecolor='black', facecolor='white')
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {fname}')

plot_qc(all_wells)
plot_growth(all_wells)
