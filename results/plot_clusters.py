"""
PATH-CLUSTER plots for notebook 3 -- reads ONLY the saved `df_trials`, no logs and no video.

The path clustering itself is done upstream (`task_*/cluster_paths.py`, run as a library inside
`build_log_df`): each trial's trajectory is reduced to four PATH-GEOMETRY features
(efficiency, speed_std, time_in_corner, effective_duration_s), standardised, and KMeans-split into
*Direct* (high efficiency, short) vs *Corner-dwelling* (low efficiency, long), with *Exploratory* as
an occasional middle group. The result is written back onto every trial as `cluster` / `cluster_name`
(+ the 2-D PCA view `cluster_pca1` / `cluster_pca2`). This module only DRAWS what is already there.

*** DESCRIPTIVE ONLY. *** These panels show the grouping and the trajectories behind it; they draw no
verdict. The behavioural CONSEQUENCES of the split (whisking, joystick, heading) are the trial-report
/ hypothesis work, not this notebook.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'common'))

# same palette as task_*/cluster_paths.CLUSTER_COLORS, kept local so this needs no task import
CLUSTER_COLORS = {'Direct': '#27ae60', 'Corner-dwelling': '#c0392b', 'Exploratory': '#9b59b6'}
_NON_CLUSTER = {'excluded', 'unclustered', 'timeout', 'nan', ''}          # not real path groups


def clustered(T):
    """The trials that carry a real path cluster (drop escape/degenerate/unclustered rows)."""
    if 'cluster_name' not in T.columns:
        return T.iloc[0:0]
    C = T[T.cluster_name.notna() & ~T.cluster_name.astype(str).str.lower().isin(_NON_CLUSTER)].copy()
    if 'analyze' in C.columns:                       # keep only analyzable trials when the flag exists
        C = C[C.analyze.fillna(True)]
    return C


def _order(names):
    """Direct, Exploratory, Corner-dwelling first (best->worst efficiency), then anything else."""
    pref = ['Direct', 'Exploratory', 'Corner-dwelling']
    return [n for n in pref if n in names] + [n for n in names if n not in pref]


def _col(name, i=0):
    return CLUSTER_COLORS.get(name, plt.cm.tab10(i % 10))


# --------------------------------------------------------------------------------------------------
def cluster_overview(C, title=''):
    """How many trials in each cluster, and the four feature means that DEFINE the split."""
    names = _order(list(C.cluster_name.unique()))
    feats = [('path_efficiency', 'efficiency (higher = direct)'),
             ('time_in_corner', 'time in corner'),
             ('speed_std', 'speed variability'),
             ('dur_s', 'trial length (s)')]
    feats = [(c, lab) for c, lab in feats if c in C.columns]
    fig, ax = plt.subplots(1, 1 + len(feats), figsize=(3.4 * (1 + len(feats)), 4.2))
    ax = np.atleast_1d(ax)

    a = ax[0]                                         # how many trials per cluster
    counts = [int((C.cluster_name == n).sum()) for n in names]
    a.bar(range(len(names)), counts, color=[_col(n, i) for i, n in enumerate(names)])
    a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    for i, c in enumerate(counts):
        a.text(i, c, str(c), ha='center', va='bottom', fontweight='bold')
    a.set_ylabel('trials'); a.set_title('trials per cluster', fontweight='bold'); a.grid(alpha=.2, axis='y')

    for a, (c, lab) in zip(ax[1:], feats):            # the defining features, mean +/- SEM
        for i, n in enumerate(names):
            v = pd.to_numeric(C.loc[C.cluster_name == n, c], errors='coerce').dropna()
            m = v.mean(); s = v.std() / np.sqrt(len(v)) if len(v) else 0
            a.bar(i, m, color=_col(n, i)); a.errorbar(i, m, yerr=s, fmt='none', ecolor='k', capsize=4)
        a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
        a.set_title(lab, fontweight='bold', fontsize=9); a.grid(alpha=.2, axis='y')

    fig.suptitle(title or 'Path clusters -- counts and the features that define them',
                 fontweight='bold', fontsize=12)
    plt.tight_layout(); return fig


def cluster_scatter(C, title=''):
    """The clustering in 2-D: the stored PCA view, and the raw efficiency-vs-corner feature plane."""
    names = _order(list(C.cluster_name.unique()))
    has_pca = {'cluster_pca1', 'cluster_pca2'} <= set(C.columns) and C.cluster_pca1.notna().any()
    has_feat = {'path_efficiency', 'time_in_corner'} <= set(C.columns)
    panels = [p for p, ok in [('pca', has_pca), ('feat', has_feat)] if ok] or ['feat']
    fig, ax = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5)); ax = np.atleast_1d(ax)

    for a, p in zip(ax, panels):
        for i, n in enumerate(names):
            g = C[C.cluster_name == n]
            if p == 'pca':
                a.scatter(g.cluster_pca1, g.cluster_pca2, s=22, alpha=.7, color=_col(n, i), label=n)
                a.set_xlabel('PCA 1'); a.set_ylabel('PCA 2')
                a.set_title('cluster space (PCA of the 4 features)', fontweight='bold')
            else:
                a.scatter(g.time_in_corner, g.path_efficiency, s=22, alpha=.7, color=_col(n, i), label=n)
                a.set_xlabel('time in corner'); a.set_ylabel('path efficiency')
                a.set_title('efficiency vs corner-dwelling', fontweight='bold')
        a.legend(fontsize=8); a.grid(alpha=.2)
    fig.suptitle(title or 'Path clusters in 2-D', fontweight='bold', fontsize=12)
    plt.tight_layout(); return fig


def example_paths(C, n=5, seed=0, title=''):
    """A few real TRAJECTORIES per cluster -- the point of the whole split. Each path is plasma-coded
    from trial start (dark) to the collection (bright); the world box is autoscaled from the coords."""
    names = _order(list(C.cluster_name.unique()))
    has_xy = {'coord_x', 'coord_y'} <= set(C.columns)
    if not has_xy:
        fig, a = plt.subplots(figsize=(6, 2)); a.axis('off')
        a.text(.5, .5, 'no coord_x/coord_y arrays in df_trials -- cannot draw trajectories',
               ha='center'); return fig
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(len(names), n, figsize=(2.5 * n, 2.6 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        g = C[C.cluster_name == name]
        take = g.sample(min(n, len(g)), random_state=int(rng.integers(1e9))) if len(g) else g
        for c in range(n):
            a = ax[r][c]; a.set_xticks([]); a.set_yticks([])
            if c >= len(take):
                a.axis('off'); continue
            row = take.iloc[c]
            xx = np.asarray(row.coord_x, float); yy = np.asarray(row.coord_y, float)
            ok = np.isfinite(xx) & np.isfinite(yy)
            xx, yy = xx[ok], yy[ok]
            if len(xx) > 1:
                a.scatter(xx, yy, c=np.arange(len(xx)), cmap='plasma', s=6)
                a.plot(xx[0], yy[0], 'o', color='k', ms=5)              # start
                a.plot(xx[-1], yy[-1], '*', color=_col(name, r), ms=12)  # collection
            a.set_aspect('equal', 'datalim'); a.invert_yaxis()
            eff = row.get('path_efficiency', np.nan)
            a.set_title(f'{row.get("outcome", "")}  eff={eff:.2f}' if np.isfinite(eff)
                        else str(row.get('outcome', '')), fontsize=7)
        ax[r][0].set_ylabel(name, color=_col(name, r), fontweight='bold', fontsize=10)
    fig.suptitle(title or 'Example trajectories per cluster  (black o = start, star = collection)',
                 fontweight='bold', fontsize=12)
    plt.tight_layout(); return fig


def cluster_composition(C, by='outcome', title=''):
    """Stacked share of each cluster within each `by` group (outcome / mouse / world)."""
    if by not in C.columns:
        fig, a = plt.subplots(figsize=(6, 2)); a.axis('off')
        a.text(.5, .5, f'no "{by}" column', ha='center'); return fig
    names = _order(list(C.cluster_name.unique()))
    groups = list(C[by].dropna().unique())
    frac = pd.DataFrame({n: [(C[(C[by] == g) & (C.cluster_name == n)].shape[0]
                             / max(1, C[C[by] == g].shape[0])) for g in groups] for n in names},
                        index=[str(g) for g in groups])
    fig, a = plt.subplots(figsize=(max(6, 1.1 * len(groups) + 3), 4.5))
    bottom = np.zeros(len(groups))
    for i, n in enumerate(names):
        a.bar(range(len(groups)), frac[n], bottom=bottom, color=_col(n, i), label=n)
        bottom += frac[n].values
    a.set_xticks(range(len(groups))); a.set_xticklabels(frac.index, rotation=25, ha='right', fontsize=8)
    a.set_ylabel('share of trials'); a.set_ylim(0, 1); a.legend(fontsize=8)
    a.set_title(title or f'cluster composition by {by}', fontweight='bold'); a.grid(alpha=.2, axis='y')
    plt.tight_layout(); return fig
