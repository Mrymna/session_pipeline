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

# same on-screen icon style as task_banish_multiplier/make_trial_report.ICON_STYLE
ICON_STYLE = {'single_reward': ('*', '#27ae60', 13, 'reward'),
              'banish': ('X', '#c0392b', 9, 'banish'),
              'unbanish': ('o', '#2980b9', 8, 'unbanish ring')}


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
    """The clusters in REAL feature units -- NOT the stored PCA.

    Clustering is done per SESSION, so `cluster_pca1/2` live in each session's own PCA basis; once
    sessions are pooled those coordinates are no longer comparable and the scatter looks like one
    unseparable blob. The raw path-geometry features (efficiency, time in corner, trial length) are
    in the same physical units across every session, so THOSE are what show the separation. Direct
    sits high-efficiency / low-corner / short; Corner-dwelling the opposite.
    """
    names = _order(list(C.cluster_name.unique()))
    planes = [('time_in_corner', 'path_efficiency', 'time in corner', 'path efficiency'),
              ('dur_s', 'path_efficiency', 'trial length (s)', 'path efficiency')]
    planes = [p for p in planes if p[0] in C.columns and p[1] in C.columns]
    if not planes:
        fig, a = plt.subplots(figsize=(6, 2)); a.axis('off')
        a.text(.5, .5, 'no path-geometry feature columns to plot', ha='center'); return fig
    fig, ax = plt.subplots(1, len(planes), figsize=(6 * len(planes), 5)); ax = np.atleast_1d(ax)
    for a, (cx, cy, lx, ly) in zip(ax, planes):
        for i, n in enumerate(names):
            g = C[C.cluster_name == n]
            a.scatter(pd.to_numeric(g[cx], errors='coerce'), pd.to_numeric(g[cy], errors='coerce'),
                      s=22, alpha=.65, color=_col(n, i), label=n)
        if cx == 'dur_s':
            a.set_xscale('log')
        a.set_xlabel(lx); a.set_ylabel(ly); a.legend(fontsize=8); a.grid(alpha=.2)
        a.set_title(f'{ly} vs {lx}', fontweight='bold')
    fig.suptitle(title or 'Path clusters in real feature units (they separate on efficiency)',
                 fontweight='bold', fontsize=12)
    plt.tight_layout(); return fig


def _turn_arrows(x, y, t, min_turn_deg=30, min_gap_s=0.3):
    """Facing arrows at direction changes >= min_turn_deg, from the PATH's own velocity direction
    (df_trials has no per-frame theta) -- a close visual proxy for the trial report's avatar-facing
    arrows. Returns x, y, u, v, and time-from-collection for plasma colouring."""
    if len(x) < 3:
        return (np.array([]),) * 5
    from scipy.ndimage import gaussian_filter1d
    xs = gaussian_filter1d(x, 1.5); ys = gaussian_filter1d(y, 1.5)
    ang = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    keep, anchor, last_t = [0], ang[0], t[0]
    for i in range(1, len(x)):
        if abs(ang[i] - anchor) >= np.radians(min_turn_deg) and (t[i] - last_t) >= min_gap_s * 1000:
            keep.append(i); anchor = ang[i]; last_t = t[i]
    keep = np.array(keep); a = ang[keep]
    return x[keep], y[keep], np.cos(a), np.sin(a), (t[keep] - t[-1]) / 1000.0


def example_paths(C, n=5, seed=0, wh=None, title=''):
    """A few real TRAJECTORIES per cluster, drawn in the SAME style as the JPAS_0168 trial_report.pdf
    path panel (log-available parts only): arena box, plasma trajectory dark=start -> bright=collection,
    facing arrows at direction changes, and the full on-screen ICON landscape with the collected icon
    ringed (* reward / X banish / o unbanish). Two rows are kept: Direct vs Corner-dwelling.

    Not shown (video-derived, absent from df_trials): licking dots. Facing arrows use the path's own
    velocity direction rather than the logged avatar theta.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle
    names = _order(list(C.cluster_name.unique()))
    if not ({'coord_x', 'coord_y'} <= set(C.columns)):
        fig, a = plt.subplots(figsize=(6, 2)); a.axis('off')
        a.text(.5, .5, 'no coord_x/coord_y arrays in df_trials -- cannot draw trajectories',
               ha='center'); return fig

    if wh is None:                                       # consistent world box across all panels
        mx = my = 0.0
        for _, r in C.iterrows():
            xx = np.asarray(r.coord_x, float); yy = np.asarray(r.coord_y, float)
            if len(xx):
                mx = max(mx, np.nanmax(xx)); my = max(my, np.nanmax(yy))
            for ic in (r.get('icons') or []):
                if isinstance(ic, dict) and ic.get('x') is not None and np.isfinite(ic['x']):
                    mx = max(mx, ic['x']); my = max(my, ic['y'])
        W = int(np.ceil((mx or 2400) / 100) * 100); H = int(np.ceil((my or 2400) / 100) * 100)
    else:
        W, H = wh

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(len(names), n, figsize=(2.8 * n, 2.9 * len(names)),
                           squeeze=False, constrained_layout=True)
    lc_last = None
    for r, name in enumerate(names):
        g = C[C.cluster_name == name]
        take = g.sample(min(n, len(g)), random_state=int(rng.integers(1e9))) if len(g) else g
        for c in range(n):
            a = ax[r][c]; a.set_xticks([]); a.set_yticks([])
            if c >= len(take):
                a.axis('off'); continue
            row = take.iloc[c]
            a.add_patch(Rectangle((0, 0), W, H, fill=False, ec='0.5', lw=1))
            x = np.asarray(row.coord_x, float); y = np.asarray(row.coord_y, float)
            t = np.asarray(row.get('coord_t_ms', np.arange(len(x))), float)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y, t = x[ok], y[ok], t[ok]
            if len(x) > 1:
                pts = np.array([x, y]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap='plasma', lw=2, zorder=3)
                lc.set_array((t[:-1] - t[-1]) / 1000.0); a.add_collection(lc); lc_last = lc
                gx, gy, u, v, gt = _turn_arrows(x, y, t)
                if len(gx):
                    a.quiver(gx, gy, u, v, gt, cmap='plasma', zorder=6, pivot='tail', scale=14,
                             width=0.013, headwidth=4, headlength=5, edgecolor='k', linewidth=0.4)
                    a.plot(gx, gy, 'o', mfc='w', mec='k', ms=2.6, mew=0.5, zorder=6.5)
                a.plot(x[0], y[0], 'o', mfc='k', mec='w', ms=7, mew=1, zorder=12)     # start
            locs = [ic.get('loc') for ic in (row.get('icons') or []) if isinstance(ic, dict)]
            for ic in (row.get('icons') or []):                                       # icon landscape
                if not isinstance(ic, dict):
                    continue
                eff, ox, oy = ic.get('effect'), ic.get('x'), ic.get('y')
                if eff not in ICON_STYLE or ox is None or oy is None or not np.isfinite(ox):
                    continue
                mk, col, msz, _ = ICON_STYLE[eff]
                a.plot(ox, oy, mk, mfc=col if mk != 'o' else 'none', mec='k', ms=msz, mew=1.2, zorder=5)
                if ic.get('loc') == row.get('target_loc'):                            # ring the collected
                    a.plot(ox, oy, 'o', mfc='none', mec='k', ms=msz + 5, mew=1.5, zorder=6)
            if (row.get('target_loc') not in locs                                     # fallback ring
                    and np.isfinite(row.get('target_x', np.nan))):
                a.plot(row.target_x, row.target_y, 'o', mfc='none', mec='k', ms=14, mew=1.5, zorder=6)
            M = 160; a.set_xlim(-M, W + M); a.set_ylim(H + M, -M); a.set_aspect('equal')
            eff = row.get('path_efficiency', np.nan)
            a.set_title(f'{row.get("outcome", "")}  eff={eff:.2f}' if np.isfinite(eff)
                        else str(row.get('outcome', '')), fontsize=7)
        ax[r][0].set_ylabel(name, color=_col(name, r), fontweight='bold', fontsize=11)
    if lc_last is not None:
        cb = fig.colorbar(lc_last, ax=ax, location='right', fraction=0.03, pad=0.01, aspect=40)
        cb.set_label('time from collection (s)  dark=start -> bright=t0', fontsize=8)
    fig.suptitle(title or 'Example trajectories per cluster  '
                 '(o start, arrows=facing at turns; * reward / X banish / o unbanish, ringed=collected)',
                 fontweight='bold', fontsize=12)
    return fig


def cluster_share_by_animal(C, title=''):
    """The 'per animal, then averaged across animals' view. For each animal, the FRACTION of its
    trials in each cluster (Direct / Corner-dwelling), one group of bars per animal; then an ALL group
    = the mean ACROSS animals (equal weight per animal, error = SEM across animals), so a prolific
    animal does not dominate. This is the cluster analogue of notebook 2's mouse_summary: aggregate
    within an animal first, then average the animals.
    """
    names = _order(list(C.cluster_name.unique()))
    animals = sorted(C.mouse.unique()) if 'mouse' in C.columns else ['all']
    share = {a: [(C[(C.mouse == a) & (C.cluster_name == n)].shape[0]
                  / max(1, C[C.mouse == a].shape[0])) for n in names] for a in animals}
    fig, a = plt.subplots(figsize=(max(7, 1.5 * (len(animals) + 1) + 2), 4.6))
    xg = np.arange(len(animals) + 1); w = 0.8 / max(1, len(names))
    for i, n in enumerate(names):
        vals = [share[a][i] for a in animals]
        allm = np.nanmean(vals); sem = np.nanstd(vals) / np.sqrt(max(1, len(animals)))
        a.bar(xg + i * w, vals + [allm], w, color=_col(n, i), label=n)
        a.errorbar(xg[-1] + i * w, allm, yerr=sem, fmt='none', ecolor='k', capsize=3)
    a.set_xticks(xg + (len(names) - 1) * w / 2)
    a.set_xticklabels([str(x) for x in animals] + ['ALL\n(mean±SEM\nacross animals)'],
                      rotation=25, ha='right', fontsize=8)
    a.set_ylabel('share of trials'); a.set_ylim(0, 1); a.legend(fontsize=8)
    a.set_title(title or 'Cluster share per animal, then averaged across animals', fontweight='bold')
    a.grid(alpha=.2, axis='y'); plt.tight_layout(); return fig


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
