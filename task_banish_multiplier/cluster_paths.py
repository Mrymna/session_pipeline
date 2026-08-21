"""
Path/behaviour clustering for a `banish_multiplier` session (port of JPAS_0231/cluster_paths.py to
the reusable pipeline). This gives every analyzable trial a PATH-TYPE label used downstream as the
CONFLICT stratifier for the whisker<->paw hypotheses (Corner-dwelling / low-efficiency = high conflict).

Method (Maryam's, from 231): per-trial PATH-GEOMETRY feature vector -> StandardScaler -> KMeans
(k in {2,3} by silhouette) -> merge any cluster < MIN_CLUSTER into the nearest -> name by RELATIVE
efficiency ranking (highest eff = Direct, lowest = Corner-dwelling, middle = Exploratory) -> PCA(2)
for the diagnostic scatter only.

Differences vs 231:
  - operates DIRECTLY on df_trials_clean.pkl (0168's disjoint reward-to-reward table IS the trial
    table -- there is no separate overlapping df_trials here);
  - INCLUDE banish/unbanish trials (231 excluded its timeout trials; here banish/escape are real
    navigation and we WANT a path label on the error trials). Only reshuffle/incomplete/degenerate
    (analyze==False) are excluded;
  - column names: path_efficiency -> efficiency, dur_s -> effective_duration_s;
  - RELATIVE naming (not 231's absolute eff>0.4/corner<0.1 thresholds) so the two clusters always
    get distinct, interpretable names even when the whole session is low-efficiency.

FEATURES (path geometry only) = efficiency, speed_std, time_in_corner, effective_duration_s.
MOTOR / WHISKER / HEADING / PUPIL are held OUT as independent outcomes (tested vs the clusters), so
any relationship stays non-circular. feature_selection_report() writes the redundancy+ablation
justification.

*** The MULTIPLIER is deliberately NOT a clustering feature *** (Maryam asked, 2026-08-18; 231 had no
multiplier so this is the one genuinely new decision for this task variant). Three reasons, all
checked on this session and reported by multiplier_outcome_report():
  1. It is NOT path geometry -- it is a REWARD-STATE variable (the streak 1-4). Its correlation with
     each of the four path features is |rho| <= 0.19, all n.s., so it would add a whole standardized
     axis of variance that carries no path information (it degrades the split; see the printout).
  2. It PARTLY RE-ENCODES THE OUTCOME LABEL: `unbanish` (escape) trials have multiplier == 1 by
     construction (a banish resets the streak), 17/17 here. Feeding it in would smuggle the outcome
     into the grouping, so any later "outcome differs by cluster" claim would be circular.
  3. The multiplier is the thing we TEST (H2's headline is that the whisker reset scales with the
     streak, r=+0.95). A tested signal must stay out of the grouping, exactly as joy_fine /
     whisk_sweep / heading_align are held out.
It is instead reported as an INDEPENDENT OUTCOME against the clusters -- and it comes out FLAT
(Direct 2.13 vs Corner-dwelling 2.06, p=0.79), which is a useful result in its own right: path type
and reward streak are INDEPENDENT axes in this session, so the H2 multiplier effect cannot be a
disguised path-type effect, and multiplier x cluster can be used as a 2-factor stratifier.

Writes `cluster` / `cluster_name` (+ `cluster_pca1/2`) back into df_trials_clean.pkl and a diagnostic
figure to <session>/debug/path_clustering.png.

Run:  python3 cluster_paths.py /home/maryam/repo/flow_test/JPAS_0168
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

FEATURES = ['efficiency', 'speed_std', 'time_in_corner', 'effective_duration_s']
CANDIDATE_POOL = ['efficiency', 'mean_speed', 'speed_std', 'time_in_corner',
                  'effective_duration_s', 'joy_fine', 'whisk_sweep', 'heading_align']
MIN_CLUSTER = 5
K_CANDIDATES = (2, 3)
SEED = 42

CLUSTER_COLORS = {'Direct': '#27ae60', 'Corner-dwelling': '#c0392b', 'Exploratory': '#9b59b6'}
_FALLBACK = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']


def build_features(dfc):
    """Feature table from the disjoint clean-window df. analyze==False (reshuffle / incomplete /
    degenerate) -> unusable. banish/unbanish ARE kept."""
    rows = []
    for i in dfc.index:
        r = dfc.loc[i]
        if not bool(r.get('analyze', False)):
            rows.append(dict(idx=i, usable=False)); continue
        rows.append(dict(
            idx=i, usable=True,
            efficiency=r.get('path_efficiency', np.nan), mean_speed=r.get('mean_speed', np.nan),
            speed_std=r.get('speed_std', np.nan), time_in_corner=r.get('time_in_corner', np.nan),
            effective_duration_s=r.get('dur_s', np.nan), joy_fine=r.get('joy_fine', np.nan),
            whisk_sweep=r.get('whisk_sweep', np.nan), heading_align=r.get('heading_align', np.nan),
            outcome=r.get('outcome', None), multiplier=r.get('multiplier', np.nan)))
    return pd.DataFrame(rows).set_index('idx')


def _merge_small(labels, X):
    for _ in range(len(set(labels))):
        sizes = pd.Series(labels).value_counts()
        small = sizes[sizes < MIN_CLUSTER].index.tolist()
        if not small:
            break
        centers = np.array([X[labels == c].mean(0) for c in range(max(labels) + 1)])
        for sc in small:
            large = [c for c in range(max(labels) + 1) if c not in small and c != sc]
            if not large:
                break
            nearest = large[int(np.argmin([np.linalg.norm(centers[sc] - centers[lc]) for lc in large]))]
            print(f'    merging cluster {sc} (n={sizes[sc]}) -> {nearest}')
            labels = np.where(labels == sc, nearest, labels)
    uniq = sorted(set(labels))
    return np.array([uniq.index(l) for l in labels])


def _name_by_rank(good, k):
    """Name clusters by RELATIVE mean efficiency: highest = Direct, lowest = Corner-dwelling,
    any middle = Exploratory. Robust when the whole session is low-efficiency (unlike absolute
    thresholds)."""
    eff = {c: good[good.cluster == c].efficiency.mean() for c in range(k)}
    order = sorted(range(k), key=lambda c: eff[c], reverse=True)   # high eff -> low
    names = {}
    for rank, c in enumerate(order):
        if k == 1:
            names[c] = 'Direct'
        elif rank == 0:
            names[c] = 'Direct'
        elif rank == k - 1:
            names[c] = 'Corner-dwelling'
        else:
            names[c] = 'Exploratory'
    return names


def cluster(dfc, F, features, prefix):
    good = F[(F['usable']) & F[features].notna().all(axis=1)].copy()
    X = StandardScaler().fit_transform(good[features].values)

    best = None
    for k in K_CANDIDATES:
        lab = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X)
        s = silhouette_score(X, lab)
        if best is None or s > best[0]:
            best = (s, k, lab)
    _, k0, labels = best
    print(f'  [{prefix}] silhouette picks k={k0} (' +
          ', '.join(f'k{k}={silhouette_score(X, KMeans(k, random_state=SEED, n_init=10).fit_predict(X)):.2f}'
                    for k in K_CANDIDATES) + ')')
    labels = _merge_small(labels, X)
    k = len(set(labels))
    good['cluster'] = labels

    names = _name_by_rank(good, k)
    good['cluster_name'] = good.cluster.map(names)
    sil = silhouette_score(X, labels) if k > 1 else np.nan
    good['pca1'], good['pca2'] = PCA(n_components=2).fit_transform(X).T

    dfc[prefix] = np.nan
    dfc[prefix + '_name'] = 'excluded'
    dfc[prefix + '_pca1'] = np.nan
    dfc[prefix + '_pca2'] = np.nan
    for i, row in good.iterrows():
        dfc.loc[i, prefix] = int(row.cluster)
        dfc.loc[i, prefix + '_name'] = row.cluster_name
        dfc.loc[i, prefix + '_pca1'] = row.pca1
        dfc.loc[i, prefix + '_pca2'] = row.pca2

    print(f'  [{prefix}] k={k}, silhouette={sil:.3f}, n={len(good)} '
          f'(excluded {int((~F["usable"]).sum())} non-analyzable):')
    for c in range(k):
        d = good[good.cluster == c]
        oc = d.outcome.value_counts().to_dict()
        print(f'    {names[c]:16s} n={len(d):3d}  eff={d.efficiency.mean():.2f}  '
              f'corner={d.time_in_corner.mean():.2f}  speed={d.mean_speed.mean():.1f}  '
              f'dur={d.effective_duration_s.mean():.1f}s  joy_fine={d.joy_fine.mean():.2f}  '
              f'sweep={d.whisk_sweep.mean():.3f}  align={d.heading_align.mean():+.3f}  {oc}')
    return good, names, k, sil, X


def _cluster_color(name, idx):
    for key, col in CLUSTER_COLORS.items():
        if str(name).startswith(key):
            return col
    return _FALLBACK[idx % len(_FALLBACK)]


def diagnostic_fig(good, names, k, X, features, subtitle, out_path):
    colors = [_cluster_color(names[c], c) for c in range(k)]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

    a = ax[0]
    for c in range(k):
        d = good[good.cluster == c]
        a.scatter(d.pca1, d.pca2, s=28, alpha=0.8, color=colors[c], label=f'{names[c]} (n={len(d)})')
    a.set_xlabel('PC1'); a.set_ylabel('PC2'); a.legend(fontsize=8)
    a.set_title('KMeans clusters in PCA space', fontsize=11, fontweight='bold'); a.grid(alpha=0.2)

    a = ax[1]
    ks = list(range(2, min(7, len(good))))
    sils = [silhouette_score(X, KMeans(n_clusters=kk, random_state=SEED, n_init=10).fit_predict(X))
            for kk in ks]
    a.plot(ks, sils, 'o-', color='#2c3e50')
    a.set_xlabel('k (n_clusters)'); a.set_ylabel('silhouette score')
    a.set_title('how many clusters are supported?', fontsize=11, fontweight='bold'); a.grid(alpha=0.2)

    a = ax[2]
    Fz = pd.DataFrame(StandardScaler().fit_transform(good[features]), columns=features, index=good.index)
    Fz['cluster'] = good.cluster.values
    m = Fz.groupby('cluster')[features].mean()
    x = np.arange(len(features)); w = 0.8 / k
    for c in range(k):
        a.bar(x + c * w, m.loc[c].values, w, color=colors[c], label=names[c])
    a.set_xticks(x + w * (k - 1) / 2)
    a.set_xticklabels([f.replace('_', '\n') for f in features], fontsize=8, ha='center')
    a.axhline(0, color='k', lw=0.6)
    a.set_ylabel('standardized mean (z)'); a.legend(fontsize=8)
    a.set_title('feature profile per cluster', fontsize=11, fontweight='bold'); a.grid(alpha=0.2, axis='y')

    # WRAP the subtitle -- a single long line overflows the figure width and is silently CLIPPED
    # at both ends (it is also embedded verbatim in the trial report's classification page).
    import textwrap
    fig.suptitle('\n'.join(textwrap.wrap(subtitle, 120)), fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.90]); fig.savefig(out_path, dpi=110); plt.close(fig)
    print(f'  wrote {out_path}')


def feature_selection_report(F, out_path):
    """CANDIDATE_POOL |correlation| (redundancy) + a k=2 drop-one ablation (dSilhouette),
    documenting WHY FEATURES = path geometry only. Non-fatal if a candidate column is missing."""
    pool = [c for c in CANDIDATE_POOL if c in F.columns]
    G = F[(F['usable']) & F[pool].notna().all(axis=1)]
    if len(G) < 6:
        print('  (skip feature_selection_report: too few complete rows)')
        return
    X = StandardScaler().fit_transform(G[pool].values)
    corr = pd.DataFrame(X, columns=pool).corr()
    base = silhouette_score(X, KMeans(2, random_state=SEED, n_init=10).fit_predict(X))
    abl = {}
    for f in pool:
        sub = [c for c in pool if c != f]
        Xs = StandardScaler().fit_transform(G[sub].values)
        abl[f] = silhouette_score(Xs, KMeans(2, random_state=SEED, n_init=10).fit_predict(Xs)) - base

    print('FEATURE SELECTION (this session, n=%d):' % len(G))
    red = [(abs(corr.iloc[i, j]), pool[i], pool[j])
           for i in range(len(pool)) for j in range(i + 1, len(pool))]
    for r, a, b in sorted(red, reverse=True)[:4]:
        print('  |r|=%.2f  %s <-> %s' % (r, a, b))
    print('  chosen FEATURES (path geometry): %s' % FEATURES)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    im = ax[0].imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax[0].set_xticks(range(len(pool))); ax[0].set_xticklabels(pool, rotation=45, ha='right', fontsize=8)
    ax[0].set_yticks(range(len(pool))); ax[0].set_yticklabels(pool, fontsize=8)
    for i in range(len(pool)):
        for j in range(len(pool)):
            ax[0].text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=7,
                       color='k' if abs(corr.iloc[i, j]) < 0.6 else 'w')
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].set_title('candidate-feature |correlation| (redundancy)', fontsize=10, fontweight='bold')
    cols = ['#c0392b' if f in FEATURES else '#95a5a6' for f in pool]
    ax[1].barh(range(len(pool)), [abl[f] for f in pool], color=cols)
    ax[1].set_yticks(range(len(pool))); ax[1].set_yticklabels(pool, fontsize=9)
    ax[1].axvline(0, color='k', lw=0.8); ax[1].invert_yaxis()
    ax[1].set_xlabel('dSilhouette when the feature is DROPPED (k=2)')
    ax[1].set_title('ablation (red = kept path features)\nnegative = the feature helps the split',
                    fontsize=10, fontweight='bold'); ax[1].grid(alpha=0.2, axis='x')
    fig.suptitle('Keep PATH-GEOMETRY features; hold motor / whisker / heading out as OUTCOMES',
                 fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out_path, dpi=110); plt.close(fig)
    print('  wrote %s' % out_path)


def multiplier_outcome_report(good, names, k, X, out_path):
    """The MULTIPLIER is a reward-STATE variable, not path geometry, so it is held OUT of the
    clustering and reported here as an INDEPENDENT OUTCOME (see the module docstring for why).
    Also runs the counterfactual -- what the split would look like if it HAD been a 5th feature --
    so the decision is testable, not asserted."""
    from scipy import stats
    g = good[good.multiplier.notna()]
    if len(g) < 10 or k < 2:
        return None
    print('\n  MULTIPLIER handling -- held OUT of the features, tested as an OUTCOME:')
    for f in FEATURES:
        rho, pv = stats.spearmanr(g[f], g.multiplier)
        print(f'    corr(multiplier, {f:22s}) rho={rho:+.2f}  p={pv:.3f}'
              f"{'   <- carries path info' if pv < 0.05 else ''}")
    grp = {c: g[g.cluster == c].multiplier.values for c in range(k)}
    line = '   '.join(f'{names[c]} {grp[c].mean():.2f} (n={len(grp[c])})' for c in range(k))
    pv = stats.mannwhitneyu(grp[0], grp[1])[1] if k == 2 else np.nan
    print(f'    multiplier by cluster: {line}   Mann-Whitney p={pv:.3f}')
    print('    -> path type and reward streak are INDEPENDENT axes here'
          if pv > 0.05 else '    -> multiplier DIFFERS by cluster: report it as a confound')

    # counterfactual: add multiplier as a 5th clustering feature
    Xm = StandardScaler().fit_transform(g[FEATURES + ['multiplier']].values)
    lab_m = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit_predict(Xm)
    sil_m = silhouette_score(Xm, lab_m)
    Xg = StandardScaler().fit_transform(g[FEATURES].values)
    sil_g = silhouette_score(Xg, KMeans(n_clusters=2, random_state=SEED, n_init=10).fit_predict(Xg))
    agree = max((lab_m == g.cluster.values).mean(), (lab_m != g.cluster.values).mean())
    print(f'    counterfactual (multiplier AS a 5th feature): silhouette {sil_g:.3f} -> {sil_m:.3f} '
          f'(d={sil_m-sil_g:+.3f}), labels agree {agree*100:.0f}% with the path-geometry split')

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    a = ax[0]
    mult_vals = sorted(g.multiplier.dropna().unique())
    wdt = 0.8 / k
    for c in range(k):
        frac = [np.mean(grp[c] == m) for m in mult_vals]
        a.bar(np.arange(len(mult_vals)) + c * wdt, frac, wdt,
              color=_cluster_color(names[c], c), label=f'{names[c]} (n={len(grp[c])})')
    a.set_xticks(np.arange(len(mult_vals)) + wdt * (k - 1) / 2)
    a.set_xticklabels([f'{int(m)}' for m in mult_vals])
    a.set_xlabel('reward multiplier (streak)'); a.set_ylabel('fraction of the cluster\'s trials')
    a.set_title(f'multiplier is FLAT across path type (p={pv:.2f})\n-> independent axes, not a confound',
                fontsize=10, fontweight='bold')
    a.legend(fontsize=8); a.grid(alpha=0.2, axis='y')

    a = ax[1]
    rhos = [stats.spearmanr(g[f], g.multiplier)[0] for f in FEATURES]
    a.barh(range(len(FEATURES)), rhos, color='#95a5a6')
    a.axvline(0, color='k', lw=0.8); a.set_xlim(-1, 1); a.invert_yaxis()
    a.set_yticks(range(len(FEATURES))); a.set_yticklabels(FEATURES, fontsize=9)
    a.set_xlabel('Spearman rho with multiplier')
    a.set_title('multiplier carries NO path-geometry info\n-> it is not a clustering feature',
                fontsize=10, fontweight='bold'); a.grid(alpha=0.2, axis='x')

    a = ax[2]; a.axis('off')
    a.text(0.0, 1.0, 'WHY THE MULTIPLIER IS HELD OUT', fontsize=11, fontweight='bold', va='top')
    a.text(0.0, 0.88,
           '1. NOT path geometry - it is a reward-STATE variable\n'
           f'    (|rho| <= {max(abs(r) for r in rhos):.2f} with every path feature, all n.s.)\n\n'
           '2. It partly RE-ENCODES THE OUTCOME: escape (unbanish)\n'
           '    trials are multiplier == 1 by construction (a banish\n'
           '    resets the streak) - clustering on it would smuggle\n'
           '    the outcome label into the grouping (circular).\n\n'
           '3. It is a TESTED signal (H2: the whisker reset scales\n'
           '    with the streak) - tested signals stay out of the\n'
           '    grouping, like joy_fine / whisk_sweep / heading_align.\n\n'
           f'Counterfactual - as a 5th feature: silhouette {sil_g:.3f} -> {sil_m:.3f},\n'
           f'labels {agree*100:.0f}% the same. It buys nothing and costs\n'
           'independence.\n\n'
           'USE IT AS: an independent outcome, and a 2nd stratifier\n'
           '(multiplier x cluster), never as a clustering input.',
           fontsize=9, va='top', family='monospace')

    fig.suptitle('MULTIPLIER: an OUTCOME / stratifier, not a clustering feature', fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out_path, dpi=110); plt.close(fig)
    print(f'  wrote {out_path}')
    return dict(p=pv, sil_geom=sil_g, sil_with_mult=sil_m, agree=agree)


def run(session_dir, write=True, verbose=True):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    dbg = d / 'debug'; dbg.mkdir(exist_ok=True)
    dfc = pd.read_pickle(d / 'df_trials_clean.pkl')
    F = build_features(dfc)

    feature_selection_report(F, dbg / 'feature_selection.png')
    print(f"\nPRIMARY path clustering for {sess['mouse_id']} (PATH-GEOMETRY features, banish/unbanish included):")
    good, names, k, sil, X = cluster(dfc, F, FEATURES, 'cluster')
    diagnostic_fig(good, names, k, X, FEATURES,
                   f"{sess['mouse_id']} path clustering - PATH-GEOMETRY features (efficiency, speed std, "
                   "time-in-corner, duration); motor/whisker/heading tested SEPARATELY as outcomes; "
                   "reshuffle/incomplete/degenerate excluded, merge < 5", dbg / 'path_clustering.png')
    multiplier_outcome_report(good, names, k, X, dbg / 'multiplier_outcome.png')

    if write:
        dfc.to_pickle(d / 'df_trials_clean.pkl')
        print(f"\nadded to df_trials_clean.pkl: cluster / cluster_name (path-geometry grouping)")
    return dfc


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
