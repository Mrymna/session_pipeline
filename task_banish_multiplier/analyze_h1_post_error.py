"""
H1 - does prior-trial difficulty (error / conflict) change the NEXT trial's whisker amplitude and
motor behaviour (post-error / post-conflict adjustment)?

Two structural facts of THIS task shape the test (both verified, see the printout):
  1. **post-ERROR is CONFOUNDED**: a banish is deterministically followed by the ESCAPE trial
     (banish -> in_banishment), so ALL 16 post-error trials ARE escape trials -- the lowest-engagement
     state. So a post-error vs post-correct difference cannot be separated from the escape state.
     -> reported, but FLAGGED, not the headline.
  2. **post-CONFLICT is the clean, well-powered test**: prev trial = Corner-dwelling (n=47) vs
     prev = Direct (n=29). Both groups have MIXED current outcomes/clusters, so it is not
     deterministically confounded. It tests "did a hard/low-efficiency previous trial change the next
     trial's behaviour" -- the defensible version of H1. We still CONTROL for current trial type
     (Direct vs Corner) because current type strongly sets whisker amplitude.

Signal limits (stated on the figure): whisker = optic-FLOW amplitude (no set-point); n is modest;
ONE session. Whisker amplitude for the tests EXCLUDES lick+groom frames (lick ~7.7 Hz leaks into the
whisker band).

Metrics per trial: whisker amplitude (mean sweep, lick/groom-excluded), FIRST-WHISK amplitude (mean
sweep in [first joystick re-steer, +1 s]), joy_fine, mean_speed, path_efficiency.

Outputs: debug/h1_post_error.png  + printed stats.
Run:  python3 analyze_h1_post_error.py /home/maryam/repo/flow_test/JPAS_0168
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_COMMON = Path(__file__).resolve().parent.parent / 'common'
sys.path.insert(0, str(_COMMON))
import fps as fpsmod       # noqa: E402
import joymove             # noqa: E402

WIN_FIRSTWHISK_S = 1.0     # window after the first re-steer for the "first-whisk" amplitude


def _mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, np.nan
    u, p = mannwhitneyu(a, b, alternative='two-sided')
    return float(np.median(a)), float(np.median(b)), float(p)


def _per_trial(d):
    sess = json.load(open(d / 'session.json'))
    log = json.load(open(d / 'log.json'))
    fps = sess['fps']
    df = pd.read_pickle(d / 'df_trials_clean.pkl')
    sweep = np.load(d / 'opticflow' / 'whisker.npz', allow_pickle=True)['sweep']
    ms = np.load(d / 'opticflow' / 'mouth_state.npz', allow_pickle=True)['mouth_state']
    excl = (ms == 1) | (ms == 2)                    # lick or groom frames -> exclude from whisker
    N = len(sweep)
    fm = fpsmod.frame_to_ms(log)
    jx, jy = joymove.stick_on_frames(log, fm)
    events, _ = joymove.movement_events(jx, jy, fps)

    win = int(round(WIN_FIRSTWHISK_S * fps))
    cyc = np.load(d / 'opticflow' / 'whisker.npz', allow_pickle=True)['cycle_whisker_right']
    rows = []
    for i, r in df.iterrows():
        s, e = int(r['start_frame']), int(min(r['end_frame'], N - 1))
        seg = np.arange(s, e)
        m = seg[~excl[seg]]                                  # clean (non-lick/groom) frames
        # whisker amplitude ONLY if enough clean whisk cycles remain (rewarded trials are lick-heavy
        # -- excluding licking can leave too few cycles to estimate amplitude; see MIN_CLEAN_FR).
        whisk_mean = float(np.nanmean(sweep[m])) if m.size >= MIN_CLEAN_FR else np.nan
        n_clean = int(m.size)
        n_cyc = float(m.size / fps * np.nanmedian(cyc[m])) if m.size else 0.0
        # first-whisk window kept ONLY as a diagnostic of the confound (n clean frames), NOT a test
        ev_in = events[(events >= s) & (events < e)]
        if ev_in.size:
            o = ev_in[0]; w = np.arange(o, min(o + win, N)); w = w[~excl[w]]
            n_clean_fw = int(w.size)
        else:
            n_clean_fw = -1
        rows.append(dict(idx=i, whisk_mean=whisk_mean, n_clean=n_clean, n_cyc=n_cyc, n_clean_fw=n_clean_fw,
                         joy_fine=r.get('joy_fine', np.nan), mean_speed=r.get('mean_speed', np.nan),
                         efficiency=r.get('path_efficiency', np.nan),
                         frac_lick=float(((ms[seg] == 1)).mean()) if seg.size else np.nan))
    P = pd.DataFrame(rows).set_index('idx')
    for c in ['prev_is_error', 'prev_is_conflict', 'prev_outcome', 'analyze', 'cluster_name', 'outcome']:
        P[c] = df[c]
    return P, sess


# NOTE the per-window "first whisk after movement" metric is DROPPED: on rewarded trials the first
# joystick move happens during consummatory LICKING, so after excluding lick frames ~0 clean whisk
# cycles remain in that window (post-correct/post-direct median 0/30) -> not estimable. The TRIAL-MEAN
# (57-105 clean cycles/trial) is the robust whisker measure and is what the tests use.
MIN_CLEAN_FR = 21     # >= ~0.7 s (~5 whisk cycles at 7 Hz) of clean frames to estimate trial whisker amp
METRICS = [('whisk_mean', 'whisker amp\n(mean sweep)'),
           ('joy_fine', 'joy_fine'), ('mean_speed', 'mean speed'), ('efficiency', 'path eff')]


def _bars(ax, A, B, la, lb, title, ca='#c0392b', cb='#7f8c8d'):
    labels, ma, mb, ps = [], [], [], []
    for col, lab in METRICS:
        x, y, p = _mw(A[col], B[col])
        labels.append(lab); ma.append(x); mb.append(y); ps.append(p)
    xs = np.arange(len(METRICS)); w = 0.38
    # normalise each metric by the pooled median so they share an axis
    for j, (col, _) in enumerate(METRICS):
        norm = np.nanmedian(np.concatenate([A[col].values, B[col].values]).astype(float))
        norm = norm if norm and np.isfinite(norm) else 1.0
        ax.bar(xs[j] - w / 2, ma[j] / norm, w, color=ca, label=la if j == 0 else None)
        ax.bar(xs[j] + w / 2, mb[j] / norm, w, color=cb, label=lb if j == 0 else None)
        star = '*' if (np.isfinite(ps[j]) and ps[j] < 0.05) else ''
        ax.text(xs[j], max(ma[j] / norm, mb[j] / norm) + 0.03,
                f'p={ps[j]:.2f}{star}' if np.isfinite(ps[j]) else 'n/a',
                ha='center', fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(1, color='k', lw=0.5, ls=':')
    ax.set_ylabel('median / pooled median'); ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')


def run(session_dir, write=True):
    d = Path(session_dir).resolve()
    P, sess = _per_trial(d)
    a = P[P.analyze]

    post_err = a[a.prev_is_error]                                   # all escape (confounded)
    post_cor = a[a.prev_outcome == 'single_reward']
    post_conf = a[a.prev_is_conflict]                              # prev Corner-dwelling (clean)
    post_dir = a[~a.prev_is_conflict & a.prev_outcome.notna()]

    print(f"=== {sess['mouse_id']} H1: post-error / post-conflict adjustment ===")
    print("  clean whisk cycles behind the TRIAL-MEAN (lick/groom excluded); "
          "first-whisk-window clean frames [/30] (dropped as a test -- lick eats it):")
    for nm, g in [('post-ERROR', post_err), ('post-CORRECT', post_cor),
                  ('post-CONFLICT', post_conf), ('post-DIRECT', post_dir)]:
        fw = g['n_clean_fw'][g['n_clean_fw'] >= 0]
        print(f"    {nm:14s} n={len(g):3d}  trial clean cycles med {g['n_cyc'].median():.0f}  "
              f"(qualifying whisk_mean {int(g['whisk_mean'].notna().sum())}/{len(g)})  "
              f"lick frac med {g['frac_lick'].median():.2f}  first-whisk clean fr med {fw.median():.0f}/30")
    print(f"  post-ERROR n={len(post_err)} (ALL escape -> CONFOUNDED) vs post-correct n={len(post_cor)}")
    for col, lab in METRICS:
        x, y, p = _mw(post_err[col], post_cor[col])
        print(f"    {lab.splitlines()[0]:16s} post-err {x:.3f} vs post-cor {y:.3f}  p={p:.3f}")
    print(f"  post-CONFLICT n={len(post_conf)} vs post-direct n={len(post_dir)}  [PRIMARY]")
    for col, lab in METRICS:
        x, y, p = _mw(post_conf[col], post_dir[col])
        print(f"    {lab.splitlines()[0]:16s} post-conf {x:.3f} vs post-dir {y:.3f}  p={p:.3f}")
    # control: post-conflict effect WITHIN each current cluster
    print("  post-CONFLICT controlled for CURRENT cluster:")
    for cur in ['Direct', 'Corner-dwelling']:
        cc = a[a.cluster_name == cur]
        x, y, p = _mw(cc[cc.prev_is_conflict].whisk_mean, cc[~cc.prev_is_conflict].whisk_mean)
        print(f"    current={cur:15s} whisk post-conf {x:.3f} vs post-dir {y:.3f}  p={p:.3f}  "
              f"(n={cc.prev_is_conflict.sum()}/{(~cc.prev_is_conflict).sum()})")

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    _bars(ax[0, 0], post_err, post_cor, 'post-error (escape)', 'post-correct',
          'post-ERROR vs post-correct  [CONFOUNDED: all post-error = escape trials]',
          ca='#c0392b', cb='#7f8c8d')
    _bars(ax[0, 1], post_conf, post_dir, 'post-conflict', 'post-direct',
          'post-CONFLICT vs post-direct  [PRIMARY, prev Corner vs Direct]',
          ca='#e67e22', cb='#2980b9')

    # per-trial dots for whisker amplitude, both contrasts
    axd = ax[1, 0]
    for j, (grp, lab, c) in enumerate([(post_err, 'post-err\n(escape)', '#c0392b'),
                                       (post_cor, 'post-cor', '#7f8c8d'),
                                       (post_conf, 'post-conf', '#e67e22'),
                                       (post_dir, 'post-dir', '#2980b9')]):
        v = grp['whisk_mean'].dropna().values
        axd.scatter(np.full(len(v), j) + np.random.uniform(-.08, .08, len(v)), v, s=22, color=c, alpha=0.7)
        axd.plot([j - .2, j + .2], [np.nanmedian(v)] * 2, color='k', lw=2)
    axd.set_xticks(range(4)); axd.set_xticklabels(['post-err\n(escape)', 'post-cor', 'post-conf', 'post-dir'], fontsize=8)
    axd.set_ylabel('whisker amp (mean sweep)'); axd.set_title('whisker amplitude per trial', fontsize=9, fontweight='bold')
    axd.grid(alpha=0.2, axis='y')

    # scatter: current whisker amp vs current path efficiency (does difficulty track whisking)
    axs = ax[1, 1]
    axs.scatter(a['efficiency'], a['whisk_mean'], s=22, c=np.where(a.prev_is_conflict, '#e67e22', '#2980b9'), alpha=0.75)
    good = a[['efficiency', 'whisk_mean']].dropna()
    if len(good) > 3:
        r = np.corrcoef(good.efficiency, good.whisk_mean)[0, 1]
        axs.set_title(f'whisker amp vs path efficiency  r={r:+.2f}  (orange=post-conflict)', fontsize=9, fontweight='bold')
    axs.set_xlabel('path efficiency'); axs.set_ylabel('whisker amp (mean sweep)'); axs.grid(alpha=0.2)

    # TWO lines: the single-line version overflowed the figure width and was clipped at both ends.
    fig.suptitle(f"{sess['mouse_id']} H1 - post-error / post-conflict whisker amplitude & motor adjustment\n"
                 f"[whisker = optic-flow amplitude, NO set-point; TRIAL-MEAN over 57-105 clean cycles "
                 f"(lick/groom excluded); first-whisk metric DROPPED (rewarded trials lick-dominated);\n"
                 f"ONE session; post-error CONFOUNDED with the escape state]",
                 fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = d / 'debug' / 'h1_post_error.png'
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  wrote {out}")
    return P


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
