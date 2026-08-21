"""
Whisker sweep AMPLITUDE in two contexts Maryam asked for (2026-08-18):

  A. FIRST HALF vs SECOND HALF of the session's trials -- does whisking effort decay as the animal
     gets tired / sated, or build as he warms up? Split by trial ORDER (not by time), so the two
     halves hold equal numbers of trials.
  B. REWARD VISIBLE vs NOT -- is he whisking more when a positive reward is actually ON HIS SCREEN?
     Visibility is the viewport test from `common/viewport.py` (session.json `view_scale`), applied
     per frame to the trial's own reward icons. This is a WITHIN-trial contrast: every trial
     contributes both a visible and a not-visible mean where it has enough clean frames, so a trial's
     overall arousal cannot drive the difference -- it is paired.

Both use the SAME whisker measure as H1/H2: `whisker.npz sweep`, with lick+groom frames EXCLUDED
(licking ~7.7 Hz sits inside the 3-13 Hz whisker band, and rewarded trials are the lick-heaviest, so
including them would import a licking artefact into a whisking claim).

⚠️ Both are DESCRIPTIVE contrasts on ONE session: (A) is confounded with anything else that drifts
over a session (satiety, engagement, the reward streak), and (B) with distance-to-reward and with
the fact that a visible reward tends to mean he is approaching it.

Outputs: debug/whisker_context.png + printed stats.
Run:  python3 analyze_whisker_context.py <session_dir>
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_COMMON = Path(__file__).resolve().parent.parent / 'common'
sys.path.insert(0, str(_COMMON))
import fps as fpsmod       # noqa: E402
import viewport as vp      # noqa: E402

MIN_CLEAN = 15             # a trial needs this many clean frames in a condition to contribute
HALF_COL = ('#2e86c1', '#8e44ad')
VIS_COL = ('#27ae60', '#7f8c8d')


def run(session_dir, write=True):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    log = json.load(open(d / 'log.json'))
    df = pd.read_pickle(d / 'df_trials_clean.pkl')
    sweep = np.load(d / 'opticflow' / 'whisker.npz', allow_pickle=True)['sweep'].astype(float)
    ms = np.load(d / 'opticflow' / 'mouth_state.npz', allow_pickle=True)['mouth_state']
    clean = ~((ms == 1) | (ms == 2))            # exclude lick + groom
    N = len(sweep)
    fm = fpsmod.frame_to_ms(log)
    C = np.array(log['coords_t[ms]/x/y'], float)
    view_w, view_h = vp.viewport(sess)
    a = df[df.analyze].sort_values('trial').reset_index(drop=True)

    print(f"=== {sess['mouse_id']} whisker sweep by CONTEXT ===")
    print(f"  viewport {view_w:.0f} x {view_h:.0f} world units (scale {vp.view_scale(sess)})")

    # ---- per-trial values ----------------------------------------------------------
    rows = []
    for i, r in a.iterrows():
        s, e = int(r['start_frame']), int(min(r['end_frame'], N))
        if e - s < MIN_CLEAN:
            continue
        cl = clean[s:e]
        sw = sweep[s:e]
        t_ms = fm[s:e]
        ax = np.interp(t_ms, C[:, 0], C[:, 1]); ay = np.interp(t_ms, C[:, 0], C[:, 2])
        # is ANY positive-reward icon of this trial inside the viewport on this frame?
        vis = np.zeros(e - s, bool)
        for ic in (r['icons'] or []):
            if ic.get('effect') == 'single_reward':
                vis |= (np.abs(ic['x'] - ax) <= view_w / 2) & (np.abs(ic['y'] - ay) <= view_h / 2)
        m_all = cl
        m_vis, m_not = cl & vis, cl & ~vis
        rows.append(dict(
            trial=int(r['trial']), outcome=r['outcome'], order=i,
            sweep_all=np.nanmean(sw[m_all]) if m_all.sum() >= MIN_CLEAN else np.nan,
            sweep_vis=np.nanmean(sw[m_vis]) if m_vis.sum() >= MIN_CLEAN else np.nan,
            sweep_not=np.nanmean(sw[m_not]) if m_not.sum() >= MIN_CLEAN else np.nan,
            frac_vis=float(vis.mean()), n_vis=int(m_vis.sum()), n_not=int(m_not.sum())))
    T = pd.DataFrame(rows)
    half = T.order < len(T) / 2
    T['half'] = np.where(half, 'first', 'second')

    # ---- A. first vs second half ---------------------------------------------------
    f1 = T[T.half == 'first'].sweep_all.dropna()
    f2 = T[T.half == 'second'].sweep_all.dropna()
    pA = mannwhitneyu(f1, f2, alternative='two-sided').pvalue
    print(f"\n  A. FIRST vs SECOND half of trials (by trial order):")
    print(f"     first  n={len(f1):2d}  sweep {f1.mean():.3f} (sd {f1.std():.3f})")
    print(f"     second n={len(f2):2d}  sweep {f2.mean():.3f} (sd {f2.std():.3f})")
    print(f"     Mann-Whitney p={pA:.3f}  -> {'DIFFERS' if pA < 0.05 else 'no difference'}")

    # ---- B. reward visible vs not (PAIRED within trial) -----------------------------
    P = T.dropna(subset=['sweep_vis', 'sweep_not'])
    pB = wilcoxon(P.sweep_vis, P.sweep_not).pvalue if len(P) >= 6 else np.nan
    print(f"\n  B. positive reward VISIBLE vs NOT (paired within trial, n={len(P)} trials with both):")
    print(f"     visible     {P.sweep_vis.mean():.3f}")
    print(f"     not visible {P.sweep_not.mean():.3f}")
    print(f"     difference  {(P.sweep_vis - P.sweep_not).mean():+.3f}   Wilcoxon p={pB:.3f}"
          f"  -> {'DIFFERS' if pB < 0.05 else 'no difference'}")
    print(f"     (a reward icon is on screen {T.frac_vis.mean()*100:.0f}% of frames on average)")

    # ---- figure ---------------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.8))

    a0 = ax[0]
    for k, (lab, v) in enumerate([('first half', f1), ('second half', f2)]):
        a0.bar(k, v.mean(), yerr=v.std() / np.sqrt(len(v)), color=HALF_COL[k], alpha=0.85, capsize=5)
        a0.scatter(np.full(len(v), k) + np.random.uniform(-.12, .12, len(v)), v,
                   s=16, color='k', alpha=0.45, zorder=3)
    a0.set_xticks([0, 1]); a0.set_xticklabels([f'first half\n(n={len(f1)})', f'second half\n(n={len(f2)})'])
    a0.set_ylabel('whisker sweep (lick/groom excluded)')
    a0.set_title(f'A. sweep by SESSION HALF\np={pA:.3f}' + ('  *' if pA < 0.05 else '  (n.s.)'),
                 fontsize=10.5, fontweight='bold')
    a0.grid(alpha=0.2, axis='y')

    a1 = ax[1]
    a1.scatter(T.order, T.sweep_all, c=np.where(T.half == 'first', HALF_COL[0], HALF_COL[1]), s=22)
    ok = T.dropna(subset=['sweep_all'])
    z = np.polyfit(ok.order, ok.sweep_all, 1)
    a1.plot(ok.order, np.polyval(z, ok.order), 'k--', lw=1.2,
            label=f'slope {z[0]:+.4f}/trial')
    a1.axvline(len(T) / 2, color='0.5', ls=':', lw=1)
    a1.set_xlabel('trial order'); a1.set_ylabel('whisker sweep')
    a1.set_title('the same, over trial order\n(is it a drift or a step?)', fontsize=10.5, fontweight='bold')
    a1.legend(fontsize=8); a1.grid(alpha=0.2)

    a2 = ax[2]
    for k, (lab, col) in enumerate([('reward\nVISIBLE', VIS_COL[0]), ('reward\nNOT visible', VIS_COL[1])]):
        v = P.sweep_vis if k == 0 else P.sweep_not
        a2.bar(k, v.mean(), yerr=v.std() / np.sqrt(len(v)), color=col, alpha=0.85, capsize=5)
    for _, r in P.iterrows():                       # paired lines: each trial is one line
        a2.plot([0, 1], [r.sweep_vis, r.sweep_not], color='k', alpha=0.22, lw=0.8, zorder=3)
    a2.set_xticks([0, 1]); a2.set_xticklabels(['reward\nVISIBLE', 'reward\nNOT visible'])
    a2.set_ylabel('whisker sweep')
    a2.set_title(f'B. is a positive reward ON SCREEN?\nPAIRED within trial (n={len(P)}), '
                 f'p={pB:.3f}' + ('  *' if pB < 0.05 else '  (n.s.)'), fontsize=10.5, fontweight='bold')
    a2.grid(alpha=0.2, axis='y')

    a3 = ax[3]
    dv = (P.sweep_vis - P.sweep_not)
    a3.hist(dv, bins=14, color='#27ae60', alpha=0.8)
    a3.axvline(0, color='k', lw=1.2)
    a3.axvline(dv.mean(), color='#c0392b', lw=2, label=f'mean {dv.mean():+.3f}')
    a3.set_xlabel('sweep(visible) - sweep(not visible), per trial')
    a3.set_ylabel('trials')
    a3.set_title('B. the paired difference\n(> 0 = whisks more with a reward on screen)',
                 fontsize=10.5, fontweight='bold')
    a3.legend(fontsize=8); a3.grid(alpha=0.2, axis='y')

    fig.suptitle(f"{sess['mouse_id']} whisker sweep by CONTEXT   [whisker = optic-flow AMPLITUDE, not "
                 f"set-point; lick+groom frames excluded; viewport {view_w:.0f}x{view_h:.0f} wu from "
                 f"session.json view_scale; ONE session - DESCRIPTIVE]", fontsize=10.5, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = d / 'debug' / 'whisker_context.png'
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n  wrote {out}")
    return T


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
