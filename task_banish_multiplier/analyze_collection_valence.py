"""
OUTCOME VALENCE at the collection instant - does whisker amplitude differ when the mouse hits a
POSITIVE (single_reward) vs a NEGATIVE (banish) icon?

This is the piece neither H1 nor H2 covers (2026-08-18):
  - H1 tests CARRY-OVER (did trial N-1 being an error change trial N)          -> not the hit;
  - H2 tests MOVEMENT-LOCKED resets (whisker follows each joystick re-steer)   -> not the hit;
  - THIS tests FEEDBACK-LOCKED valence: sweep around t=0 = the collection, by outcome.
It bears directly on H2's interpretation: H2 found banish (0.19) ~ mult-4 reward (0.19) and read that
as AROUSAL, not error. If valence per se mattered, it should show at the feedback moment.

*** HARD LIMIT - THE POST-COLLECTION WINDOW IS NOT ESTIMABLE FOR REWARD TRIALS. ***
Licking is ~7.7 Hz, inside the 3-13 Hz whisker band, so lick frames MUST be excluded. Measured
lick/groom fraction of the window, by outcome:
        window            single_reward   banish   unbanish
        -0.5 .. 0 s  (approach)   0.37      0.40      0.33     <- MATCHED -> comparable
         0 .. +0.5 s (feedback)   0.96      0.35      0.26     <- reward is ~all licking
        +0.5 .. +1.5 s           0.96      0.24      0.15     <- ditto
A reward IS consummatory licking, and a banish delivers no reward, so the asymmetry is structural and
cannot be gated away: after t=0 there are almost no clean whisker frames on reward trials. Reporting a
post-hit reward-vs-banish difference would be reporting the lick detector. So:
  - the STATISTICAL TEST uses the APPROACH window only (lick/groom matched, both ~0.4);
  - the post-hit half is still PLOTTED, but greyed with its per-lag usable-event count, so the
    reader sees where it dies instead of reading an artifact.
(Same trap as H1's dropped "first whisk after the first movement" metric.)

Outputs: debug/collection_valence.png + printed stats.
Run:  python3 analyze_collection_valence.py /home/maryam/repo/flow_test/JPAS_0168
"""
from pathlib import Path
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_COMMON = Path(__file__).resolve().parent.parent / 'common'
sys.path.insert(0, str(_COMMON))

WIN_S = 1.5                      # +- window around the collection instant
BASE_LO_S, BASE_HI_S = -1.5, -1.0   # pre-onset baseline
TEST_LO_S, TEST_HI_S = -0.5, 0.0    # the APPROACH window the test runs on (lick-matched)
MIN_CLEAN = 4                    # a trial needs >= this many clean frames in the test window
MIN_EVENTS_LAG = 5               # don't DRAW a lag supported by fewer events than this: post-reward
                                 # the clean count falls to ~2, and a 2-event mean looks like a huge
                                 # transient. Masking is what stops the figure showing the artifact
                                 # the middle panel is there to warn about.
N_BOOT = 4000
SEED = 0

OUT_COL = {'single_reward': '#27ae60', 'banish': '#c0392b', 'unbanish': '#2980b9'}
# DISPLAY labels: every reward in this task carries a multiplier (there is no plain "single"
# reward), so `single_reward` is shown as REWARD. The data column is unchanged.
OUT_LAB = {'single_reward': 'POSITIVE (reward)', 'banish': 'NEGATIVE (banish)',
           'unbanish': 'escape (unbanish)'}


def _eta(sig, events, w, excl=None):
    """Event-triggered matrix, lick/groom frames set to NaN (kept as rows so the per-lag usable
    count can be shown, rather than silently dropping events)."""
    N = len(sig)
    rows = []
    for e in events:
        if e - w < 0 or e + w >= N:
            continue
        seg = sig[e - w:e + w + 1].astype(float).copy()
        if excl is not None:
            seg[excl[e - w:e + w + 1]] = np.nan
        rows.append(seg)
    return np.array(rows)


def run(session_dir, write=True):
    rng = np.random.default_rng(SEED)
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    fps = sess['fps']
    df = pd.read_pickle(d / 'df_trials_clean.pkl')
    sweep = np.load(d / 'opticflow' / 'whisker.npz', allow_pickle=True)['sweep'].astype(float)
    ms = np.load(d / 'opticflow' / 'mouth_state.npz', allow_pickle=True)['mouth_state']
    excl = (ms == 1) | (ms == 2)
    N = len(sweep)

    w = int(round(WIN_S * fps))
    lags = np.arange(-w, w + 1) / fps
    base_m = (lags >= BASE_LO_S) & (lags <= BASE_HI_S)
    test_m = (lags >= TEST_LO_S) & (lags < TEST_HI_S)

    a = df[df.analyze]
    print(f"=== {sess['mouse_id']} OUTCOME VALENCE at the collection instant ===")
    print(f"  whisker sweep, +-{WIN_S}s around t=0 = collection; lick/groom frames excluded (NaN).")
    print(f"  TEST window = APPROACH [{TEST_LO_S}, {TEST_HI_S}) s (lick/groom matched across outcomes);"
          f" post-hit NOT tested (reward window is ~96% licking).\n")

    res = {}
    for oc in ['single_reward', 'banish', 'unbanish']:
        ev = np.array([int(r.end_frame) for r in a[a.outcome == oc].itertuples()])
        M_cl = _eta(sweep, ev, w, excl)          # lick/groom -> NaN
        M_raw = _eta(sweep, ev, w, None)         # transparency: nothing excluded
        if len(M_cl) == 0:
            continue
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN lags are EXPECTED post-reward
            base = np.nanmean(M_cl[:, base_m], axis=1, keepdims=True)
            Mb = M_cl - base
            mu = np.nanmean(Mb, axis=0)
            n_lag = np.sum(np.isfinite(Mb), axis=0)                 # usable events per lag
            # per-trial approach-window value (the test statistic), gated on enough clean frames
            n_clean = np.sum(np.isfinite(M_cl[:, test_m]), axis=1)
            per_trial = np.where(n_clean >= MIN_CLEAN, np.nanmean(Mb[:, test_m], axis=1), np.nan)
            raw_mu = np.nanmean(M_raw - np.nanmean(M_raw[:, base_m], axis=1, keepdims=True), axis=0)
        res[oc] = dict(mu=mu, n_lag=n_lag, per_trial=per_trial, raw_mu=raw_mu,
                       n=len(M_cl), n_ok=int(np.sum(np.isfinite(per_trial))),
                       lickfrac_post=float(np.mean(excl[[min(e + w, N - 1) for e in ev]])))
        v = per_trial[np.isfinite(per_trial)]
        print(f"  {OUT_LAB[oc]:26s} n={len(M_cl):2d} collections, {len(v):2d} with >= {MIN_CLEAN} "
              f"clean frames in the approach window;  sweep = {v.mean():+.3f} (sd {v.std():.3f})")

    # --- the test: POSITIVE vs NEGATIVE in the approach window -------------------------
    pos = res['single_reward']['per_trial']; pos = pos[np.isfinite(pos)]
    neg = res['banish']['per_trial']; neg = neg[np.isfinite(neg)]
    pv = mannwhitneyu(pos, neg, alternative='two-sided').pvalue
    diff = pos.mean() - neg.mean()
    bd = np.array([rng.choice(pos, len(pos)).mean() - rng.choice(neg, len(neg)).mean()
                   for _ in range(N_BOOT)])
    ci = np.percentile(bd, [2.5, 97.5])
    print(f"\n  APPROACH-window test  POSITIVE vs NEGATIVE:")
    print(f"    reward {pos.mean():+.3f} (n={len(pos)})  vs  banish {neg.mean():+.3f} (n={len(neg)})")
    print(f"    difference {diff:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  Mann-Whitney p={pv:.3f}"
          f"   -> {'DIFFERS' if pv < 0.05 else 'NO valence difference'}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))

    a0 = ax[0]
    for oc, r in res.items():
        mu = np.where(r['n_lag'] >= MIN_EVENTS_LAG, r['mu'], np.nan)   # hide under-supported lags
        a0.plot(lags, mu, color=OUT_COL[oc], lw=2, label=f"{OUT_LAB[oc]} (n={r['n']})")
    a0.axvspan(TEST_LO_S, TEST_HI_S, color='#2ecc71', alpha=0.10)
    a0.axvspan(0, WIN_S, color='0.5', alpha=0.18)
    a0.text(WIN_S * 0.52, a0.get_ylim()[0], 'POST-HIT: NOT ESTIMABLE for reward\n(window ~96% licking;'
            f'\nlags with < {MIN_EVENTS_LAG} clean events are not drawn)',
            ha='center', va='bottom', fontsize=8, color='#333', style='italic')
    a0.text((TEST_LO_S + TEST_HI_S) / 2, a0.get_ylim()[0], 'TEST\nwindow', ha='center', va='bottom',
            fontsize=8, color='#1a7', fontweight='bold')
    a0.axvline(0, color='k', ls='--', lw=1); a0.axhline(0, color='k', lw=0.5)
    a0.set_xlabel('time from COLLECTION (s)'); a0.set_ylabel('whisker sweep (baseline-sub)')
    a0.set_title('whisker amplitude around the collection instant\n(lick/groom frames excluded)',
                 fontsize=10, fontweight='bold')
    a0.legend(fontsize=7.5, loc='upper left'); a0.grid(alpha=0.2)

    a1 = ax[1]
    for oc, r in res.items():
        a1.plot(lags, r['n_lag'] / max(r['n'], 1), color=OUT_COL[oc], lw=1.8, label=OUT_LAB[oc])
    a1.axvline(0, color='k', ls='--', lw=1)
    a1.axvspan(TEST_LO_S, TEST_HI_S, color='#2ecc71', alpha=0.10)
    a1.set_ylim(0, 1.02)
    a1.set_xlabel('time from COLLECTION (s)'); a1.set_ylabel('fraction of events with a CLEAN frame')
    a1.set_title('WHY the post-hit half is unusable\nreward collapses to ~0 clean frames after t=0',
                 fontsize=10, fontweight='bold', color='#c0392b')
    a1.legend(fontsize=7.5); a1.grid(alpha=0.2)

    a2 = ax[2]
    groups = [('single_reward', pos), ('banish', neg)]
    for k, (oc, v) in enumerate(groups):
        a2.bar(k, v.mean(), yerr=v.std() / np.sqrt(len(v)), color=OUT_COL[oc], alpha=0.85, capsize=5)
        a2.scatter(np.full(len(v), k) + np.random.uniform(-.12, .12, len(v)), v,
                   s=16, color='k', alpha=0.45, zorder=3)
    a2.axhline(0, color='k', lw=0.6)
    a2.set_xticks([0, 1])
    a2.set_xticklabels([f'POSITIVE\nreward (any streak)\n(n={len(pos)})', f'NEGATIVE\nbanish\n(n={len(neg)})'],
                       fontsize=8.5)
    a2.set_ylabel('whisker sweep in the approach (baseline-sub)')
    a2.set_title(f'APPROACH window [{TEST_LO_S}, {TEST_HI_S}) s\ndiff {diff:+.3f} '
                 f'[{ci[0]:+.3f}, {ci[1]:+.3f}]  p={pv:.3f}'
                 f"{'  *' if pv < 0.05 else '  (n.s.)'}", fontsize=10, fontweight='bold')
    a2.grid(alpha=0.2, axis='y')

    fig.suptitle(f"{sess['mouse_id']}  OUTCOME VALENCE at the collection instant - does whisker "
                 'amplitude differ hitting a POSITIVE vs a NEGATIVE icon?\n'
                 '[whisker = optic-flow AMPLITUDE, not a set-point; 30 fps -> 33 ms resolution; '
                 'ONE session; banish n=10 usable -> UNDERPOWERED]',
                 fontsize=10.5, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = d / 'debug' / 'collection_valence.png'
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n  wrote {out}")
    return res


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
