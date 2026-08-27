"""
CROSS-SESSION performance tracking -- compare the same animal across days/weeks.

    python3 compare_sessions.py <session_dir> [<session_dir> ...]
    python3 compare_sessions.py --glob '/home/maryam/repo/flow_test/JPAS_*'
        -> <first session>/debug/session_comparison.png  (+ a printed table)
        -> --out DIR  to write elsewhere

HOW SHOULD SESSIONS BE COMPARED?  (the question was "box plots per day of D?", 2026-08-18)
--------------------------------------------------------------------------------------
Short answer: **box plots are right for the per-TRIAL measures, and wrong for D.**

  D IS ONE NUMBER PER SESSION, not a sample. A box plot needs a distribution; D is a single
  proportion-derived scalar, so its honest display is a POINT WITH A CONFIDENCE INTERVAL
  (Wilson interval on the underlying hit count, mapped through D). Box-plotting bootstrap
  replicates of D would draw a fat, reassuring box out of what is really n = (pos+neg) trials --
  it would look like data and it is not.
  -> panel 1: D per session as a LEARNING CURVE with CIs and a chance line at 0.

  WITHIN-SESSION BLOCKS give D a real spread, if you want one. Splitting each session into
  blocks of trials yields several D values per day, which IS box-plottable and shows within-day
  drift. It is noisy (few trials per block) so it is shown as points, not a box.
  -> panel 2: D per block, so a day that started well and fell apart is visible.

  PER-TRIAL QUANTITIES ARE WHERE BOX PLOTS BELONG -- trial duration, path efficiency, whisker
  sweep all have one value per TRIAL, so each session genuinely has a distribution.
  -> panels 4-6: box plot per session.

  *** THE ONE THAT MATTERS MOST: COMPARE D, NOT RAW ACCURACY. ***
  The chance level is not a constant across sessions -- it depends on how often each icon type
  was on screen, which depends on the world, the view scale and how the animal moved. Two
  sessions with the same raw accuracy can sit on opposite sides of chance. D already divides
  that out, which is exactly what makes it the comparable quantity across days.

Also plotted: throughput (rewards/min, units/min) and the session's chance level itself, so a
change in D can be read against a change in the task's own difficulty.
"""
from pathlib import Path
import argparse
import glob as globmod
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / 'common'))
import analyze_performance as ap     # noqa: E402
import fps as fpsmod                 # noqa: E402

N_BLOCKS = 4          # within-session blocks for the per-block D
MIN_BLOCK = 6         # a block needs this many pos+neg collections to give a usable D


def _wilson(k, n, z=1.96):
    """Wilson score interval for a proportion -- correct at small n, unlike the normal
    approximation, which is the regime every one of these sessions is in."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def load_session(d):
    d = Path(d).resolve()
    sess = json.load(open(d / 'session.json'))
    blocks = ap.run(str(d), write=False)
    B = blocks['ALL']
    df = pd.read_pickle(d / 'df_trials_clean.pkl')
    a = df[df.analyze].sort_values('trial').reset_index(drop=True)
    n = B['pos'] + B['neg']
    lo, hi = _wilson(B['pos'], n)
    ch = B['chance']
    to_D = lambda p: (p - ch) / (1 - ch)
    # per-block D (within-session drift)
    blk = []
    edges = np.linspace(0, len(a), N_BLOCKS + 1).astype(int)
    log = json.load(open(d / 'log.json'))
    fm = fpsmod.frame_to_ms(log)
    C = np.array(log['coords_t[ms]/x/y'], float)
    for i in range(N_BLOCKS):
        sub = a.iloc[edges[i]:edges[i + 1]]
        p_ = int((sub.outcome == 'single_reward').sum()); n_ = int((sub.outcome == 'banish').sum())
        if p_ + n_ < MIN_BLOCK:
            continue
        _, _, ch_b = ap._visibility_chance(sub, log, sess, fm, C)
        if not np.isfinite(ch_b) or ch_b >= 1:
            continue
        blk.append((p_ / (p_ + n_) - ch_b) / (1 - ch_b))
    return dict(name=sess['mouse_id'], date=str(sess.get('date', ''))[:10], dir=d,
                D=B['D'], D_lo=to_D(lo), D_hi=to_D(hi), p=B['p'], chance=ch, acc=B['acc'],
                pos=B['pos'], neg=B['neg'], units=B['units'], dur=B['dur_min'],
                rew_min=B['rew_min'], units_min=B['units_min'], n_trials=len(a),
                blocks=blk, dur_s=a.dur_s.values, eff=a.path_efficiency.values,
                sweep=a.whisk_sweep.values)


def run(dirs, out_dir=None):
    S = [load_session(d) for d in dirs]
    S.sort(key=lambda s: (s['date'], s['name']))
    labs = [f"{s['name']}\n{s['date']}" if s['date'] else s['name'] for s in S]
    x = np.arange(len(S))

    print(f"\n=== CROSS-SESSION COMPARISON ({len(S)} session"
          f"{'s' if len(S) != 1 else ''}) ===")
    print(f"  {'session':16s} {'trials':>6s} {'pos':>4s} {'neg':>4s} {'acc':>6s} {'chance':>7s} "
          f"{'D':>7s} {'95% CI':>16s} {'p':>6s} {'rew/min':>8s} {'units/min':>10s}")
    for s in S:
        ci = f"[{s['D_lo']:+.2f},{s['D_hi']:+.2f}]"
        print(f"  {s['name']:16s} {s['n_trials']:6d} {s['pos']:4d} {s['neg']:4d} {s['acc']:6.3f} "
              f"{s['chance']:7.3f} {s['D']:+7.3f} {ci:>16s} {s['p']:6.3f} "
              f"{s['rew_min']:8.2f} {s['units_min']:10.2f}")
    if len(S) == 1:
        print("\n  NOTE: one session -- the panels below are the template. Add more session dirs\n"
              "  to see the learning curve; the per-block panel already shows within-day drift.")

    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))

    # 1. D learning curve -- POINT + CI, not a box plot (D is one number per session)
    a0 = ax[0, 0]
    a0.axhspan(0, 1.05, color='#27ae60', alpha=0.07)
    a0.axhspan(-1.05, 0, color='#c0392b', alpha=0.07)
    a0.errorbar(x, [s['D'] for s in S],
                yerr=[[s['D'] - s['D_lo'] for s in S], [s['D_hi'] - s['D'] for s in S]],
                fmt='o-', ms=9, lw=2, capsize=5, color='#2c3e50', ecolor='#7f8c8d')
    for i, s in enumerate(S):
        if s['p'] < 0.05:
            a0.annotate('*', (i, s['D_hi']), ha='center', fontsize=16, color='#27ae60')
    a0.axhline(0, color='k', lw=1.6)
    a0.set_xticks(x); a0.set_xticklabels(labs, fontsize=8)
    a0.set_ylabel('discrimination score D'); a0.set_ylim(-1.05, 1.05)
    a0.set_title('1. LEARNING CURVE: D per session\npoint + 95% confidence interval (D is ONE number per session, '
                 'so no box plot)', fontsize=10.5, fontweight='bold')
    a0.text(0.02, 0.95, 'above 0 = discriminating', transform=a0.transAxes, fontsize=8,
            color='#27ae60', va='top')
    a0.text(0.02, 0.05, 'below 0 = drawn to the hazard', transform=a0.transAxes, fontsize=8,
            color='#c0392b', va='bottom')
    a0.grid(alpha=0.2)

    # 2. within-session blocks -- here a spread DOES exist
    a1 = ax[0, 1]
    for i, s in enumerate(S):
        if s['blocks']:
            a1.scatter(np.full(len(s['blocks']), i) + np.linspace(-.16, .16, len(s['blocks'])),
                       s['blocks'], s=45, color='#8e44ad', zorder=3)
            a1.plot([i - .2, i + .2], [np.mean(s['blocks'])] * 2, color='k', lw=2)
    a1.axhline(0, color='k', lw=1.6)
    a1.set_xticks(x); a1.set_xticklabels(labs, fontsize=8)
    a1.set_ylabel('D per block of trials')
    a1.set_title(f'2. WITHIN-session drift\nD per block ({N_BLOCKS} blocks/session), bar = mean',
                 fontsize=10.5, fontweight='bold')
    a1.grid(alpha=0.2)

    # 3. the chance level itself -- task difficulty is NOT constant across days
    a2 = ax[0, 2]
    a2.plot(x, [s['chance'] for s in S], 's--', color='#f39c12', lw=2, ms=8, label='chance level')
    a2.plot(x, [s['acc'] for s in S], 'o-', color='#27ae60', lw=2, ms=8, label='observed accuracy')
    a2.set_xticks(x); a2.set_xticklabels(labs, fontsize=8)
    a2.set_ylabel('P(collect positive)'); a2.set_ylim(0, 1)
    a2.set_title('3. WHY raw accuracy is not comparable\nthe chance level moves between sessions',
                 fontsize=10.5, fontweight='bold')
    a2.legend(fontsize=8); a2.grid(alpha=0.2)

    # 4-6. per-TRIAL distributions -- box plots belong here
    for a, key, lab, col in [(ax[1, 0], 'dur_s', 'trial duration (s)', '#2e86c1'),
                             (ax[1, 1], 'eff', 'path efficiency', '#16a085'),
                             (ax[1, 2], 'sweep', 'whisker sweep', '#8e44ad')]:
        data = [pd.Series(s[key]).dropna().values for s in S]
        bp = a.boxplot(data, positions=x, widths=0.55, patch_artist=True, showfliers=False)
        for b in bp['boxes']:
            b.set_facecolor(col); b.set_alpha(0.55)
        for med in bp['medians']:
            med.set_color('k'); med.set_linewidth(2)
        for i, v in enumerate(data):
            a.scatter(np.full(len(v), i) + np.random.uniform(-.16, .16, len(v)), v,
                      s=8, color='k', alpha=0.25, zorder=3)
        a.set_xticks(x); a.set_xticklabels(labs, fontsize=8)
        a.set_ylabel(lab)
        a.set_title(f'{4 + list(ax[1]).index(a)}. {lab} per trial\nBOX PLOT is right here: '
                    'one value per TRIAL', fontsize=10.5, fontweight='bold')
        a.grid(alpha=0.2, axis='y')

    fig.suptitle('Cross-session performance\n'
                 'compare D, NOT raw accuracy: the chance level depends on how often each icon was '
                 'on screen, so it moves between sessions\n'
                 'point + CI for D (one number per session);  box plots for per-TRIAL measures',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    outdir = Path(out_dir) if out_dir else (S[0]['dir'] / 'debug')
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / 'session_comparison.png'
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f'\n  wrote {out}')
    return S


def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument('session_dirs', nargs='*')
    ap_.add_argument('--glob', help="e.g. '/path/JPAS_*' (dirs containing session.json)")
    ap_.add_argument('--out', help='directory for the figure')
    a = ap_.parse_args()
    dirs = list(a.session_dirs)
    if a.glob:
        dirs += [p for p in sorted(globmod.glob(a.glob)) if (Path(p) / 'session.json').exists()]
    dirs = [d for d in dict.fromkeys(dirs) if (Path(d) / 'session.json').exists()]
    if not dirs:
        sys.exit('no session dirs found (need session.json + df_trials_clean.pkl in each)')
    run(dirs, a.out)


if __name__ == '__main__':
    main()
