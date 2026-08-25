"""
PATTERN plots across every session of one animal -- the "is there anything here?" step.

*** DESCRIPTIVE ONLY, BY DESIGN. ***
These panels plot each session as a point on a line and draw no verdicts: no fitted trend line, no
significance star, no "improving"/"declining" label. That is deliberate. Looking at a curve and
then choosing how to bin it is how a pattern gets manufactured -- pick the split after seeing the
data and almost any series will oblige. So the order is: LOOK here first, then, if something looks
real, test it with the block tests, whose bins are fixed in advance.

The one comparison that IS quantified is FIRST vs SECOND half, because it is a fixed, pre-declared
split rather than one chosen by eye -- and it is the specific question asked ("is he only good in
the first half?"). Even there the output is the two numbers, their intervals and a p-value, with
no adjective attached.

Two halves of WHAT, kept separate because they answer different questions:
  within-session  each session's own trials split at its midpoint, then pooled across sessions.
                  -> does he fade DURING a session (fatigue, satiety)?
  across-sessions the first half of the DAYS vs the second half.
                  -> did he change over training?
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'common'))
import perf_from_log as pfl          # noqa: E402

CHANCE_C, EXO_C, OBS_C = '#7f8c8d', '#f39c12', '#2c3e50'


def _wilson(k, n):
    return pfl._wilson(k, n) if n else (np.nan, np.nan)


def _thin(a, labels, every=None):
    n = len(labels)
    every = every or max(1, int(np.ceil(n / 14)))
    a.set_xticks(np.arange(n))
    a.set_xticklabels([l if i % every == 0 else '' for i, l in enumerate(labels)],
                      fontsize=7, rotation=45, ha='right')


def session_lines(R, out_path=None, title=''):
    """One point per session for every performance quantity, in one column of shared-x panels.

    Shared x means a feature at session k lines up vertically across panels, so a bump in D can be
    read against what throughput and session length were doing at the same time.
    """
    x = np.arange(len(R))
    fig, ax = plt.subplots(6, 1, figsize=(max(11, 0.42 * len(R) + 6), 15), sharex=True)

    a = ax[0]
    a.axhspan(0, 1.05, color='#27ae60', alpha=.06); a.axhspan(-1.05, 0, color='#c0392b', alpha=.06)
    # numpy, not Series: matplotlib warns (and will later raise) on a one-row frame
    yerr = np.vstack([(R.D - R.D_lo).clip(lower=0).to_numpy(float),
                      (R.D_hi - R.D).clip(lower=0).to_numpy(float)])
    a.errorbar(x, R.D.to_numpy(float), yerr=yerr,
               fmt='o-', ms=6, lw=1.6, capsize=3, color=OBS_C, label='D (visibility-weighted)')
    if 'D_exo' in R:
        a.plot(x, R.D_exo, 's--', ms=5, color=EXO_C, label='D (spawn geometry)')
    a.axhline(0, color='k', lw=1)
    a.set_ylabel('D'); a.legend(fontsize=8, ncol=2)
    a.set_title('DISCRIMINATION  (0 = chance, 1 = perfect; band = 95% confidence interval)',
                fontsize=10, fontweight='bold'); a.grid(alpha=.2)

    a = ax[1]
    a.plot(x, R.acc, 'o-', color=OBS_C, label='observed')
    a.plot(x, R.chance, 'v--', color=CHANCE_C, label='chance (visibility)')
    if 'chance_exo' in R:
        a.plot(x, R.chance_exo, '^:', color=EXO_C, label='chance (spawn geometry)')
    a.set_ylabel('P(positive)'); a.legend(fontsize=8, ncol=3)
    a.set_title('OBSERVED vs its OWN chance level -- the gap between them IS D',
                fontsize=10, fontweight='bold'); a.grid(alpha=.2)

    a = ax[2]
    a.plot(x, R.coll_per_min, 'o-', color='#2980b9', label='collections / min')
    a.plot(x, R.drops_per_min, 's-', color='#8e44ad', label='reward drops / min')
    a.set_ylabel('per minute'); a.legend(fontsize=8)
    a.set_title('THROUGHPUT -- how often he scored vs how much he was paid',
                fontsize=10, fontweight='bold'); a.grid(alpha=.2)

    a = ax[3]
    a.plot(x, R.n_trials, 'o-', color='#16a085', label='scored collections')
    a.set_ylabel('n'); a.legend(fontsize=8)
    a.set_title('HOW MUCH DATA each session contributes (a low point makes its D uncertain)',
                fontsize=10, fontweight='bold'); a.grid(alpha=.2)

    a = ax[4]
    a.plot(x, R.wall_min, 'o-', color='#d35400', label='session length (wall min)')
    a.set_ylabel('minutes'); a.legend(fontsize=8)
    a.set_title('SESSION LENGTH', fontsize=10, fontweight='bold'); a.grid(alpha=.2)

    a = ax[5]
    if 'drops' in R:
        a.bar(x, R.drops, color='#27ae60', alpha=.8)
    a.set_ylabel('drops'); a.set_title('TOTAL REWARD DROPS earned', fontsize=10, fontweight='bold')
    a.grid(alpha=.2, axis='y')
    _thin(a, list(R.label))
    a.set_xlabel('session')

    fig.suptitle(title or 'Per-session pattern -- descriptive, no trend fitted',
                 fontweight='bold', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    if out_path:
        fig.savefig(out_path, dpi=110); print(f'  wrote {out_path}')
    return fig


def half_split(C, R=None, out_path=None):
    """FIRST vs SECOND half, two ways. `C` = the df_log collections table for ONE animal.

    Within-session halves are computed per session and then POOLED, rather than splitting the
    pooled stream once: sessions differ in length, so one long session would otherwise dominate
    both halves and the comparison would partly be about that session.
    """
    C = C[C.valence != 0].copy()             # escapes offer no choice
    out = {}

    # --- within session -------------------------------------------------------------------
    firsts, seconds = [], []
    for s, g in C.groupby('session'):
        g = g.sort_values('t_ms')
        h = len(g) // 2
        if h < 2:
            continue
        firsts.append(g.iloc[:h]); seconds.append(g.iloc[h:])
    F = pd.concat(firsts) if firsts else C.iloc[:0]
    S = pd.concat(seconds) if seconds else C.iloc[:0]
    for tag, g in (('within_first', F), ('within_second', S)):
        k, n = int(g.is_positive.sum()), len(g)
        lo, hi = _wilson(k, n)
        out[tag] = dict(k=k, n=n, p=k / n if n else np.nan, lo=lo, hi=hi)

    # per-session first/second, so the spread across sessions is visible too
    per = []
    for s, g in C.groupby('session'):
        g = g.sort_values('t_ms'); h = len(g) // 2
        if h < 2:
            continue
        per.append(dict(session=s, day=g.day.iloc[0],
                        first=g.iloc[:h].is_positive.mean(),
                        second=g.iloc[h:].is_positive.mean(),
                        n=len(g)))
    P = pd.DataFrame(per)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    a = ax[0]
    for i, (tag, lbl) in enumerate((('within_first', 'first half\nof each session'),
                                    ('within_second', 'second half\nof each session'))):
        d = out[tag]
        a.bar(i, d['p'], color=['#2980b9', '#c0392b'][i], width=.55)
        a.errorbar(i, d['p'], yerr=[[d['p'] - d['lo']], [d['hi'] - d['p']]],
                   fmt='none', ecolor='k', capsize=5, lw=1.4)
        a.text(i, 0.02, f"{d['p']:.3f}\nn={d['n']}", ha='center', fontsize=9, color='w',
               fontweight='bold')
        a.set_xticks([0, 1])
    a.set_xticklabels(['first half\nof each session', 'second half\nof each session'], fontsize=9)
    a.set_ylabel('P(collected a reward)'); a.set_ylim(0, 1)
    a.set_title('WITHIN SESSION, pooled over all sessions\n(bars = 95% confidence interval)',
                fontsize=10, fontweight='bold'); a.grid(alpha=.2, axis='y')

    a = ax[1]
    if len(P):
        for _, r in P.iterrows():
            a.plot([0, 1], [r['first'], r['second']], '-', color='#95a5a6', lw=.9, alpha=.7)
        a.plot(np.zeros(len(P)), P['first'], 'o', color='#2980b9', ms=6)
        a.plot(np.ones(len(P)), P['second'], 'o', color='#c0392b', ms=6)
        a.plot([0, 1], [P['first'].mean(), P['second'].mean()], 'o-', color='k', lw=2.4, ms=9,
               label='mean')
        a.legend(fontsize=8)
    a.set_xticks([0, 1]); a.set_xticklabels(['first half', 'second half'])
    a.set_ylabel('P(collected a reward)'); a.set_ylim(0, 1)
    a.set_title('EACH SESSION separately\n(one line per session)', fontsize=10, fontweight='bold')
    a.grid(alpha=.2, axis='y')

    a = ax[2]
    if R is not None and len(R) >= 2:
        h = len(R) // 2
        for i, (lbl, g) in enumerate((('first half\nof the days', R.iloc[:h]),
                                      ('second half\nof the days', R.iloc[h:]))):
            a.bar(i, g.D.mean(), color=['#2980b9', '#c0392b'][i], width=.55)
            a.plot(np.full(len(g), i) + np.random.RandomState(0).normal(0, .05, len(g)),
                   g.D, 'o', color='k', ms=4, alpha=.6)
            out[f'days_{"first" if i == 0 else "second"}'] = dict(D=float(g.D.mean()), n=len(g))
        a.axhline(0, color='k', lw=1)
        a.set_xticks([0, 1])
        a.set_xticklabels(['first half\nof the days', 'second half\nof the days'], fontsize=9)
    a.set_ylabel('D'); a.set_title('ACROSS SESSIONS: D by half of the training days\n'
                                   '(dots = individual sessions)',
                                   fontsize=10, fontweight='bold'); a.grid(alpha=.2, axis='y')

    fig.suptitle('FIRST vs SECOND half -- a fixed split, stated without interpretation',
                 fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    if out_path:
        fig.savefig(out_path, dpi=110); print(f'  wrote {out_path}')
    return fig, out, P


# ---------------------------------------------------------------------------------------------
# DESCRIPTIVE overview -- "what have I actually got?", before any performance question is asked.
# These read df_sessions / df_trials straight from the saved dataset; nothing is scored or fitted.
# ---------------------------------------------------------------------------------------------

def overview(X, T=None, out_path=None):
    """How many sessions per mouse per world, how long they run, how many trials they carry.

    Deliberately the FIRST thing to look at: a D curve is unreadable without knowing that a point
    is one 40-minute session of 80 trials and the next is a 10-minute session of 12. Panels (d)
    and (e) are what tell you whether two points are comparable at all.
    """
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))

    a = ax[0, 0]
    ct = X.pivot_table(index='mouse', columns='world_id', values='session',
                       aggfunc='count').fillna(0)
    ct.plot(kind='bar', stacked=True, ax=a, colormap='tab20', width=.7)
    a.set_ylabel('sessions'); a.set_xlabel('')
    a.legend(fontsize=7, title='world', ncol=2)
    a.set_title('(a) SESSIONS per mouse per WORLD', fontsize=10, fontweight='bold')
    a.tick_params(axis='x', rotation=0); a.grid(alpha=.2, axis='y')

    a = ax[0, 1]
    ct2 = X.pivot_table(index='mouse', columns='task', values='session',
                        aggfunc='count').fillna(0)
    ct2.plot(kind='bar', stacked=True, ax=a, colormap='Set2', width=.7)
    a.set_ylabel('sessions'); a.set_xlabel('')
    a.legend(fontsize=7, title='protocol')
    a.set_title('(b) SESSIONS per mouse per PROTOCOL', fontsize=10, fontweight='bold')
    a.tick_params(axis='x', rotation=0); a.grid(alpha=.2, axis='y')

    a = ax[0, 2]
    for m, g in X.groupby('mouse'):
        g = g.sort_values('day')
        for task, gg in g.groupby('task'):
            a.plot(pd.to_datetime(gg.day), [m] * len(gg), 'o', ms=7, alpha=.8, label=task)
    h, l = a.get_legend_handles_labels()
    seen = dict(zip(l, h))
    a.legend(seen.values(), seen.keys(), fontsize=7)
    a.set_title('(c) TIMELINE -- which protocol, which day', fontsize=10, fontweight='bold')
    a.tick_params(axis='x', rotation=45, labelsize=7); a.grid(alpha=.2)

    a = ax[1, 0]
    a.bar(range(len(X)), X.wall_min.to_numpy(float), color='#d35400')
    _thin(a, list(X.label))
    a.set_ylabel('minutes')
    a.set_title('(d) SESSION LENGTH (wall clock)', fontsize=10, fontweight='bold')
    a.grid(alpha=.2, axis='y')

    a = ax[1, 1]
    a.bar(range(len(X)), X.n_trials_total.to_numpy(float), color='#16a085')
    _thin(a, list(X.label))
    a.set_ylabel('trials')
    a.set_title('(e) TRIALS per session', fontsize=10, fontweight='bold')
    a.grid(alpha=.2, axis='y')

    a = ax[1, 2]
    if T is not None and 'outcome' in T:
        oc = T.pivot_table(index='task', columns='outcome', values='session',
                           aggfunc='count').fillna(0)
        oc = oc.div(oc.sum(axis=1), axis=0)
        oc.plot(kind='barh', stacked=True, ax=a, colormap='tab20', width=.6)
        a.legend(fontsize=7, ncol=2, loc='lower right')
        a.set_xlabel('fraction of trials'); a.set_ylabel('')
    a.set_title('(f) OUTCOME MIX per protocol', fontsize=10, fontweight='bold')
    a.grid(alpha=.2, axis='x')

    fig.suptitle('What is in the dataset -- descriptive only, nothing scored',
                 fontweight='bold', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    if out_path:
        fig.savefig(out_path, dpi=110); print(f'  wrote {out_path}')
    return fig


def pooled_criterion(B, out_path=None, title=''):
    """The criterion with EVERY session of a world pooled into one block (blocks_of(size=None)).

    The un-binned answer, which is the one to read first. `B` is a one-row blocks table; the panel
    shows CONFLICT (hazard was nearer -- he must override proximity) against CONTROL (reward was
    nearer -- the easy case), with Wilson intervals. The two must SEPARATE for the animal to be
    recognising the icon: if both move together he has merely stopped following proximity.
    """
    fig, a = plt.subplots(figsize=(7.5, 5.5))
    r = B.iloc[0]
    for i, (k, lo, hi, lbl, col) in enumerate((
            ('control_p', 'control_lo', 'control_hi', 'CONTROL\nreward nearer (easy)', '#7f8c8d'),
            ('conflict_p', 'conflict_lo', 'conflict_hi',
             'CONFLICT\nhazard nearer (must override)', '#e67e22'))):
        a.bar(i, r[k], color=col, width=.55)
        a.errorbar(i, r[k], yerr=[[r[k] - r[lo]], [r[hi] - r[k]]], fmt='none',
                   ecolor='k', capsize=6, lw=1.6)
        n = int(r['n_control'] if i == 0 else r['n_conflict'])
        kk = int(r['k_control'] if i == 0 else r['k_conflict'])
        a.text(i, 0.03, f'{r[k]:.3f}\n{kk}/{n}', ha='center', color='w', fontweight='bold')
        a.set_xticks([0, 1])
    a.set_xticklabels(['CONTROL\nreward nearer (easy)',
                       'CONFLICT\nhazard nearer (must override)'], fontsize=9)
    a.axhline(0.5, ls=':', color='k', lw=1)
    a.set_ylim(0, 1); a.set_ylabel('P(collected the reward)')
    a.set_title((title or 'THE CRITERION, all sessions pooled') +
                f'\n{int(r.n_sessions)} sessions  |  gap (control - conflict) = '
                f'{r.control_p - r.conflict_p:+.3f}', fontsize=11, fontweight='bold')
    a.grid(alpha=.2, axis='y')
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=110); print(f'  wrote {out_path}')
    return fig
