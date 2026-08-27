"""
STREAK / COMBO analysis -- what the discrimination score D cannot see.

D is a POOLED RATE: of everything he collected, what fraction was positive. It is completely blind
to ORDER. These two sequences give an identical D:

    R R R R R R B B B B        (long good patch, then a bad patch)
    R B R R B R B R R B        (evenly mixed)

but they describe different animals. The streak analysis asks whether outcomes are INDEPENDENT --
i.e. whether the next collection depends on how the recent ones went.

Task relevance: consecutive rewards build the multiplier 1 -> 4 (it resets on a negative outcome),
so a streak is worth more than the same number of scattered rewards. `analyse()` therefore also
reports the reward DROPS actually earned versus what the same collections would have earned if they
had arrived in a random order.

*** THE NON-CIRCULAR FRAMING MATTERS HERE. *** "Performance vs multiplier" is partly circular,
because reaching multiplier 4 MEANS three prior successes -- the multiplier is a relabelling of the
streak, not an independent variable. The two questions that are NOT circular are:
  1. does P(next is positive) change with the CURRENT run length?
  2. is the run-length distribution different from what independence predicts?
Both are answered below, with the small-n caveats made explicit.

STATISTICS
  - Wald-Wolfowitz RUNS TEST. Under independence the number of runs has a known mean/variance given
    the counts. MORE runs than expected = alternation (a positive outcome is followed by a negative
    more often than chance); FEWER = clustering (good and bad patches).
  - P(next positive | current run length k), with Wilson intervals -- n falls off fast with k, so the
    intervals are what to read, not the point estimates.
  - Observed run lengths vs the GEOMETRIC null, whose mean is 1/(1-p) for an overall positive rate p.

Task-agnostic: pass the ordered boolean sequence (True = positive outcome).
"""
import numpy as np
from scipy.stats import norm


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def runs_of(seq):
    """Lengths of the consecutive-True runs."""
    out, cur = [], 0
    for x in seq:
        if x:
            cur += 1
        elif cur:
            out.append(cur); cur = 0
    if cur:
        out.append(cur)
    return np.array(out, int)


def analyse(seq, mult_cap=4):
    """`seq` = ordered bools, True = positive outcome. Returns every statistic the figure needs."""
    seq = np.asarray(seq, bool)
    n1, n2 = int(seq.sum()), int((~seq).sum())
    N = n1 + n2
    p = n1 / N if N else np.nan

    # --- Wald-Wolfowitz runs test -------------------------------------------------------
    R = 1 + int((seq[1:] != seq[:-1]).sum()) if N > 1 else 0
    ER = 2 * n1 * n2 / N + 1 if N else np.nan
    VR = (2 * n1 * n2 * (2 * n1 * n2 - N) / (N ** 2 * (N - 1))) if N > 1 else np.nan
    z = (R - ER) / np.sqrt(VR) if VR and VR > 0 else np.nan
    p_runs = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan

    # --- P(next positive | current run length) ------------------------------------------
    by_run = {}
    run = 0
    for x in seq:
        by_run.setdefault(run, []).append(bool(x))
        run = run + 1 if x else 0
    cond = {}
    for k, v in by_run.items():
        v = np.array(v)
        lo, hi = _wilson(int(v.sum()), len(v))
        cond[k] = dict(n=len(v), p=float(v.mean()), lo=lo, hi=hi)

    # --- run lengths vs the geometric null ----------------------------------------------
    rl = runs_of(seq)
    geo_mean = 1 / (1 - p) if 0 < p < 1 else np.nan

    # --- what the streaks were WORTH (multiplier caps at mult_cap) -----------------------
    def drops(s):
        tot, m = 0, 1
        for x in s:
            if x:
                tot += m; m = min(m + 1, mult_cap)
            else:
                m = 1
        return tot
    earned = drops(seq)
    rng = np.random.default_rng(0)
    shuffled = np.array([drops(rng.permutation(seq)) for _ in range(2000)])

    return dict(n=N, n_pos=n1, n_neg=n2, p=p, seq=seq,
                runs=R, runs_exp=ER, runs_sd=np.sqrt(VR) if VR and VR > 0 else np.nan,
                runs_z=z, runs_p=p_runs,
                cond=cond, run_lengths=rl, run_mean=float(rl.mean()) if len(rl) else np.nan,
                geo_mean=geo_mean,
                drops_earned=earned, drops_shuffled=shuffled,
                drops_shuffled_mean=float(shuffled.mean()),
                drops_pct=float((shuffled < earned).mean()))


def verdict(st):
    """One-line reading of the runs test, with the direction spelled out."""
    if not np.isfinite(st['runs_p']):
        return 'not enough collections to test'
    if st['runs_p'] >= 0.05:
        return (f"consistent with INDEPENDENT outcomes (runs {st['runs']} vs "
                f"{st['runs_exp']:.1f} expected, p={st['runs_p']:.3f}) -- no evidence of "
                f"good/bad patches beyond chance")
    direction = 'ALTERNATION (a positive is followed by a negative more often than chance)' \
        if st['runs'] > st['runs_exp'] else 'CLUSTERING (outcomes come in patches)'
    return (f"outcomes are NOT independent: {direction}; runs {st['runs']} vs "
            f"{st['runs_exp']:.1f} expected, p={st['runs_p']:.3f}")


def figure(st, mouse_id, pos_label='reward', neg_label='banishment', mult_cap=4):
    """The streak page. Every panel is lettered (a)-(e) and titled in plain language: the earlier
    version used jargon ("run", "Wilson CI", "n falls off fast") that did not explain itself."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    C_POS, C_NEG, C_NULL = '#27ae60', '#c0392b', '#95a5a6'
    seq = st['seq']
    mid = len(seq) // 2
    r1, r2 = runs_of(seq[:mid]), runs_of(seq[mid:])

    fig = plt.figure(figsize=(18, 11.6))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.8, 1, 1], hspace=0.75, wspace=0.22,
                  top=0.875, bottom=0.055, left=0.06, right=0.98)

    # (a) the sequence itself
    a = fig.add_subplot(gs[0, :])
    x = np.arange(len(seq))
    a.bar(x[seq], np.ones(seq.sum()), color=C_POS, width=0.85)
    a.bar(x[~seq], -np.ones((~seq).sum()), color=C_NEG, width=0.85)
    m, mult = 1, []
    for v in seq:
        if v:
            mult.append(m); m = min(m + 1, mult_cap)
        else:
            mult.append(0); m = 1
    a.plot(x, np.array(mult) / mult_cap, color='#f39c12', lw=1.4, marker='o', ms=3,
           label=f'multiplier earned at that collection (1-{mult_cap}, rescaled to fit)')
    a.axvline(mid - 0.5, color='0.4', ls=':', lw=1.5)
    a.axhline(0, color='k', lw=1)
    a.set_xlabel('collection number, in the order they happened'); a.set_yticks([])
    a.set_title(f'(a)  every collection in order:  green up = {pos_label},  red down = {neg_label}'
                f"   ({st['n_pos']} / {st['n_neg']})\n"
                f'FIRST half: {len(r1)} streaks averaging {r1.mean():.1f} in a row (longest {r1.max()})'
                f'      SECOND half: {len(r2)} streaks averaging {r2.mean():.1f} (longest {r2.max()})',
                fontsize=11, fontweight='bold')
    a.legend(fontsize=8, loc='lower right'); a.margins(x=0.01)

    # (b) does the next outcome depend on the current streak?
    a = fig.add_subplot(gs[1, 0])
    ks = sorted(k for k, v in st['cond'].items() if v['n'] >= 3)
    pv = [st['cond'][k]['p'] for k in ks]
    lo = [st['cond'][k]['p'] - st['cond'][k]['lo'] for k in ks]
    hi = [st['cond'][k]['hi'] - st['cond'][k]['p'] for k in ks]
    a.errorbar(ks, pv, yerr=[lo, hi], fmt='o-', ms=9, lw=2, capsize=5, color='#2c3e50')
    a.axhline(st['p'], color=C_NULL, ls='--', lw=2,
              label=f"his overall rate {st['p']:.3f}")
    # values on ONE fixed level near the top -- offsetting them from each point put some of
    # them on top of the line/error bars
    for k, v in zip(ks, pv):
        a.text(k, 1.10, f"{v:.2f}\n(n={st['cond'][k]['n']})", ha='center', va='top', fontsize=8.5)
    a.set_xlabel(f'how many {pos_label}s he has just collected IN A ROW')
    a.set_ylabel(f'P(the NEXT one is a {pos_label})')
    a.set_ylim(0, 1.22); a.set_xticks(ks)
    a.set_title(f'(b)  after N {pos_label}s in a row, how often is the NEXT one a {pos_label}?\n'
                'flat line = the streak makes no difference.  Vertical bars = the range of true\n'
                'values the data allow (95% confidence interval); they widen as cases shrink',
                fontsize=10, fontweight='bold')
    a.legend(fontsize=8, loc='lower left'); a.grid(alpha=0.2)

    # (c) how long are the streaks, vs independence
    a = fig.add_subplot(gs[1, 1])
    rl = st['run_lengths']
    if len(rl):
        mx = int(rl.max())
        a.hist(rl, bins=np.arange(1, mx + 2) - 0.5, color=C_POS, alpha=0.85,
               label=f'his {len(rl)} actual streaks')
        kk = np.arange(1, mx + 1)
        a.plot(kk, len(rl) * (st['p'] ** (kk - 1)) * (1 - st['p']), 'o--', color=C_NULL, lw=2, ms=7,
               label='what independent coin flips would give')
    a.set_xlabel(f'length of a streak (how many {pos_label}s in a row)')
    a.set_ylabel('how many streaks of that length')
    shorter = st['run_mean'] < st['geo_mean']
    a.set_title(f"(c)  how long are his {pos_label} streaks?\n"
                f"his average {st['run_mean']:.2f} in a row, vs {st['geo_mean']:.2f} if every\n"
                f"collection were an independent coin flip at {st['p']:.2f}\n"
                + ('SHORTER = he switches more often than chance'
                   if shorter else 'LONGER = his outcomes come in patches'),
                fontsize=10, fontweight='bold')
    a.legend(fontsize=8); a.grid(alpha=0.2, axis='y')

    # (d) what the ORDER was worth
    a = fig.add_subplot(gs[2, 0])
    a.hist(st['drops_shuffled'], bins=30, color=C_NULL, alpha=0.85,
           label=f"the SAME {st['n_pos']} {pos_label}s + {st['n_neg']} {neg_label}s,\nreshuffled into a random order (2000x)")
    a.axvline(st['drops_earned'], color=C_POS, lw=3,
              label=f"what HIS order actually earned: {st['drops_earned']}")
    a.set_xlabel(f'reward drops earned over the session (multiplier caps at {mult_cap})')
    a.set_ylabel('number of shuffles')
    lost = st['drops_shuffled_mean'] - st['drops_earned']
    a.set_title(f"(d)  did the ORDER of his collections cost him reward?\n"
                f"same collections, only the order changed. A random order pays "
                f"{st['drops_shuffled_mean']:.0f} on average;\nhis actual order earned "
                f"{st['drops_earned']} -- {'LOST' if lost > 0 else 'GAINED'} {abs(lost):.0f} drops.\n"
                f"Only {st['drops_pct']*100:.0f}% of random orders would pay this little: his "
                f"successes are too scattered\nto reach the high multipliers. This is about ORDER "
                f"only -- the collections themselves are identical.",
                fontsize=9.5, fontweight='bold')
    a.legend(fontsize=8); a.grid(alpha=0.2, axis='y')

    # (e) the runs test, explained
    a = fig.add_subplot(gs[2, 1]); a.axis('off')
    a.set_title('(e)  are his outcomes independent of each other?', fontsize=10.5, fontweight='bold')
    a.text(0.0, 0.97,
           "THE RUNS TEST (Wald-Wolfowitz), in plain terms:\n"
           "split the sequence into BLOCKS of the same outcome --\n"
           "  R R R B B R  ->  [RRR][BB][R]  =  3 blocks.\n"
           f"With {st['n_pos']} {pos_label}s and {st['n_neg']} {neg_label}s, independent outcomes\n"
           f"would give about {st['runs_exp']:.1f} blocks.\n\n"
           f"   he actually has        {st['runs']} blocks\n"
           f"   expected if independent {st['runs_exp']:.1f}  +-{st['runs_sd']:.1f}\n"
           f"   p = {st['runs_p']:.3f}\n\n"
           "MORE blocks = switches more than chance; FEWER = patches.",
           transform=a.transAxes, va='top', ha='left', fontsize=8.5, family='monospace',
           linespacing=1.35)
    a.text(0.0, 0.12, verdict(st), transform=a.transAxes, va='top', ha='left', fontsize=9,
           color='#c0392b' if st['runs_p'] < 0.05 else '#2c3e50', wrap=True,
           bbox=dict(boxstyle='round', fc='#f7f7f7', ec='0.7'))

    fig.suptitle(f'{mouse_id}  STREAKS / COMBO -- what the discrimination score D cannot see\n'
                 f'{pos_label}s collected IN A ROW build the multiplier (caps at {mult_cap}); a '
                 f'{neg_label} resets it to 1,\nso the same collections in a different ORDER pay a '
                 f'different amount. D is a pooled rate and is blind to order.',
                 fontsize=12, fontweight='bold')
    return fig
