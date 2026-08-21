"""
PERFORMANCE SCORECARD for a `banish_multiplier` session -- how well did the animal do today, and
did it change between the first and second half?

THE CENTRAL PROBLEM, and why a raw hit-rate is misleading here
--------------------------------------------------------------
Each normal trial puts 2 reward icons and 1 banish icon in the world, but the mouse sees only what is
inside his viewport, and the two types are NOT on screen equally often. Measured on this session:
reward icons are visible 0.69 of the time per frame, banish icons 0.28. So an animal walking around
with ZERO ability to tell them apart, collecting whatever he bumps into, already lands ~71% positive.

So the discrimination metric is measured against a **VISIBILITY-WEIGHTED CHANCE BASELINE**:

    chance = mean visible reward icons / (mean visible reward + mean visible banish icons)

*** REPORTED AS A RANGE OVER A MEMORY ASSUMPTION. *** The line above assumes ZERO memory -- an icon
counts only while it is on screen right now. But he wanders, sees an icon, then parks in a corner
where it is off screen and still knows where it is. Counting an icon from the moment he has seen it
(PERFECT memory) gives the other bound; measured here 85% of reward and 84% of banishment icons are
seen at least once in their trial, so that is very nearly the whole board -- which is why the
"2 rewards : 1 banish = 0.667" figure is NOT a naive baseline but essentially the perfect-memory one
(0.671 here). The two are the ends of an assumption, not a right and a wrong answer:
    zero memory     chance 0.713 -> D +0.013
    perfect memory  chance 0.671 -> D +0.139
Real memory over a 10-50 s trial lies between, and both ends are non-significant here.

and the headline number is how much of the available headroom above that chance he captured:

    DISCRIMINATION SCORE  D = (observed - chance) / (1 - chance)
        D = 0   collecting in proportion to what is on screen -- no evidence he tells them apart
        D = 1   perfect avoidance of the banishment
        D < 0   worse than chance (actively drawn to the banishment)

THREE AXES, DELIBERATELY NOT BLENDED INTO ONE NUMBER
----------------------------------------------------
  1. DISCRIMINATION  did he choose? (D, above)                -> the LEARNING metric
  2. THROUGHPUT      reward COLLECTIONS/min and reward DROPS/min -> how much he earned
  3. EFFICIENCY      trial duration, path efficiency           -> how directly he got there
They move independently -- this session has flat discrimination and a halved throughput -- and a
single blended score would hide exactly that.

SHOULD THE MULTIPLIER BE IN THE SCORE?  Split it:
  - NOT in DISCRIMINATION. The multiplier is a deterministic function of the reward STREAK, so
    weighting hits by it counts one success twice (once as a hit, once as the streak it built) and
    makes the metric super-linear in accuracy. It adds nothing about whether he can tell the icons
    apart.
  - YES in THROUGHPUT, as reward DROPS/min alongside COLLECTIONS/min. A "drop" is one delivery of
    the reward pump: a collection made at streak 3 pays 3 drops, so DROPS = sum of the multiplier
    over the rewarded trials (95 here -- the rig's own dashboard reads "num rew drops: 95", which is
    what this was checked against). Drops are what he actually earned, and because streaks pay more
    than scattered hits the GAP between collections/min and drops/min is itself informative -- but
    only while the two are reported separately.

Outputs: debug/performance.png + printed scorecard.
Run:  python3 analyze_performance.py <session_dir>
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_COMMON = Path(__file__).resolve().parent.parent / 'common'
sys.path.insert(0, str(_COMMON))
import fps as fpsmod       # noqa: E402
import viewport as vp      # noqa: E402

POS, NEG, ESC = 'single_reward', 'banish', 'unbanish'
C_POS, C_NEG, C_ESC = '#27ae60', '#c0392b', '#2980b9'
C_HALF = ('#2e86c1', '#8e44ad')


def _visibility_chance(sub, log, sess, fm, C):
    """Availability-weighted chance P(positive), under TWO memory assumptions.

    ZERO memory  -- an icon counts only while it is on screen right now.
    PERFECT memory -- it counts once he has seen it at any point this trial, because he wanders,
                      sees an icon, then parks in a corner where it is off screen but still knows
                      where it is. Measured here: 85% of reward and 84% of banishment icons are seen
                      at least once in their own trial, so this is very nearly the full board.

    Real memory over a 10-50 s trial lies between, so the two are reported as a RANGE. They are the
    two ends of a memory assumption, NOT a right and a wrong baseline.
    Only NORMAL-world trials -- the banishment world has no reward/banish choice to make."""
    vw, vh = vp.viewport(sess)
    vr = vb = 0.0          # ZERO-memory: fraction of frames the icon is on screen
    mr = mb = 0.0          # PERFECT-memory: was it EVER on screen this trial
    n = 0
    for _, r in sub[sub.world == 'NORMAL'].iterrows():
        s, e = int(r['start_frame']), int(min(r['end_frame'], len(fm)))
        if e - s < 5:
            continue
        t = fm[s:e]
        ax = np.interp(t, C[:, 0], C[:, 1]); ay = np.interp(t, C[:, 0], C[:, 2])
        for ic in (r['icons'] or []):
            seen = (np.abs(ic['x'] - ax) <= vw / 2) & (np.abs(ic['y'] - ay) <= vh / 2)
            if ic['effect'] == POS:
                vr += float(seen.mean()); mr += float(seen.any())
            elif ic['effect'] == NEG:
                vb += float(seen.mean()); mb += float(seen.any())
        n += 1
    if n == 0 or (vr + vb) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    ch = vr / (vr + vb)
    ch_mem = mr / (mr + mb) if (mr + mb) > 0 else np.nan
    return vr / n, vb / n, ch, mr / n, mb / n, ch_mem


def _wander(sub):
    """(median path length, median displacement, median start->target distance) in world units.

    These are what make a change in `path_efficiency` interpretable, because efficiency is
    displacement / path length and so it moves if EITHER term moves. Separating them says whether
    the animal WANDERED more (path length moved, displacement did not) or whether the TASK changed
    (targets spawned further away). Measured: efficiency correlates -0.95 with path length but only
    +0.07 with target distance, i.e. it is a wandering measure, not a distance measure."""
    plen, disp, sdist = [], [], []
    for _, r in sub.iterrows():
        x = np.asarray(r['coord_x'], float); y = np.asarray(r['coord_y'], float)
        if len(x) < 2:
            continue
        plen.append(np.nansum(np.hypot(np.diff(x), np.diff(y))))
        disp.append(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        if np.isfinite(r.get('target_x', np.nan)):
            sdist.append(np.hypot(r['target_x'] - x[0], r['target_y'] - y[0]))
    med = lambda v: float(np.nanmedian(v)) if len(v) else np.nan
    return med(plen), med(disp), med(sdist)


def _heading_to_visible(sub, log, sess, fm, C):
    """Mean forward ALIGNMENT toward each icon TYPE, over the frames where that icon is on screen.

    align = cos(theta - bearing_to_icon): +1 = driving straight at it, 0 = perpendicular,
    -1 = driving straight away. Only frames where the icon is inside the viewport count, because
    "was he heading toward it" is only meaningful while he could see it.

    Reward vs banish separately: if he discriminates, he should point TOWARD visible rewards and
    AWAY from visible banish icons, i.e. reward alignment > banish alignment.
    """
    vw, vh = vp.viewport(sess)
    A = np.array(log['angles_t[ms]/theta'], float)
    acc = {POS: [], NEG: []}
    for _, r in sub[sub.world == 'NORMAL'].iterrows():
        st, en = int(r['start_frame']), int(min(r['end_frame'], len(fm)))
        if en - st < 5:
            continue
        t = fm[st:en]
        ax = np.interp(t, C[:, 0], C[:, 1]); ay = np.interp(t, C[:, 0], C[:, 2])
        th = np.interp(t, A[:, 0], A[:, 1])
        for ic in (r['icons'] or []):
            if ic.get('effect') not in (POS, NEG):
                continue
            seen = (np.abs(ic['x'] - ax) <= vw / 2) & (np.abs(ic['y'] - ay) <= vh / 2)
            if seen.sum() < 3:
                continue
            bearing = np.arctan2(ic['y'] - ay, ic['x'] - ax)
            acc[ic['effect']].append(float(np.mean(np.cos(th[seen] - bearing[seen]))))
    m = lambda v: float(np.mean(v)) if v else np.nan
    return m(acc[POS]), m(acc[NEG]), len(acc[POS]), len(acc[NEG])


def _block(sub, log, sess, fm, C):
    p = int((sub.outcome == POS).sum()); n = int((sub.outcome == NEG).sum())
    esc = int((sub.outcome == ESC).sum())
    units = int(sub[sub.outcome == POS].multiplier.sum())
    dur_min = (sub.end_ms.max() - sub.start_ms.min()) / 60000
    vr, vb, chance, mr, mb, chance_mem = _visibility_chance(sub, log, sess, fm, C)
    acc = p / (p + n) if (p + n) else np.nan
    D = (acc - chance) / (1 - chance) if np.isfinite(chance) and chance < 1 else np.nan
    pv = binomtest(p, p + n, chance).pvalue if (p + n) and np.isfinite(chance) else np.nan
    D_mem = (acc - chance_mem) / (1 - chance_mem) if np.isfinite(chance_mem) and chance_mem < 1 else np.nan
    pv_mem = binomtest(p, p + n, chance_mem).pvalue if (p + n) and np.isfinite(chance_mem) else np.nan
    plen, disp, sdist = _wander(sub)
    al_r, al_b, n_r, n_b = _heading_to_visible(sub, log, sess, fm, C)
    return dict(align_reward=al_r, align_banish=al_b, n_align_r=n_r, n_align_b=n_b,
                mean_speed=float(sub.mean_speed.mean()),pos=p, neg=n, esc=esc, units=units, dur_min=dur_min, acc=acc,
                vis_r=vr, vis_b=vb, chance=chance, D=D, p=pv,
                mem_r=mr, mem_b=mb, chance_mem=chance_mem, D_mem=D_mem, p_mem=pv_mem,
                rew_min=p / dur_min, units_min=units / dur_min,
                med_dur=float(sub.dur_s.median()),
                eff=float(sub.path_efficiency.mean()),
                path_len=plen, disp=disp, start_dist=sdist,
                mult=float(sub[sub.outcome == POS].multiplier.mean()))


def run(session_dir, write=True, return_fig=False):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    log = json.load(open(d / 'log.json'))
    df = pd.read_pickle(d / 'df_trials_clean.pkl')
    a = df[df.analyze].sort_values('trial').reset_index(drop=True)
    fm = fpsmod.frame_to_ms(log)
    C = np.array(log['coords_t[ms]/x/y'], float)

    mid = len(a) / 2
    blocks = {'first': _block(a[a.index < mid], log, sess, fm, C),
              'second': _block(a[a.index >= mid], log, sess, fm, C),
              'ALL': _block(a, log, sess, fm, C)}
    B = blocks['ALL']

    print(f"=== {sess['mouse_id']} PERFORMANCE SCORECARD ===")
    print(f"  session {B['dur_min']:.1f} min, {len(a)} analyzable trials\n")
    print(f"  {'':7s} {'pos':>4s} {'neg':>4s} {'esc':>4s} {'acc':>6s} {'chance':>7s} {'D':>7s} "
          f"{'p':>6s} {'rew/min':>8s} {'units/min':>10s} {'s/trial':>8s} {'eff':>5s}")
    for k in ('first', 'second', 'ALL'):
        b = blocks[k]
        print(f"  {k:7s} {b['pos']:4d} {b['neg']:4d} {b['esc']:4d} {b['acc']:6.3f} {b['chance']:7.3f} "
              f"{b['D']:+7.3f} {b['p']:6.3f} {b['rew_min']:8.2f} {b['units_min']:10.2f} "
              f"{b['med_dur']:8.1f} {b['eff']:5.2f}")

    f, s_ = blocks['first'], blocks['second']
    p_acc = fisher_exact([[f['pos'], f['neg']], [s_['pos'], s_['neg']]])[1]
    d1 = a[a.index < mid].dur_s; d2 = a[a.index >= mid].dur_s
    p_dur = mannwhitneyu(d1, d2).pvalue
    # SECOND baseline -- exogenous, so it cannot absorb his avoidance skill (see viewport.py)
    exo = vp.spawn_geometry_chance(a, log, ['single_reward'], ['banish'])
    D_exo = (B['acc'] - exo) / (1 - exo)
    p_exo = binomtest(B['pos'], B['pos'] + B['neg'], exo).pvalue
    print(f"\n  ROBUSTNESS -- the visibility baseline is ENDOGENOUS (steering away from the banishment")
    print(f"  makes it visible less often, which the baseline would absorb). Exogenous check, from")
    print(f"  the spawn configuration + his position when the batch appeared:")
    print(f"    nearest-at-spawn is positive {exo:.3f} -> D = {D_exo:+.3f}, p = {p_exo:.3f}")
    print(f"    -> the 'at chance' read {'HOLDS' if p_exo > 0.05 else 'does NOT hold'} under both baselines")
    print(f"\n  DISCRIMINATION: observed {B['acc']:.3f} vs visibility-chance {B['chance']:.3f} "
          f"-> D = {B['D']:+.3f}, binomial p = {B['p']:.3f}")
    print(f"     {'AT CHANCE - no evidence he tells reward from banishment' if B['p'] > 0.05 else 'ABOVE CHANCE'}")
    print(f"     PERFECT-MEMORY bound: chance {B['chance_mem']:.3f} -> D = {B['D_mem']:+.3f}, "
          f"p = {B['p_mem']:.3f}   (he also knows icons he has already seen)")
    print(f"     => D = {B['D']:+.3f} (zero memory) to {B['D_mem']:+.3f} (perfect memory); "
          f"{'at chance either way' if max(B['p'], B['p_mem']) > 0.05 else 'see p-values'}")
    print(f"  first vs second half: accuracy {f['acc']:.3f} -> {s_['acc']:.3f} (Fisher p={p_acc:.3f})")
    print(f"                        rewards/min {f['rew_min']:.2f} -> {s_['rew_min']:.2f}")
    print(f"                        units/min   {f['units_min']:.2f} -> {s_['units_min']:.2f}")
    print(f"                        s/trial     {f['med_dur']:.1f} -> {s_['med_dur']:.1f} "
          f"(Mann-Whitney p={p_dur:.3f})")

    # ---------------- figure ----------------
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(18, 19.2))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1.42], hspace=0.62, wspace=0.26,
                  top=0.930, bottom=0.070, left=0.045, right=0.985)
    ax = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)])

    # 1. counts
    a0 = ax[0, 0]
    x = np.arange(3); w = 0.36
    for k, (lab, col) in enumerate([('first', C_HALF[0]), ('second', C_HALF[1])]):
        b = blocks[lab]
        a0.bar(x + k * w, [b['pos'], b['neg'], b['esc']], w, color=col, alpha=0.85,
               label=lab + ' half')
    a0.set_xticks(x + w / 2); a0.set_xticklabels(['positive\n(reward)', 'negative\n(banish)', 'escape\n(unbanish)'])
    a0.set_ylabel('collections'); a0.legend(fontsize=8)
    a0.set_title(f"(a)  collections by type\ntotal {B['pos']} pos / {B['neg']} neg / {B['esc']} esc",
                 fontsize=10.5, fontweight='bold'); a0.grid(alpha=0.2, axis='y')

    # 2. THE key panel: observed vs the two candidate baselines
    a1 = ax[0, 1]
    labs = ['first', 'second', 'ALL']
    xlab = ['first half', 'second half', 'WHOLE SESSION']   # axis labels, spelled out
    obs = [blocks[k]['acc'] for k in labs]; ch = [blocks[k]['chance'] for k in labs]
    xx = np.arange(3)
    a1.bar(xx - 0.18, obs, 0.36, color='#27ae60', alpha=0.9, label='observed P(positive)')
    a1.bar(xx + 0.18, ch, 0.36, color='#95a5a6', alpha=0.9, label='visibility-weighted CHANCE')
    a1.axhline(2 / 3, color='#c0392b', ls='--', lw=1.4, label='perfect-memory baseline (he remembers what he saw)')
    for i, k in enumerate(labs):
        a1.text(i, max(obs[i], ch[i]) + 0.02, f"p={blocks[k]['p']:.2f}", ha='center', fontsize=8)
    a1.set_xticks(xx); a1.set_xticklabels(xlab, fontsize=9); a1.set_ylim(0, 1.0)
    a1.set_ylabel('P(collect positive | collected)')
    a1.set_title('(b)  DISCRIMINATION vs the right baseline\nreward icons are on screen MORE than banish '
                 'icons', fontsize=10.5, fontweight='bold')
    a1.legend(fontsize=7.5, loc='lower left'); a1.grid(alpha=0.2, axis='y')

    # 3. the headline index
    a2 = ax[0, 2]
    Ds = [blocks[k]['D'] for k in labs]
    a2.bar(xx, Ds, 0.5, color=['#2e86c1', '#8e44ad', '#f39c12'], alpha=0.9)
    a2.axhline(0, color='k', lw=1.2)
    a2.set_xticks(xx); a2.set_xticklabels(xlab, fontsize=9); a2.set_ylim(-0.5, 1.0)
    a2.text(0.5, 0.93, 'D = 1  perfect avoidance of the banishment', transform=a2.transAxes,
            ha='center', fontsize=8, color='#333')
    a2.text(0.5, 0.06, 'D = 0  collecting in proportion to what is on screen',
            transform=a2.transAxes, ha='center', fontsize=8, color='#333')
    a2.set_ylabel('D = (observed - chance) / (1 - chance)')
    a2.text(0.5, 0.50, 'D = DISCRIMINATION SCORE\nthe fraction of the available headroom\n'
            'above chance that he captured',
            transform=a2.transAxes, ha='center', va='center', fontsize=9, style='italic',
            color='#555', bbox=dict(boxstyle='round', fc='#f7f7f7', ec='0.8', alpha=0.9))
    a2.set_title(f"(c)  DISCRIMINATION SCORE  D\nsession D = {B['D']:+.3f}  "
                 f"({'AT CHANCE' if B['p'] > 0.05 else 'above chance'})",
                 fontsize=10.5, fontweight='bold',
                 color='#c0392b' if B['p'] > 0.05 else '#27ae60')
    a2.grid(alpha=0.2, axis='y')

    # 4. throughput
    a3 = ax[1, 0]
    # "units" was opaque -- it is reward DROPS, the rig's own word (its dashboard reads
    # "num rew drops: 95"). A collection made at streak 3 pays 3 drops, so DROPS > COLLECTIONS.
    a3.bar(xx - 0.18, [blocks[k]['rew_min'] for k in labs], 0.36, color='#27ae60',
           alpha=0.9, label='reward COLLECTIONS / min  (how OFTEN he scored)')
    a3.bar(xx + 0.18, [blocks[k]['units_min'] for k in labs], 0.36, color='#f39c12',
           alpha=0.9, label='reward DROPS / min  (how MUCH he was paid)')
    for i, k in enumerate(labs):
        a3.text(i - 0.18, blocks[k]['rew_min'], f"{blocks[k]['rew_min']:.2f}", ha='center',
                va='bottom', fontsize=8)
        a3.text(i + 0.18, blocks[k]['units_min'], f"{blocks[k]['units_min']:.2f}", ha='center',
                va='bottom', fontsize=8)
    a3.set_xticks(xx); a3.set_xticklabels(xlab, fontsize=9); a3.set_ylabel('per minute')
    a3.set_title('(d)  THROUGHPUT: how OFTEN he scored vs how MUCH he was paid\n'
                 f"{B['pos']} collections -> {B['units']} drops in {B['dur_min']:.0f} min\n"
                 f"(mean {B['units']/B['pos']:.1f} drops each = the streak multiplier)",
                 fontsize=9.5, fontweight='bold')
    a3.legend(fontsize=7.5); a3.grid(alpha=0.2, axis='y'); a3.margins(y=0.16)

    # 5. PATH EFFICIENCY (higher = better) with time per trial on a twin axis (LOWER = better).
    #    This panel used to be titled "EFFICIENCY" while plotting SECONDS, so a taller second-half
    #    bar (slower = worse) read as "efficiency improved" -- it misled a reader on p.15. Both
    #    series are now drawn with their good direction stated, and the efficiency series is the
    #    one the title names.
    a4 = ax[1, 1]
    a4.bar([-0.16, 0.84], [f['eff'], s_['eff']], 0.32, color=C_HALF, alpha=0.9)
    a4.set_xticks([0, 1]); a4.set_xticklabels(['first half', 'second half'], fontsize=9)
    a4.set_ylabel('path efficiency  (HIGHER = better)', color='#2c3e50')
    a4.set_ylim(0, max(f['eff'], s_['eff']) * 1.7)
    a4t = a4.twinx()
    a4t.bar([0.16, 1.16], [f['med_dur'], s_['med_dur']], 0.32, color='#bdc3c7', alpha=0.95,
            edgecolor='#7f8c8d')
    a4t.set_ylabel('median trial duration, s  (LOWER = better)', color='#7f8c8d')
    a4t.tick_params(axis='y', colors='#7f8c8d')
    a4t.set_ylim(0, max(f['med_dur'], s_['med_dur']) * 1.7)
    eff_dir = 'WORSE' if s_['eff'] < f['eff'] else 'BETTER'
    a4.set_title(f"(e)  TIME OF TRIAL  {f['med_dur']:.0f}s -> {s_['med_dur']:.0f}s "
                 f"(p={p_dur:.3f}{'  *' if p_dur < 0.05 else '  n.s.'})\n"
                 f"coloured = path efficiency {f['eff']:.2f} -> {s_['eff']:.2f} ({eff_dir});  "
                 f"the two are ONE axis, r~-0.9",
                 fontsize=9.5, fontweight='bold')
    a4.legend(handles=[Patch(facecolor=C_HALF[0], label='path efficiency (higher better)'),
                       Patch(facecolor='#bdc3c7', edgecolor='#7f8c8d',
                             label='time per trial (lower better)')],
              fontsize=7, loc='upper left')
    a4.grid(alpha=0.2, axis='y')

    # 6. cumulative earning curve -- the classic performance plot
    a5 = ax[1, 2]
    t0 = a.start_ms.min()
    tm = (a.end_ms - t0) / 60000
    cum_pos = (a.outcome == POS).cumsum()
    cum_units = np.where(a.outcome == POS, a.multiplier.fillna(0), 0).cumsum()
    cum_neg = (a.outcome == NEG).cumsum()
    a5.plot(tm, cum_units, color='#f39c12', lw=2, label='cumulative reward UNITS')
    a5.plot(tm, cum_pos, color='#27ae60', lw=2, label='cumulative rewards')
    a5.plot(tm, cum_neg, color='#c0392b', lw=2, label='cumulative banishes')
    a5.axvline(tm.iloc[int(mid)], color='0.5', ls=':', lw=1.2, label='half split')
    a5.set_xlabel('session time (min)'); a5.set_ylabel('cumulative count')
    a5.set_title('(f)  EARNING CURVE\nflattening = the second-half slow-down', fontsize=10.5,
                 fontweight='bold')
    a5.legend(fontsize=7.5); a5.grid(alpha=0.2)

    # ---- the SCORECARD: every parameter on one line, first / second / whole session ----------
    # ---- ROW 3: speed, heading toward what he can SEE, and what the chance baseline IS ------
    a6 = ax[2, 0]
    sp = [blocks[k]['mean_speed'] for k in labs]
    a6.bar(xx, sp, 0.5, color=['#2e86c1', '#8e44ad', '#f39c12'], alpha=0.9)
    for i, v in enumerate(sp):
        a6.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
    d1s = a[a.index < mid].mean_speed.dropna(); d2s = a[a.index >= mid].mean_speed.dropna()
    p_sp = mannwhitneyu(d1s, d2s).pvalue
    a6.set_xticks(xx); a6.set_xticklabels(xlab, fontsize=9)
    a6.set_ylabel('mean speed (world units per second)')
    a6.set_title(f"(g)  SPEED of the animal\n{sp[0]:.0f} -> {sp[1]:.0f} world units/s  (p={p_sp:.3f}"
                 f"{'  *' if p_sp < 0.05 else '  n.s.'})\n"
                 f"the world is {int(sess['world_width'])} world units across, so 400 wu/s "
                 f"crosses it in ~{sess['world_width']/400:.0f} s",
                 fontsize=9.5, fontweight='bold')
    a6.grid(alpha=0.2, axis='y'); a6.margins(y=0.16)

    a7 = ax[2, 1]
    wid = 0.36
    a7.bar(np.arange(2) - wid / 2, [f['align_reward'], s_['align_reward']], wid,
           color='#27ae60', alpha=0.9, label=f"toward visible REWARD (n={f['n_align_r']+s_['n_align_r']})")
    a7.bar(np.arange(2) + wid / 2, [f['align_banish'], s_['align_banish']], wid,
           color='#c0392b', alpha=0.9, label=f"toward visible BANISH (n={f['n_align_b']+s_['n_align_b']})")
    a7.axhline(0, color='k', lw=1.2)
    a7.set_xticks(np.arange(2)); a7.set_xticklabels(['first half', 'second half'], fontsize=9)
    a7.set_ylabel('alignment = cos(heading error)')
    a7.text(0.98, 0.96, 'toward', transform=a7.transAxes, ha='right', va='top', fontsize=8, color='#2c7')
    a7.text(0.98, 0.04, 'away', transform=a7.transAxes, ha='right', va='bottom', fontsize=8, color='#c33')
    a7.set_title('(h)  is he HEADING toward the icons he can SEE?\n'
                 'heading error = the angle between the way he is pointing and the\n'
                 'direction to the icon (its BEARING); shown as cos(error):\n'
                 '+1 = straight at it, 0 = sideways, -1 = straight away',
                 fontsize=9, fontweight='bold')
    a7.legend(fontsize=7.5); a7.grid(alpha=0.2, axis='y')

    a8 = ax[2, 2]; a8.axis('off')
    a8.set_title('(i)  what "visibility-weighted chance" means', fontsize=10.5, fontweight='bold')
    a8.text(0.0, 1.0,
            "He can only choose among the icons that are ON HIS SCREEN,\n"
            "and the two types are not on screen equally often.\n\n"
            "PER FRAME, for every icon of the trial, ask: is it inside\n"
            f"the viewport?  |icon.x - avatar.x| <= {vp.viewport(sess)[0]/2:.0f}  and\n"
            f"|icon.y - avatar.y| <= {vp.viewport(sess)[1]/2:.0f}   (world units;\n"
            f"viewport = 800/{vp.view_scale(sess)} x 600/{vp.view_scale(sess)}).\n\n"
            "Average over frames -> each icon's visible fraction;\n"
            "sum by type -> icons of that type on screen per frame:\n\n"
            f"      REWARD {B['vis_r']:.2f}      BANISH {B['vis_b']:.2f}\n\n"
            "         chance = REWARD / (REWARD + BANISH)\n"
            f"                = {B['vis_r']:.2f} / ({B['vis_r']:.2f} + {B['vis_b']:.2f}) = {B['chance']:.3f}\n\n"
            "That is the rate an animal with NO discrimination would\n"
            "score, just by bumping into whatever is on screen.\n"
            f"He scored {B['acc']:.3f}, hence D = {B['D']:+.3f}.\n\n"
            "That assumes ZERO MEMORY (only what is on screen NOW).\n"
            "He also knows icons he has ALREADY SEEN this trial\n"
            f"this trial ({B['mem_r']/2*100:.0f}% of rewards, {B['mem_b']*100:.0f}% of banishments):\n\n"
            f"      REWARD {B['mem_r']:.2f}      BANISHMENT {B['mem_b']:.2f}   seen per trial\n\n"
            f"   chance = {B['mem_r']:.2f} / ({B['mem_r']:.2f} + {B['mem_b']:.2f}) = {B['chance_mem']:.3f}\n"
            f"   -> D = {B['D_mem']:+.3f}\n"
            f"Real memory is between: D = {B['D']:+.3f} to {B['D_mem']:+.3f}.",
            transform=a8.transAxes, va='top', ha='left', fontsize=7.9, family='monospace',
            linespacing=1.26,
            bbox=dict(boxstyle='round', fc='#f7f7f7', ec='0.6', alpha=0.97))

    axs = fig.add_subplot(gs[3, :]); axs.axis('off')
    def _f(k, key, fmt):
        return fmt.format(blocks[k][key])
    rows = [
        ['rewards per SECOND',          *[f"{blocks[k]['rew_min']/60:.4f}" for k in labs], 'throughput'],
        ['rewards per MINUTE',          *[f"{blocks[k]['rew_min']:.2f}" for k in labs], 'throughput'],
        ['reward DROPS per minute (x multiplier)',
                                        *[f"{blocks[k]['units_min']:.2f}" for k in labs], 'throughput'],
        ['n POSITIVE (reward)',         *[f"{blocks[k]['pos']}" for k in labs], 'count'],
        ['n NEGATIVE (banish)',         *[f"{blocks[k]['neg']}" for k in labs], 'count'],
        ['n ESCAPE (unbanish)',         *[f"{blocks[k]['esc']}" for k in labs], 'count'],
        ['total reward DROPS earned',   *[f"{blocks[k]['units']}" for k in labs], 'count'],
        ['P(hit POSITIVE | collected)', *[f"{blocks[k]['acc']:.3f}" for k in labs], 'CHOICE'],
        ['P(hit NEGATIVE | collected)', *[f"{1-blocks[k]['acc']:.3f}" for k in labs], 'CHOICE'],
        ['  chance from VISIBILITY',    *[f"{blocks[k]['chance']:.3f}" for k in labs], 'CHOICE'],
        ['  DISCRIMINATION SCORE  D',   *[f"{blocks[k]['D']:+.3f}" for k in labs], 'CHOICE'],
        ['  p vs chance (binomial)',    *[f"{blocks[k]['p']:.3f}" for k in labs], 'CHOICE'],
        ['  DISCRIMINATION SCORE  D (perfect memory)',
                                        *[f"{blocks[k]['D_mem']:+.3f}" for k in labs], 'CHOICE'],
        ['  p vs chance (perfect memory)',
                                        *[f"{blocks[k]['p_mem']:.3f}" for k in labs], 'CHOICE'],
        ['median trial duration (s)  [lower better]',
                                        *[f"{blocks[k]['med_dur']:.1f}" for k in labs], 'efficiency'],
        ['path efficiency  [higher better]',
                                        *[f"{blocks[k]['eff']:.2f}" for k in labs], 'efficiency'],
        ['  median PATH LENGTH (wu)',   *[f"{blocks[k]['path_len']:.0f}" for k in labs], 'efficiency'],
        ['  median DISPLACEMENT (wu)',  *[f"{blocks[k]['disp']:.0f}" for k in labs], 'efficiency'],
        ['  median START->TARGET dist (wu)',
                                        *[f"{blocks[k]['start_dist']:.0f}" for k in labs], 'efficiency'],
        ['mean multiplier on rewards',  *[f"{blocks[k]['mult']:.2f}" for k in labs], 'efficiency'],
    ]
    tab = axs.table(cellText=[r[:4] for r in rows],
                    colLabels=['parameter', 'first half', 'second half', 'WHOLE SESSION'],
                    colWidths=[0.40, 0.20, 0.20, 0.20], loc='upper center', cellLoc='center')
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.12)
    grp_col = {'throughput': '#fdf3e3', 'count': '#eef4fa', 'CHOICE': '#e9f7ef',
               'efficiency': '#f4f0f7'}
    for (rr, cc), cell in tab.get_celld().items():
        cell.set_edgecolor('#ccc')
        if rr == 0:
            cell.set_facecolor('#34495e'); cell.set_text_props(color='w', fontweight='bold')
            continue
        grp = rows[rr - 1][4]
        cell.set_facecolor(grp_col[grp])
        if cc == 0:
            cell.set_text_props(ha='left')
        if grp == 'CHOICE' and 'DISCRIMINATION' in rows[rr - 1][0]:
            cell.set_text_props(fontweight='bold')
    axs.set_title('SCORECARD  -  green = CHOICE (the learning metric), orange = THROUGHPUT, '
                  'blue = counts, purple = EFFICIENCY', fontsize=10.5, fontweight='bold', pad=3)
    both_ns = max(B['p'], B['p_mem']) > 0.05
    verdict = (f"CHOICE IS AT CHANCE:  D = {B['D']:+.3f} (zero memory) to {B['D_mem']:+.3f} "
               f"(perfect memory) -- at chance either way." if both_ns else
               f"CHOICE IS ABOVE CHANCE:  D = {B['D']:+.3f} to {B['D_mem']:+.3f}.")
    fig.text(0.5, 0.030, verdict, ha='center', va='bottom', fontsize=11,
             color='#c0392b' if B['p'] > 0.05 else '#27ae60', fontweight='bold')
    fig.suptitle(f"{sess['mouse_id']} PERFORMANCE SCORECARD\n"
                 f"discrimination is scored against a VISIBILITY-WEIGHTED chance baseline "
                 f"(reward icons on screen {B['vis_r']:.2f} vs banish {B['vis_b']:.2f} per frame), "
                 f"not the 2:1 icon-count baseline\n"
                 f"the multiplier is EXCLUDED from accuracy (it double-counts a streak) and "
                 f"INCLUDED in units/min (it is what he actually earned)",
                 fontsize=11, fontweight='bold', y=0.985)
    out = d / 'debug' / 'performance.png'
    fig.savefig(out, dpi=110)
    if return_fig:
        return blocks, fig          # caller writes it into a VECTOR pdf (sharp at any zoom)
    plt.close(fig)
    print(f"\n  wrote {out}")
    return blocks


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
