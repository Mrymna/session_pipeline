"""
PERFORMANCE SCORECARD for a `timeout_double` session (JPAS_0231-class).

Same logic as the `banish_multiplier` scorecard -- the animal's choice is scored against a
VISIBILITY-WEIGHTED chance baseline, because he can only choose among the icons that are actually
on his screen and the types are not on screen equally often. Scoring against the "2 good : 1 bad"
icon count would credit him with discrimination he has not shown.

    DISCRIMINATION SCORE   D = (observed - chance) / (1 - chance)
        0 = collecting in proportion to what is on screen, 1 = perfect avoidance of the hazard.

TWO THINGS DIFFER FROM THE BANISH TASK
--------------------------------------
1. REWARD UNITS. There is no multiplier; instead `double_reward` pays twice. Verified against the
   log's actual delivery stream (`rewards_t[ms]`): 163 deliveries = 61 single x1 + 51 double x2,
   so units = 1*single + 2*double is measured, not assumed. As in the banish fork the units are
   kept OUT of the accuracy metric (they measure size of prize, not quality of choice) and IN the
   throughput metric (they are what he actually earned).

2. THE TIMEOUT FREEZE MAKES "PER MINUTE" AMBIGUOUS -- so BOTH rates are reported:
     rewards / WALL minute    = what he earned per minute of session. The freeze counts against
                                him, which is correct: the penalty IS part of his performance.
     rewards / ACTIVE minute  = the rate while he could actually play (freeze time removed).
   The GAP between them is the cost of his own errors. Reporting only the wall rate hides how
   much of a slow session was penalty rather than sloth; reporting only the active rate forgives
   the penalty entirely.

Outputs: debug/performance.png + printed scorecard.
Run:  python3 analyze_performance.py /home/maryam/repo/flow_test/JPAS_0231_pipeline
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

_COMMON = Path(__file__).resolve().parent.parent / 'common'
sys.path.insert(0, str(_COMMON))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fps as fpsmod       # noqa: E402
import viewport as vp      # noqa: E402
import task_spec as ts     # noqa: E402

UNITS = {'single_reward': 1, 'double_reward': 2}     # verified against rewards_t[ms]
C_HALF = ('#2e86c1', '#8e44ad')


def _visibility_chance(sub, log, sess, fm, C):
    """Availability-weighted chance P(positive): how often each icon TYPE was on screen."""
    vw, vh = vp.viewport(sess)
    vpos = vneg = 0.0      # ZERO-memory: fraction of frames on screen
    mpos = mneg = 0.0      # PERFECT-memory: ever on screen this trial
    n = 0
    for _, r in sub.iterrows():
        s, e = int(r['start_frame']), int(min(r['end_frame'], len(fm)))
        if e - s < 5:
            continue
        t = fm[s:e]
        ax = np.interp(t, C[:, 0], C[:, 1]); ay = np.interp(t, C[:, 0], C[:, 2])
        for ic in (r['icons'] or []):
            seen = (np.abs(ic['x'] - ax) <= vw / 2) & (np.abs(ic['y'] - ay) <= vh / 2)
            if ic['effect'] in ts.POSITIVE:
                vpos += float(seen.mean()); mpos += float(seen.any())
            elif ic['effect'] in ts.NEGATIVE:
                vneg += float(seen.mean()); mneg += float(seen.any())
        n += 1
    if n == 0 or (vpos + vneg) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    ch_mem = mpos / (mpos + mneg) if (mpos + mneg) > 0 else np.nan
    return vpos / n, vneg / n, vpos / (vpos + vneg), mpos / n, mneg / n, ch_mem


def _block(sub, log, sess, fm, C):
    pos = int(sub.outcome.isin(ts.POSITIVE).sum())
    neg = int(sub.outcome.isin(ts.NEGATIVE).sum())
    single = int((sub.outcome == 'single_reward').sum())
    double = int((sub.outcome == 'double_reward').sum())
    units = single * UNITS['single_reward'] + double * UNITS['double_reward']
    wall_min = (sub.end_ms.max() - sub.start_ms.min()) / 60000
    active_min = sub.active_s.sum() / 60
    freeze_min = sub.freeze_ms.sum() / 60000
    vpos, vneg, chance, mpos, mneg, chance_mem = _visibility_chance(sub, log, sess, fm, C)
    acc = pos / (pos + neg) if (pos + neg) else np.nan
    D = (acc - chance) / (1 - chance) if np.isfinite(chance) and chance < 1 else np.nan
    p = binomtest(pos, pos + neg, chance).pvalue if (pos + neg) and np.isfinite(chance) else np.nan
    D_mem = (acc - chance_mem) / (1 - chance_mem) if np.isfinite(chance_mem) and chance_mem < 1 else np.nan
    p_mem = binomtest(pos, pos + neg, chance_mem).pvalue if (pos + neg) and np.isfinite(chance_mem) else np.nan
    return dict(pos=pos, neg=neg, single=single, double=double, units=units,
                wall_min=wall_min, active_min=active_min, freeze_min=freeze_min,
                acc=acc, vis_pos=vpos, vis_neg=vneg, chance=chance, D=D, p=p,
                mem_pos=mpos, mem_neg=mneg, chance_mem=chance_mem, D_mem=D_mem, p_mem=p_mem,
                rew_wall=pos / wall_min, rew_active=pos / max(active_min, 1e-9),
                units_wall=units / wall_min, units_active=units / max(active_min, 1e-9),
                med_dur=float(sub.active_s.median()), eff=float(sub.path_efficiency.mean()))


def run(session_dir, write=True):
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
    labs = ['first', 'second', 'ALL']

    print(f"=== {sess['mouse_id']} PERFORMANCE SCORECARD ({ts.TASK}) ===")
    print(f"  {B['wall_min']:.1f} min wall, {B['active_min']:.1f} min ACTIVE "
          f"({B['freeze_min']:.1f} min lost to timeout freezes), {len(a)} analyzable trials\n")
    print(f"  {'':7s} {'sgl':>4s} {'dbl':>4s} {'TO':>4s} {'units':>6s} {'acc':>6s} {'chance':>7s} "
          f"{'D':>7s} {'p':>6s} {'rew/wall':>9s} {'rew/actv':>9s}")
    for k in labs:
        b = blocks[k]
        print(f"  {k:7s} {b['single']:4d} {b['double']:4d} {b['neg']:4d} {b['units']:6d} "
              f"{b['acc']:6.3f} {b['chance']:7.3f} {b['D']:+7.3f} {b['p']:6.3f} "
              f"{b['rew_wall']:9.2f} {b['rew_active']:9.2f}")

    # SECOND baseline -- exogenous, so it cannot absorb his avoidance skill (see viewport.py)
    exo = vp.spawn_geometry_chance(a, log, ts.POSITIVE, ts.NEGATIVE)
    D_exo = (B['acc'] - exo) / (1 - exo)
    p_exo = binomtest(B['pos'], B['pos'] + B['neg'], exo).pvalue
    print(f"\n  ROBUSTNESS -- the visibility baseline is ENDOGENOUS (steering away from the hazard")
    print(f"  makes it visible less often, which the baseline would absorb). Exogenous check, from")
    print(f"  the spawn configuration + his position when the batch appeared:")
    print(f"    nearest-at-spawn is positive {exo:.3f} -> D = {D_exo:+.3f}, p = {p_exo:.3f}")
    print(f"    -> the 'at chance' read {'HOLDS' if p_exo > 0.05 else 'does NOT hold'} under both baselines")
    print(f"\n  DISCRIMINATION: observed {B['acc']:.3f} vs visibility-chance {B['chance']:.3f} "
          f"-> D = {B['D']:+.3f}, binomial p = {B['p']:.3f}")
    print(f"     {'AT CHANCE - no evidence he avoids the hazard' if B['p'] > 0.05 else 'ABOVE CHANCE - he is avoiding the hazard'}")
    print(f"     PERFECT-MEMORY bound: chance {B['chance_mem']:.3f} -> D = {B['D_mem']:+.3f}, "
          f"p = {B['p_mem']:.3f}")
    print(f"     => D = {B['D']:+.3f} (zero memory) to {B['D_mem']:+.3f} (perfect memory)")
    print(f"  positive icons on screen {B['vis_pos']:.2f}/frame vs negative {B['vis_neg']:.2f}")
    print(f"  FREEZE COST: {B['freeze_min']:.1f} of {B['wall_min']:.1f} min "
          f"({100*B['freeze_min']/B['wall_min']:.0f}%) lost to timeout penalties -> "
          f"rewards/min {B['rew_wall']:.2f} wall vs {B['rew_active']:.2f} active")

    f, s_ = blocks['first'], blocks['second']
    p_acc = fisher_exact([[f['pos'], f['neg']], [s_['pos'], s_['neg']]])[1]
    p_dur = mannwhitneyu(a[a.index < mid].active_s, a[a.index >= mid].active_s).pvalue

    # ---------------- figure ----------------
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(18, 14.4))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.05], hspace=0.40, wspace=0.26,
                  top=0.900, bottom=0.062, left=0.045, right=0.985)
    ax = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
    xx = np.arange(3)

    a0 = ax[0, 0]
    w = 0.36
    for k, (lab, col) in enumerate([('first', C_HALF[0]), ('second', C_HALF[1])]):
        b = blocks[lab]
        a0.bar(np.arange(3) + k * w, [b['single'], b['double'], b['neg']], w, color=col, alpha=0.85,
               label=lab)
    a0.set_xticks(np.arange(3) + w / 2)
    a0.set_xticklabels(['single\nreward', 'double\nreward', 'TIMEOUT'])
    a0.set_ylabel('collections'); a0.legend(fontsize=8)
    a0.set_title(f"collections by type\n{B['single']} single / {B['double']} double / {B['neg']} timeout",
                 fontsize=10.5, fontweight='bold'); a0.grid(alpha=0.2, axis='y')

    a1 = ax[0, 1]
    obs = [blocks[k]['acc'] for k in labs]; ch = [blocks[k]['chance'] for k in labs]
    a1.bar(xx - 0.18, obs, 0.36, color='#27ae60', alpha=0.9, label='observed P(positive)')
    a1.bar(xx + 0.18, ch, 0.36, color='#95a5a6', alpha=0.9, label='visibility-weighted CHANCE')
    a1.axhline(2 / 3, color='#c0392b', ls='--', lw=1.4, label='perfect-memory baseline (he remembers what he saw)')
    for i, k in enumerate(labs):
        a1.text(i, max(obs[i], ch[i]) + 0.02, f"p={blocks[k]['p']:.2f}", ha='center', fontsize=8)
    a1.set_xticks(xx); a1.set_xticklabels(labs); a1.set_ylim(0, 1)
    a1.set_ylabel('P(collect positive | collected)')
    a1.set_title('DISCRIMINATION vs the right baseline', fontsize=10.5, fontweight='bold')
    a1.legend(fontsize=7.5, loc='lower left'); a1.grid(alpha=0.2, axis='y')

    a2 = ax[0, 2]
    a2.bar(xx, [blocks[k]['D'] for k in labs], 0.5,
           color=['#2e86c1', '#8e44ad', '#f39c12'], alpha=0.9)
    a2.axhline(0, color='k', lw=1.2); a2.set_ylim(-0.5, 1.0)
    a2.set_xticks(xx); a2.set_xticklabels(labs)
    a2.set_ylabel('D = (observed - chance) / (1 - chance)')
    a2.text(0.5, 0.50, 'D = DISCRIMINATION SCORE\nthe fraction of the available headroom\n'
            'above chance that he captured', transform=a2.transAxes, ha='center', va='center',
            fontsize=9, style='italic', color='#555',
            bbox=dict(boxstyle='round', fc='#f7f7f7', ec='0.8', alpha=0.9))
    a2.set_title(f"DISCRIMINATION SCORE  D\nsession D = {B['D']:+.3f}  "
                 f"({'AT CHANCE' if B['p'] > 0.05 else 'above chance'})", fontsize=10.5,
                 fontweight='bold', color='#c0392b' if B['p'] > 0.05 else '#27ae60')
    a2.grid(alpha=0.2, axis='y')

    a3 = ax[1, 0]
    a3.bar(xx - 0.18, [blocks[k]['rew_wall'] for k in labs], 0.36, color='#27ae60', alpha=0.9,
           label='rewards / WALL min')
    a3.bar(xx + 0.18, [blocks[k]['rew_active'] for k in labs], 0.36, color='#f39c12', alpha=0.9,
           label='rewards / ACTIVE min (freeze removed)')
    a3.set_xticks(xx); a3.set_xticklabels(labs); a3.set_ylabel('per minute')
    a3.set_title(f"THROUGHPUT -- the gap IS the penalty cost\n{B['pos']} rewards = {B['units']} "
                 f"drops in {B['wall_min']:.0f} min wall / {B['active_min']:.0f} min active",
                 fontsize=10.5, fontweight='bold')
    a3.legend(fontsize=7.5); a3.grid(alpha=0.2, axis='y')

    a4 = ax[1, 1]
    a4.bar([0, 1], [f['freeze_min'], s_['freeze_min']], 0.5, color=C_HALF, alpha=0.9)
    a4.set_xticks([0, 1]); a4.set_xticklabels(['first', 'second'])
    a4.set_ylabel('minutes lost to timeout freeze')
    a4.set_title(f"TIMEOUT PENALTY COST\n{B['freeze_min']:.1f} of {B['wall_min']:.1f} min "
                 f"({100*B['freeze_min']/B['wall_min']:.0f}% of the session)",
                 fontsize=10.5, fontweight='bold', color='#c0392b')
    a4.grid(alpha=0.2, axis='y')

    a5 = ax[1, 2]
    t0 = a.start_ms.min()
    tm = (a.end_ms - t0) / 60000
    cum_units = np.where(a.outcome == 'single_reward', 1,
                         np.where(a.outcome == 'double_reward', 2, 0)).cumsum()
    a5.plot(tm, cum_units, color='#f39c12', lw=2, label='cumulative reward DROPS')
    a5.plot(tm, a.outcome.isin(ts.POSITIVE).cumsum(), color='#27ae60', lw=2,
            label='cumulative rewards')
    a5.plot(tm, (a.outcome == 'timeout').cumsum(), color='#c0392b', lw=2,
            label='cumulative timeouts')
    a5.axvline(tm.iloc[int(mid)], color='0.5', ls=':', lw=1.2, label='half split')
    a5.set_xlabel('session time (min)'); a5.set_ylabel('cumulative count')
    a5.set_title('EARNING CURVE', fontsize=10.5, fontweight='bold')
    a5.legend(fontsize=7.5); a5.grid(alpha=0.2)

    axs = fig.add_subplot(gs[2, :]); axs.axis('off')
    rows = [
        ['rewards per SECOND (wall)', *[f"{blocks[k]['rew_wall']/60:.4f}" for k in labs], 'throughput'],
        ['rewards per WALL minute', *[f"{blocks[k]['rew_wall']:.2f}" for k in labs], 'throughput'],
        ['rewards per ACTIVE minute', *[f"{blocks[k]['rew_active']:.2f}" for k in labs], 'throughput'],
        ['reward DROPS per wall minute', *[f"{blocks[k]['units_wall']:.2f}" for k in labs], 'throughput'],
        ['n single_reward', *[f"{blocks[k]['single']}" for k in labs], 'count'],
        ['n double_reward', *[f"{blocks[k]['double']}" for k in labs], 'count'],
        ['n TIMEOUT (negative)', *[f"{blocks[k]['neg']}" for k in labs], 'count'],
        ['total reward DROPS (1x single + 2x double)', *[f"{blocks[k]['units']}" for k in labs], 'count'],
        ['P(hit POSITIVE | collected)', *[f"{blocks[k]['acc']:.3f}" for k in labs], 'CHOICE'],
        ['P(hit TIMEOUT | collected)', *[f"{1-blocks[k]['acc']:.3f}" for k in labs], 'CHOICE'],
        ['  chance from VISIBILITY', *[f"{blocks[k]['chance']:.3f}" for k in labs], 'CHOICE'],
        ['  DISCRIMINATION SCORE  D', *[f"{blocks[k]['D']:+.3f}" for k in labs], 'CHOICE'],
        ['  p vs chance (binomial)', *[f"{blocks[k]['p']:.3f}" for k in labs], 'CHOICE'],
        ['  DISCRIMINATION SCORE  D (perfect memory)',
                                        *[f"{blocks[k]['D_mem']:+.3f}" for k in labs], 'CHOICE'],
        ['minutes lost to timeout FREEZE', *[f"{blocks[k]['freeze_min']:.1f}" for k in labs], 'efficiency'],
        ['median ACTIVE trial duration (s)', *[f"{blocks[k]['med_dur']:.1f}" for k in labs], 'efficiency'],
        ['path efficiency', *[f"{blocks[k]['eff']:.2f}" for k in labs], 'efficiency'],
    ]
    tab = axs.table(cellText=[r[:4] for r in rows],
                    colLabels=['parameter', 'first half', 'second half', 'WHOLE SESSION'],
                    colWidths=[0.40, 0.20, 0.20, 0.20], loc='upper center', cellLoc='center')
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.18)
    grp_col = {'throughput': '#fdf3e3', 'count': '#eef4fa', 'CHOICE': '#e9f7ef',
               'efficiency': '#f4f0f7'}
    for (rr, cc), cell in tab.get_celld().items():
        cell.set_edgecolor('#ccc')
        if rr == 0:
            cell.set_facecolor('#34495e'); cell.set_text_props(color='w', fontweight='bold'); continue
        cell.set_facecolor(grp_col[rows[rr - 1][4]])
        if cc == 0:
            cell.set_text_props(ha='left')
        if 'DISCRIMINATION' in rows[rr - 1][0]:
            cell.set_text_props(fontweight='bold')
    axs.set_title('SCORECARD  -  green = CHOICE (the learning metric), orange = THROUGHPUT, '
                  'blue = counts, purple = EFFICIENCY', fontsize=10.5, fontweight='bold', pad=14)

    verdict = ('CHOICE IS AT CHANCE: he collects in proportion to what is on screen.'
               if B['p'] > 0.05 else
               'CHOICE IS ABOVE CHANCE: he is avoiding the timeout.')
    fig.text(0.5, 0.028, f"VERDICT  {verdict}", ha='center', va='bottom', fontsize=11,
             color='#c0392b' if B['p'] > 0.05 else '#27ae60', fontweight='bold')
    fig.text(0.5, 0.006, f"accuracy {f['acc']:.3f} -> {s_['acc']:.3f} (Fisher p={p_acc:.3f});  "
             f"active trial {f['med_dur']:.1f}s -> {s_['med_dur']:.1f}s (p={p_dur:.3f});  "
             f"{B['freeze_min']:.0f} min lost to timeout penalties.",
             ha='center', va='bottom', fontsize=9.5, color='#333')
    fig.suptitle(f"{sess['mouse_id']} PERFORMANCE SCORECARD  ({ts.TASK})\n"
                 f"discrimination is scored against a VISIBILITY-WEIGHTED chance baseline "
                 f"(positive icons on screen {B['vis_pos']:.2f} vs negative {B['vis_neg']:.2f} "
                 f"per frame), not the 2:1 icon count\n"
                 f"reward DROPS (1x single + 2x double, verified against the delivery log) are "
                 f"OUT of accuracy and IN throughput; both WALL and ACTIVE rates are shown",
                 fontsize=11, fontweight='bold', y=0.985)
    out = d / 'debug' / 'performance.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n  wrote {out}")
    return blocks


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
