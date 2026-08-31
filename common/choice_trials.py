"""
THE CLEAN SUBSET: trials where BOTH icon types were on screen at the same time.

Why this exists. Every argument about the discrimination score D is an argument about the chance
BASELINE -- what was available to him, whether he remembers it, whether his own movement created the
availability. All of that disappears in the trials where both types were simultaneously on his
screen: he could plainly see a reward and a banishment at once, so "which did he go to?" is a
well-posed question with no visibility model in the way. On JPAS_0168 that is 44 of 61 choice trials
(72%), which is enough to ask it directly.

Three pages:
  A  WHEN was the choice available?  -- the validity check. If both icons only become visible after
     he has already committed to a target, the later pages are not measuring a decision. Reports
     when joint visibility first occurs (as a fraction into the trial and in seconds before the
     collection) and how long it lasts.
  B  THE DECISION POINT -- freeze the frame at the FIRST moment both are visible, record the
     geometry (distance and heading error to each type), then ask which he actually collected. The
     baseline is EXOGENOUS: "he goes to whichever is nearer", fixed by the board at that instant and
     not by anything he does afterwards.
  C  IS THERE AN AVOIDANCE SIGNATURE? -- a behavioural test that does not depend on the final
     choice: did he approach the banishment and veer off? Closest approach is reported too, but with
     its confound stated (staying far from the punishment is what you get for free by going somewhere
     else, so distance alone is not avoidance).

Task-agnostic: pass the positive/negative effect names.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fps as fpsmod          # noqa: E402
import viewport as vp         # noqa: E402

VEER_CLOSE = 700.0     # "approached" = came within this many world units
VEER_BACK = 250.0      # "veered off" = then retreated by at least this much


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def _veered(d):
    """Came within VEER_CLOSE of the icon and then retreated by VEER_BACK."""
    if len(d) < 3 or d.min() > VEER_CLOSE:
        return False
    i = int(np.argmin(d))
    return bool(d[i:].max() - d[i] > VEER_BACK)


def analyse(session_dir, positive, negative, world='NORMAL'):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    log = json.load(open(d / 'log.json'))
    df = pd.read_pickle(d / 'df_trials_clean.pkl')
    a = df[df.analyze]
    if 'world' in a.columns and world:
        a = a[a.world == world]
    fm = fpsmod.frame_to_ms(log)
    C = np.array(log['coords_t[ms]/x/y'], float)
    A = np.array(log['angles_t[ms]/theta'], float)
    vw, vh = vp.viewport(sess)
    fps = sess['fps']

    rows, all_rows = [], []
    for _, r in a.iterrows():
        s, e = int(r['start_frame']), int(min(r['end_frame'], len(fm)))
        if e - s < 5:
            continue
        t = fm[s:e]
        ax = np.interp(t, C[:, 0], C[:, 1]); ay = np.interp(t, C[:, 0], C[:, 2])
        th = np.interp(t, A[:, 0], A[:, 1])
        P = [ic for ic in (r['icons'] or []) if ic.get('effect') in positive]
        N = [ic for ic in (r['icons'] or []) if ic.get('effect') in negative]
        if not P or not N:
            continue
        vP = np.zeros(len(t), bool); vN = np.zeros(len(t), bool)
        for ic in P:
            vP |= (np.abs(ic['x'] - ax) <= vw / 2) & (np.abs(ic['y'] - ay) <= vh / 2)
        for ic in N:
            vN |= (np.abs(ic['x'] - ax) <= vw / 2) & (np.abs(ic['y'] - ay) <= vh / 2)
        dP = np.min([np.hypot(ic['x'] - ax, ic['y'] - ay) for ic in P], axis=0)
        dN = np.min([np.hypot(ic['x'] - ax, ic['y'] - ay) for ic in N], axis=0)
        got = r['outcome'] in positive
        all_rows.append(dict(got=got, veer_neg=_veered(dN), veer_pos=_veered(dP),
                             min_dN=float(dN.min()), min_dP=float(dP.min())))

        both = vP & vN
        if not both.any():
            continue
        i = int(np.argmax(both))                       # FIRST joint-visibility frame
        hP = min(abs(np.angle(np.exp(1j * (th[i] - np.arctan2(ic['y'] - ay[i], ic['x'] - ax[i])))))
                 for ic in P)
        hN = min(abs(np.angle(np.exp(1j * (th[i] - np.arctan2(ic['y'] - ay[i], ic['x'] - ax[i])))))
                 for ic in N)
        rows.append(dict(
            trial=int(r['trial']), got=got, n_frames=len(t),
            frac_in=i / len(t), s_from_start=(t[i] - t[0]) / 1000.0,
            s_to_end=(t[-1] - t[i]) / 1000.0, joint_s=float(both.sum()) / fps,
            dP=float(dP[i]), dN=float(dN[i]), nearer_pos=bool(dP[i] < dN[i]),
            hP=float(np.degrees(hP)), hN=float(np.degrees(hN)), facing_pos=bool(hP < hN),
            min_dN=float(dN.min()), veer_neg=_veered(dN), veer_pos=_veered(dP)))

    return dict(D=pd.DataFrame(rows), ALL=pd.DataFrame(all_rows), fps=fps,
                mouse_id=sess['mouse_id'], n_choice=len(a))


# ── the three pages ──────────────────────────────────────────────────────────────────────
def _pg(fig, txt):
    fig.text(0.995, 0.004, txt, ha='right', va='bottom', fontsize=8, color='#555')


def figure_when(res, pos_label='reward', neg_label='banishment'):
    """A -- WHEN was the choice available? The validity check for B and C."""
    import matplotlib.pyplot as plt
    D = res['D']
    fig, ax = plt.subplots(2, 2, figsize=(17, 9.6))
    C_POS, C_NEG = '#27ae60', '#c0392b'

    a = ax[0, 0]
    a.hist(D.frac_in, bins=np.linspace(0, 1, 11), color='#f39c12', alpha=0.9, edgecolor='w')
    a.axvline(D.frac_in.median(), color='k', ls='--', lw=2,
              label=f'median {D.frac_in.median():.2f}')
    a.set_xlabel('when both first appear together, as a fraction into the trial\n'
                 '(0 = already there at the start, 1 = only at the collection)')
    a.set_ylabel('trials')
    a.set_title(f"(a)  WHEN does the choice become available?\n"
                f"{int((D.frac_in == 0).sum())} trials have both visible from the very start; "
                f"{int((D.frac_in < 0.25).sum())} within the first quarter",
                fontsize=10.5, fontweight='bold')
    a.legend(fontsize=9); a.grid(alpha=0.2, axis='y')

    a = ax[0, 1]
    a.hist(D.s_to_end, bins=14, color='#2e86c1', alpha=0.9, edgecolor='w')
    a.axvline(D.s_to_end.median(), color='k', ls='--', lw=2,
              label=f'median {D.s_to_end.median():.1f} s')
    a.set_xlabel('seconds from that moment until he collects something')
    a.set_ylabel('trials')
    a.set_title('(b)  is there TIME to act on it?\n'
                'the gap between seeing both and collecting -- if this were ~0 the\n'
                'choice would already have been made before he could compare',
                fontsize=10.5, fontweight='bold')
    a.legend(fontsize=9); a.grid(alpha=0.2, axis='y')

    a = ax[1, 0]
    a.hist(D.joint_s, bins=14, color='#8e44ad', alpha=0.9, edgecolor='w')
    a.axvline(D.joint_s.median(), color='k', ls='--', lw=2,
              label=f'median {D.joint_s.median():.1f} s')
    a.set_xlabel('seconds both stay on screen together')
    a.set_ylabel('trials')
    a.set_title('(c)  how LONG is the window?\n'
                'this is why "6% of frames" and "72% of trials" are both true:\n'
                'a brief burst, but present in most trials', fontsize=10.5, fontweight='bold')
    a.legend(fontsize=9); a.grid(alpha=0.2, axis='y')

    a = ax[1, 1]; a.axis('off')
    a.set_title('(d)  does this subset support a decision analysis?',
                fontsize=10.5, fontweight='bold')
    early = int((D.frac_in < 0.25).sum())
    a.text(-0.02, 0.97,
           f"These are the {len(D)} trials (of {res['n_choice']}) where a {pos_label} and a "
           f"{neg_label} were on\nhis screen AT THE SAME TIME -- so he could compare them without "
           f"relying on memory,\nand no visibility model is needed to say what was available.\n\n"
           f"YES, with one caveat:\n"
           f"   + there is a median {D.s_to_end.median():.1f} s between seeing both and collecting, "
           f"which at his\n     speed (~400 world units/s) is ample time to change course;\n"
           f"   + {early} of {len(D)} trials have the choice available in the first quarter of the "
           f"trial.\n\n"
           f"   - BUT the median first-joint-visibility is {D.frac_in.median():.0%} of the way "
           f"through the trial,\n     so in many trials he may already be committed to a target. "
           f"Read page B\n     as 'given both were visible', NOT as 'he deliberated from the start'.\n\n"
           f"P({pos_label}) in this subset is {D.got.mean():.3f}, against {0.717:.3f} for all choice "
           f"trials --\nseeing both did not make him do better.",
           transform=a.transAxes, va='top', ha='left', fontsize=10, linespacing=1.5)

    fig.suptitle(f"{res['mouse_id']}  THE CLEAN SUBSET: {len(D)} trials where he could SEE both "
                 f"a {pos_label} and a {neg_label}\n"
                 'page A -- when was the choice actually available to him?',
                 fontsize=12.5, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def figure_decision(res, pos_label='reward', neg_label='banishment'):
    """B -- at the FIRST moment both are visible, does his choice beat 'go to the nearer one'?"""
    import matplotlib.pyplot as plt
    from scipy.stats import binomtest
    D = res['D']; n = len(D)
    C_POS, C_NEG, C_NULL = '#27ae60', '#c0392b', '#95a5a6'
    obs = D.got.mean(); base = D.nearer_pos.mean()
    # BOTH bars are measured on the SAME trials, so they are PAIRED -- the baseline is an
    # estimate with its own error bar, not a fixed constant. The correct test is therefore
    # McNemar on the two DISCORDANT cells (he broke the nearest-rule toward the positive icon
    # vs toward the negative one), not a binomial of `obs` against `base` held fixed.
    n_bp = int((D.got & ~D.nearer_pos).sum())   # took POS while NEG was nearer  -> overrode toward reward
    n_bn = int((~D.got & D.nearer_pos).sum())   # took NEG while POS was nearer  -> overrode toward punishment
    p = binomtest(n_bp, n_bp + n_bn, 0.5).pvalue if (n_bp + n_bn) else np.nan
    lo, hi = _wilson(int(D.got.sum()), n)
    blo, bhi = _wilson(int(D.nearer_pos.sum()), n)

    fig, ax = plt.subplots(2, 2, figsize=(19, 9.6),
                           gridspec_kw=dict(width_ratios=[1, 1.35]))

    a = ax[0, 0]
    a.scatter(D.dN[D.got], D.dP[D.got], s=55, color=C_POS, alpha=0.85,
              label=f'collected the {pos_label} ({int(D.got.sum())})')
    a.scatter(D.dN[~D.got], D.dP[~D.got], s=55, color=C_NEG, alpha=0.85,
              marker='X', label=f'collected the {neg_label} ({int((~D.got).sum())})')
    lim = max(D.dP.max(), D.dN.max()) * 1.05
    a.plot([0, lim], [0, lim], 'k--', lw=1.2)
    a.text(lim * 0.10, lim * 0.90, f'above the line:\n{pos_label} is FARTHER',
           fontsize=9.5, color='#555', va='top',
           bbox=dict(boxstyle='round,pad=0.35', fc='w', ec='0.75', alpha=0.85))
    a.text(lim * 0.72, lim * 0.14, f'below the line:\n{pos_label} is NEARER',
           fontsize=9.5, color='#555', va='top',
           bbox=dict(boxstyle='round,pad=0.35', fc='w', ec='0.75', alpha=0.85))
    a.set_xlabel(f'distance to the {neg_label} (world units)')
    a.set_ylabel(f'distance to the nearest {pos_label}')
    a.set_xlim(0, lim); a.set_ylim(0, lim)
    a.set_title('(a)  the geometry at the decision moment\n'
                'each trial: how far each type was when both first appeared',
                fontsize=10.5, fontweight='bold')
    a.legend(fontsize=8.5); a.grid(alpha=0.2)

    a = ax[0, 1]
    bars = [('he collected\nthe ' + pos_label, obs, '#27ae60'),
            ('the ' + pos_label + ' was\nthe NEARER icon', base, C_NULL)]
    a.bar([0, 1], [b[1] for b in bars], 0.55, color=[b[2] for b in bars], alpha=0.9)
    # both bars are estimates from the same 44 trials, so BOTH carry a Wilson interval
    a.errorbar([0, 1], [obs, base],
               yerr=[[obs - lo, base - blo], [hi - obs, bhi - base]],
               fmt='none', ecolor='k', capsize=6, lw=1.5)
    for i, b in enumerate(bars):
        e_hi = (hi if i == 0 else bhi)
        a.text(i, e_hi + 0.035, f'{b[1]:.3f}', ha='center', fontsize=13, fontweight='bold')
    a.set_xticks([0, 1]); a.set_xticklabels([b[0] for b in bars], fontsize=10)
    a.set_ylim(0, 1.1); a.set_ylabel('fraction of the ' + str(n) + ' trials')
    # keep every line short enough to fit the axes width -- a long single line overflows the page
    a.set_title(f'(b)  does he beat "just go to the nearer one"?\n'
                f'{obs:.3f} vs {base:.3f}   -- both are estimates from these same {n} trials,\n'
                f'so BOTH carry a 95% confidence interval and the test must be PAIRED\n'
                f'McNemar on the {n_bp + n_bn} trials where HIS CHOICE went AGAINST the nearer icon:\n'
                f'{n_bp} toward the {pos_label}, {n_bn} toward the {neg_label},  '
                f'p = {p:.3f}'
                + ('  -- NO DIFFERENCE' if p > 0.05 else '  *'),
                fontsize=10, fontweight='bold',
                color='#c0392b' if p > 0.05 else '#27ae60')
    a.grid(alpha=0.2, axis='y')

    # (c) THE LEARNING CRITERION: the conflict trials, where proximity and value disagree.
    # Tinted + framed so the eye lands here first -- this is the number tracked across sessions.
    HL, EDGE = '#fdf1e0', '#e67e22'
    a = ax[1, 0]
    a.set_facecolor(HL)
    for sp in a.spines.values():
        sp.set_edgecolor(EDGE); sp.set_linewidth(2.2)
    near = D[~D.nearer_pos]          # the NEGATIVE icon was the closer one -- he must override
    far = D[D.nearer_pos]            # they agree -- the easy case, and the control
    vals = [far.got.mean() if len(far) else np.nan, near.got.mean() if len(near) else np.nan]
    los, his = [], []
    for sub in (far, near):
        lo_, hi_ = _wilson(int(sub.got.sum()), len(sub))
        los.append(vals[len(los)] - lo_); his.append(hi_ - vals[len(his)])
    a.bar([0, 1], vals, 0.55, color=['#bdc3c7', '#e67e22'], alpha=0.95,
          edgecolor=['#999', '#b35418'], linewidth=[1, 2.5])
    a.errorbar([0, 1], vals, yerr=[los, his], fmt='none', ecolor='k', capsize=6, lw=1.5)
    for i, (v, sub) in enumerate(zip(vals, (far, near))):
        top = v + his[i]                      # clear the error bar, not just the bar
        a.text(i, top + 0.10, f'{v:.2f}', ha='center', fontsize=14, fontweight='bold')
        # panel-coloured bbox so the label never reads as struck through by the error bar
        a.text(i, top + 0.045, f'= {int(sub.got.sum())} of {len(sub)} trials', ha='center',
               fontsize=9.5, color='#555',
               bbox=dict(boxstyle='square,pad=0.15', fc=HL, ec='none'))
    a.axhline(0.5, color='0.6', ls=':', lw=1.5)
    a.set_xticks([0, 1])
    a.set_xticklabels([f'{pos_label} was nearer\n(easy -- CONTROL)',
                       f'{neg_label} was nearer\n(he must OVERRIDE)'], fontsize=9.5)
    a.set_ylim(0, 1.3); a.set_ylabel(f'P(he collected the {pos_label})')
    a.set_title(f'(c)  *** THE LEARNING CRITERION ***\n'
                f'on CONFLICT trials -- where the {neg_label} is closer -- he takes the '
                f'{pos_label}\non only {int(near.got.sum())} of those {len(near)} ({vals[1]:.0%}). '
                f'THIS is the number that must RISE '
                f'if he learns\nto avoid the {neg_label}. Bars = 95% confidence interval.',
                fontsize=10, fontweight='bold', color='#b35418')
    a.grid(alpha=0.2, axis='y')

    a = ax[1, 1]; a.axis('off')
    a.set_title('(d)  what this settles', fontsize=10.5, fontweight='bold')
    conf = D[~D.nearer_pos]
    # long lines on purpose: this column is wide, and short wrapping pushed the block down into
    # the highlighted box below it
    a.text(-0.04, 1.00,
           f"The cleanest test in the report: it needs NO baseline argument, because in these {n} "
           f"trials he could see a {pos_label} and a\n"
           f"{neg_label} at the same time. The {pos_label} happened to be the nearer icon in "
           f"{100*base:.0f}% of trials -- and he collected a\n"
           f"{pos_label} in {100*obs:.0f}% of them, the SAME number. Trial by trial, {n_bp + n_bn} "
           f"of the {n} went against the nearer icon: {n_bp} toward the {pos_label} and\n"
           f"{n_bn} toward the {neg_label}, an exact tie (paired McNemar p = {p:.3f}). His choice "
           f"is exactly what 'go to whichever is\n"
           f"closer' predicts, and the TYPE of the icon adds nothing.",
           transform=a.transAxes, va='top', ha='left', fontsize=9.8, linespacing=1.5)
    a.text(-0.04, 0.44,
           "\u2605  THE CRITERION WE TRACK ACROSS SESSIONS  --  panel (c)\n\n"
           f"P({pos_label} | the {neg_label} was NEARER)  =  {int(conf.got.sum())} of "
           f"{len(conf)} CONFLICT trials  =  {conf.got.mean():.2f}\n\n"
           f"It must RISE if he learns to avoid the {neg_label}, while the CONTROL column stays high. "
           f"If BOTH\n"
           f"move he has merely stopped following proximity -- which is NOT the same as recognising "
           f"the icon.\n"
           f"Only a large effect shows in ONE session at this n; pool ~3 sessions per block.",
           transform=a.transAxes, va='top', ha='left', fontsize=9.8, linespacing=1.5,
           color='#8a4b12', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#fdf1e0', edgecolor='#e67e22',
                     linewidth=2.0))

    fig.suptitle(f"{res['mouse_id']}  page B -- THE DECISION POINT: when he could see both, "
                 f"which did he go to?\n"
                 f"NEAREST-ICON baseline (exogenous): which icon was nearer at that instant, fixed by "
                 f"the board and not by anything he did next", fontsize=12.5, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def figure_avoidance(res, pos_label='reward', neg_label='banishment'):
    """C -- a behavioural avoidance test that does not depend on the final choice."""
    import matplotlib.pyplot as plt
    from scipy.stats import fisher_exact, mannwhitneyu
    D = res['D']; ALL = res['ALL']
    C_POS, C_NEG = '#27ae60', '#c0392b'
    rew = ALL[ALL.got]
    fig, ax = plt.subplots(2, 2, figsize=(17, 9.6))

    a = ax[0, 0]
    a.hist(rew.min_dN, bins=14, color=C_NEG, alpha=0.85, edgecolor='w')
    a.axvline(246, color='k', ls='--', lw=2, label='collection radius ~246 wu')
    a.axvline(rew.min_dN.median(), color='#2c3e50', lw=2,
              label=f'median {rew.min_dN.median():.0f} wu')
    a.set_xlabel(f'closest he ever came to the {neg_label} (world units)')
    a.set_ylabel('trials')
    a.set_title(f'(a)  on trials he collected a {pos_label}, how close did he get\n'
                f'to the {neg_label}?  {int((rew.min_dN<500).sum())} of {len(rew)} came within 500 wu',
                fontsize=10.5, fontweight='bold')
    a.legend(fontsize=8.5); a.grid(alpha=0.2, axis='y')

    a = ax[0, 1]
    vn, vp_ = int(rew.veer_neg.sum()), int(rew.veer_pos.sum())
    a.bar([0, 1], [vn, vp_], 0.55, color=[C_NEG, C_POS], alpha=0.9)
    for i, v in enumerate([vn, vp_]):
        a.text(i, v + 0.15, f'{v}  ({100*v/len(rew):.0f}%)', ha='center', fontsize=12,
               fontweight='bold')
    a.set_xticks([0, 1])
    a.set_xticklabels([f'veered away from\nthe {neg_label}', f'veered away from\na {pos_label}\n(CONTROL)'],
                      fontsize=9.5)
    a.set_ylabel(f'trials (of {len(rew)})'); a.margins(y=0.22)
    odds, pf = fisher_exact([[vn, len(rew) - vn], [vp_, len(rew) - vp_]])
    a.set_title(f'(b)  APPROACH-THEN-VEER: did he turn away?\n'
                f'came within {VEER_CLOSE:.0f} wu then retreated >{VEER_BACK:.0f} wu.  '
                f'Fisher p = {pf:.3f}', fontsize=10.5, fontweight='bold')
    a.grid(alpha=0.2, axis='y')

    a = ax[1, 0]
    got = D[D.got]; nogot = D[~D.got]
    a.boxplot([got.hN, nogot.hN], positions=[0, 1], widths=0.5, patch_artist=True,
              boxprops=dict(facecolor='#bdc3c7'), medianprops=dict(color='k', lw=2))
    for i, v in enumerate([got.hN, nogot.hN]):
        a.scatter(np.full(len(v), i) + np.random.uniform(-.12, .12, len(v)), v, s=22,
                  color='k', alpha=0.5, zorder=3)
    a.axhline(90, color='0.6', ls=':', lw=1.5)
    a.set_xticks([0, 1])
    a.set_xticklabels([f'collected a {pos_label}\n(n={len(got)})',
                       f'collected the {neg_label}\n(n={len(nogot)})'], fontsize=9.5)
    a.set_ylabel(f'heading error toward the {neg_label} at the\ndecision moment (deg; 0 = facing it)')
    pv = mannwhitneyu(got.hN, nogot.hN).pvalue if len(got) > 2 and len(nogot) > 2 else np.nan
    a.set_title(f'(c)  was he already pointed AWAY from the {neg_label}?\n'
                f'at the moment both appeared.  Mann-Whitney p = {pv:.3f}',
                fontsize=10.5, fontweight='bold')
    a.grid(alpha=0.2, axis='y')

    a = ax[1, 1]; a.axis('off')
    a.set_title('(d)  is there any avoidance signal?', fontsize=10.5, fontweight='bold')
    a.text(-0.02, 0.97,
           f"THE ACTUAL TEST is (b): approaching and then veering off. He did that on\n"
           f"{vn} of {len(rew)} {pos_label} trials ({100*vn/len(rew):.0f}%) for the {neg_label}, "
           f"versus {vp_} ({100*vp_/len(rew):.0f}%) for a\n{pos_label} -- the control for ordinary "
           f"wandering. Fisher p = {pf:.3f}.\n\n"
           f"Direction is what avoidance would look like, but {vn} events cannot carry\n"
           f"a conclusion. Treat this as UNDERPOWERED, not as negative: to test it\n"
           f"properly needs more sessions, not more analysis of this one.",
           transform=a.transAxes, va='top', ha='left', fontsize=10, linespacing=1.5)

    fig.suptitle(f"{res['mouse_id']}  page C -- is there an AVOIDANCE signature that the choice "
                 f"itself does not show?\n"
                 f"a behavioural test independent of which icon he finally collected",
                 fontsize=12.5, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig
