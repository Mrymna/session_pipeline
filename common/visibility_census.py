"""
VISIBILITY CENSUS -- the motivating page: what is actually on the mouse's screen, frame by frame.

This exists to answer "why bother with a visibility-weighted baseline at all?". The answer is that
the animal almost never sees the whole board: on JPAS_0168 he sees NOTHING for 45% of frames and
both icon types together for only 6%. A raw hit-rate silently assumes he was choosing between all
the icons all the time; this page shows how far from true that is.

Task-agnostic: pass the positive/negative effect names.
"""
import json
import numpy as np


def census(session_dir, positive, negative, world='NORMAL'):
    """Per-frame counts of visible positive / negative icons over the choice trials."""
    from pathlib import Path
    import pandas as pd
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fps as fpsmod
    import viewport as vp

    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    log = json.load(open(d / 'log.json'))
    df = pd.read_pickle(d / 'df_trials_clean.pkl')
    a = df[df.analyze]
    if 'world' in a.columns and world:
        a = a[a.world == world]
    fm = fpsmod.frame_to_ms(log)
    C = np.array(log['coords_t[ms]/x/y'], float)
    vw, vh = vp.viewport(sess)

    nP, nN = [], []
    tri = dict(both=0, pos_only=0, neg_only=0, neither=0, simultaneous=0, n=0,
               seq_pos_first=0, seq_neg_first=0)
    for _, r in a.iterrows():
        s, e = int(r['start_frame']), int(min(r['end_frame'], len(fm)))
        if e - s < 5:
            continue
        t = fm[s:e]
        ax = np.interp(t, C[:, 0], C[:, 1]); ay = np.interp(t, C[:, 0], C[:, 2])
        vp_, vn_ = np.zeros(len(t)), np.zeros(len(t))
        for ic in (r['icons'] or []):
            if ic.get('effect') not in set(positive) | set(negative):
                continue
            m = (np.abs(ic['x'] - ax) <= vw / 2) & (np.abs(ic['y'] - ay) <= vh / 2)
            if ic['effect'] in positive:
                vp_ += m
            else:
                vn_ += m
        nP.append(vp_); nN.append(vn_)
        # trial-level: what did he see AT ANY POINT during this trial? (the frame view says what
        # is on screen at an instant; this says what the trial as a whole made available)
        R, N_ = vp_ > 0, vn_ > 0
        tri['n'] += 1
        nboth = int((R & N_).sum())
        if nboth:
            tri.setdefault('both_frames', []).append(nboth)
        tri.setdefault('trial_frames', []).append(len(R))
        tri['simultaneous'] += bool((R & N_).any())
        if R.any() and N_.any():
            tri['both'] += 1
            # could he COMPARE them directly, or only one at a time (which needs memory)?
            if not (R & N_).any():
                if np.argmax(R) < np.argmax(N_):
                    tri['seq_pos_first'] += 1
                else:
                    tri['seq_neg_first'] += 1
        elif R.any():
            tri['pos_only'] += 1
        elif N_.any():
            tri['neg_only'] += 1
        else:
            tri['neither'] += 1
    P = np.concatenate(nP) if nP else np.array([])
    N = np.concatenate(nN) if nN else np.array([])
    # session-level context: the census covers ONLY the choice trials, and a reader needs to know
    # what fraction of the session that is -- otherwise "6% of frames" gets read as 6% of the day.
    all_min = sess['n_frames'] / sess['fps'] / 60
    dur = (df.end_frame - df.start_frame) / sess['fps'] / 60
    an = df[df.analyze]
    other_min = float(dur[df.analyze & (df.world != world)].sum()) if 'world' in df.columns else 0.0
    non_min = float(dur[~df.analyze].sum())
    return dict(P=P, N=N, trials=tri, n_frames=len(P), fps=sess['fps'], mouse_id=sess['mouse_id'],
                view_w=vw, view_h=vh, session_min=all_min,
                census_min=len(P) / sess['fps'] / 60,
                other_world_min=other_min, non_analyzable_min=non_min,
                world_w=sess['world_width'], world_h=sess['world_height'])


def figure(cs, pos_label='reward', neg_label='banishment'):
    import matplotlib.pyplot as plt
    P, N = cs['P'], cs['N']
    cats = [('nothing at all', (P == 0) & (N == 0), '#4a4a4a'),
            (f'{pos_label}(s) only', (P > 0) & (N == 0), '#27ae60'),
            (f'{neg_label} only', (P == 0) & (N > 0), '#c0392b'),
            (f'BOTH -- a real choice', (P > 0) & (N > 0), '#f39c12')]
    frac = [c[1].mean() * 100 for c in cats]

    fig, axg = plt.subplots(2, 2, figsize=(19, 10.5),
                            gridspec_kw=dict(width_ratios=[1, 1.45]))
    ax = axg.ravel()

    a = ax[0]
    wedges, _, txt = a.pie(frac, colors=[c[2] for c in cats], autopct='%1.1f%%',
                           startangle=90, textprops=dict(color='w', fontsize=11,
                                                         fontweight='bold'),
                           wedgeprops=dict(edgecolor='w', linewidth=1.5))
    a.legend(wedges, [c[0] for c in cats], fontsize=9.5, loc='center right',
             bbox_to_anchor=(-0.02, 0.5), frameon=False)
    a.set_title('(a)  what is on his screen, frame by frame\n'
                f"the {cs['trials']['n']} CHOICE trials ADDED TOGETHER = "
                f"{cs['census_min']:.0f} min ({cs['n_frames']} frames)\n"
                f"they are scattered through the {cs['session_min']:.0f} min session, "
                f"not one block",
                fontsize=10.5, fontweight='bold')

    a = ax[1]
    tr0 = cs['trials']
    tot0 = max(tr0['n'], 1)
    comp = [('both AT THE SAME TIME\n(a direct comparison)', tr0['simultaneous'], '#f39c12'),
            (f'one at a time: {pos_label} first\n(comparing needs MEMORY)',
             tr0['seq_pos_first'], '#f7c66b'),
            (f'one at a time: {neg_label} first\n(comparing needs MEMORY)',
             tr0['seq_neg_first'], '#f7c66b'),
            (f'only {pos_label}, ever\n(no choice to make)', tr0['pos_only'], '#27ae60'),
            (f'only {neg_label}, ever\n(no choice to make)', tr0['neg_only'], '#c0392b'),
            ('neither, ever', tr0['neither'], '#4a4a4a')]
    yy = np.arange(len(comp))
    a.barh(yy, [c[1] for c in comp], color=[c[2] for c in comp], alpha=0.9)
    for i, c in enumerate(comp):
        a.text(c[1] + 0.7, i, f'{c[1]}  ({100*c[1]/tot0:.0f}%)', va='center', fontsize=10,
               fontweight='bold')
    a.set_yticks(yy); a.set_yticklabels([c[0] for c in comp], fontsize=8.5)
    a.invert_yaxis(); a.set_xlabel('number of trials')
    a.set_xlim(0, max(c[1] for c in comp) * 1.32)
    a.set_title('(b)  could he COMPARE the two types at all?\n'
                f"in {100*tr0['simultaneous']/tot0:.0f}% of trials both were on screen together;\n"
                f"{100*(tr0['seq_pos_first']+tr0['seq_neg_first'])/tot0:.0f}% only one at a time, "
                f"{100*(tr0['pos_only']+tr0['neg_only'])/tot0:.0f}% only ever one type",
                fontsize=10.5, fontweight='bold')
    a.grid(alpha=0.2, axis='x')

    # (c) the TRIAL view -- same pie treatment as (a), so the two are directly comparable
    tr = cs['trials']
    a = ax[2]
    tcats = [('both, at some point', tr['both'], '#f39c12'),
             (f'only {pos_label} ever', tr['pos_only'], '#27ae60'),
             (f'only {neg_label} ever', tr['neg_only'], '#c0392b'),
             ('neither ever', tr['neither'], '#4a4a4a')]
    shown = [c for c in tcats if c[1] > 0]      # a 0-size wedge just clutters the pie
    tot = max(tr['n'], 1)
    wedges2, _, _ = a.pie([c[1] for c in shown], colors=[c[2] for c in shown],
                          autopct=lambda pct: f'{pct:.0f}%\n({round(pct*tot/100)})',
                          startangle=90, pctdistance=0.74,
                          textprops=dict(color='w', fontsize=10.5, fontweight='bold'),
                          wedgeprops=dict(edgecolor='w', linewidth=1.5))
    zero = [c[0] for c in tcats if c[1] == 0]
    a.legend(wedges2, [c[0] for c in shown], fontsize=9.5, loc='center right',
             bbox_to_anchor=(-0.02, 0.5), frameon=False)
    a.set_title(f"(c)  per TRIAL: what was available at ANY point\n"
                f"in {100*tr['both']/tot:.0f}% of the {tr['n']} choice trials he met BOTH types\n"
                f"({100*tr['simultaneous']/tot:.0f}% saw them on screen at the SAME time"
                + (f";  {', '.join(zero)}: 0)" if zero else ")"),
                fontsize=10.5, fontweight='bold')

    a = ax[3]; a.axis('off')
    a.set_title('(d)  why this matters', fontsize=12, fontweight='bold')
    tr1 = cs['trials']; t1 = max(tr1['n'], 1)
    tri_bf = np.array(tr1.get('both_frames', [0]))
    tri_tf = np.array(tr1.get('trial_frames', [1]))
    # longer lines, bigger type, fewer of them -- the narrow hand-wrapped column wasted the panel
    a.text(-0.02, 0.98,
           f"The board always holds the same icons, but his SCREEN shows only a "
           f"{cs['view_w']:.0f} x {cs['view_h']:.0f} window\n"
           f"of the {cs['world_w']:.0f} x {cs['world_h']:.0f} world, centred on him. "
           f"So what he can choose between keeps changing as he moves.\n\n"
           f"AT ANY INSTANT he sees NOTHING for {frac[0]:.0f}% of the time and both types together "
           f"for only {frac[3]:.0f}%.\n"
           f"OVER A WHOLE TRIAL, though, he meets both types in {100*tr1['both']/t1:.0f}% of trials. "
           f"Those are not in conflict:\n"
           f"the both-visible frames come in SHORT BURSTS -- about "
           f"{np.median(tri_bf)/cs['fps']:.1f} s inside a typical {np.median(tri_tf)/cs['fps']:.0f} s "
           f"trial -- and a\ntrial counts here if it contains even ONE such frame. Brief in "
           f"DURATION, present in most TRIALS.\n\n"
           f"Panel (b) sharpens that: in {100*tr1['simultaneous']/t1:.0f}% of trials the two were on "
           f"screen TOGETHER, so no memory was\n"
           f"needed to compare them; only "
           f"{100*(tr1['seq_pos_first']+tr1['seq_neg_first'])/t1:.0f}% required holding one in mind, "
           f"and {100*(tr1['pos_only']+tr1['neg_only'])/t1:.0f}% offered no comparison at all.\n\n"
           f"A raw hit-rate assumes he was choosing between all the icons all the time. He was not.\n"
           f"That is why the chance baseline has to be weighted by what was actually VISIBLE,\n"
           f"which is what the next page builds.\n\n"
           f"WHERE THE SESSION GOES ({cs['session_min']:.0f} min total). These are TOTALS -- the "
           f"trial types alternate all session:\n"
           f"     {cs['census_min']:.0f} min   {tr1['n']} choice trials "
           f"(~{cs['census_min']*60/t1:.0f} s each)   <- everything above\n"
           f"     {cs['other_world_min']:.0f} min   escape trials in the banishment world "
           f"(only escape rings there -- no choice to score)\n"
           f"     {cs['non_analyzable_min']:.0f} min   reshuffle / incomplete",
           transform=a.transAxes, va='top', ha='left', fontsize=9.7, linespacing=1.45)

    fig.suptitle(f"{cs['mouse_id']}  WHAT THE MOUSE CAN ACTUALLY SEE\n"
                 'the motivation for everything that follows: he rarely sees the whole board',
                 fontsize=12.5, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig
