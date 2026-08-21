"""
Standalone PERFORMANCE PDF for one session -- separate from `trial_report.pdf`, which has grown
large enough that the performance material is hard to find in it.

*** WRITTEN AS VECTOR PDF, NOT AS AN EMBEDDED PNG. *** The scorecard inside `trial_report.pdf` is a
110-dpi raster that is then placed as an image on the page, so zooming in blurs the text -- which is
exactly what happens to the "what visibility-weighted chance means" box. Here the figures go
straight into a PdfPages, so every label stays sharp at any zoom and the text is selectable.

Pages (motivation -> metric -> result -> what the metric cannot see):
  1. WHAT HE CAN ACTUALLY SEE -- he sees nothing at all for ~45% of frames and both icon types
     together for only ~6%, which is why a raw hit-rate will not do
  2. how the discrimination score is built: what observed / chance / D each count, and why chance
     is reported as a range over a memory assumption
  3. the performance scorecard (counts, discrimination, throughput, time, speed, heading toward
     visible icons, and the scorecard table)
  4-6. THE CLEAN SUBSET -- the trials where both icon types were on screen at once, where no
     visibility/memory/endogeneity argument applies: (4) when the choice was available, (5) the
     decision point vs an exogenous "go to the nearer one" baseline, (6) an avoidance signature
  7. STREAKS / COMBO -- the ordering information D is blind to: does the next outcome depend on the
     current run, are run lengths different from independence, and what the ORDER was worth in
     reward drops (consecutive rewards build the multiplier, a negative resets it)

Run:  python3 make_performance_pdf.py <session_dir>   -> <session_dir>/performance_report.pdf
"""
from pathlib import Path
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_performance as ap                    # noqa: E402
import make_visibility_chance_explainer as vex      # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'common'))
import streaks                                      # noqa: E402
import visibility_census as vcen                    # noqa: E402
import choice_trials as cht                         # noqa: E402
import pandas as pd                                 # noqa: E402


def build(session_dir, out_name='performance_report.pdf'):
    d = Path(session_dir).resolve()
    outp = d / out_name
    with PdfPages(outp) as pdf:
        # MOTIVATION FIRST: show that he rarely sees the whole board, which is WHY the chance
        # baseline has to be visibility-weighted at all.
        try:
            cs = vcen.census(str(d), ['single_reward'], ['banish'])
            fig0 = vcen.figure(cs, pos_label='reward', neg_label='banishment')
            (d / 'debug').mkdir(exist_ok=True)
            fig0.savefig(d / 'debug' / 'visibility_census.png', dpi=110)
            fig0.text(0.995, 0.004, 'p. 1', ha='right', va='bottom', fontsize=8, color='#555')
            pdf.savefig(fig0); plt.close(fig0)
            print('  + p.1  what the mouse can actually see')
        except Exception as ex:
            print(f'  !! census page skipped: {type(ex).__name__}: {ex}')

        # THEN THE METRIC: define it, then show the session's numbers against it.
        try:
            fig2 = vex.run(str(d), write=False, return_fig=True)
            fig2.text(0.995, 0.004, 'p. 2', ha='right', va='bottom', fontsize=8, color='#888')
            pdf.savefig(fig2, facecolor=fig2.get_facecolor()); plt.close(fig2)
            print('  + p.2  how the discrimination score is built')
        except Exception as ex:
            print(f'  !! explainer page skipped: {type(ex).__name__}: {ex}')

        blocks, fig = ap.run(str(d), write=False, return_fig=True)
        fig.text(0.995, 0.004, 'p. 3', ha='right', va='bottom', fontsize=8, color='#555')
        pdf.savefig(fig); plt.close(fig)
        print('  + p.3  performance scorecard')

        # p.4-6 THE CLEAN SUBSET: the trials where he could SEE both types at once. This is where
        # the baseline arguments of p.2 stop mattering, so it belongs right after the number they
        # qualify (p.3) and before the ordering analysis.
        try:
            res = cht.analyse(str(d), ['single_reward'], ['banish'])
            for k, (fn, nm) in enumerate([(cht.figure_when, 'when the choice was available'),
                                          (cht.figure_decision, 'the decision point'),
                                          (cht.figure_avoidance, 'avoidance signature')]):
                figc = fn(res, pos_label='reward', neg_label='banishment')
                figc.savefig(d / 'debug' / f'choice_{"abc"[k]}.png', dpi=110)
                figc.text(0.995, 0.004, f'p. {4+k}', ha='right', va='bottom', fontsize=8,
                          color='#555')
                pdf.savefig(figc); plt.close(figc)
                print(f'  + p.{4+k}  {nm}')
        except Exception as ex:
            print(f'  !! clean-subset pages skipped: {type(ex).__name__}: {ex}')

        # p.7 STREAKS: the ordering information D is blind to (consecutive rewards build the
        # multiplier, so the same collections in a different order pay differently).
        try:
            df = pd.read_pickle(d / 'df_trials_clean.pkl')
            tr = df[df.analyze].sort_values('trial')
            seq = (tr[tr.outcome.isin(['single_reward', 'banish'])].outcome
                   == 'single_reward').values
            st = streaks.analyse(seq)
            fig3 = streaks.figure(st, json.load(open(d / 'session.json'))['mouse_id'],
                                  pos_label='reward', neg_label='banishment')
            # save the PNG BEFORE stamping the page number -- make_trial_report.py embeds this
            # image and adds its own stamp, so a burned-in "p. 3" would show up twice there
            (d / 'debug').mkdir(exist_ok=True)
            fig3.savefig(d / 'debug' / 'streaks.png', dpi=110)
            fig3.text(0.995, 0.004, 'p. 7', ha='right', va='bottom', fontsize=8, color='#555')
            pdf.savefig(fig3); plt.close(fig3)
            print(f'  + p.7  streaks / combo  ({streaks.verdict(st)[:60]}...)')
        except Exception as ex:
            print(f'  !! streak page skipped: {type(ex).__name__}: {ex}')

    print(f'wrote {outp}  (VECTOR -- text stays sharp when you zoom)')
    return outp


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else '.')
