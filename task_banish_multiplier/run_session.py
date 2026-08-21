"""
Batch runner + MANIFEST for a `banish_multiplier` session (JPAS_0168-class).

This file is the single source of truth for WHAT the pipeline consists of, in what ORDER, what each
step writes, and which steps need a HUMAN to look at a picture before the result is trusted. The
validation notebook (`validate_session.ipynb`) reads the same STEPS list, so the two cannot drift.

    python3 run_session.py <session_dir> --list          # manifest + what already exists
    python3 run_session.py <session_dir>                 # run everything missing, pausing at QC gates
    python3 run_session.py <session_dir> --all           # re-run everything (recompute)
    python3 run_session.py <session_dir> --only flow,pupil
    python3 run_session.py <session_dir> --from trials   # this step and all later ones
    python3 run_session.py <session_dir> --yes           # don't pause at the human-QC gates

HUMAN-QC GATES (`qc=True`) are the steps a new animal can silently break -- the pupil fit, the blink
threshold and the lick/groom split are all per-animal tuned (see CLAUDE.md "Reusable-pipeline
design"). The runner stops at them and names the picture to open. Everything else is task-agnostic
core that runs from session.json untouched.
"""
from pathlib import Path
import argparse
import importlib.util
import sys
import time

ROOT = Path(__file__).resolve().parent
COMMON = ROOT.parent / 'common'
DET = COMMON / 'detectors'


def _load(path, name=None):
    """Import a pipeline module by file path (they are plain scripts, not a package)."""
    for p in (str(COMMON), str(DET), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    name = name or Path(path).stem
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# name, script, callable, outputs (relative to session dir), qc?, one-line what/why
STEPS = [
    dict(key='session', script=COMMON / 'session_config.py', fn='build_session',
         args='mouse_dir', outputs=['session.json', 'opticflow/camera_check.json'], qc=False,
         desc='discover log/videos/ROIs, classify the task, camera-shift check -> session.json '
              '(the contract every later step reads)'),
    dict(key='flow', script=COMMON / 'compute_roi_flow.py', fn='run',
         args='dir', outputs=['opticflow/opticflow_paw_mag.npy',
                              'opticflow/opticflow_whisker_right_mag.npy'], qc=False,
         desc='per-ROI Farneback optic flow, 8 metrics x 7 ROIs (~34 min; the slow step)'),
    dict(key='pupil', script=DET / 'segment_pupil.py', fn='run',
         args='dir', outputs=['opticflow/pupil_track.npz'], qc=True,
         desc='glint-anchored dark-iris fit -> radius/cx/cy/glint. PER-ANIMAL TUNED',
         qc_hint='overlay_grid() contact sheet on RAW crops -- debug/pupil_fit_check.png. '
                 'Check the circle sits on the iris, not the lid or the socket shadow.'),
    dict(key='eye', script=DET / 'compute_eye_events.py', fn='run',
         args='dir', outputs=['opticflow/eye_events.npz'], qc=True,
         desc='blink + eye-squint + eye position (P-CR). NO dilation/constriction for an '
              'approx-iris animal',
         qc_hint='blink count/rate in the notebook + an annotate_clip window over a known closure. '
                 'The eye_mean threshold is PER-ANIMAL.'),
    dict(key='whisker', script=COMMON / 'whisksweep.py', fn='run',
         args='dir', outputs=['opticflow/whisker.npz'], qc=False,
         desc='whisker sweep (amplitude) + cycle (Hz): 3-13 Hz band-pass, principal axis, Hilbert'),
    dict(key='groom', script=DET / 'detect_grooming.py', fn='run',
         args='dir', outputs=['opticflow/groom_mask.npy', 'opticflow/groom_mask_clean.npy'], qc=True,
         desc='paw & whisker flow both >p75, GATED on joystick stillness',
         qc_hint='annotate_mouth_clip.py over a grooming bout -- the paw must be visibly UP at the '
                 'face. Check the joystick-still gate did not leave locomotion in.'),
    dict(key='lick', script=DET / 'detect_licking.py', fn='run',
         args='dir', outputs=['opticflow/licking.npz'], qc=True,
         desc='mouth fy 5-12 Hz rhythm envelope -> lick bouts',
         qc_hint='within-bout frequency must match the spectral peak (~7-8 Hz). If it does not, the '
                 'band or the fy sign is wrong for this animal.'),
    dict(key='mouth', script=DET / 'compute_mouth_state.py', fn='run',
         args='dir', outputs=['opticflow/mouth_state.npz'], qc=True,
         desc='single mouth axis 0 CLOSED / 1 LICKING / 2 GROOMING, GROOMING PRIORITISED '
              '(a raised paw sweeps the mouth ROI and false-fires the flow lick detector)',
         qc_hint='annotate_mouth_clip.py on an overlap window -- every reassigned frame should show '
                 'the paw up.'),
    dict(key='saccade', script=DET / 'detect_saccades.py', fn='run',
         args='dir', outputs=['opticflow/saccades.npz'], qc=False,
         desc='APPROXIMATE P-CR fast-step-that-holds. Flagged approximate wherever it is used'),
    dict(key='trials', script=ROOT / 'build_trials.py', fn='build',
         args='dir', outputs=['df_trials_clean.pkl'], qc=False,
         desc='TRIAL = SPAWN BATCH (one collection -> the next). Outcomes single_reward(+multiplier)'
              ' / banish / unbanish / reshuffle / incomplete'),
    dict(key='enrich', script=ROOT / 'enrich_trials.py', fn='run',
         args='dir', outputs=['df_trials_clean.pkl'], qc=False,
         desc='join the detectors onto the trials: frac_blink/squint/grooming/whisking, '
              'radius_mean, whisk_sweep/cycle, joy_fine'),
    dict(key='cluster', script=ROOT / 'cluster_paths.py', fn='run',
         args='dir', outputs=['debug/path_clustering.png', 'debug/feature_selection.png',
                              'debug/multiplier_outcome.png'], qc=False,
         desc='KMeans on PATH GEOMETRY only -> Direct / Corner-dwelling. Motor/whisker/heading/'
              'pupil/MULTIPLIER held out as OUTCOMES so later tests stay non-circular'),
    dict(key='labels', script=ROOT / 'label_trials.py', fn='run',
         args='dir', outputs=['debug/overshoot_check.png'], qc=False,
         desc='error (= banish) / conflict (= Corner-dwelling) labels + collection-radius geometry'),
    dict(key='h1', script=ROOT / 'analyze_h1_post_error.py', fn='run',
         args='dir', outputs=['debug/h1_post_error.png'], qc=False,
         desc='H1 CARRY-OVER: did the previous trial (error/conflict) change this one?'),
    dict(key='h2', script=ROOT / 'analyze_h2_whisker_reset.py', fn='run',
         args='dir', outputs=['debug/h2_whisker_reset.png'], qc=False,
         desc='H2 MOVEMENT-LOCKED: whisker reset around each joystick re-steer, by reward streak'),
    dict(key='valence', script=ROOT / 'analyze_collection_valence.py', fn='run',
         args='dir', outputs=['debug/collection_valence.png'], qc=False,
         desc='FEEDBACK-LOCKED: whisker amplitude hitting a POSITIVE vs NEGATIVE icon '
              '(approach window only -- post-hit is ~96% licking on reward trials)'),
    dict(key='wcontext', script=ROOT / 'analyze_whisker_context.py', fn='run',
         args='dir', outputs=['debug/whisker_context.png'], qc=False,
         desc='whisker sweep by CONTEXT: first vs second half of trials, and positive-reward-'
              'VISIBLE vs not (paired within trial, viewport-gated)'),
    dict(key='perf', script=ROOT / 'analyze_performance.py', fn='run',
         args='dir', outputs=['debug/performance.png'], qc=False,
         desc='PERFORMANCE scorecard: discrimination vs a VISIBILITY-WEIGHTED chance baseline, '
              'throughput (rewards/min + multiplier-weighted units/min), efficiency, by session half'),
    dict(key='explain', script=ROOT / 'make_visibility_chance_explainer.py', fn='run',
         args='dir', outputs=['debug/visibility_chance_explainer.png'], qc=False,
         desc='teaching figure: why the chance baseline is VISIBILITY-weighted and what D means'),
    # MUST run before `report`: make_trial_report embeds visibility_census / choice_a-c / streaks
    # as PNGs and only this step writes them. _embed_png SKIPS a missing PNG with a print, so
    # without this the report silently loses 5 pages instead of failing.
    dict(key='perfpdf', script=ROOT / 'make_performance_pdf.py', fn='build',
         args='dir',
         outputs=['performance_report.pdf', 'debug/visibility_census.png', 'debug/choice_a.png',
                  'debug/choice_b.png', 'debug/choice_c.png', 'debug/streaks.png'], qc=False,
         desc='the standalone 4-page VECTOR performance PDF (census -> chance explainer -> '
              'scorecard -> clean both-visible subset -> streaks), and the PNGs the trial report '
              'embeds for the same block'),
    dict(key='report', script=ROOT / 'make_trial_report.py', fn='Report', args='report',
         outputs=['trial_report.pdf'], qc=False,
         desc='the PDF: 13 analysis pages (classification -> features -> multiplier -> summaries -> '
              'correlations -> finding-6 -> H1/H2/valence) + one page per analyzable trial'),
]

QC_ONLY = [k['key'] for k in STEPS if k['qc']]


def _exists(d, step):
    return [(o, (d / o).exists()) for o in step['outputs']]


def list_steps(d):
    print(f"\nPIPELINE MANIFEST  (task_banish_multiplier)   session: {d}")
    print(f"{'':2} {'step':9} {'QC':3} outputs")
    print('-' * 110)
    for i, st in enumerate(STEPS, 1):
        ex = _exists(d, st)
        mark = 'OK ' if all(e for _, e in ex) else ('part' if any(e for _, e in ex) else ' -- ')
        print(f"{i:2} {st['key']:9} {'QC' if st['qc'] else '  ':3} [{mark}] "
              + ', '.join(o for o, _ in ex))
        print(f"{'':16}{st['desc']}")
        for o, e in ex:
            if not e:
                print(f"{'':16}  MISSING: {o}")
    print('-' * 110)
    print(f"human-QC gates: {', '.join(QC_ONLY)}  (per-animal tuning -- look at the picture)")
    print("\nQC TOOLS (not pipeline steps -- run them by hand at the gates, or via the notebook):")
    print("  validate_session.ipynb                    per-step checks, expected-vs-actual, PASS/FAIL")
    print("  detectors/annotate_clip.py       --style approx|stable   eye/iris validation clip")
    print("  detectors/annotate_mouth_clip.py --style flow|pixel      lick-vs-groom validation clip")
    print("  reconstruct_view.py --test|--trial N|--full [--vision mouse]  the MOUSE MONITOR VIEW")
    print("  compare_sessions.py <dirA> <dirB> ... | --glob 'DIR/JPAS_*'   CROSS-SESSION tracking")
    print("    (style is auto-detected from the session; a mismatched style is REFUSED, not faked)")


def run_step(d, st, verbose=True):
    t0 = time.time()
    print(f"\n=== [{st['key']}] {st['desc'][:80]} ===", flush=True)
    mod = _load(st['script'])
    if st['args'] == 'report':
        mod.Report(str(d)).build()
    elif st['args'] == 'mouse_dir':
        getattr(mod, st['fn'])(d.name, str(d))
    else:
        getattr(mod, st['fn'])(str(d))
    print(f"    [{st['key']}] done in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('session_dir')
    ap.add_argument('--list', action='store_true', help='print the manifest and exit')
    ap.add_argument('--all', action='store_true', help='re-run every step, even if outputs exist')
    ap.add_argument('--only', help='comma-separated step keys')
    ap.add_argument('--from', dest='from_', help='start at this step key and run the rest')
    ap.add_argument('--skip', help='comma-separated step keys to skip')
    ap.add_argument('--yes', action='store_true', help='do not pause at human-QC gates')
    a = ap.parse_args()
    d = Path(a.session_dir).resolve()

    if a.list:
        list_steps(d)
        return

    steps = STEPS
    if a.from_:
        keys = [s['key'] for s in STEPS]
        if a.from_ not in keys:
            sys.exit(f"unknown --from '{a.from_}'. keys: {', '.join(keys)}")
        steps = STEPS[keys.index(a.from_):]
    if a.only:
        want = set(a.only.split(','))
        steps = [s for s in steps if s['key'] in want]
    if a.skip:
        skip = set(a.skip.split(','))
        steps = [s for s in steps if s['key'] not in skip]

    ran, skipped = [], []
    for st in steps:
        if not a.all and all(e for _, e in _exists(d, st)):
            skipped.append(st['key']); continue
        run_step(d, st)
        ran.append(st['key'])
        if st['qc'] and not a.yes:
            print(f"\n  *** HUMAN-QC GATE [{st['key']}] ***\n  {st.get('qc_hint', '')}")
            try:
                if input('  looks right? [y/N] ').strip().lower() not in ('y', 'yes'):
                    sys.exit(f"  stopped at the {st['key']} QC gate -- tune it before continuing.")
            except EOFError:                      # non-interactive: warn, keep going
                print('  (non-interactive: continuing WITHOUT QC confirmation)')
    print(f"\nran: {', '.join(ran) or 'nothing'}")
    if skipped:
        print(f"already present (use --all to recompute): {', '.join(skipped)}")


if __name__ == '__main__':
    main()
