"""
Batch runner + MANIFEST for a `timeout_double` session (JPAS_0231-class).

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
         desc='discover log/videos/ROIs, classify the task, camera-shift check -> session.json. '
              'NOTE: view_scale is NOT derivable from the log and must be set per session '
              '(JPAS_0231 = 0.56); build_session preserves it across rebuilds'),
    dict(key='flow', script=COMMON / 'compute_roi_flow.py', fn='run',
         args='dir', outputs=['opticflow/opticflow_paw_mag.npy',
                              'opticflow/opticflow_whisker_right_mag.npy'], qc=False,
         desc='per-ROI Farneback optic flow, 8 metrics x 7 ROIs (the slow step, ~30 min)'),
    dict(key='pupil', script=DET / 'segment_pupil.py', fn='run',
         args='dir', outputs=['opticflow/pupil_track.npz'], qc=True,
         desc='pupil fit -> radius/cx/cy/glint. PER-ANIMAL TUNED',
         qc_hint='overlay contact sheet on RAW crops. JPAS_0231 has a RELATIVELY STABLE iris (unlike '
                 '0168), so this session may support dilation/constriction events -- set '
                 'session.json iris_quality accordingly and use annotate_clip --style stable.'),
    dict(key='eye', script=DET / 'compute_eye_events.py', fn='run',
         args='dir', outputs=['opticflow/eye_events.npz'], qc=True,
         desc='blink + squint + eye position (P-CR)',
         qc_hint='blink rate + a clip over a known closure; the eye_mean threshold is PER-ANIMAL.'),
    dict(key='whisker', script=COMMON / 'whisksweep.py', fn='run',
         args='dir', outputs=['opticflow/whisker.npz'], qc=False,
         desc='whisker sweep + cycle: 3-13 Hz band-pass, principal axis, Hilbert'),
    dict(key='groom', script=DET / 'detect_grooming.py', fn='run',
         args='dir', outputs=['opticflow/groom_mask.npy', 'opticflow/groom_mask_clean.npy'], qc=True,
         desc='paw & whisker flow both >p75, GATED on joystick stillness',
         qc_hint='annotate_mouth_clip.py over a bout -- the paw must be visibly UP at the face.'),
    dict(key='lick', script=DET / 'detect_licking.py', fn='run',
         args='dir', outputs=['opticflow/licking.npz'], qc=True,
         desc='mouth fy 5-12 Hz rhythm envelope -> lick bouts',
         qc_hint='within-bout frequency must match the spectral peak (~7-8 Hz).'),
    dict(key='mouth', script=DET / 'compute_mouth_state.py', fn='run',
         args='dir', outputs=['opticflow/mouth_state.npz'], qc=True,
         desc='single mouth axis CLOSED / LICKING / GROOMING, GROOMING PRIORITISED',
         qc_hint='annotate_mouth_clip.py on an overlap window -- reassigned frames need the paw up.'),
    dict(key='saccade', script=DET / 'detect_saccades.py', fn='run',
         args='dir', outputs=['opticflow/saccades.npz'], qc=False,
         desc='APPROXIMATE P-CR fast-step-that-holds'),
    dict(key='trials', script=ROOT / 'build_trials.py', fn='build',
         args='dir', outputs=['df_trials_clean.pkl'], qc=False,
         desc='TRIAL = SPAWN BATCH. Outcomes single_reward / double_reward / timeout. Detects the '
              '~14 s TIMEOUT FREEZE from the coord blackout and excludes it from all geometry'),
    dict(key='perf', script=ROOT / 'analyze_performance.py', fn='run',
         args='dir', outputs=['debug/performance.png'], qc=False,
         desc='PERFORMANCE scorecard: discrimination vs a VISIBILITY-WEIGHTED chance baseline '
              '(+ an exogenous spawn-geometry cross-check), WALL and ACTIVE throughput, freeze cost'),
]

QC_ONLY = [k['key'] for k in STEPS if k['qc']]


def _exists(d, step):
    return [(o, (d / o).exists()) for o in step['outputs']]


def list_steps(d):
    print(f"\nPIPELINE MANIFEST  (task_timeout_double)   session: {d}")
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
    print("    (style is auto-detected from the session; a mismatched style is REFUSED, not faked)")
    print("  NOT YET FORKED for this task: enrich/cluster/labels/h1/h2/valence/wcontext/report/")
    print("    reconstruct_view -- see task_banish_multiplier for the versions to port.")


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
