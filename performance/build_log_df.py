"""
Build the LOG-ONLY dataset: `df_sessions` + `df_trials`, for ALL mice, from `log.json` and nothing
else. This is THE dataset; everything downstream reads it rather than re-reading the server.

*** ONE CALL, TWO TABLES. ***
    df_sessions, df_trials = build_all(MAIN_DIR)
There is deliberately no third table and no separate scoring step: the performance columns (D,
chance, the conflict criterion, throughput) are merged INTO `df_sessions` here. An earlier version
handed back a session index, a score table and a collections table with overlapping columns, and it
was impossible to tell which one was the dataset.

*** THE TRIAL LOGIC IS NOT REIMPLEMENTED HERE. ***
`task_*/build_trials.py` already defines a trial correctly -- a SPAWN BATCH, not a collection, so
batches that end without one (a mid-session reshuffle, the session-end tail) are real rows. It also
carries the on-screen `icons` list, the NORMAL/BANISH_WORLD column (the log does not record the
world swap, so that column is the authority) and `start_frame`/`end_frame`, which is what a later
optic-flow join needs. Those builders are called through optional keyword args (`sess=`, `dfc=`,
`write=False`, `figures=False`) so nothing is read from or written to the session directory --
raw server folders have no `session.json` and must not gain a `df_trials_clean.pkl`.

`cluster_paths` and `label_trials` are log-only too (clustering uses four path-geometry features;
labels use coords + icons), so they run here as well and their columns land in `df_trials`.

*** WHAT IS NOT HERE. *** Anything needing the video: pupil, blink, whisker, lick, groom, saccade,
per-ROI flow. Those join onto `df_trials` later via `start_frame`/`end_frame`.

    from build_log_df import build_all, save, load
    df_sessions, df_trials = build_all(MAIN_DIR)
    save(df_sessions, OUT / 'df_sessions.pkl'); save(df_trials, OUT / 'df_trials.pkl')
"""
from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / 'common'))
import session_index as sidx          # noqa: E402
import perf_from_log as pfl           # noqa: E402
import geom                           # noqa: E402

BUILD_VERSION = 2

# NOTE `world_sig`, not `world`. The trial table already has a `world` column meaning which world
# that TRIAL happened in ('NORMAL' / 'BANISH_WORLD') -- a different thing from the world the SESSION
# was recorded on, and the authority for it, since the log does not record the world swap.
ID_COLS = ['session', 'mouse', 'day', 'time', 'task', 'world_sig', 'world_id', 'view_scale', 'dir']
_ID_SOURCE = {'world_sig': 'world'}     # identity column -> the discover-table column it comes from

VALENCE = {'single_reward': +1, 'double_reward': +1, 'timeout': -1, 'banish': -1, 'unbanish': 0}
# `unbanish` is NEUTRAL: an escape is a collection but pays nothing and is not a hazard hit, so
# counting it either way corrupts the hit rate. Reward drops: the streak multiplier where the task
# has one, else the effect's fixed pay.
FIXED_DROPS = {'single_reward': 1, 'double_reward': 2, 'timeout': 0, 'banish': 0, 'unbanish': 0}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Trial construction is task-specific and lives with the task. A protocol with no entry here falls
# back to the generic builder below and is TAGGED as such, so it is visible which rows were built
# by the validated task logic and which by the fallback.
TASK_MODULES = {
    'banish_multiplier': _ROOT / 'task_banish_multiplier',
}


def _task_pipeline(task):
    """(build_trials, cluster_paths, label_trials) modules for a task, or None."""
    d = TASK_MODULES.get(task)
    if d is None:
        return None
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    return (_load(f'_bt_{task}', d / 'build_trials.py'),
            _load(f'_cp_{task}', d / 'cluster_paths.py'),
            _load(f'_lt_{task}', d / 'label_trials.py'))


def _sess_from_row(row, log):
    """The session.json fields the task builders read, recovered from the log + the index row.

    A raw server session folder has a log.json but no session.json -- the video pipeline was never
    run on it -- and everything the builders actually use is in the log anyway.
    """
    w = (log.get('worlds') or [{}])[0]
    return dict(world_width=int(w.get('width', 0)), world_height=int(w.get('height', 0)),
                mouse_id=row.get('mouse'), camera_moved=False,
                view_scale=row.get('view_scale'), task_type=row.get('task'))


def _generic_trials(row, log, sess):
    """Fallback trial table for a protocol with no dedicated builder.

    Trials are collection-to-collection, which is the common subset every protocol shares. It does
    NOT reproduce the spawn-batch definition (so a batch ending without a collection is absent) and
    it has no task-specific columns. Rows are tagged `builder='generic'` so a downstream analysis
    can tell these apart from trials built by validated task logic instead of assuming they match.
    """
    t_xy, x_xy, y_xy = geom.load_coords(log)
    t_th, theta = geom.load_theta(log)
    W, H = sess['world_width'], sess['world_height']
    coll = sorted(log.get('collected') or [], key=lambda c: c.get('time', 0))
    t0_session = float(t_xy[0]) if len(t_xy) else 0.0
    rows, prev = [], None
    for k, c in enumerate(coll):
        t1 = c.get('time')
        t0 = prev if prev is not None else t0_session
        tt, xx, yy = geom.slice_track(t_xy, x_xy, y_xy, t0, t1)
        msp, ssp = geom.speed_stats(tt, xx, yy)
        _, align = geom.heading_to_target(t_th, theta, t_xy, x_xy, y_xy,
                                          (c.get('x'), c.get('y')), t0, t1)
        rows.append(dict(
            trial=k, start_ms=t0, end_ms=t1, dur_s=(t1 - t0) / 1000.0,
            outcome=c.get('effect'), multiplier=c.get('multiplier', np.nan),
            target_x=c.get('x'), target_y=c.get('y'), target_loc=c.get('loc'),
            path_efficiency=geom.path_efficiency(xx, yy),
            time_in_corner=geom.time_in_corner(xx, yy, W, H),
            mean_speed=msp, speed_std=ssp, heading_align=align,
            coord_t_ms=tt, coord_x=xx, coord_y=yy,
            degenerate=False, analyze=True))
        prev = t1
    return pd.DataFrame(rows)


def session_trials(row, verbose=False):
    """Trial table for ONE session: task builder -> clustering -> error/conflict labels."""
    log = json.load(open(row['log']))
    sess = _sess_from_row(row, log)
    pipe = _task_pipeline(row['task'])
    if pipe is None:
        T = _generic_trials(row, log, sess)
        T['builder'] = 'generic'
    else:
        bt, cp, lt = pipe
        T = bt.build(str(row['dir']), write=False, verbose=verbose, sess=sess)
        T['builder'] = row['task']
        # clustering and labels are log-only; a session too small to cluster keeps its trials
        try:
            T = cp.run(str(row['dir']), write=False, verbose=verbose, dfc=T, sess=sess, figures=False)
        except Exception as ex:
            T['cluster_name'] = 'unclustered'
            T.attrs['cluster_error'] = f'{type(ex).__name__}: {ex}'
        try:
            T = lt.run(str(row['dir']), write=False, verbose=verbose, dfc=T, sess=sess, figures=False)
        except Exception as ex:
            T.attrs['label_error'] = f'{type(ex).__name__}: {ex}'
    # derived columns every downstream plot wants, stated ONCE here so they cannot disagree
    # between tables. `t_ms` is the collection instant (the trial's end), which is what
    # event-aligned and ordering analyses key on.
    if 'outcome' in T:
        T['valence'] = T.outcome.map(VALENCE)
        T['is_positive'] = T.valence.eq(1)
        T['is_negative'] = T.valence.eq(-1)
        T['drops'] = [
            (m if (o == 'single_reward' and pd.notna(m)) else FIXED_DROPS.get(o, 0))
            for o, m in zip(T.outcome, T.get('multiplier', pd.Series([np.nan] * len(T))))]
    if 'end_ms' in T:
        T['t_ms'] = T['end_ms']
    T['label'] = row.get('label')
    for c in reversed(ID_COLS):
        T.insert(0, c, row.get(_ID_SOURCE.get(c, c)))
    return T


def session_row(row, T):
    """One session row: log facts + PERFORMANCE, merged. `T` is that session's trial table."""
    log = json.load(open(row['log']))
    ed = log.get('experiment_data', {}) or {}
    coords = np.asarray(log.get('coords_t[ms]/x/y') or [[0]], float)

    rec = {c: row.get(_ID_SOURCE.get(c, c)) for c in ID_COLS}
    rec.update(
        name=row.get('name'), label=row.get('label'), marker=row.get('marker', ''), mouse_src=row.get('mouse_src', ''),
        use=row.get('use', True), note=row.get('note', ''), warn=row.get('warn', ''),
        is_dup_day=bool(row.get('is_dup_day', False)),
        keep_of_day=bool(row.get('keep_of_day', True)),
        effects=row.get('effects', ''), effects_offered=row.get('effects_offered', ''),
        never_collected=row.get('never_collected', ''),
        task_description=pfl.TASK_DESCRIPTION.get(row.get('task'), ''),
        has_multiplier=pfl.log_has_multiplier(log),
        n_trials_total=len(T),
        n_analyzable=int(T['analyze'].sum()) if 'analyze' in T else np.nan,
        wall_min=(coords[-1, 0] - coords[0, 0]) / 60000.0 if len(coords) > 1 else np.nan,
        weight_g=ed.get('weight (g)', np.nan),
        weight_baseline_g=ed.get('original weight (g)', np.nan),
        weight_fraction=ed.get('weight_fraction', np.nan))

    # counts straight from the trial table, per outcome
    if 'outcome' in T:
        for eff, n in T.outcome.value_counts().items():
            rec[f'n_{eff}'] = int(n)

    # PERFORMANCE -- merged in, not returned as a separate table. Unscorable sessions keep their
    # identity and get NaN here plus a reason, rather than vanishing from the dataset.
    rec['perf_error'] = ''
    try:
        p = pfl.score_log(row['log'], view_scale=row.get('view_scale'),
                          mouse_id=row.get('mouse'), label=row.get('label'))
        for k, v in p.items():
            if k not in ('task', 'mouse', 'label', 'log', 'view_scale'):
                rec[k] = v
    except Exception as ex:
        rec['perf_error'] = f'{type(ex).__name__}: {ex}'
    return rec


def build(S, progress=True, verbose=True):
    """(df_sessions, df_trials) from a `discover` table. EVERY discovered session becomes a row."""
    todo = [r for _, r in S.iterrows()]
    bar = sidx._progress(todo, f'building {len(todo)} session(s)', enabled=progress,
                         labels=[str(r.get('session', r['name'])) for r in todo])
    tframes, srows, failed = [], [], []
    for r in bar:
        if hasattr(bar, 'set_postfix_str'):
            bar.set_postfix_str(str(r.get('session', r['name']))[:32])
        try:
            T = session_trials(r)
            tframes.append(T)
            srows.append(session_row(r, T))
        except Exception as ex:
            failed.append((r.get('session', r['name']), f'{type(ex).__name__}: {ex}'))
    df_trials = pd.concat(tframes, ignore_index=True) if tframes else pd.DataFrame()
    df_sessions = pd.DataFrame(srows)
    if len(df_sessions):
        df_sessions = df_sessions.sort_values(['mouse', 'day', 'time']).reset_index(drop=True)
    if verbose:
        report(df_sessions, df_trials, failed)
    return df_sessions, df_trials


def build_all(main_dir, pattern='*', progress=True, verbose=True, one_per_day=True, **kw):
    """Every mouse under MAIN_DIR. `require_single_animal` does NOT apply -- that guard protects a
    within-animal learning curve; a stored dataset spanning animals is the intended thing."""
    main_dir = Path(main_dir).expanduser()
    A = sidx.list_animals(main_dir, pattern=pattern)
    if not len(A):
        return pd.DataFrame(), pd.DataFrame()
    frames = []
    for _, a in A.iterrows():
        if verbose:
            print(f"\n--- {a.animal} ({a.n_sessions} session folder(s)) ---")
        Si = sidx.discover(a.path, progress=progress, **kw)
        if one_per_day and len(Si):
            # FLAG the duplicates, do not drop them: the build keeps everything and filtering is an
            # explicit choice at analysis time.
            Si = sidx.one_per_day(Si, keep='last', verbose=verbose, mark_only=True)
        frames.append(Si)
    S = pd.concat(frames, ignore_index=True)
    sidx.protocol_census(S)          # assigns world_id in place
    return build(S, progress=progress, verbose=verbose)


def report(X, T, failed=()):
    print('\n=== df_log (log files only) ===')
    if not len(X):
        print('  nothing built'); return
    print(f'  {len(X)} session(s), {len(T)} trial(s), {X.mouse.nunique()} animal(s)')
    print(f'  days      : {X.day.min()} .. {X.day.max()}')
    print(f'  protocols : {X.task.value_counts().to_dict()}')
    print(f'  worlds    : {X.world_id.value_counts().to_dict()}')
    if 'builder' in T:
        print(f'  builders  : {T.builder.value_counts().to_dict()}')
    if 'outcome' in T:
        print(f'  outcomes  : {T.outcome.value_counts().to_dict()}')
    ok = X.perf_error.eq('') if 'perf_error' in X else pd.Series([True] * len(X))
    print(f'  scored    : {int(ok.sum())} of {len(X)} '
          f'({int((~ok).sum())} without performance -- see perf_error)')
    print('  per animal:')
    for m, g in X.groupby('mouse'):
        print(f'    {m:12s} {len(g):3d} session(s)  {int(g.n_trials_total.sum()):5d} trials  '
              f'{g.day.min()} .. {g.day.max()}')
    if failed:
        print(f'  FAILED {len(failed)}:')
        for n, why in failed:
            print(f'    -- {n}: {why}')


def _array_cols(D):
    return [c for c in D.columns if D[c].map(lambda v: isinstance(v, (np.ndarray, list))).any()]


def save(D, path, verbose=True):
    """Persist, stamped with what built it, so a stale cache is recognisable rather than merely old.

    `.pkl` keeps everything including the per-trial coordinate arrays and the `icons` lists.
    `.parquet`/`.csv` cannot hold a column of arrays, so those are dropped and the loss is stated.
    """
    path = Path(path)
    D = D.copy()
    D.attrs['built'] = dict(when=time.strftime('%Y-%m-%d %H:%M:%S'), build_version=BUILD_VERSION,
                            n_rows=len(D),
                            animals=sorted(D.mouse.dropna().unique().tolist()) if 'mouse' in D else [],
                            tasks=sorted(D.task.dropna().unique().tolist()) if 'task' in D else [])
    arr = _array_cols(D)
    if path.suffix in ('.parquet', '.csv') and arr:
        if verbose:
            print(f'  ! {path.suffix} cannot store array columns -- dropping {len(arr)}: '
                  f'{", ".join(arr)}   (use .pkl to keep them)')
        D = D.drop(columns=arr)
    if path.suffix == '.parquet':
        D.to_parquet(path, index=False)
    elif path.suffix == '.csv':
        D.to_csv(path, index=False)
    else:
        D.to_pickle(path)
    if verbose:
        print(f'  wrote {path}  ({len(D)} rows x {len(D.columns)} cols, '
              f'{path.stat().st_size / 1e6:.2f} MB)')
    return path


def load(path, verbose=True):
    path = Path(path)
    D = (pd.read_parquet(path) if path.suffix == '.parquet' else
         pd.read_csv(path) if path.suffix == '.csv' else pd.read_pickle(path))
    b = D.attrs.get('built')
    if verbose:
        if b:
            stale = b.get('build_version') != BUILD_VERSION
            print(f'  {path.name}: {b["n_rows"]} rows, built {b["when"]} (v{b["build_version"]})'
                  + (f'  !! STALE: module is now v{BUILD_VERSION} -- rebuild' if stale else ''))
        else:
            print(f'  {path.name}: {len(D)} rows (no build stamp)')
    return D
