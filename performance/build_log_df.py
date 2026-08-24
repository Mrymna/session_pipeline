"""
`df_log` -- ONE table of what the logs say, for ALL mice, built from `log.json` and nothing else.

*** SCOPE: this folder is for PERFORMANCE work and is deliberately self-contained. ***
It does NOT read, write, import or depend on `df_trials_clean.pkl` or on either task fork's
`build_trials.py`. That is the point: the trial pipeline is validated, per-session, and needs the
video pipeline to have been run; performance across many days needs neither, and coupling the two
would mean a change here could disturb a table that a lot of analysis already rests on. Nothing in
this folder can affect df_trials_clean.

WHAT IT IS BUILT FROM. Every row comes from the `collected` list in a log -- the record of each
icon the animal actually took. That list has the SAME shape in both protocols (time, effect,
texture, x, y, loc, ID, t_spawned, and `multiplier` where the task has one), so this needs no
task-specific trial logic and works on any session, including protocols not yet written.

TWO GRAINS, because performance questions ask at both:
  collections_df(S)  one row per COLLECTION -- the atom. What he took, when, where, what it paid.
                     This is what streak / order / choice-sequence analysis needs.
  sessions_df(S)     one row per SESSION -- identity, protocol, world, counts per effect, duration,
                     drops, and the animal's weight where the rig recorded it.

VALENCE AND PAY are stated once, here, so every downstream number agrees:
  positive  single_reward, double_reward      negative  timeout, banish
  neutral   unbanish  -- an ESCAPE from the banishment world. It is a collection, but it pays
            nothing and is not a hazard hit, so counting it either way would corrupt the hit rate.
            It is kept as its own category rather than dropped.
  drops     the MEASURED reward unit: banish_multiplier pays the multiplier (the streak), and
            timeout_double pays 1 for single and 2 for double (confirmed against the delivery
            stream `rewards_t[ms]`: 163 = 61x1 + 51x2 on JPAS_0231).

    from build_log_df import build_all, save, load
    C, X = build_all(MAIN_DIR)            # collections, sessions -- every mouse
    save(C, 'df_log_collections.pkl'); save(X, 'df_log_sessions.pkl')
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

_COMMON = Path(__file__).resolve().parent.parent / 'common'
sys.path.insert(0, str(_COMMON))
import session_index as sidx          # noqa: E402
import geom                           # noqa: E402  -- the SAME geometry df_trials_clean uses

BUILD_VERSION = 1

# effect -> valence. Distinct effect names across protocols, so one table covers both.
VALENCE = {'single_reward': +1, 'double_reward': +1,
           'timeout': -1, 'banish': -1,
           'unbanish': 0}
FIXED_DROPS = {'single_reward': 1, 'double_reward': 2, 'timeout': 0, 'banish': 0, 'unbanish': 0}

ID_COLS = ['session', 'mouse', 'day', 'time', 'task', 'world_sig', 'view_scale', 'dir']
_ID_SOURCE = {'world_sig': 'world', 'time': 'time'}


def _ids(row):
    return {c: row.get(_ID_SOURCE.get(c, c), None) for c in ID_COLS}


def _drops(effect, multiplier):
    """Reward units a collection paid. Where the task has a multiplier the pay IS the multiplier
    (the streak), so it is used when present rather than assumed to be 1."""
    if effect in ('single_reward',) and multiplier is not None and not pd.isna(multiplier):
        return float(multiplier)
    return float(FIXED_DROPS.get(effect, 0))


def session_collections(row, geometry=True):
    """One row per collection for ONE session -- and, with `geometry`, the TRIAL that led to it.

    *** THE TRIAL WINDOW IS THE TIME BETWEEN TWO COLLECTIONS. ***
    Each collection closes a disjoint window `(previous collection, this collection]`, which is
    the same definition `df_trials_clean` uses -- in these protocols a collection immediately
    triggers the next spawn batch, so "spawn batch to spawn batch" and "collection to collection"
    are the same boundaries. Computing it here means the two tables can be COMPARED rather than
    taken on trust, and the geometry below calls the very same `common/geom` functions the trial
    pipeline calls, so a disagreement would be a real disagreement and not a difference of method.

    The first collection of a session has no preceding one; its window opens at the first logged
    coordinate, and `is_first` marks it so it can be excluded from any window-length statistic.

    NOT identical to df_trials_clean by construction, and should not be assumed so: that table
    also drops reshuffles (a spawn batch with no collection) and marks degenerate windows. Those
    are trial-pipeline decisions and are deliberately not reproduced here -- this stays a plain
    record of the log. Compare on `dur_s` and the geometry columns.
    """
    L = json.load(open(row['log']))
    ids = _ids(row)
    coll = sorted(L.get('collected', []) or [], key=lambda c: c.get('time', 0))
    # Spawn-batch times, used only to FLAG a window that spans a board reshuffle (a batch that
    # produced no collection). df_trials_clean treats such a reshuffle as its own trial, so a
    # collection-to-collection window silently swallows it -- on JPAS_0168 that is the one trial
    # out of 77 where the two tables disagree, by 101 s. Flagging it makes the comparison exact
    # without re-implementing the trial builder's rules here.
    batch_t = sorted({sp.get('time') for sp in (L.get('spawns') or []) if sp.get('time') is not None})
    if geometry:
        t_xy, x_xy, y_xy = geom.load_coords(L)
        t_th, theta = geom.load_theta(L)
        w = (L.get('worlds') or [{}])[0]
        W, H = int(w.get('width', 0)), int(w.get('height', 0))
        t0_session = float(t_xy[0]) if len(t_xy) else 0.0
    rows = []
    prev_t = None
    for k, c in enumerate(coll):
        eff = c.get('effect')
        mult = c.get('multiplier')
        t_end = c.get('time')
        rec = dict(**ids, n=k, t_ms=t_end,
                   effect=eff, texture=c.get('texture'),
                   valence=VALENCE.get(eff, np.nan),
                   is_positive=VALENCE.get(eff, 0) > 0,
                   is_negative=VALENCE.get(eff, 0) < 0,
                   multiplier=mult, drops=_drops(eff, mult),
                   x=c.get('x'), y=c.get('y'), loc=c.get('loc'),
                   icon_id=c.get('ID'), t_spawned=c.get('t_spawned'),
                   x_char=c.get('x_char'), y_char=c.get('y_char'))
        if geometry:
            t_start = prev_t if prev_t is not None else t0_session
            rec.update(t_start_ms=t_start, t_end_ms=t_end,
                       dur_s=(t_end - t_start) / 1000.0, is_first=prev_t is None)
            tt, xx, yy = geom.slice_track(t_xy, x_xy, y_xy, t_start, t_end)
            msp, ssp = geom.speed_stats(tt, xx, yy)
            _, align = geom.heading_to_target(t_th, theta, t_xy, x_xy, y_xy,
                                              (c.get('x'), c.get('y')), t_start, t_end)
            n_batches = sum(1 for b in batch_t if t_start < b < t_end)
            rec.update(n_coords=len(tt),
                       spans_reshuffle=n_batches > 0,
                       n_reshuffles=n_batches,
                       path_efficiency=geom.path_efficiency(xx, yy),
                       time_in_corner=geom.time_in_corner(xx, yy, W, H),
                       mean_speed=msp, speed_std=ssp, heading_align=align)
        rows.append(rec)
        prev_t = t_end
    return pd.DataFrame(rows)


def session_summary(row, C=None):
    """One row per session: identity, protocol, world, per-effect counts, duration, pay, weight."""
    L = json.load(open(row['log']))
    ids = _ids(row)
    C = session_collections(row) if C is None else C
    ed = L.get('experiment_data', {}) or {}
    coords = np.asarray(L.get('coords_t[ms]/x/y') or [[0]], float)
    dur_min = (coords[-1, 0] - coords[0, 0]) / 60000.0 if len(coords) > 1 else np.nan

    rec = dict(**ids, n_collected=len(C), wall_min=dur_min,
               drops=float(C.drops.sum()) if len(C) else 0.0,
               n_positive=int(C.is_positive.sum()) if len(C) else 0,
               n_negative=int(C.is_negative.sum()) if len(C) else 0,
               n_neutral=int((C.valence == 0).sum()) if len(C) else 0)
    # raw hit rate. NOT a performance score on its own -- it ignores what was on screen, which is
    # what the D score in common/perf_from_log.py exists to correct for. Kept because it is what
    # the log literally says, and because D is only interpretable next to it.
    denom = rec['n_positive'] + rec['n_negative']
    rec['acc_raw'] = rec['n_positive'] / denom if denom else np.nan
    rec['max_multiplier'] = float(C.multiplier.max()) if len(C) and C.multiplier.notna().any() else np.nan
    for eff in sorted(VALENCE):
        rec[f'n_{eff}'] = int((C.effect == eff).sum()) if len(C) else 0
    # welfare / motivation covariates, present only where the rig recorded them
    for k, col in (('weight (g)', 'weight_g'), ('original weight (g)', 'weight_baseline_g'),
                   ('weight_fraction', 'weight_fraction')):
        rec[col] = ed.get(k, np.nan)
    return rec


def build(S, progress=True, verbose=True):
    """Build both grains from a `discover` table. Returns (collections, sessions).

    Every DISCOVERED session is included, not just `use=True`: `use` encodes fitness for the D
    score (it needs a view_scale), but a session with no view_scale still has a perfectly good
    record of what the animal collected. Dropping those here would silently shrink the dataset for
    reasons that do not apply to it. The `use` flag and its `note` travel with each row so a
    downstream filter is an explicit choice.
    """
    todo = [r for _, r in S.iterrows()]
    bar = sidx._progress(todo, f'reading {len(todo)} log(s)', enabled=progress,
                         labels=[str(r.get('session', r['name'])) for r in todo])
    cframes, srows, failed = [], [], []
    for r in bar:
        if hasattr(bar, 'set_postfix_str'):
            bar.set_postfix_str(str(r.get('session', r['name']))[:32])
        try:
            C = session_collections(r)
            C['use'] = r.get('use', True)
            C['note'] = r.get('note', '')
            cframes.append(C)
            s = session_summary(r, C)
            s['use'] = r.get('use', True)
            s['note'] = r.get('note', '')
            srows.append(s)
        except Exception as ex:
            failed.append((r.get('session', r['name']), f'{type(ex).__name__}: {ex}'))

    C = pd.concat(cframes, ignore_index=True) if cframes else pd.DataFrame()
    X = pd.DataFrame(srows)
    if len(X):
        X = X.sort_values(['mouse', 'day', 'time']).reset_index(drop=True)
    if verbose:
        _report(C, X, failed)
    return C, X


def build_all(main_dir, pattern='*', progress=True, verbose=True, **kw):
    """Every mouse under MAIN_DIR. This is the ALL-MICE entry point.

    `require_single_animal` deliberately does NOT apply here -- that guard protects a learning
    CURVE, which is a within-animal measure. A stored dataset spanning animals is the intended
    thing; the `mouse` column keeps them separable.
    """
    main_dir = Path(main_dir).expanduser()
    A = sidx.list_animals(main_dir, pattern=pattern)
    if not len(A):
        return pd.DataFrame(), pd.DataFrame()
    frames = []
    for _, a in A.iterrows():
        if verbose:
            print(f"\n--- {a.animal} ({a.n_sessions} session folder(s)) ---")
        frames.append(sidx.discover(a.path, progress=progress, **kw))
    S = pd.concat(frames, ignore_index=True)
    return build(S, progress=progress, verbose=verbose)


def _report(C, X, failed):
    print('\n=== df_log (log files only) ===')
    if not len(X):
        print('  nothing built'); return
    print(f'  {len(X)} session(s), {len(C)} collection(s), '
          f'{X.mouse.nunique()} animal(s): {sorted(X.mouse.dropna().unique())}')
    print(f'  days     : {X.day.min()} .. {X.day.max()}')
    print(f'  protocols: {X.task.value_counts().to_dict()}')
    print(f'  worlds   : {X.world_sig.nunique()}')
    print(f'  effects  : {C.effect.value_counts().to_dict()}')
    print(f'  drops    : {X.drops.sum():.0f} total')
    print('  per animal:')
    for m, g in X.groupby('mouse'):
        print(f'    {m:12s} {len(g):3d} session(s)  {int(g.n_collected.sum()):5d} collections  '
              f'raw acc {g.acc_raw.mean():.3f}  {g.day.min()} .. {g.day.max()}')
    if failed:
        print(f'  FAILED {len(failed)}:')
        for n, why in failed:
            print(f'    -- {n}: {why}')


def _array_cols(D):
    return [c for c in D.columns if D[c].map(lambda v: isinstance(v, (np.ndarray, list))).any()]


def save(D, path, verbose=True):
    """Persist, stamped with what built it, so a stale cache is recognisable rather than merely old.

    Format follows the extension: .pkl / .parquet / .csv. These tables are flat (no per-trial
    coordinate arrays), so all three round-trip -- but only .pkl carries the stamp, since parquet
    and csv drop `DataFrame.attrs`.
    """
    path = Path(path)
    D = D.copy()
    D.attrs['built'] = dict(when=time.strftime('%Y-%m-%d %H:%M:%S'), build_version=BUILD_VERSION,
                            n_rows=len(D),
                            animals=sorted(D.mouse.dropna().unique().tolist()) if 'mouse' in D else [],
                            tasks=sorted(D.task.dropna().unique().tolist()) if 'task' in D else [])
    arr = _array_cols(D)
    if path.suffix in ('.parquet', '.csv') and arr:
        print(f'  ! {path.suffix} cannot store array columns -- dropping: {", ".join(arr)}')
        D = D.drop(columns=arr)
    if path.suffix == '.parquet':
        D.to_parquet(path, index=False)
    elif path.suffix == '.csv':
        D.to_csv(path, index=False)
        if verbose:
            print('  note: .csv does not carry the build stamp (use .pkl to keep it)')
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


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    C, X = build_all(root)
