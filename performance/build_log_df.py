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


def session_collections(row):
    """One row per collection for ONE session."""
    L = json.load(open(row['log']))
    ids = _ids(row)
    coll = sorted(L.get('collected', []) or [], key=lambda c: c.get('time', 0))
    rows = []
    for k, c in enumerate(coll):
        eff = c.get('effect')
        mult = c.get('multiplier')
        rows.append(dict(**ids, n=k, t_ms=c.get('time'),
                         effect=eff, texture=c.get('texture'),
                         valence=VALENCE.get(eff, np.nan),
                         is_positive=VALENCE.get(eff, 0) > 0,
                         is_negative=VALENCE.get(eff, 0) < 0,
                         multiplier=mult, drops=_drops(eff, mult),
                         x=c.get('x'), y=c.get('y'), loc=c.get('loc'),
                         icon_id=c.get('ID'), t_spawned=c.get('t_spawned'),
                         x_char=c.get('x_char'), y_char=c.get('y_char')))
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
