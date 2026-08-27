"""
Error / conflict labels on df_trials_clean (prerequisite for the whisker<->paw hypotheses H1/H2).

Run cluster_paths.py FIRST (this needs `cluster_name`).

What the data says (JPAS_0168, verified before thresholding -- 231 house rule):
  - The collection radius is a HARD ~246 world units (avatar collects any icon within ~246; the
    collected-target closest approach is 245+-3 across ALL outcomes). So there is no graded "how
    close did it get" -- a trial either collects (reaches 246) or not.
  - On single_reward trials the mouse stays FAR from banish icons (min dist median ~996, min ~820)
    => a banish NEAR-MISS essentially never happens; the banishments are well separated. So geometric
    "near-miss to a banish" flags ~0 trials and is NOT used.
  - OVERSHOOT of the collected target (approach<400 -> retreat>700 -> return) flags only ~4/43
    single_reward trials and the set is threshold-unstable => recorded for transparency but too
    sparse to stratify on.

Therefore:
  - the DISCRETE error = `banish` (hit the banishment, n=17); correct = single_reward.
  - the well-powered CONFLICT / difficulty stratifier = the Corner-dwelling path cluster (n=47),
    which is low-efficiency, long, corner-dwelling, negative heading alignment.
  - H1's post-error test (prev trial = banish, n<=17) is complemented by a better-powered
    post-CONFLICT test (prev trial = Corner-dwelling).

Columns added to df_trials_clean.pkl:
  min_dist_target, min_dist_banish (world units) ; overshoot (bool) ;
  is_error_banish, is_error, is_correct, is_conflict (bool) ;
  prev_is_error, prev_is_conflict (bool, from the preceding trial in index order).

Run:  python3 label_trials.py /home/maryam/repo/flow_test/JPAS_0168
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

COLLISION_R = 246.0     # world units, derived empirically (median collected-target closest approach)
APPROACH = 400.0        # overshoot: first-approach band
RETREAT = 700.0         # overshoot: must retreat past this after approaching, then return to collect


def _dist_to_target(r):
    x = np.asarray(r['coord_x'], float); y = np.asarray(r['coord_y'], float)
    if x.size < 3 or not np.isfinite(r.get('target_x', np.nan)):
        return None
    return np.hypot(x - r['target_x'], y - r['target_y'])


def _min_dist_banish(r):
    x = np.asarray(r['coord_x'], float); y = np.asarray(r['coord_y'], float)
    bx = [ic['x'] for ic in (r['icons'] or []) if ic.get('effect') == 'banish' and np.isfinite(ic.get('x', np.nan))]
    by = [ic['y'] for ic in (r['icons'] or []) if ic.get('effect') == 'banish' and np.isfinite(ic.get('x', np.nan))]
    if x.size < 2 or not bx:
        return np.nan
    return float(min(np.nanmin(np.hypot(x - bxi, y - byi)) for bxi, byi in zip(bx, by)))


def _overshoot(r):
    d = _dist_to_target(r)
    if d is None or r.get('outcome') != 'single_reward':
        return False
    below = np.where(d < APPROACH)[0]
    if below.size == 0:
        return False
    tail = d[below[0]:]
    return bool(tail.max() > RETREAT and d[-1] < COLLISION_R + 60)


def run(session_dir, write=True, verbose=True, dfc=None, sess=None, figures=True):
    """Add error / conflict labels to one session's trial table.

    `dfc` / `sess` (optional) take the trial table and the session fields IN MEMORY instead of
    reading `df_trials_clean.pkl` and `session.json`; `figures=False` skips the diagnostic PNGs.
    Together these let the performance folder label a raw server session without reading or
    writing anything in the session directory. Passing nothing keeps the original behaviour.
    """
    d = Path(session_dir).resolve()
    if sess is None:
        sess = json.load(open(d / 'session.json'))
    df = dfc if dfc is not None else pd.read_pickle(d / "df_trials_clean.pkl")
    if 'cluster_name' not in df.columns:
        raise SystemExit('run cluster_paths.py first (need cluster_name for the conflict stratifier)')

    df['min_dist_target'] = [np.nanmin(_dist_to_target(r)) if _dist_to_target(r) is not None else np.nan
                             for _, r in df.iterrows()]
    df['min_dist_banish'] = [_min_dist_banish(r) for _, r in df.iterrows()]
    df['overshoot'] = [_overshoot(r) for _, r in df.iterrows()]

    df['is_error_banish'] = df['outcome'].eq('banish')
    # PRIMARY error = banish only (clean). overshoot is kept separate: the overshoot check
    # (debug/overshoot_check.png) showed it conflates 2 genuine terminal overshoots (trials 41, 46)
    # with 2 long Corner-dwelling WANDERING trials (0, 51), so it is NOT folded into is_error.
    df['is_error'] = df['is_error_banish'] & df['analyze']
    df['is_error_broad'] = (df['is_error_banish'] | df['overshoot']) & df['analyze']   # optional
    df['is_correct'] = df['outcome'].eq('single_reward') & ~df['overshoot'] & df['analyze']
    df['is_conflict'] = df['cluster_name'].eq('Corner-dwelling') & df['analyze']

    # previous-trial (index order) state, for the post-error / post-conflict tests
    df['prev_is_error'] = df['is_error_banish'].shift(1, fill_value=False)
    df['prev_is_conflict'] = df['cluster_name'].shift(1).eq('Corner-dwelling')

    if verbose:
        a = df[df.analyze]
        print(f"=== {sess['mouse_id']} trial labels ===")
        print(f"  collision radius R = {COLLISION_R:.0f} world units")
        print(f"  ERROR (banish)      : {int(df['is_error_banish'].sum())}")
        print(f"  overshoot (recorded): {int(df['overshoot'].sum())}  (too sparse to stratify)")
        print(f"  is_error (banish)   : {int(df['is_error'].sum())}   is_correct: {int(df['is_correct'].sum())}"
              f"   is_error_broad(+OS): {int(df['is_error_broad'].sum())}")
        print(f"  CONFLICT (Corner)   : {int(df['is_conflict'].sum())} / {len(a)} analyzable")
        print(f"  min_dist_banish on single_reward: median {a.loc[a.outcome=='single_reward','min_dist_banish'].median():.0f}"
              f" (=> no banish near-miss)")
        print(f"  post-ERROR trials    : {int((df['prev_is_error'] & df['analyze']).sum())}")
        print(f"  post-CONFLICT trials : {int((df['prev_is_conflict'] & df['analyze']).sum())}")

    if write:
        df.to_pickle(d / 'df_trials_clean.pkl')
        if verbose:
            print(f"  wrote {d / 'df_trials_clean.pkl'}")
    return df


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
