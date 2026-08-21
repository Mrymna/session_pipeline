"""
Canonical heading-to-reward ("green arrow" direction) measure, shared by build_trial_dfs.py,
build_clean_trials.py, cluster_paths.py (via the df column) and make_trial_report.py.

The avatar's heading is log.json `angles_t[ms]/theta` (radians, the green direction arrow).
Combined with the avatar position (`coords_t[ms]/x/y`) and the trial's reward position
(reward_x/reward_y, world 0-2000) it gives a GOAL-DIRECTEDNESS signal:

    bearing(t)       = atan2(reward_y - y(t), reward_x - x(t))   # direction TO the reward
    heading_error(t) = wrap(theta(t) - bearing(t))              # 0 = pointing at reward
    R                = |mean(exp(i*heading_error))|  in [0, 1]  # resultant length

TWO complementary readouts of the same mean resultant vector (magnitude R at angle mu):
  R          = |mean(exp(i*err))|      in [0, 1]   -- MEAN VECTOR LENGTH of the heading
               error: how CONCENTRATED the heading-to-reward angle is (does NOT say toward
               vs away: a heading held steadily AWAY from the reward also gives a high R).
  alignment  = mean(cos(err)) = R*cos(mu)  in [-1, +1]  -- FORWARD ALIGNMENT to the reward:
               +1 = always pointing AT the reward, -1 = always pointing AWAY, 0 = perpendicular
               / scattered. This is the behaviourally meaningful "faces the reward" measure and
               the stronger cluster separator.
Both are path-GEOMETRY / goal measures, independent of the motor features (joy_fine, whisker),
so they are non-circular axes for classification and validation. Measured per cluster:
Direct  R~0.57, alignment~+0.38 (toward);  Corner-dwelling R~0.39, alignment~-0.23 (AWAY).

Timeout trials have no collected reward; if reward_x/reward_y is NaN the functions return
empty / NaN and the caller should treat heading_R as missing.
"""
from pathlib import Path
import json
import numpy as np

_ROOT = Path(__file__).resolve().parent


def load_arrays(log=None, path=None):
    """Return (theta_t Nx2 [t_ms, rad], coords Mx3 [t_ms, x, y]) from a loaded log dict
    or from log.json on disk."""
    if log is None:
        log = json.load(open(path or _ROOT / 'log.json'))
    theta_t = np.asarray(log['angles_t[ms]/theta'], float)
    coords = np.asarray(log['coords_t[ms]/x/y'], float)
    return theta_t, coords


def heading_error(theta_t, coords, reward_xy, s_ms, e_ms):
    """Per-frame wrapped heading error over [s_ms, e_ms] (evaluated on theta timestamps,
    avatar position interpolated from coords). Returns (t_ms, err_rad); empty if no samples
    or the reward position is unknown."""
    rx, ry = reward_xy
    if not (np.isfinite(rx) and np.isfinite(ry)):
        return np.array([]), np.array([])
    m = (theta_t[:, 0] >= s_ms) & (theta_t[:, 0] <= e_ms)
    tt, th = theta_t[m, 0], theta_t[m, 1]
    if tt.size == 0:
        return tt, np.array([])
    x = np.interp(tt, coords[:, 0], coords[:, 1])
    y = np.interp(tt, coords[:, 0], coords[:, 2])
    bearing = np.arctan2(ry - y, rx - x)
    err = np.angle(np.exp(1j * (th - bearing)))
    return tt, err


def resultant_length(err):
    """MEAN VECTOR LENGTH R = |mean(exp(i*err))| in [0, 1] (NaN if empty). Concentration of
    the heading-error angle; does NOT distinguish toward from away."""
    err = np.asarray(err, float)
    return float(np.abs(np.mean(np.exp(1j * err)))) if err.size else np.nan


def forward_alignment(err):
    """FORWARD ALIGNMENT = mean(cos(err)) in [-1, +1] (NaN if empty). +1 = always facing the
    reward, -1 = always facing away, 0 = perpendicular / scattered."""
    err = np.asarray(err, float)
    return float(np.mean(np.cos(err))) if err.size else np.nan


def row_heading(row, theta_t, coords, s_ms=None, e_ms=None):
    """Heading error + R (mean vector length) + alignment (mean cos) for one trial row. Uses
    the row's trial_start_ms/trial_end_ms unless s_ms/e_ms are given (clean-window bounds).
    Returns (t_ms, err, R, alignment)."""
    s = row['trial_start_ms'] if s_ms is None else s_ms
    e = row['trial_end_ms'] if e_ms is None else e_ms
    tt, err = heading_error(theta_t, coords, (row.get('reward_x'), row.get('reward_y')), s, e)
    return tt, err, resultant_length(err), forward_alignment(err)


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_pickle(_ROOT / 'df_trials.pkl')
    theta_t, coords = load_arrays()
    byR, byA = {}, {}
    for i in df.index:
        nm = df.loc[i].get('cluster_name')
        if nm in ('Direct', 'Corner-dwelling'):
            _, err, R, al = row_heading(df.loc[i], theta_t, coords)
            if np.isfinite(R):
                byR.setdefault(nm, []).append(R); byA.setdefault(nm, []).append(al)
    for nm in byR:
        print('%-16s R(vector length) = %.3f   alignment(mean cos) = %+.3f (n=%d)'
              % (nm, np.mean(byR[nm]), np.mean(byA[nm]), len(byR[nm])))
