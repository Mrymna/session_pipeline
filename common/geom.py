"""
Task-agnostic path-geometry + heading helpers, computed from the log alone.
World size is passed in (2000 for timeout_double, 2400 for banish_multiplier) -- never
hardcoded. All times are the log ms clock (coords_t[ms]/x/y, angles_t[ms]/theta).
"""
import numpy as np


def load_coords(log):
    """(t_ms, x, y) avatar trajectory in world units."""
    a = np.asarray(log['coords_t[ms]/x/y'], float)
    return a[:, 0], a[:, 1], a[:, 2]


def load_theta(log):
    """(t_ms, theta_rad) avatar heading (the green direction arrow)."""
    a = np.asarray(log['angles_t[ms]/theta'], float)
    return a[:, 0], a[:, 1]


def slice_track(t, x, y, s_ms, e_ms):
    m = (t >= s_ms) & (t <= e_ms)
    return t[m], x[m], y[m]


def path_efficiency(x, y):
    """net straight-line displacement / total path length, in [0, 1] (1 = straight)."""
    if len(x) < 2:
        return np.nan
    steps = np.hypot(np.diff(x), np.diff(y))
    total = steps.sum()
    net = np.hypot(x[-1] - x[0], y[-1] - y[0])
    return float(net / total) if total > 0 else np.nan


def time_in_corner(x, y, world_w, world_h, frac=0.2, t=None):
    """Fraction in any of the four world corners (each a frac x frac box).

    ⚠️ SAMPLE-weighted by default, and that UNDER-COUNTS dwelling. The rig logs `coords` only
    while the avatar is moving -- ~31% (JPAS_0168) to 36% (JPAS_0231) of session time falls in
    coord gaps > 2 s -- so an animal that PARKS in a corner emits almost no samples there and
    its corner time is under-stated. Measured on JPAS_0168: time-weighted 0.339 vs
    sample-weighted 0.243 (mean +0.096; 33/77 trials shift by >0.10), though the two correlate
    at r=0.96 so the RANKING is largely preserved.

    Pass `t` (the sample timestamps, ms) to weight each sample by the time it represents, which
    is the correct measure of "time in corner". Left OFF by default so the JPAS_0168 numbers
    already published do not change silently -- turn it on for BOTH tasks together, or the two
    become incomparable.
    """
    if len(x) == 0:
        return np.nan
    cw, ch = world_w * frac, world_h * frac
    in_corner = (((x < cw) | (x > world_w - cw)) & ((y < ch) | (y > world_h - ch)))
    if t is None:
        return float(np.mean(in_corner))
    t = np.asarray(t, float)
    dt = np.diff(np.concatenate([t, t[-1:]]))       # time each sample stands for
    if dt.sum() <= 0:
        return float(np.mean(in_corner))
    return float(np.average(in_corner, weights=dt))


def speed_stats(t, x, y):
    """(mean_speed, speed_std) in world units / second."""
    if len(x) < 2:
        return np.nan, np.nan
    dt = np.diff(t) / 1000.0
    dt[dt <= 0] = np.nan
    sp = np.hypot(np.diff(x), np.diff(y)) / dt
    return float(np.nanmean(sp)), float(np.nanstd(sp))


def heading_to_target(t_th, theta, t_xy, x, y, target_xy, s_ms, e_ms):
    """Per-frame heading error toward a FIXED target over [s_ms, e_ms], plus its
    forward-alignment scalar. err = wrap(theta - bearing_to_target); alignment = mean(cos err)
    in [-1, +1] (+1 = always facing the target). Avatar position interpolated onto theta stamps.
    Returns (err_array, alignment)."""
    tx, ty = target_xy
    if not (np.isfinite(tx) and np.isfinite(ty)):
        return np.array([]), np.nan
    m = (t_th >= s_ms) & (t_th <= e_ms)
    tt, th = t_th[m], theta[m]
    if tt.size == 0:
        return np.array([]), np.nan
    ax = np.interp(tt, t_xy, x)
    ay = np.interp(tt, t_xy, y)
    bearing = np.arctan2(ty - ay, tx - ax)
    err = np.angle(np.exp(1j * (th - bearing)))
    return err, float(np.mean(np.cos(err)))
