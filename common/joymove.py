"""
Joystick ("paw") movement events on the eye-frame clock -- shared by the whisker<->paw hypotheses.

This joystick is HELD/active almost continuously (only ~10% of the session is |stick|<2), so a
"still -> moving" onset does not fit. The meaningful motor event is a RE-STEER: a burst in stick
VELOCITY (|d stick/dt|) after relative quiet -- the same signal 231's whisker<->joystick lead/lag
analysis used. We detect velocity-burst PEAKS (find_peaks on smoothed |d stick/dt|), which is exactly
the event 231 aligned to.

Thresholds are ADAPTIVE (median + k*MAD of the speed) so the detector ports across animals without
retuning.

Functions:
  stick_on_frames(log, frame_ms) -> (jx, jy) interpolated onto the per-frame ms grid.
  movement_events(jx, jy, fps, k, min_gap_s) -> frame indices of re-steer velocity-burst peaks.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def stick_on_frames(log, frame_ms):
    """Interpolate joystick x,y onto the per-frame ms grid (frame_ms from fps.frame_to_ms)."""
    joy = np.asarray(log['joystick_t[ms]/x/y'], float)
    t, x, y = joy[:, 0], joy[:, 1], joy[:, 2]
    jx = np.interp(frame_ms, t, x)
    jy = np.interp(frame_ms, t, y)
    return jx, jy


def stick_speed(jx, jy, sigma=1.0):
    """Per-frame stick speed |d stick/dt| (units/frame), lightly smoothed (zero-phase)."""
    v = np.hypot(np.gradient(jx), np.gradient(jy))
    return gaussian_filter1d(v, sigma) if sigma else v


def movement_events(jx, jy, fps, k=1.5, min_gap_s=0.3, sigma=1.0):
    """Frame indices of re-steer velocity-burst peaks.
    height = median(speed) + k*1.4826*MAD ; peaks >= min_gap_s apart."""
    v = stick_speed(jx, jy, sigma)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    thr = med + k * mad
    dist = max(int(round(min_gap_s * fps)), 1)
    peaks, _ = find_peaks(v, height=thr, distance=dist)
    return peaks, dict(speed=v, thr=float(thr), med=float(med), mad=float(mad))
