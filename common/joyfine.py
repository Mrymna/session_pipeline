"""
Canonical definition of joystick "fine movement" (joy_fine), shared by
build_trial_dfs.py, cluster_paths.py and make_trial_report.py so all four call
sites agree.

joy_fine = local variability of how far the stick is pushed, in the MOTOR BAND:

    mag       = hypot(joystick_x, joystick_y)          # how far pushed from centre
    mag_lp    = low-pass(mag, < LP Hz)                  # keep motor band, drop sensor jitter
    fine(t)   = mag_lp.rolling(WIN, center=True).std()  # local wiggle over ~1/3 s
    joy_fine  = mean_t fine(t)

Why the low-pass: a mouse forelimb cannot correct faster than ~10 Hz, so anything
above that in the joystick trace is sensor jitter, not movement. Band-limiting first
(analogous to the whisker 3-13 Hz band-pass) makes joy_fine a measure of real fine
motor control rather than noise.

APPROXIMATION: the joystick is sampled slightly irregularly (median dt 17 ms ->
~59 Hz, some gaps ~34 ms). We treat the samples as uniform at FS for the filter
design; this is good enough for a comparative per-trial measure and matches the
joystick-fine explainer. A rigorous version would resample to a uniform grid first.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FS  = 58.8     # joystick sample rate (Hz); median dt 17 ms
LP  = 9.0      # motor-band low-pass cutoff (Hz)
WIN = 20       # rolling-window length in samples (~1/3 s)
_MIN = 15      # need > filtfilt padlen (~12) for a stable low-pass


def _lowpass(x):
    b, a = butter(3, LP / (FS / 2), btype='low')
    return filtfilt(b, a, x)


def fine_trace(jx, jy):
    """Per-sample low-passed rolling-std trace of |joystick|. NaNs if too short."""
    jx = np.asarray(jx, float)
    jy = np.asarray(jy, float)
    mag = np.hypot(jx, jy)
    if mag.size < _MIN:
        return np.full(mag.size, np.nan)
    return pd.Series(_lowpass(mag)).rolling(WIN, center=True).std().to_numpy()


def fine_scalar(jx, jy):
    """Per-trial joy_fine = mean of the low-passed rolling-std trace (NaN if short)."""
    tr = fine_trace(jx, jy)
    return float(np.nanmean(tr)) if np.isfinite(tr).any() else np.nan
