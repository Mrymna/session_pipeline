"""
APPROXIMATE saccade estimate from the P-CR vector (port of JPAS_0231/pcr/detect_saccades_pcr.py).

P-CR = pupil_centre - corneal_reflection(glint). Because the glint is fixed on the cornea while the
pupil moves when the eye rotates, P-CR is HEAD-MOTION INVARIANT by construction (a head shift moves
pupil and glint together and cancels). A saccade is a FAST step (~1-2 frames) in P-CR that then
HOLDS; single-frame jitter reverts (fails the hold) and pupil-size drift is a slow ramp (never trips
the fast step). Both `cx/cy` and `glint_x/y` come from segment_pupil's pupil_track.npz -- no separate
cr_track needed here.

** APPROXIMATE for this mouse ** (Maryam asked for an estimate): (1) the pupil centre itself is only
approximate here (the fit is approximate), so P-CR is noisier than 231's; (2) there is no
head_motion.npz for this session, so the numerical head-motion sanity guards 231 used (corr with
|head|, % in top-decile head) CANNOT be computed -- the method is head-clean by construction but that
is not independently verified. Blink + squint + failed-fit frames are excluded (a squint resizes the
fit and would fake a step). Treat the count as an order-of-magnitude estimate, not a firm number.

Run:  python3 detect_saccades.py <session_dir>   -> opticflow/saccades.npz
"""
from pathlib import Path
import sys
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

K = 3          # frames averaged on each side of the step
GAP = 2        # frames skipped across the transition (the saccade itself)
K_STEP = 6     # MAD multiplier on the fast-step magnitude
MIN_GAP = 6    # frames, min spacing between saccades
HOLD_FR = 0.5  # the displacement must persist to at least this fraction of the step


def _interp(a):
    a = np.asarray(a, float)
    i = np.arange(len(a)); m = ~np.isnan(a)
    return np.interp(i, i[m], a[m]) if m.any() else a


def _fast_step(sig):
    N = len(sig); o = K + GAP
    pad = np.pad(sig, o, mode='edge')
    b = np.empty(N); a = np.empty(N)
    for i in range(N):
        j = i + o
        b[i] = pad[j - GAP - K:j - GAP].mean()
        a[i] = pad[j + GAP:j + GAP + K].mean()
    return a - b


def _hold(sig):
    N = len(sig); pad = np.pad(sig, 15, mode='edge'); o = 15
    h = np.empty(N)
    for i in range(N):
        j = i + o
        h[i] = pad[j + 2:j + 12].mean() - pad[j - K:j].mean()
    return h


def run(session_dir, write=True, verbose=True):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    fps = sess['fps']
    tr = np.load(d / 'opticflow' / 'pupil_track.npz', allow_pickle=True)
    ev = np.load(d / 'opticflow' / 'eye_events.npz', allow_pickle=True)
    N = len(tr['radius'])

    px, py = _interp(tr['cx']), _interp(tr['cy'])
    gx, gy = _interp(tr['glint_x']), _interp(tr['glint_y'])
    pcrx, pcry = px - gx, py - gy
    # exclude blink, squint and failed-fit frames (a squint resizes the fit -> fake step)
    unusable = ev['blink'] | ev['squint'] | np.isnan(tr['radius'])

    sx, sy = _fast_step(pcrx), _fast_step(pcry)
    fast = gaussian_filter1d(np.hypot(sx, sy), 0.7)
    held = np.hypot(_hold(pcrx), _hold(pcry))

    good = ~unusable
    m = np.median(fast[good])
    thr = m + K_STEP * 1.4826 * np.median(np.abs(fast[good] - m))
    sc = fast.copy(); sc[unusable] = 0
    cand, _ = find_peaks(sc, height=thr, distance=MIN_GAP)
    saccade = np.array([int(p) for p in cand
                        if not unusable[max(0, p - 3):p + 4].any()
                        and held[p] > HOLD_FR * fast[p]], dtype=int)
    ang = np.arctan2(sy[saccade], sx[saccade])
    amp = fast[saccade]

    if verbose:
        print(f"=== {sess['mouse_id']} saccade ESTIMATE (P-CR, approximate) ===")
        print(f"  saccades : {len(saccade)}  ({len(saccade)/(N/fps/60):.1f}/min)   thr={thr:.2f}px")
        print(f"  P-CR baseline offset |pupil-glint| median = {np.median(np.hypot(pcrx,pcry)):.1f}px")
        print("  NOTE approximate: pupil centre is approximate here; no head_motion.npz to verify "
              "head-cleanliness numerically (P-CR is head-invariant by construction).")

    if write:
        out = d / 'opticflow' / 'saccades.npz'
        np.savez(out, saccade=saccade, fast=fast, held=held, step_x=sx, step_y=sy,
                 angle=ang, amp=amp, thr=float(thr), pcr_x=pcrx, pcr_y=pcry,
                 unusable=unusable, fps=fps, approximate=True)
        if verbose:
            print(f"  wrote {out}")
    return dict(saccade=saccade, angle=ang, amp=amp)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
