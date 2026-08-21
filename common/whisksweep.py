"""
Whisker SWEEP + CYCLE for a session -- port of JPAS_0231/whisksweep.py, generalised to this
pipeline: FPS comes from session.json (not hardcoded) and it runs per whisker ROI (this session has
whisker_left + whisker_right; 231 had one).

Whisking = rhythmic protraction/retraction. In the whisker-ROI optic flow it is a COHERENT
oscillation that is DIAGONAL (not horizontal): PCA on the band-passed (x, y) flow gives a principal
axis, and projecting onto it recovers more sweep amplitude than x alone. Pipeline per ROI:

    bx, by = bandpass(whisker_x, whisker_y)     # butter 3-13 Hz, filtfilt (zero-phase)
    axis   = principal eigenvector of cov([bx; by])
    proj   = bx*ax0 + by*ax1
    sweep  = gaussian(|hilbert(proj)|, 2)       # whisking AMPLITUDE (the robust readout)
    cycle  = gaussian(d(unwrap angle)/dt * FPS/2pi, 3)   # instantaneous rate (Hz)

CAVEAT (Nyquist): FPS/2 Hz Nyquist; at 30 fps -> 15 Hz, so CYCLE(Hz) is trustworthy only below
~13 Hz. SWEEP (amplitude) is the more robust readout. CAVEAT (band-pass, not a head model): head
motion is slow/transient so band-limiting to 3-13 Hz removes most of it; BUT licking (~7.5 Hz) is
in-band and can leak -- cross-check the lick mask when sweep is used near reward.

`run()` writes opticflow/whisker.npz: per-ROI sweep/cycle + a COMBINED sweep (mean of the pads) +
a WHISKING bout mask (sustained high sweep). Run: python3 whisksweep.py <session_dir>
"""
from pathlib import Path
import sys
import json
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d, binary_dilation

BAND = (3.0, 13.0)      # whisking band (Hz)
SMOOTH = 2.0            # sweep envelope smoothing (frames)
WHISK_K = 1.0          # whisking bout: sweep > median + K*MAD ...
WHISK_MIN = 6          # ... sustained >= this many frames (0.2s @30fps)
WHISK_GAP = 4          # ... bridging gaps <= this many frames


def _bandpass(sig, fps):
    b, a = butter(3, [BAND[0] / (fps / 2), BAND[1] / (fps / 2)], btype='band')
    return filtfilt(b, a, np.asarray(sig, float))


def principal_axis(bx, by):
    """Unit vector of the dominant oscillation axis + major/minor eigenvalue ratio (1-D-ness).
    Sign made deterministic (largest-|component| positive) for reproducibility."""
    C = np.cov(np.vstack([bx, by]))
    evals, evecs = np.linalg.eigh(C)
    pax = evecs[:, -1]
    if pax[int(np.argmax(np.abs(pax)))] < 0:
        pax = -pax
    return pax, float(evals[-1] / max(evals[0], 1e-12))


def decompose(wx, wy, fps):
    """Whisker decomposition along the principal whisking axis. Returns dict of session-length
    arrays (sweep, cycle, proj) plus axis/axis_deg/ratio scalars."""
    bx, by = _bandpass(wx, fps), _bandpass(wy, fps)
    pax, ratio = principal_axis(bx, by)
    proj = bx * pax[0] + by * pax[1]
    analytic = hilbert(proj)
    sweep = gaussian_filter1d(np.abs(analytic), SMOOTH)
    phase = np.unwrap(np.angle(analytic))
    cycle = gaussian_filter1d(np.diff(phase, prepend=phase[0]) * fps / (2 * np.pi), 3)
    return dict(sweep=sweep, cycle=cycle, proj=proj, bx=bx, by=by,
                axis=pax, axis_deg=float(np.degrees(np.arctan2(pax[1], pax[0]))), ratio=ratio)


def _whisking_bouts(sweep, k=WHISK_K, min_run=WHISK_MIN, gap=WHISK_GAP):
    """Active-whisking mask: sweep above a robust threshold, sustained, small gaps bridged."""
    med = np.median(sweep)
    mad = np.median(np.abs(sweep - med)) + 1e-9
    m = sweep > med + k * mad
    # bridge short gaps
    d = np.diff(m.astype(int))
    on = sorted(list(np.where(d == 1)[0] + 1) + ([0] if m[0] else []))
    off = sorted(list(np.where(d == -1)[0] + 1) + ([len(m)] if m[-1] else []))
    for i in range(len(off) - 1):
        if on[i + 1] - off[i] <= gap:
            m[off[i]:on[i + 1]] = True
    # drop short runs
    d = np.diff(m.astype(int))
    on = sorted(list(np.where(d == 1)[0] + 1) + ([0] if m[0] else []))
    off = sorted(list(np.where(d == -1)[0] + 1) + ([len(m)] if m[-1] else []))
    for s, e in zip(on, off):
        if e - s < min_run:
            m[s:e] = False
    return m


def _whisker_rois(sess):
    return [k for k in sess['rois'] if k.startswith('whisker')]


def run(session_dir, write=True, verbose=True):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    fps = sess['fps']
    od = d / 'opticflow'
    rois = _whisker_rois(sess)

    out = {}
    sweeps = []
    for roi in rois:
        wx = np.load(od / f'opticflow_{roi}_x.npy').astype(float)
        wy = np.load(od / f'opticflow_{roi}_y.npy').astype(float)
        dec = decompose(wx, wy, fps)
        out[f'sweep_{roi}'] = dec['sweep'].astype(np.float32)
        out[f'cycle_{roi}'] = dec['cycle'].astype(np.float32)
        out[f'axis_deg_{roi}'] = dec['axis_deg']
        out[f'ratio_{roi}'] = dec['ratio']
        sweeps.append(dec['sweep'])
        if verbose:
            mag = np.load(od / f'opticflow_{roi}_mag.npy').astype(float)
            print(f'  {roi:14s} axis={dec["axis_deg"]:.0f}deg  1-D ratio={dec["ratio"]:.1f}  '
                  f'mean sweep={dec["sweep"].mean():.3f}  median cycle={np.median(dec["cycle"]):.1f}Hz  '
                  f'corr(sweep,mag)={np.corrcoef(dec["sweep"], mag)[0, 1]:.2f}')

    # COMBINED sweep across pads = the overall whisking-effort signal; bouts from it
    sweep_comb = np.mean(sweeps, axis=0).astype(np.float32)
    whisking = _whisking_bouts(sweep_comb)
    out['sweep'] = sweep_comb                     # canonical combined sweep
    out['whisking'] = whisking
    out['rois'] = rois
    out['fps'] = fps

    if verbose:
        nb = int(np.diff((np.r_[0, whisking.astype(int), 0])).clip(min=0).sum())
        print(f"=== {sess['mouse_id']} whisker sweep/cycle ===")
        print(f"  combined sweep mean {sweep_comb.mean():.3f}   "
              f"WHISKING {whisking.sum()} frames ({100 * whisking.mean():.1f}%), {nb} bouts")

    if write:
        np.savez(od / 'whisker.npz', **out)
        if verbose:
            print(f"  wrote {od / 'whisker.npz'}")
    return out


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
