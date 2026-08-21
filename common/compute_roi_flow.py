"""
Per-ROI optic flow for a session -- the "generate optic flow files" pipeline step.

Ports the JPAS_0231 recipe (metadata: farneback pyr_scale .5 / levels 3 / winsize 15 /
iterations 3 / poly_n 5 / poly_sigma 1.2, computed on the frame scaled x0.5) and the exact 8-metric
formulas from `Optic_flow_002.ipynb` "CELL 6" (kept byte-for-byte in `pupil_simulator.compute_metrics`):
    mag, x, y, angle, coherence, variance, divergence, radial.
Full-frame Farneback on the scaled grey frame, Gaussian-blur the flow, then per ROI crop (ROI bbox
scaled) and average the metrics. Writes `opticflow_{roi}_{metric}.npy` (len == n_frames; frame 0 = 0)
plus `opticflow_metadata_<mouse>.json`, matching the 231 layout so downstream reads are identical.

Downstream that needs this: grooming (paw+whisker |flow|), licking/whisker/facial-activity, and the
fixed-ROI eye flow.  Run:  python3 compute_roi_flow.py /home/maryam/repo/flow_test/JPAS_0168
"""
from pathlib import Path
import sys
import json
import time
import numpy as np
import cv2

SCALE = 0.5
PAD = 16     # px, pad each scaled ROI before Farneback+blur so the flow/blur has context, then
             # slice the interior back out. Per-ROI crops (vs one full-frame flow) are ~3x faster
             # and more correct for a per-ROI metric (no smoothing shared across ROI boundaries).
FBK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
METRICS = ['mag', 'x', 'y', 'angle', 'coherence', 'variance', 'divergence', 'radial']


def compute_metrics(vx, vy):
    """The eight metrics, exact formulas from JPAS_0231 (pupil_simulator.compute_metrics)."""
    h, w = vx.shape
    mag = np.hypot(vx, vy)
    mfx = float(vx.mean()); mfy = float(vy.mean()); mmag = float(mag.mean())
    coh = float(np.hypot(mfx, mfy) / (mmag + 1e-10))
    var = float(mag.var())
    div = float(np.gradient(vx, axis=1).mean() + np.gradient(vy, axis=0).mean())
    ys, xs = np.meshgrid(np.arange(h) - h / 2, np.arange(w) - w / 2, indexing='ij')
    r = np.hypot(xs, ys) + 1e-10
    rx, ry = xs / r, ys / r
    rad = float((vx * rx + vy * ry).mean())
    return dict(mag=mmag, x=mfx, y=mfy, angle=float(np.arctan2(mfy, mfx)),
                coherence=coh, variance=var, divergence=div, radial=rad)


def run(session_dir, lo=0, hi=None, write=True, progress=True):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    rois = {k: [int(round(c * SCALE)) for c in v] for k, v in sess['rois'].items()}

    cap = cv2.VideoCapture(sess['video_noreflection'])
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    SW, SH = int(sess['width'] * SCALE), int(sess['height'] * SCALE)
    hi = N if hi is None else min(hi, N)
    arr = {roi: {m: np.zeros(N, np.float32) for m in METRICS} for roi in rois}

    # padded crop coords per ROI (clamped) + the interior slice back to the true ROI
    pads = {}
    for roi, (x1, y1, x2, y2) in rois.items():
        px1, py1 = max(0, x1 - PAD), max(0, y1 - PAD)
        px2, py2 = min(SW, x2 + PAD), min(SH, y2 + PAD)
        pads[roi] = (px1, py1, px2, py2, x1 - px1, y1 - py1, x2 - x1, y2 - y1)

    def scaled_gray(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(lo))
    ok, img = cap.read()
    if not ok:
        raise IOError('cannot read first frame')
    prev = scaled_gray(img)
    t0 = time.time()
    for f in range(lo + 1, hi):
        ok, img = cap.read()
        if not ok:
            hi = f
            break
        curr = scaled_gray(img)
        for roi, (px1, py1, px2, py2, ox, oy, w, h) in pads.items():
            raw = cv2.calcOpticalFlowFarneback(prev[py1:py2, px1:px2], curr[py1:py2, px1:px2],
                                               None, **FBK)
            vx = cv2.GaussianBlur(raw[..., 0], (15, 15), 0).astype(np.float64)
            vy = cv2.GaussianBlur(raw[..., 1], (15, 15), 0).astype(np.float64)
            met = compute_metrics(vx[oy:oy + h, ox:ox + w], vy[oy:oy + h, ox:ox + w])
            for m in METRICS:
                arr[roi][m][f] = met[m]
        prev = curr
        if progress and (f - lo) % 5000 == 0:
            print('  %d/%d  %.0fs' % (f - lo, hi - lo, time.time() - t0), flush=True)
    cap.release()

    outdir = d / 'opticflow'
    if write:
        outdir.mkdir(exist_ok=True)
        for roi in rois:
            for m in METRICS:
                np.save(outdir / f'opticflow_{roi}_{m}.npy', arr[roi][m])
        meta = dict(mouse_id=sess['mouse_id'], video_path=sess['video_noreflection'],
                    total_frames=N, fps=sess['fps'],
                    original_resolution=[sess['width'], sess['height']], scale=SCALE,
                    scaled_resolution=[int(sess['width'] * SCALE), int(sess['height'] * SCALE)],
                    farneback_params={k: v for k, v in FBK.items() if k != 'flags'},
                    pad=PAD, metrics=METRICS, rois_original=sess['rois'], rois_scaled=rois,
                    note=('frame 0 = 0 (no previous frame); flow computed per-ROI on a PAD-padded '
                          'crop then sliced to the ROI; metrics per pupil_simulator.compute_metrics'))
        json.dump(meta, open(outdir / f'opticflow_metadata_{sess["mouse_id"]}.json', 'w'), indent=2)
    print('roi flow %d-%d done  ->  %d rois x %d metrics  (%s)' % (
        lo, hi, len(rois), len(METRICS), outdir if write else 'not written'))
    # quick sanity: paw/whisker mag ranges
    for roi in ('paw', 'whisker_left', 'whisker_right', 'mouth'):
        if roi in arr:
            mg = arr[roi]['mag'][lo + 1:hi]
            print('  %-14s mag  med %.3f  p95 %.3f  max %.3f' % (
                roi, np.median(mg), np.percentile(mg, 95), mg.max()))
    return arr


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
