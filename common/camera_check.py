"""
Pre-flight CAMERA / VIEW-STABILITY sanity check.

Purpose: catch sessions where the framing of the mouse changes partway through -- a real
camera move OR (more commonly) the head-fixed mouse repositioning its head -- because then
the FIXED ROI boxes (eye, whiskers, ...) no longer cover their targets for the whole session,
which silently corrupts every fixed-ROI signal. This must run BEFORE df_trials_clean or any
optic-flow generation so a bad session is flagged up front, not discovered downstream.

It is a fast, approximate heuristic (samples ~120 frames, no full decode). It watches the EYE
region, whose framing is the first thing to drift, via two complementary signals:
  - eye-box MEAN BRIGHTNESS  -- rises when bright snout/cheek fur enters the box as the head
    shifts (this is what cleanly caught JPAS_0168_cameraMoved: 151 -> 170 at ~frame 50000);
  - dark-pupil CENTROID position in a widely-padded eye region -- where the eye actually sits.
A sustained late-vs-early change in either flags `moved`, with the onset frame/time reported.

Real example (JPAS_0168_cameraMoved): brightness +12.7% at frame ~50000 (~28 min, ~67% through,
trial ~53/74) -> moved=True.
"""
from pathlib import Path
import numpy as np


def _rolling_median(a, w=3):
    a = np.asarray(a, float)
    out = np.full_like(a, np.nan)
    h = w // 2
    for i in range(len(a)):
        lo, hi = max(0, i - h), min(len(a), i + h + 1)
        out[i] = np.nanmedian(a[lo:hi])
    return out


def _onset_frame(fr, sig, early, late):
    """First sampled frame where a rolling median of `sig` crosses the early<->late midpoint
    and stays on the late side for the rest of the session (a SUSTAINED shift, not a blip)."""
    mid = (early + late) / 2.0
    rm = _rolling_median(sig, 3)
    up = late > early
    on_late = (rm > mid) if up else (rm < mid)
    for i in range(len(rm)):
        if on_late[i] and np.all(on_late[i:][~np.isnan(rm[i:])]):
            return int(fr[i])
    return None


def detect_view_shift(video_path, eye_bbox, fps, n_frames=None, n_samples=120,
                      pad_frac=0.6, bright_rel_thresh=0.08, pos_frac_thresh=0.30):
    """Sample the session and decide whether the eye framing drifts. Returns a report dict:
      moved (bool), onset_frame, onset_s, onset_frac (fraction through the session),
      brightness_rel_change, pos_shift_px, and the raw sampled traces (for a diagnostic plot).
    Thresholds: brightness relative change > `bright_rel_thresh` (default 8%), OR pupil-centroid
    drift > `pos_frac_thresh` x the shorter eye-box side (default 30%)."""
    import cv2
    x1, y1, x2, y2 = eye_bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_frac), int(bh * pad_frac)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f'cannot open {video_path}')
    N = n_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    X1, Y1 = max(0, x1 - px), max(0, y1 - py)
    X2, Y2 = min(W, x2 + px), min(H, y2 + py)

    samples = np.linspace(int(N * 0.01), int(N * 0.99), n_samples).astype(int)
    fr_idx, meanB, cx, cy = [], [], [], []
    for f in samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        wide = g[Y1:Y2, X1:X2].astype(float)
        thr = np.percentile(wide, 15)                 # darkest ~15% = pupil
        ys, xs = np.where(wide < thr)
        fr_idx.append(int(f))
        meanB.append(float(g[y1:y2, x1:x2].mean()))
        cx.append(float(xs.mean() + X1) if len(xs) else np.nan)
        cy.append(float(ys.mean() + Y1) if len(ys) else np.nan)
    cap.release()

    fr = np.array(fr_idx)
    meanB, cx, cy = np.array(meanB), np.array(cx), np.array(cy)
    q = max(3, len(fr) // 4)                           # early / late quartile windows
    eB, lB = np.nanmedian(meanB[:q]), np.nanmedian(meanB[-q:])
    eX, lX = np.nanmedian(cx[:q]), np.nanmedian(cx[-q:])
    eY, lY = np.nanmedian(cy[:q]), np.nanmedian(cy[-q:])

    bright_rel = abs(lB - eB) / eB if eB else 0.0
    pos_shift = float(np.hypot(lX - eX, lY - eY))
    moved_bright = bright_rel > bright_rel_thresh
    moved_pos = pos_shift > pos_frac_thresh * min(bw, bh)
    moved = bool(moved_bright or moved_pos)

    onset_frame = None
    if moved:
        if moved_bright:
            onset_frame = _onset_frame(fr, meanB, eB, lB)
        else:
            sig, e, l = (cy, eY, lY) if abs(lY - eY) >= abs(lX - eX) else (cx, eX, lX)
            onset_frame = _onset_frame(fr, sig, e, l)

    onset_s = (onset_frame / fps) if onset_frame is not None else None
    onset_frac = (onset_frame / N) if onset_frame is not None else None

    return dict(
        moved=moved, moved_by=('brightness' if moved_bright else 'position' if moved_pos else None),
        onset_frame=onset_frame, onset_s=onset_s, onset_frac=onset_frac,
        brightness_rel_change=round(bright_rel, 4),
        brightness_early=round(eB, 1), brightness_late=round(lB, 1),
        pos_shift_px=round(pos_shift, 1), eye_box_min_side=int(min(bw, bh)),
        n_samples=len(fr),
        _traces=dict(frame=fr.tolist(), meanB=meanB.tolist(), cx=cx.tolist(), cy=cy.tolist()),
    )


if __name__ == '__main__':
    import sys, json
    d = Path(sys.argv[1])
    sess = json.load(open(d / 'session.json'))
    rep = detect_view_shift(sess['video_noreflection'], sess['rois']['left_eye'], sess['fps'],
                            n_frames=sess['n_frames'])
    rep.pop('_traces')
    print(json.dumps(rep, indent=2))
