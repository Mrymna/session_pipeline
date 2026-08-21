"""
Eye-detector VALIDATION CLIP: renders a window of the session with the fitted pupil overlaid
on the eye, a live radius(t) trace, and per-frame state pills (OPEN / BLINK / UNRELIABLE) plus
DILATION / CONSTRICTION excursion bands -- so a human can eyeball whether the pupil fit + events
are correct before the full session is trusted.

Events (computed on the window):
  - unusable   = fit not ok (round-pupil gate failed)
  - BLINK      = a sustained run of unusable frames (>= BLINK_MIN_RUN): the eye closes to a slit,
                 the round pupil collapses (dark_frac ~0.1, axis_ratio low) -> not a size change
  - UNRELIABLE = an isolated unusable frame
  - DILATION / CONSTRICTION = turning-point excursions of the blink-bridged, smoothed radius by
                 >= MIN_AMP px (rise = dilation, fall = constriction)

Run: python3 build_eye_clip.py <session_dir> [lo] [hi]
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_pupil as sp

BLINK_MIN_RUN = 2       # >= this many consecutive not-ok frames = a blink
MIN_AMP = 1.0           # px radius excursion to count a dilation / constriction
FPS_OUT = 30
BLUE, VERM, GRAY = (185, 114, 0), (0, 94, 213), (150, 150, 150)   # BGR: dil / con / blink


def _bridge_smooth(radius, ok):
    s = pd.Series(np.where(ok, radius, np.nan))
    s = s.interpolate(limit_direction='both')
    return s.rolling(5, center=True, min_periods=1).mean().to_numpy()


def _runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j - 1)); i = j
        else:
            i += 1
    return out


def _turning_points(r, min_amp):
    """Return dilation and constriction spans as (start,end) index pairs, from the extrema of r."""
    d = np.diff(r)
    sign = np.sign(d)
    idx = [0] + [k + 1 for k in range(len(sign) - 1) if sign[k] != 0 and sign[k + 1] != 0
                 and sign[k] != sign[k + 1]] + [len(r) - 1]
    dil, con = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        if r[b] - r[a] >= min_amp:
            dil.append((a, b))
        elif r[a] - r[b] >= min_amp:
            con.append((a, b))
    return dil, con


def build(session_dir, lo=700, hi=1100):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    eb = sess['rois']['left_eye']
    fps = sess['fps']

    o = sp.run(sess['video_noreflection'], eb, lo=lo, hi=hi, progress=True)
    ok = o['ok']; radius = o['radius']
    unusable = ~ok
    blink = np.zeros(len(ok), bool)
    for a, b in _runs(unusable):
        if b - a + 1 >= BLINK_MIN_RUN:
            blink[a:b + 1] = True
    unreliable = unusable & ~blink
    rad_s = _bridge_smooth(radius, ok & ~blink)
    dil, con = _turning_points(rad_s, MIN_AMP)
    dil_mask = np.zeros(len(ok), bool); con_mask = np.zeros(len(ok), bool)
    for a, b in dil:
        dil_mask[a:b + 1] = True
    for a, b in con:
        con_mask[a:b + 1] = True

    # trial context
    try:
        tdf = pd.read_pickle(d / 'df_trials_clean.pkl')
    except Exception:
        tdf = None

    X1, Y1, X2, Y2 = o['search_box']
    ZOOM = 3
    ew, eh = (X2 - X1) * ZOOM, (Y2 - Y1) * ZOOM
    PANEL_W = 380
    Wc, Hc = 20 + ew + 20 + PANEL_W, 20 + eh + 170
    cap = cv2.VideoCapture(sess['video_noreflection'])
    outdir = d / 'test'; outdir.mkdir(exist_ok=True)
    outp = outdir / f'eye_clip_{lo}_{hi}.mp4'
    vw = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*'mp4v'), FPS_OUT, (Wc, Hc))

    rmin, rmax = np.nanmin(rad_s) - 1, np.nanmax(rad_s) + 1
    trace_x0, trace_w = 30, Wc - 60
    trace_y0, trace_h = 20 + eh + 40, 100

    def rx(i):
        return int(trace_x0 + trace_w * i / max(1, len(ok) - 1))

    def ry(v):
        return int(trace_y0 + trace_h * (1 - (v - rmin) / (rmax - rmin)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(lo))
    for i in range(len(ok)):
        okr, fr = cap.read()
        if not okr:
            break
        f = lo + i
        canvas = np.full((Hc, Wc, 3), 24, np.uint8)
        # eye inset (zoomed) with fit
        eye = fr[Y1:Y2, X1:X2]
        eye = cv2.resize(eye, (ew, eh), interpolation=cv2.INTER_NEAREST)
        if ok[i] and np.isfinite(radius[i]):
            cx, cy = int((o['cx'][i] - X1) * ZOOM), int((o['cy'][i] - Y1) * ZOOM)
            col = VERM if con_mask[i] else BLUE if dil_mask[i] else (0, 230, 0)
            cv2.circle(eye, (cx, cy), int(radius[i] * ZOOM), col, 2)
            cv2.circle(eye, (cx, cy), 2, (0, 0, 255), -1)
        elif blink[i]:
            cv2.putText(eye, 'BLINK', (ew // 2 - 60, eh // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, GRAY, 3)
        canvas[20:20 + eh, 20:20 + ew] = eye

        # header / pills
        x0 = 20 + ew + 20
        state = ('BLINK', GRAY) if blink[i] else ('UNRELIABLE', (0, 165, 255)) if unreliable[i] else \
                ('CONSTRICTION', VERM) if con_mask[i] else ('DILATION', BLUE) if dil_mask[i] else ('OPEN', (0, 230, 0))
        cv2.putText(canvas, sess['mouse_id'], (x0, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        cv2.putText(canvas, 'frame %d  (%.1fs)' % (f, f / fps), (x0, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(canvas, 'radius %.1f px' % (rad_s[i]), (x0, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(canvas, state[0], (x0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, state[1], 3)
        if tdf is not None:
            tr = tdf[(tdf.start_frame <= f) & (tdf.end_frame >= f)]
            if len(tr):
                rr = tr.iloc[0]
                cv2.putText(canvas, 'trial %d  %s%s' % (rr.trial, rr.outcome,
                            '  [BANISH WORLD]' if rr.in_banishment else ''),
                            (x0, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 180), 1)

        # radius trace with event bands
        cv2.rectangle(canvas, (trace_x0, trace_y0), (trace_x0 + trace_w, trace_y0 + trace_h), (60, 60, 60), 1)
        for a, b in _runs(blink):
            cv2.rectangle(canvas, (rx(a), trace_y0), (rx(b), trace_y0 + trace_h), (70, 70, 70), -1)
        for msk, c in ((con_mask, VERM), (dil_mask, BLUE)):
            for a, b in _runs(msk):
                cv2.rectangle(canvas, (rx(a), trace_y0 + trace_h - 6), (rx(b), trace_y0 + trace_h), c, -1)
        pts = [(rx(k), ry(rad_s[k])) for k in range(len(rad_s))]
        for k in range(1, len(pts)):
            cv2.line(canvas, pts[k - 1], pts[k], (220, 220, 220), 1)
        cv2.line(canvas, (rx(i), trace_y0), (rx(i), trace_y0 + trace_h), (0, 255, 255), 1)
        cv2.putText(canvas, 'pupil radius (px)  blue=dilation orange=constriction gray=blink',
                    (trace_x0, trace_y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
        vw.write(canvas)
    cap.release(); vw.release()

    print('=== eye clip %s frames %d-%d ===' % (sess['mouse_id'], lo, hi))
    print('  ok=%.1f%%  blink frames=%d (%d blinks)  unreliable=%d' % (
        100 * ok.mean(), int(blink.sum()), len(_runs(blink)), int(unreliable.sum())))
    print('  dilations=%d  constrictions=%d  radius[min,med,max]=%.1f/%.1f/%.1f' % (
        len(dil), len(con), np.nanmin(rad_s), np.nanmedian(rad_s), np.nanmax(rad_s)))
    print('  wrote', outp)
    return outp


if __name__ == '__main__':
    a = sys.argv
    build(a[1], int(a[2]) if len(a) > 2 else 700, int(a[3]) if len(a) > 3 else 1100)
