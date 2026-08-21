"""
LICK vs GROOM validation clip -- the mouth-state detector, in TWO STYLES.

Why this clip exists: licking and grooming are the pair this pipeline most easily confuses, and the
confusion is not symmetric. Grooming REQUIRES the paw raised to the face, and a raised paw sweeps
across the mouth ROI -- which makes the optic-flow lick detector FALSE-fire. That is why
`compute_mouth_state` gives GROOMING priority over LICKING. This clip is how a human checks that
call frame by frame: every frame the priority rule REASSIGNED is marked, and the reviewer's job is
simply "is the paw up?".

  STYLE "flow"  (JPAS_0168-class -- flow-derived mouth state)
      Licking from the mouth-ROI `fy` 5-12 Hz rhythm; grooming from paw & whisker |flow| gated on
      joystick stillness. Shows live bars for paw / whisker / mouth flow (the three signals the two
      detectors actually run on), the mouth-`fy` trace, and a REASSIGNED marker on every frame where
      the flow lick detector fired but grooming won. 0 CLOSED / 1 LICKING / 2 GROOMING.
      *** There is NO pixel tongue signal in this style: 231's `detect_tongue` does NOT transfer --
      the tongue is too faint and the spout too close, so no mouth-ROI patch oscillates at the lick
      rhythm (tested: dominant 0.4 Hz, not ~7 Hz). So no TONGUE_OUT pill. ***

  STYLE "pixel"  (JPAS_0231-class -- direct pixel mouth read)
      Adds the three pixel features 231 could measure -- `tongue_out` (bright fraction of a muzzle
      patch, off the metal spout bar), `spout_dark` (snout forward at the spout) and `paw_mouth`
      (paw fur in the mouth region, the feature that separates a grooming ONSET from a lick) --
      drawn as live feature boxes, plus the 4-state axis 0 CLOSED / 1 TONGUE_OUT / 2 LICKING /
      3 GROOMING. TONGUE_OUT is the anticipatory state: tongue out at the spout but not a completed
      lick.

Style is AUTO-DETECTED (tongue.npz present -> pixel, else flow) and can be forced with --style.
Forcing "pixel" without tongue.npz is REFUSED rather than faked.

Run:  python3 annotate_mouth_clip.py <session_dir> [lo] [hi] [--style flow|pixel]
"""
from pathlib import Path
import sys
import json
import numpy as np
import cv2

from annotate_clip import pill, draw_rois, TEAL, BROWN, GREY, DIM, AMBER, GREEN

WHITE = (245, 245, 245)
PAW_COL = (200, 140, 60)
WHK_COL = (90, 200, 90)
MOU_COL = (150, 160, 40)
RED = (60, 60, 240)

STATE_FLOW = {0: ('CLOSED', DIM), 1: ('LICKING', TEAL), 2: ('GROOMING', BROWN)}
STATE_PIXEL = {0: ('CLOSED', DIM), 1: ('TONGUE OUT', AMBER), 2: ('LICKING', TEAL),
               3: ('GROOMING', BROWN)}


def resolve_style(d, forced=None):
    has_tongue = (d / 'opticflow' / 'tongue.npz').exists()
    auto = 'pixel' if has_tongue else 'flow'
    style = forced or auto
    if style not in ('flow', 'pixel'):
        sys.exit(f"unknown style '{style}' -- choose flow or pixel")
    if style == 'pixel' and not has_tongue:
        sys.exit("style 'pixel' needs opticflow/tongue.npz (231's direct pixel mouth read), and "
                 "this session has none. For a JPAS_0168-class session that is EXPECTED: the "
                 "tongue is too faint and the spout too close for a pixel patch to work "
                 "(no mouth-ROI patch oscillates at the lick rhythm). Use --style flow.")
    return style


def bar(img, x, y, w, h, val, lo, hi, col, label):
    """A live horizontal level bar for one signal, so the viewer can see WHICH signal is driving
    the current label rather than trusting the pill."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (35, 35, 35), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)
    frac = 0.0 if not np.isfinite(val) else float(np.clip((val - lo) / max(hi - lo, 1e-9), 0, 1))
    cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), col, -1)
    cv2.putText(img, f'{label} {np.nan_to_num(val):.2f}', (x + 6, y + int(h * 0.75)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)


def trace(img, x, y, w, h, f, sig, lo, hi, col, label, state=None, states=None):
    """Windowed trace with the mouth STATE shaded behind it, so a lick bout and a grooming bout are
    visually distinct even where the raw signal looks similar."""
    seg = np.asarray(sig[lo:hi], float)
    n = hi - lo
    vmin, vmax = np.nanpercentile(seg, 2), np.nanpercentile(seg, 98)
    span = max(vmax - vmin, 1e-9)
    cv2.rectangle(img, (x, y), (x + w, y + h), (28, 28, 28), -1)
    xof = lambda i: x + int(i / max(n - 1, 1) * w)
    if state is not None:
        for i in range(n):
            s = int(state[lo + i])
            if s:
                c = states[s][1]
                cv2.line(img, (xof(i), y), (xof(i), y + h),
                         tuple(int(v * 0.35) for v in c), 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)
    pts = [(xof(i), y + int(h * (vmax - np.clip(np.nan_to_num(seg[i], nan=vmin), vmin, vmax)) / span))
           for i in range(n)]
    for i in range(1, n):
        cv2.line(img, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)
    cv2.line(img, (xof(f - lo), y), (xof(f - lo), y + h), WHITE, 1)
    cv2.putText(img, label, (x + 6, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1,
                cv2.LINE_AA)


def feature_box(img, crop, x0, y0, zoom, col, label, value):
    """231's pixel-feature panels: show the actual patch the number is computed from, so a bad
    patch placement is visible instead of hiding inside a scalar."""
    disp = cv2.resize(crop, (crop.shape[1] * zoom, crop.shape[0] * zoom),
                      interpolation=cv2.INTER_NEAREST)
    if disp.ndim == 2:
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    H, W = disp.shape[:2]
    img[y0:y0 + H, x0:x0 + W] = disp
    cv2.rectangle(img, (x0, y0), (x0 + W, y0 + H), col, 2)
    cv2.putText(img, f'{label} {np.nan_to_num(value):.2f}', (x0, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return W


def main(session_dir, lo, hi, forced_style=None):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    rois = json.load(open(sess['roi_config']))
    style = resolve_style(d, forced_style)
    od = d / 'opticflow'
    ms = np.load(od / 'mouth_state.npz', allow_pickle=True)
    state = ms['mouth_state']
    n = len(state)
    states = STATE_PIXEL if style == 'pixel' else STATE_FLOW
    print(f"  style = {style}")

    # the raw signals the two detectors run on
    paw = np.load(od / 'opticflow_paw_mag.npy').astype(float)
    whk = np.maximum(np.load(od / 'opticflow_whisker_left_mag.npy').astype(float),
                     np.load(od / 'opticflow_whisker_right_mag.npy').astype(float))
    mouth_fy = np.load(od / 'opticflow_mouth_y.npy').astype(float)
    groom = np.load(od / 'groom_mask_clean.npy')
    lick_flow = np.load(od / 'licking.npz', allow_pickle=True)['lick_bout'].astype(bool) \
        if (od / 'licking.npz').exists() else np.zeros(n, bool)
    # frames the GROOMING-priority rule took away from the flow lick detector -- the whole reason
    # this clip exists. The reviewer should see the paw UP on every one of them.
    reassigned = lick_flow[:n] & (state == (3 if style == 'pixel' else 2))

    tongue = spout = pawm = None
    if style == 'pixel':
        tz = np.load(od / 'tongue.npz', allow_pickle=True)
        tongue, spout, pawm = tz['tongue_out'], tz['spout_dark'], tz['paw_mouth']

    p75_paw, p75_whk = np.nanpercentile(paw, 75), np.nanpercentile(whk, 75)
    mouth_roi = [int(v) for v in sess['rois']['mouth']]

    src = cv2.VideoCapture(sess['video_original'])
    fps = src.get(cv2.CAP_PROP_FPS) or sess['fps']
    W, H = int(src.get(3)), int(src.get(4))
    src.set(cv2.CAP_PROP_POS_FRAMES, lo)
    outdir = d / 'test'; outdir.mkdir(exist_ok=True)
    outp = outdir / f'mouth_clip_{style}_{lo}_{hi}.mp4'
    wr = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for i in range(hi - lo):
        ok, img = src.read()
        if not ok:
            break
        f = lo + i
        draw_rois(img, rois)
        ph = 150
        ov = img.copy(); cv2.rectangle(ov, (0, 0), (W, ph), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.66, img, 0.34, 0, img)
        cv2.putText(img, f"{sess['mouse_id']}  LICK vs GROOM   [{style} mouth state]", (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        lab, col = states[int(state[f])]
        cv2.putText(img, f'frame {f}    state: {lab}', (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    col, 2, cv2.LINE_AA)

        x, y, w, h = 16, 68, 150, 26
        for sv, (slab, scol) in states.items():
            pill(img, x, y, w, h, slab, int(state[f]) == sv, scol)
            x += w + 8
        if reassigned[f]:
            cv2.putText(img, 'REASSIGNED lick -> groom  (paw should be UP)', (16, ph - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)

        # the driving signals, with their detector thresholds marked in the label
        bx = W - 330
        bar(img, bx, 14, 310, 20, paw[f], 0, np.nanpercentile(paw, 99), PAW_COL,
            f'paw |flow|  (p75={p75_paw:.2f})')
        bar(img, bx, 38, 310, 20, whk[f], 0, np.nanpercentile(whk, 99), WHK_COL,
            f'whisker |flow| (p75={p75_whk:.2f})')
        bar(img, bx, 62, 310, 20, abs(mouth_fy[f]), 0, np.nanpercentile(abs(mouth_fy), 99), MOU_COL,
            'mouth |fy| (lick rhythm)')
        trace(img, bx, 88, 310, 52, f, mouth_fy, lo, hi, MOU_COL,
              'mouth fy  (bg = mouth state)', state, states)

        if style == 'pixel':
            mc = img[mouth_roi[1]:mouth_roi[3], mouth_roi[0]:mouth_roi[2]]
            zx = 16
            for arr, lab_, c in [(tongue, 'tongue_out', AMBER), (spout, 'spout_dark', GREY),
                                 (pawm, 'paw_mouth', PAW_COL)]:
                zx += feature_box(img, mc, zx, H - mc.shape[0] * 2 - 16, 2, c, lab_, arr[f]) + 14
        wr.write(img)
    wr.release(); src.release()
    print(f'wrote {outp}')
    for sv, (slab, _) in states.items():
        print(f'  {slab:11s} {int((state[lo:hi] == sv).sum()):4d} frames')
    print(f'  REASSIGNED (flow said lick, paw said groom): {int(reassigned[lo:hi].sum())} frames')
    return outp


if __name__ == '__main__':
    a = [v for v in sys.argv[1:] if not v.startswith('--')]
    style = None
    for v in sys.argv[1:]:
        if v.startswith('--style'):
            style = v.split('=', 1)[1] if '=' in v else sys.argv[sys.argv.index(v) + 1]
    main(a[0], int(a[1]) if len(a) > 1 else 72800, int(a[2]) if len(a) > 2 else 73000, style)
