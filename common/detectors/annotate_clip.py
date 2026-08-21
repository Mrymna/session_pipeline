"""
Annotated eye/face VALIDATION CLIP -- the JPAS_0231 layout, in TWO STYLES.

Why two styles: how much you are allowed to SAY about the pupil depends on how well it can be
measured in that animal, and that differs per mouse. Drawing a confident circle and DILATION /
CONSTRICTION pills on an animal whose iris is only approximately fittable is the single easiest way
to make a bad signal look good, so the clip renderer refuses to do it.

  STYLE "approx"  (JPAS_0168-class -- UNSTABLE / approximate iris)
      The iris is a low-contrast dark almond and only the glint is rock-solid. Position and BLINK
      are reliable; SIZE is not. So: the fitted circle is drawn AMBER + dashed with a "~" prefix on
      the radius, the header carries a standing IRIS SIZE APPROXIMATE warning, the sparkline is
      labelled approx, and there are NO dilation/constriction pills at all -- because this build
      deliberately does not emit those events.
      Pills: BLINK / EYE SQUINT / LICKING / GROOMING / WHISKING.

  STYLE "stable"  (JPAS_0231-class -- relatively STABLE iris)
      The fit is trustworthy enough to classify size changes. So: solid YELLOW circle, the radius
      sparkline is SHADED with the dilation (blue) and constriction (vermillion) spans, and the
      DILATION / CONSTRICTION pills are lit. It also carries 231's two mutually-exclusive status
      slots -- orange IRIS BEHIND GLINT (the crescent misfit) and amber EYE SQUINT -- so a viewer
      can tell a real constriction from an artifact of the fit.
      Pills: BLINK / DILATION / CONSTRICTION / LICKING / GROOMING / WHISKING.

The style is AUTO-DETECTED from what the session actually provides (`iris_quality` in session.json
if set, else: dil/con spans present -> stable, absent -> approx) and can be forced with --style.
Forcing "stable" on a session whose eye_events.npz has no dil/con spans is REFUSED, not faked.

Renders from the ORIGINAL (glint-intact) video -- the same one the fit runs on.

Run:  python3 annotate_clip.py <session_dir> [lo] [hi] [--style approx|stable]
"""
from pathlib import Path
import sys
import json
import numpy as np
import cv2

GREY = (200, 200, 200)
BROWN = (60, 120, 180)     # grooming pill (BGR)
TEAL = (150, 160, 40)      # licking pill (BGR)
AMBER = (60, 200, 235)     # eye-squint / approx-iris (BGR, matches 231 SQUINT_BOX)
ORANGE = (40, 130, 240)    # 231 ORANGE_BOX: "IRIS BEHIND GLINT"
GREEN = (90, 200, 90)      # whisking pill (BGR)
BLUE = (180, 114, 0)       # dilation (Okabe-Ito #0072B2 in BGR)
VERM = (0, 94, 213)        # constriction (Okabe-Ito #D55E00 in BGR)
YEL = (0, 255, 255)
DIM = (90, 90, 90)
ZOOM = 5


# ── style table ──────────────────────────────────────────────────────────────────────
STYLES = {
    'approx': dict(
        circle_col=AMBER, dashed=True, radius_prefix='~',
        banner='IRIS SIZE APPROXIMATE - eye POSITION + BLINK are reliable, pupil SIZE is NOT',
        spark_label='pupil radius (px)  [APPROX - not classified]',
        show_dilcon=False, show_unreliable_box=False),
    'stable': dict(
        circle_col=YEL, dashed=False, radius_prefix='',
        banner=None,
        spark_label='radius (px)  blue=dilation orange=constriction',
        show_dilcon=True, show_unreliable_box=True),
}


def resolve_style(sess, ev, forced=None):
    """Pick the clip style. Explicit session.json `iris_quality` wins, then an explicit --style,
    then auto-detection from whether the events file actually carries dilation/constriction spans."""
    has_dilcon = ('dil_spans' in ev.files and len(ev['dil_spans'])) or \
                 ('con_spans' in ev.files and len(ev['con_spans']))
    auto = 'stable' if has_dilcon else 'approx'
    style = forced or sess.get('iris_quality') or auto
    if style not in STYLES:
        sys.exit(f"unknown style '{style}'. choose from {list(STYLES)}")
    if style == 'stable' and not has_dilcon:
        sys.exit("style 'stable' needs dilation/constriction spans in eye_events.npz, and this "
                 "session has none -- that is deliberate for an approximate-iris animal. "
                 "Use --style approx (or produce dil/con spans first).")
    if forced is None and style != auto:
        print(f"  note: session.json iris_quality='{style}' overrides the auto-detected '{auto}'")
    return style


def pill(img, x, y, w, h, label, active, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), color if active else (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), color if active else DIM, 1, cv2.LINE_AA)
    tc = (15, 15, 15) if active else DIM
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(img, label, (x + (w - tw) // 2, y + int(h * 0.66)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, tc, 2, cv2.LINE_AA)


def status_box(img, x, y, w, h, label, active, color):
    """231's status slot: a thin outlined box under the pill row (IRIS BEHIND GLINT / EYE SQUINT)."""
    cv2.rectangle(img, (x, y), (x + w, y + h), color if active else (30, 30, 30), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), color if active else DIM, 1, cv2.LINE_AA)
    cv2.putText(img, label, (x + 8, y + int(h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (15, 15, 15) if active else DIM, 1, cv2.LINE_AA)


def _span_mask(spans, n):
    m = np.zeros(n, bool)
    for s, e in np.atleast_2d(spans) if len(spans) else []:
        m[int(s):int(e) + 1] = True
    return m


def sparkline(img, x, y, w, h, f, radius, blink, lo, hi, label, dil=None, con=None):
    """Radius trace over the clip window. In `stable` style the dilation / constriction spans are
    shaded behind it, so the viewer sees WHY a pill is lit; in `approx` style there is nothing to
    shade because those events are not emitted."""
    seg = radius[lo:hi]
    rmin, rmax = np.nanpercentile(seg, 2), np.nanpercentile(seg, 98)
    span = max(rmax - rmin, 1e-3)
    n = hi - lo
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
    xof = lambda i: x + int(i / max(n - 1, 1) * w)
    for i in range(n):
        if dil is not None and dil[lo + i]:
            cv2.line(img, (xof(i), y), (xof(i), y + h), (70, 45, 0), 1)
        if con is not None and con[lo + i]:
            cv2.line(img, (xof(i), y), (xof(i), y + h), (0, 35, 75), 1)
        if blink[lo + i]:
            cv2.line(img, (xof(i), y), (xof(i), y + h), (110, 60, 60), 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)
    pts = [(xof(i), y + int(h * (rmax - np.clip(np.nan_to_num(seg[i], nan=rmin), rmin, rmax)) / span))
           for i in range(n)]
    for i in range(1, n):
        cv2.line(img, pts[i - 1], pts[i], YEL, 1, cv2.LINE_AA)
    ci = xof(f - lo)
    cv2.line(img, (ci, y), (ci, y + h), (255, 255, 255), 1)
    cv2.putText(img, label, (x + 6, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (150, 150, 150), 1, cv2.LINE_AA)


def draw_rois(img, rois):
    for name, cfg in rois.items():
        x1, y1, x2, y2 = cfg['bbox']
        col = tuple(cfg['color_bgr'])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, name, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def _dashed_circle(img, c, r, col, thick=2, n_dash=16):
    """A dashed circle -- the visual cue that the radius is an ESTIMATE. cv2 has no dashed circle,
    so it is drawn as alternating arcs."""
    step = 360 / n_dash
    for k in range(0, n_dash, 2):
        cv2.ellipse(img, c, (r, r), 0, k * step, (k + 1) * step, col, thick, cv2.LINE_AA)


def draw_eye_inset(img, g_eye, eye, cx, cy, r, glx, gly, x0, y0, style,
                   blink=False, squint=False, unreliable=False):
    """Zoomed left_eye crop: fitted pupil circle + red glint marker. On a blink the eye is closed,
    so a circle would be meaningless -> BLINK label instead. Colour/line style follow the style."""
    st = STYLES[style]
    disp = cv2.cvtColor(g_eye, cv2.COLOR_GRAY2BGR)
    disp = cv2.resize(disp, (g_eye.shape[1] * ZOOM, g_eye.shape[0] * ZOOM),
                      interpolation=cv2.INTER_NEAREST)
    if blink:
        cv2.putText(disp, 'BLINK', (disp.shape[1] // 2 - 60, disp.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREY, 3, cv2.LINE_AA)
    elif unreliable and st['show_unreliable_box']:
        cv2.rectangle(disp, (2, 2), (disp.shape[1] - 3, disp.shape[0] - 3), ORANGE, 3)
        cv2.putText(disp, 'IRIS BEHIND GLINT', (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    ORANGE, 2, cv2.LINE_AA)
    else:
        col = AMBER if (squint or st['dashed']) else st['circle_col']
        if not np.isnan(glx):
            cv2.drawMarker(disp, (int(round((glx - eye[0]) * ZOOM)), int(round((gly - eye[1]) * ZOOM))),
                           (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        if not np.isnan(r):
            ccx = int(round((cx - eye[0]) * ZOOM)); ccy = int(round((cy - eye[1]) * ZOOM))
            if st['dashed']:
                _dashed_circle(disp, (ccx, ccy), int(round(r * ZOOM)), col)
            else:
                cv2.circle(disp, (ccx, ccy), int(round(r * ZOOM)), col, 2, cv2.LINE_AA)
            cv2.drawMarker(disp, (ccx, ccy), col, cv2.MARKER_CROSS, 12, 1)
        if squint:
            cv2.putText(disp, 'EYE SQUINT', (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMBER, 2, cv2.LINE_AA)
    H, W = disp.shape[:2]
    img[y0:y0 + H, x0:x0 + W] = disp
    cv2.rectangle(img, (x0, y0), (x0 + W, y0 + H), st['circle_col'], 1)
    cap = ('left_eye + fitted pupil (AMBER DASHED = approximate) / glint (red)' if st['dashed']
           else 'left_eye + fitted pupil (yellow) / glint (red)')
    cv2.putText(img, cap, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, st['circle_col'], 1, cv2.LINE_AA)


def main(session_dir, lo, hi, forced_style=None):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    rois = json.load(open(sess['roi_config']))
    eye = [int(v) for v in sess['rois']['left_eye']]
    tr = np.load(d / 'opticflow' / 'pupil_track.npz', allow_pickle=True)
    ev = np.load(d / 'opticflow' / 'eye_events.npz', allow_pickle=True)
    style = resolve_style(sess, ev, forced_style)
    st = STYLES[style]
    print(f"  style = {style}  ({'approximate iris - no dil/con' if style == 'approx' else 'stable iris - dil/con shown'})")

    p_cx, p_cy, p_r = tr['cx'], tr['cy'], tr['radius']
    glx, gly = tr['glint_x'], tr['glint_y']
    radius = ev['radius']; blink = ev['blink']; n = len(radius)
    squint = ev['squint'] if 'squint' in ev.files else np.zeros(n, bool)
    unrel = ev['fit_unreliable'] if 'fit_unreliable' in ev.files else np.zeros(n, bool)
    dil = _span_mask(ev['dil_spans'], n) if st['show_dilcon'] and 'dil_spans' in ev.files else None
    con = _span_mask(ev['con_spans'], n) if st['show_dilcon'] and 'con_spans' in ev.files else None

    mspath = d / 'opticflow' / 'mouth_state.npz'
    if mspath.exists():                            # grooming-priority mouth state
        ms = np.load(mspath, allow_pickle=True)['mouth_state']
        lick, groom = ms == 1, ms == 2
    else:
        lick = np.zeros(n, bool)
        groom = np.load(d / 'opticflow' / 'groom_mask_clean.npy')
    wpath = d / 'opticflow' / 'whisker.npz'
    if wpath.exists():
        wz = np.load(wpath, allow_pickle=True)
        whisking = wz['whisking']
    else:
        whisking = np.zeros(n, bool)

    src = cv2.VideoCapture(sess['video_original'])
    fps = src.get(cv2.CAP_PROP_FPS) or sess['fps']
    W, H = int(src.get(3)), int(src.get(4))
    src.set(cv2.CAP_PROP_POS_FRAMES, lo)
    outdir = d / 'test'; outdir.mkdir(exist_ok=True)
    outp = outdir / f'eye_clip_{style}_{lo}_{hi}.mp4'
    wr = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for i in range(hi - lo):
        ok, img = src.read()
        if not ok:
            break
        f = lo + i
        g_eye = cv2.cvtColor(img[eye[1]:eye[3], eye[0]:eye[2]], cv2.COLOR_BGR2GRAY).copy()
        draw_rois(img, rois)
        ph = 170 if st['show_unreliable_box'] or st['banner'] else 140
        ov = img.copy(); cv2.rectangle(ov, (0, 0), (W, ph), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.62, img, 0.38, 0, img)
        cv2.putText(img, f"{sess['mouse_id']}  EYE + FACE DETECTION   [{style} iris]", (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(img, f"frame {f}   radius {st['radius_prefix']}{np.nan_to_num(radius[f]):4.1f}px",
                    (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        is_blink = bool(blink[f]); is_squint = bool(squint[f])
        x, y, w, h, g = 16, 70, 140, 26, 8
        pill(img, x, y, w, h, 'BLINK', is_blink, GREY); x += w + g
        if st['show_dilcon']:
            pill(img, x, y, w, h, 'DILATION', bool(dil[f]) if dil is not None else False, BLUE); x += w + g
            pill(img, x, y, w, h, 'CONSTRICT', bool(con[f]) if con is not None else False, VERM); x += w + g
        else:
            pill(img, x, y, w, h, 'EYE SQUINT', is_squint, AMBER); x += w + g
        pill(img, x, y, w, h, 'LICKING', bool(lick[f]), TEAL); x += w + g
        pill(img, x, y, w, h, 'GROOMING', bool(groom[f]), BROWN); x += w + g
        pill(img, x, y, w, h, 'WHISKING', bool(whisking[f]), GREEN)
        if st['show_unreliable_box']:
            # 231's two mutually-exclusive slots: a crescent misfit vs a mild eye-shrink
            status_box(img, 16, y + h + 6, 190, 22, 'IRIS BEHIND GLINT',
                       bool(unrel[f]) and not is_squint, ORANGE)
            status_box(img, 214, y + h + 6, 150, 22, 'EYE SQUINT', is_squint, AMBER)
        if st['banner']:
            cv2.putText(img, st['banner'], (16, ph - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        AMBER, 1, cv2.LINE_AA)
        sparkline(img, W - 360, 14, 340, 74, f, radius, blink, lo, hi, st['spark_label'], dil, con)
        ih = (eye[3] - eye[1]) * ZOOM
        draw_eye_inset(img, g_eye, eye, p_cx[f], p_cy[f], p_r[f], glx[f], gly[f],
                       16, H - ih - 16, style, blink=is_blink, squint=is_squint,
                       unreliable=bool(unrel[f]))
        wr.write(img)
    wr.release(); src.release()
    print(f'wrote {outp}')
    print(f'  in clip: {int(blink[lo:hi].sum())} blink, {int(squint[lo:hi].sum())} squint, '
          f'{int(lick[lo:hi].sum())} licking, {int(groom[lo:hi].sum())} grooming frames')
    return outp


if __name__ == '__main__':
    a = [v for v in sys.argv[1:] if not v.startswith('--')]
    style = None
    for v in sys.argv[1:]:
        if v.startswith('--style'):
            style = v.split('=', 1)[1] if '=' in v else sys.argv[sys.argv.index(v) + 1]
    main(a[0], int(a[1]) if len(a) > 1 else 800, int(a[2]) if len(a) > 2 else 1200, style)
