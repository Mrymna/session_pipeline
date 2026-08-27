"""
Per-frame pupil/iris segmentation for a session -- JPAS_0168-class (bright-glint dark eye).

Established interactively (2026-08-11): this mouse's eye is a low-contrast DARK
almond and the only rock-solid landmark is the bright corneal **glint**. The pupil/iris is a
dark disc and the **glint sits INSIDE it** (near centre-right when dilated, on the lower-right
edge when constricted). So the fit:
  - CENTRES the search window on the glint (not on a fixed fovea point), and
  - FILLS the glint area as pupil (erasing it drifted the centre left), and
  - uses an ABSOLUTE-anchored threshold `thr = p5(window) + MARG` -- NOT a fixed percentile: a
    fixed fraction of a mostly-dark window pins the radius and cannot show dilation, whereas the
    absolute anchor lets the disc grow when the pupil dilates (validated on f4522: the pupil core
    darkens 46->33 and the disc grows to r~10, x-extent [30,52] ~ the marked [29,54]).
  - convex-hull ellipse, roundness floor, radius cap; pick the LARGEST valid dark blob near the
    glint.

Runs on the ORIGINAL (glint-intact) video -- the glint is the anchor here, so it must be present
(the noreflection video erased it). The detected glint doubles as the corneal-reflection track
(glint_x/y), so a separate track_cr.py is not needed for this session.

** ACCURACY IS INHERENTLY LIMITED for this mouse ** (per QC): position + blink are reliable,
pupil SIZE (dilation/constriction) is only APPROXIMATE -- the iris top hides under the lid on some
frames (e.g. f22176 reads over-size) and cannot always be pinned (e.g. f42352). Validate on RAW
crops (contrast filters make it harder to read), never trust size without eyeballing.

Entry points:
  fit_pupil(gray_eye, sess_gate, prior)     -> (cx,cy,r,axr, glint) in eye-crop coords, or None
  overlay_grid(session_dir, frames, out)    -> eyeball-validation contact sheet (USE FIRST)
  run(session_dir, lo, hi)                  -> per-frame arrays -> opticflow/pupil_track.npz
"""
from pathlib import Path
import sys
import json
import time
import numpy as np
import cv2

# --- tuned constants (glint-centred, absolute-anchored; retune via overlay_grid) ----------
GATE_FRAC = None    # gate defaults to the fovea-box centre (computed per session)
GLINT_SEARCH_R = 22    # px, look for the glint within this of the fovea-centre gate
MARG = 26.0            # absolute threshold above the window's darkest core (p5)
SEARCH_R = 15          # px, dark-blob search window radius around the glint
DGATE = 10             # px, blob centroid must be within this of the glint
MIN_AREA = 14          # px, smallest dark blob considered
MIN_AXR = 0.50         # ellipse minor/major (roundness gate)
R_RANGE = (4.0, 14.0)  # px radius cap
GLINT_FILL_R = 5       # px, fill the glint disc as pupil so the blob spans it


def eye_fov(sess):
    eye = tuple(int(v) for v in sess['rois']['left_eye'])
    fov = tuple(int(v) for v in sess['rois']['left_fovea'])
    return eye, fov


def fovea_centre(eye, fov):
    """Fovea-box centre in eye-crop coords -- the fallback anchor / glint search seed."""
    return ((fov[0] + fov[2]) / 2.0 - eye[0], (fov[1] + fov[3]) / 2.0 - eye[1])


def find_glint(gb, gate, search_r=GLINT_SEARCH_R):
    """Brightest saturated blob within `search_r` of the fovea-centre gate. Returns (gx,gy) or None."""
    H, W = gb.shape
    yy, xx = np.ogrid[:H, :W]
    win = (xx - gate[0]) ** 2 + (yy - gate[1]) ** 2 <= search_r ** 2
    if not win.any():
        return None
    thr = max(200.0, np.percentile(gb[win], 99.5))
    bw = ((gb >= thr) & win).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return float(cent[i][0]), float(cent[i][1])


def fit_pupil(gray_eye, gate, prior=None, marg=MARG, search_r=SEARCH_R, dgate=DGATE,
              min_area=MIN_AREA, min_axr=MIN_AXR, r_range=R_RANGE):
    """Fit the dark pupil/iris disc, centred on the glint. `gate` = fovea-centre anchor (eye-crop
    coords), `prior` = previous good centre (used only if the glint is missing). Returns
    (cx, cy, r, axr, glint) where glint = (gx,gy) or None, or None if no valid fit."""
    gb = cv2.GaussianBlur(gray_eye, (3, 3), 0)
    gl = find_glint(gb, gate)
    if gl is not None:
        anchor = gl
    elif prior is not None and np.isfinite(prior[0]):
        anchor = prior
    else:
        anchor = gate

    H, W = gb.shape
    yy, xx = np.ogrid[:H, :W]
    win = (xx - anchor[0]) ** 2 + (yy - anchor[1]) ** 2 <= search_r ** 2
    if win.sum() < min_area:
        return None
    core = np.percentile(gb[win].astype(np.float32), 5)
    thr = core + marg
    dark = ((gb <= thr) & win).astype(np.uint8)
    if gl is not None:                                   # the glint is INSIDE the pupil
        cv2.circle(dark, (int(round(gl[0])), int(round(gl[1]))), GLINT_FILL_R, 1, -1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark)
    best = None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        cx, cy = cent[i]
        if np.hypot(cx - anchor[0], cy - anchor[1]) > dgate:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        if len(hull) < 5:
            continue
        (ex, ey), (MA, ma), _ = cv2.fitEllipse(hull)
        r = (MA + ma) / 4.0
        axr = min(MA, ma) / max(MA, ma) if max(MA, ma) > 0 else 0.0
        if axr < min_axr:
            continue
        area = stats[i, cv2.CC_STAT_AREA]
        if best is None or area > best[0]:              # largest valid dark blob near the glint
            best = (area, ex, ey, r, axr)
    if best is None:
        return None
    _, ex, ey, r, axr = best
    r = float(min(max(r, r_range[0]), r_range[1]))
    return ex, ey, r, axr, gl


def _load(session_dir):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    return d, sess


def _video(sess):
    """Fit runs on the ORIGINAL (glint-intact) video -- the glint is the anchor."""
    return sess.get('video_original') or sess['video_noreflection']


def overlay_grid(session_dir, frames, out_path=None, zoom=9, cols=6, **kw):
    """Draw the per-frame fit (fovea box + fitted circle + glint marker) on `frames` -> contact
    sheet for eyeball validation. Uses RAW crops (no contrast filter)."""
    d, sess = _load(session_dir)
    eye, fov = eye_fov(sess)
    gate = fovea_centre(eye, fov)
    fx0, fy0 = fov[0] - eye[0], fov[1] - eye[1]
    fx1, fy1 = fov[2] - eye[0], fov[3] - eye[1]
    W, H = eye[2] - eye[0], eye[3] - eye[1]
    cap = cv2.VideoCapture(_video(sess))
    tiles = []
    prior = None
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr[eye[1]:eye[3], eye[0]:eye[2]], cv2.COLOR_BGR2GRAY)
        disp = cv2.resize(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), (W * zoom, H * zoom),
                          interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(disp, (fx0 * zoom, fy0 * zoom), (fx1 * zoom, fy1 * zoom), (120, 120, 0), 1)
        fit = fit_pupil(g, gate, prior=prior, **kw)
        if fit is not None:
            ex, ey, r, axr, gl = fit
            if gl is not None:
                cv2.drawMarker(disp, (int(gl[0] * zoom), int(gl[1] * zoom)), (0, 0, 255),
                               cv2.MARKER_TILTED_CROSS, 12, 1)
            cv2.circle(disp, (int(ex * zoom), int(ey * zoom)), int(r * zoom), (0, 255, 255), 2)
            cv2.drawMarker(disp, (int(ex * zoom), int(ey * zoom)), (0, 255, 255), cv2.MARKER_CROSS, 8, 1)
            lbl = 'r%.1f' % r
            prior = (ex, ey)
        else:
            lbl = 'NO FIT'
        cv2.putText(disp, 'f%d %s' % (f, lbl), (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
        tiles.append(disp)
    cap.release()
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles) - len(tiles) % cols, cols)]
    out_path = out_path or (d / 'debug' / 'pupil_fit_check.png')
    Path(out_path).parent.mkdir(exist_ok=True)
    if rows:
        cv2.imwrite(str(out_path), np.vstack(rows))
    return out_path


def run(session_dir, lo=0, hi=None, prior_reset=8, write=True, progress=True):
    """Sequential per-frame fit with prior tracking -> opticflow/pupil_track.npz."""
    d, sess = _load(session_dir)
    eye, fov = eye_fov(sess)
    gate = fovea_centre(eye, fov)
    fy0, fy1 = fov[1] - eye[1], fov[3] - eye[1]
    fx0, fx1 = fov[0] - eye[0], fov[2] - eye[0]

    cap = cv2.VideoCapture(_video(sess))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hi = N if hi is None else min(hi, N)
    radius = np.full(N, np.nan); cxs = np.full(N, np.nan); cys = np.full(N, np.nan)
    axr = np.full(N, np.nan)
    glx = np.full(N, np.nan); gly = np.full(N, np.nan); glpres = np.zeros(N, bool)
    fovmed = np.zeros(N, np.float32); eyemean = np.zeros(N, np.float32)

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(lo))
    prior, bad = None, 0
    t0 = time.time()
    for f in range(lo, hi):
        ok, img = cap.read()
        if not ok:
            hi = f
            break
        g = cv2.cvtColor(img[eye[1]:eye[3], eye[0]:eye[2]], cv2.COLOR_BGR2GRAY)
        fovmed[f] = np.median(g[fy0:fy1, fx0:fx1]); eyemean[f] = g.mean()
        gb = cv2.GaussianBlur(g, (3, 3), 0)
        gl = find_glint(gb, gate)
        if gl is not None:
            glx[f] = gl[0] + eye[0]; gly[f] = gl[1] + eye[1]; glpres[f] = True
        fit = fit_pupil(g, gate, prior=prior)
        if fit is not None:
            ex, ey, r, a, _ = fit
            cxs[f] = ex + eye[0]; cys[f] = ey + eye[1]; radius[f] = r; axr[f] = a
            prior, bad = (ex, ey), 0
        else:
            bad += 1
            if bad >= prior_reset:
                prior = None
        if progress and (f - lo) % 10000 == 0:
            print('  %d/%d  %.0fs' % (f - lo, hi - lo, time.time() - t0), flush=True)
    cap.release()

    out = d / 'opticflow' / 'pupil_track.npz'
    if write:
        out.parent.mkdir(exist_ok=True)
        np.savez(out, radius=radius, cx=cxs, cy=cys, axis_ratio=axr,
                 glint_x=glx, glint_y=gly, glint_present=glpres,
                 fovea_med=fovmed, eye_mean=eyemean,
                 params=dict(eye=eye, fov=fov, marg=MARG, search_r=SEARCH_R, dgate=DGATE,
                             r_range=R_RANGE, min_axr=MIN_AXR,
                             fitter='glint-centred fill, absolute-anchored (JPAS_0168)'))
    valid = np.mean(~np.isnan(radius[lo:hi])) * 100
    print('fit %d-%d: valid %.1f%%  radius median %.1f (p5 %.1f - p95 %.1f)  glint %.1f%%  -> %s' % (
        lo, hi, valid, np.nanmedian(radius[lo:hi]),
        np.nanpercentile(radius[lo:hi], 5), np.nanpercentile(radius[lo:hi], 95),
        100 * glpres[lo:hi].mean(), out if write else '(not written)'))
    return dict(radius=radius, cx=cxs, cy=cys, axis_ratio=axr, glint_x=glx, glint_y=gly,
                glint_present=glpres, fovea_med=fovmed, eye_mean=eyemean)


if __name__ == '__main__':
    sd = sys.argv[1]
    d, sess = _load(sd)
    if len(sys.argv) > 2 and sys.argv[2] == 'run':
        run(sd)
    else:
        frames = np.linspace(2000, sess['n_frames'] - 2000, 30).astype(int)
        print('overlay ->', overlay_grid(sd, frames))
