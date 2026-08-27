"""
Eye events for a session -- BLINK + POSITION only (2026-08-11).

Scope this build: pupil SIZE is only APPROXIMATE for this mouse (the iris hides under the lid on
some frames, e.g. f22176; can't always be pinned, e.g. f42352), so we do NOT classify
dilation/constriction here. We emit the reliable signals: BLINK and eye POSITION.

BLINK (231's brightness idea, retuned for this mouse): the eye is a DARK almond, so when the
lid/fur closes over it the eye-box mean brightness rises well above its own baseline. We use the
whole `left_eye` box mean (`eye_mean`), NOT the small upper `left_fovea` box -- validated on this
session, the fovea box barely moves during a blink (86+/-3) while eye_mean jumps 133->150-169 on
real closures (frames ~870/9976/33482/72814, eyeballed). We do NOT use a brightness-*derivative*
cue (231: it fired on constrictions and gave false blinks). NOTE the glint is NOT a reliable blink
corroborator here -- the glint finder still latches onto a bright lid/fur pixel on a fully closed
eye (glint_present stays ~100%), so blink rests on eye_mean alone. This mouse BLINKS RARELY
(~13 events / 43 min), consistent with the JPAS_0231 mouse.

POSITION: pupil centre (cx,cy) and P-CR = pupil_centre - glint (head-motion-invariant eye position;
the glint from segment_pupil IS the corneal reflection).

Reads opticflow/pupil_track.npz ; writes opticflow/eye_events.npz.
Run:  python3 compute_eye_events.py /home/maryam/repo/flow_test/JPAS_0168
"""
from pathlib import Path
import sys
import json
import numpy as np
from scipy.ndimage import binary_dilation

# A closure raises eye_mean. We first find the VALIDATED closure region (eye_mean over baseline,
# dilated -- the frames confirmed to be all blinks/squeezes), then PARTITION each closure into
# a full-BLINK core vs mild-SQUINT shoulders (231's "EYE SQUINT") by the pupil-fit shape: on a blink
# the fit fails or goes elongated, on a squint the pupil stays round. This keeps the blink-event
# count at the validated ~13 rather than inventing new events from fit noise.
BRIGHT_OVER_BASE = 12     # eye_mean over baseline = a closure (validated: 162 frames / 13 events)
CLOSURE_DILATE = 2        # widen the closure region this many frames each side (captures shoulders)
BLINK_OVER = 18           # strongly elevated eye_mean = full blink even if the fitter latched round
AXR_BLINK = 0.72          # pupil axis-ratio below this (or nan fit) within a closure = full blink


def _spans(mask):
    d = np.diff(mask.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return list(zip(starts, ends))


def run(session_dir, write=True, verbose=True):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    z = np.load(d / 'opticflow' / 'pupil_track.npz', allow_pickle=True)
    eye_mean = z['eye_mean'].astype(float)
    fovea_med = z['fovea_med'].astype(float)
    glint_present = z['glint_present'].astype(bool)
    radius = z['radius'].astype(float)
    axr = z['axis_ratio'].astype(float)
    cx, cy = z['cx'].astype(float), z['cy'].astype(float)
    glx, gly = z['glint_x'].astype(float), z['glint_y'].astype(float)
    N = len(eye_mean)
    fps = sess['fps']

    # ── BLINK vs SQUINT: find the validated closure region, then partition by fit shape ─
    base = float(np.median(eye_mean))
    is_bright = eye_mean > base + BRIGHT_OVER_BASE                 # a closure (validated cue)
    closure = binary_dilation(is_bright, iterations=CLOSURE_DILATE)
    # within a closure: full blink where the pupil fit fails / goes elongated / eye_mean very high
    blink = closure & (np.isnan(radius) | (axr < AXR_BLINK) | (eye_mean > base + BLINK_OVER))
    squint = closure & ~blink                                     # shoulders: pupil still round

    # ── POSITION: pupil centre and P-CR (pupil - glint) ────────────────────────────────
    pcr_x = np.where(glint_present, cx - glx, np.nan)
    pcr_y = np.where(glint_present, cy - gly, np.nan)

    # a frame is unusable if the eye is blinking OR the pupil fit failed (glint always found here)
    unusable = blink | np.isnan(radius)

    blink_spans = _spans(blink)
    squint_spans = _spans(squint)
    counts = dict(blink_frames=int(blink.sum()), blink_events=len(blink_spans),
                  squint_frames=int(squint.sum()), squint_events=len(squint_spans),
                  glint_present=float(glint_present.mean()),
                  fit_valid=float(np.mean(~np.isnan(radius))),
                  unusable=float(unusable.mean()))

    if verbose:
        print(f"=== {sess['mouse_id']} eye events (blink + squint + position) ===")
        print(f"  eye_mean baseline : {base:.1f}  (closure if > {base + BRIGHT_OVER_BASE:.1f}; "
              f"full blink if fit fails/elongated or > {base + BLINK_OVER:.1f})")
        print(f"  BLINK             : {counts['blink_frames']} frames "
              f"({100 * blink.mean():.2f}%), {counts['blink_events']} events")
        print(f"  EYE SQUINT        : {counts['squint_frames']} frames "
              f"({100 * squint.mean():.2f}%), {counts['squint_events']} events")
        print(f"  glint present     : {100 * glint_present.mean():.1f}%   "
              f"fit valid {100 * counts['fit_valid']:.1f}%   unusable {100 * unusable.mean():.1f}%")

    if write:
        out = d / 'opticflow' / 'eye_events.npz'
        np.savez(out, blink=blink, squint=squint, unusable=unusable, is_bright=is_bright,
                 cx=cx, cy=cy, pcr_x=pcr_x, pcr_y=pcr_y,
                 radius=radius, axis_ratio=axr, glint_present=glint_present,
                 eye_mean=eye_mean, fovea_med=fovea_med,
                 blink_spans=np.array(blink_spans).reshape(-1, 2),
                 squint_spans=np.array(squint_spans).reshape(-1, 2),
                 baseline=base, counts=counts, fps=fps)
        if verbose:
            print(f"  wrote {out}")
    return dict(blink=blink, unusable=unusable, pcr_x=pcr_x, pcr_y=pcr_y, counts=counts)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
