"""
Corneal-reflection (IR glint) removal -- STEP 0 of the optic-flow pipeline.

The eye video is lit by IR LEDs, which put a near-white specular glint on the cornea. That glint
sits over the pupil and confuses both the pupil fit and the per-ROI eye flow, so it is inpainted OUT
before any tracking runs -- producing the `*_noreflection.mp4` that every downstream step
(`compute_roi_flow`, `segment_pupil`, `build_session`) reads. The ORIGINAL (glint-intact) video is
kept too: the corneal glint is the one rock-solid landmark for the pupil anchor, so `segment_pupil`
reads the original while the flow reads the noreflection copy (they are frame-aligned).

Method (tuned on JPAS_0168, matches `JPAS_0168/Remove_reflection_00.ipynb`): the glint is the only
near-white thing in the eye crop, so threshold `> 240`, dilate with a 3x3 kernel once, and
`cv2.inpaint(..., INPAINT_TELEA)` -- a fixed absolute threshold works because an IR specular is
always near-saturated regardless of where it lands. Only the eye ROIs are touched; the rest of the
frame is copied through unchanged.

Run:  python3 remove_reflection.py <session_dir>          # uses roi_config left_eye/left_fovea
      python3 remove_reflection.py <in.mp4> <out.mp4> x1,y1,x2,y2 [x1,y1,x2,y2 ...]
"""
from pathlib import Path
import sys
import json
import time
import numpy as np
import cv2

THRESHOLD = 240      # a pixel brighter than this in the eye crop is the IR glint (near-white)
KERNEL = 3           # dilation kernel side (px)
DILATE_ITER = 1      # grow the mask by this many iterations so the glint's soft edge is covered
INPAINT_RADIUS = 5   # cv2.inpaint neighbourhood (px)


def reflection_mask(crop, threshold=THRESHOLD, kernel=KERNEL, dilate_iter=DILATE_ITER):
    """Binary mask of the IR glint inside an eye crop (uint8, 0/255)."""
    k = np.ones((kernel, kernel), np.uint8)
    _, m = cv2.threshold(crop, threshold, 255, cv2.THRESH_BINARY)
    return cv2.dilate(m, k, iterations=dilate_iter)


def remove_reflection_auto(crop, threshold=THRESHOLD, kernel=KERNEL,
                           dilate_iter=DILATE_ITER, inpaint_radius=INPAINT_RADIUS):
    """(cleaned_crop, mask) -- inpaint the IR glint out of a single grayscale eye crop."""
    mask = reflection_mask(crop, threshold, kernel, dilate_iter)
    cleaned = cv2.inpaint(crop, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned, mask


def preview(video_path, eye_bbox, frame=1000, **kw):
    """Grab one frame and return (crop, mask, cleaned) for the given eye bbox -- for eyeballing
    the threshold before committing to a full-session pass."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise IOError(f'cannot read frame {frame} of {video_path}')
    x1, y1, x2, y2 = eye_bbox
    crop = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2].copy()
    cleaned, mask = remove_reflection_auto(crop, **kw)
    return crop, mask, cleaned


def generate_noreflection_video(video_path, output_path, eye_bboxes,
                                threshold=THRESHOLD, kernel=KERNEL, dilate_iter=DILATE_ITER,
                                inpaint_radius=INPAINT_RADIUS, lo=0, hi=None, progress=True):
    """Write a full-frame GRAYSCALE video with the IR glint inpainted out of each eye bbox.

    Frame-aligned with the input (same fps, same frame count), so all frame indices stay
    interchangeable between the original and the noreflection copy. `eye_bboxes` is a list of
    (x1,y1,x2,y2) in original-video pixels -- typically [left_eye, left_fovea]."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hi = total if hi is None else min(hi, total)
    k = np.ones((kernel, kernel), np.uint8)

    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h),
                          isColor=False)
    if not out.isOpened():
        raise IOError(f'cannot open VideoWriter for {output_path}')

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(lo))
    t0 = time.time()
    n = 0
    for i in range(lo, hi):
        ok, img = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for x1, y1, x2, y2 in eye_bboxes:
            crop = gray[y1:y2, x1:x2].copy()
            _, m = cv2.threshold(crop, threshold, 255, cv2.THRESH_BINARY)
            m = cv2.dilate(m, k, iterations=dilate_iter)
            gray[y1:y2, x1:x2] = cv2.inpaint(crop, m, inpaint_radius, cv2.INPAINT_TELEA)
        out.write(gray)
        n += 1
        if progress and (i - lo) % 5000 == 0:
            el = time.time() - t0
            eta = el / max(n, 1) * (hi - i)
            print(f'  frame {i - lo}/{hi - lo}  {el:.0f}s  ETA {eta:.0f}s', flush=True)
    cap.release()
    out.release()
    print(f'noreflection video {lo}-{hi} ({n} frames) -> {output_path}  ({time.time()-t0:.0f}s)')
    return Path(output_path)


def _eye_bboxes_from_roi_config(session_dir):
    """[left_eye, left_fovea] bboxes (px) from the session's roi_config, whichever eye ROIs exist."""
    d = Path(session_dir).resolve()
    cands = sorted(d.glob('roi_config*.json'))
    if not cands:
        raise FileNotFoundError(f'no roi_config*.json in {d}')
    cfg = json.load(open(cands[0]))
    names = [n for n in ('left_eye', 'left_fovea', 'right_eye', 'right_fovea', 'eye', 'fovea')
             if n in cfg]
    if not names:
        raise ValueError(f'no eye/fovea ROI in {cands[0].name}; keys: {list(cfg)}')
    return [tuple(cfg[n]['bbox']) for n in names], names


if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1].endswith('.mp4'):
        inp, outp = sys.argv[1], sys.argv[2]
        boxes = [tuple(int(v) for v in a.split(',')) for a in sys.argv[3:]]
        generate_noreflection_video(inp, outp, boxes)
    else:
        d = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
        orig = sorted(p for p in d.glob('*.mp4')
                      if not any(t in p.name.lower() for t in ('noreflection', 'ui', 'viz')))
        if not orig:
            sys.exit(f'no original (with-reflection) *.mp4 found in {d}')
        boxes, names = _eye_bboxes_from_roi_config(d)
        outp = d / f'{d.name}_noreflection.mp4'
        print(f'original: {orig[0].name}\neye ROIs: {names}')
        generate_noreflection_video(orig[0], outp, boxes)
