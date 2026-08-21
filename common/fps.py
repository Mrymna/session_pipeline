"""
fps + frame-sync helpers, shared by both task pipelines.

The AUTHORITATIVE frame rate for a session is derived from the LOG's camera table
(`cameras (t_ms/#frame/vid_time)`), which maps every recorded eye-video frame to a
millisecond stamp on the same clock as coords / joystick / collections. That is the
number every downstream resample uses -- NEVER a hardcoded value (an earlier notebook
carried a stale 55.0 from a different mouse). The video container's own CAP_PROP_FPS is
only used as a CROSS-CHECK.
"""
from pathlib import Path
import numpy as np


# The camera table key is NOT spelled the same in every rig version: JPAS_0168 writes
# 'cameras (t_ms/#frame/vid_time)' while JPAS_0231 writes 'cameras (t ms/#frame/vid time)'
# (spaces, not underscores). Match on the prefix instead of a literal, or the whole pipeline
# fails at step 1 on a session recorded by a different build.
_CAM_PREFIX = 'cameras ('


def _camera_key(log):
    for k in log:
        if k.startswith(_CAM_PREFIX) and isinstance(log[k], dict) and log[k]:
            return k
    return None


def _camera_table(log):
    """Return (camera_name, rows) for the single eye camera in the log. rows = list of
    [t_ms, frame_index, vid_time_str]."""
    key = _camera_key(log)
    cam = log.get(key, {}) if key else {}
    if not cam:
        raise KeyError("log has no 'cameras (...)' table (looked for any key starting "
                       f"'{_CAM_PREFIX}'; found: {sorted(log)[:12]})")
    if len(cam) > 1:
        # more than one camera -> caller must disambiguate; we take the longest table
        name = max(cam, key=lambda k: len(cam[k]))
    else:
        name = next(iter(cam))
    return name, cam[name]


def fps_from_log(log):
    """Frame rate from the camera table: (n_frames - 1) * 1000 / (t_last - t_first).
    Returns dict(fps, n_frames, camera_name, span_s, median_dt_ms)."""
    name, rows = _camera_table(log)
    t = np.asarray([r[0] for r in rows], float)
    n = len(rows)
    span_ms = float(t[-1] - t[0])
    fps = (n - 1) * 1000.0 / span_ms if span_ms > 0 else float('nan')
    return dict(fps=fps, n_frames=n, camera_name=name,
                span_s=span_ms / 1000.0, median_dt_ms=float(np.median(np.diff(t))))


def fps_from_video(video_path):
    """Container-reported (fps, n_frames, width, height) via OpenCV. Cross-check only."""
    import cv2
    c = cv2.VideoCapture(str(video_path))
    if not c.isOpened():
        raise IOError(f'cannot open video {video_path}')
    out = dict(fps=float(c.get(cv2.CAP_PROP_FPS)),
               n_frames=int(c.get(cv2.CAP_PROP_FRAME_COUNT)),
               width=int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
               height=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    c.release()
    return out


def frame_to_ms(log):
    """Per-frame millisecond stamps (length n_frames) from the camera table, so any
    per-frame signal can be placed on the log's ms clock (coords/joystick/collections)."""
    _, rows = _camera_table(log)
    return np.asarray([r[0] for r in rows], float)


def check_fps(log, video_path, tol=0.02):
    """Compare log-derived fps to the video container's. Returns (ok, report_dict).
    ok is True when they agree within `tol` (relative) AND frame counts match."""
    lg = fps_from_log(log)
    vd = fps_from_video(video_path)
    rel = abs(lg['fps'] - vd['fps']) / lg['fps'] if lg['fps'] else float('inf')
    frames_match = (lg['n_frames'] == vd['n_frames'])
    ok = (rel <= tol) and frames_match
    return ok, dict(fps_log=lg['fps'], fps_video=vd['fps'], rel_diff=rel,
                    n_frames_log=lg['n_frames'], n_frames_video=vd['n_frames'],
                    frames_match=frames_match, camera_name=lg['camera_name'],
                    span_s=lg['span_s'])


if __name__ == '__main__':
    import sys, json
    d = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    log = json.load(open(d / 'log.json'))
    print('fps_from_log:', fps_from_log(log))
