"""
Discover + validate a session and write a `session.json` descriptor -- the single contract
that every downstream step (optic-flow, detectors, trial builder) reads. Task-AGNOSTIC:
it also classifies the task variant from the log so the right task pipeline can pick it up.

Usage:
    from session_config import build_session
    sess = build_session('JPAS_0168', '/home/maryam/repo/flow_test/JPAS_0168')

or:  python3 session_config.py /home/maryam/repo/flow_test/JPAS_0168 JPAS_0168
"""
from pathlib import Path
import json
import numpy as np

import fps as fpsmod
import camera_check

# --- task classification: which collection effects define each variant --------------
TASK_SIGNATURES = {
    'timeout_double':    {'timeout', 'double_reward'},   # JPAS_0231-style
    'banish_multiplier': {'banish', 'unbanish'},         # JPAS_0168 / 299-style
}


def _find_videos(d):
    """Classify the mp4s in the directory into (original, noreflection, ui, viz, other)."""
    vids = {'original': None, 'noreflection': None, 'ui': None, 'viz': None, 'other': []}
    for p in sorted(d.glob('*.mp4')):
        nm = p.name.lower()
        if 'noreflection' in nm:
            vids['noreflection'] = p
        elif 'viz' in nm or 'opticflow_viz' in nm:
            vids['viz'] = p
        elif nm.startswith('ui') or 'ui-' in nm or nm == 'ui.mp4':
            vids['ui'] = p
        else:
            vids['other'].append(p)
    # the "original" (with-reflection) eye video = the non-noreflection/viz/ui candidate.
    # if several, prefer one whose frame count matches the noreflection video.
    cands = vids['other']
    if len(cands) == 1:
        vids['original'] = cands[0]
    elif len(cands) > 1 and vids['noreflection'] is not None:
        nref = fpsmod.fps_from_video(vids['noreflection'])['n_frames']
        match = [c for c in cands if fpsmod.fps_from_video(c)['n_frames'] == nref]
        vids['original'] = match[0] if match else cands[0]
    elif cands:
        vids['original'] = cands[0]
    return vids


def _classify_task(effects_present):
    for name, sig in TASK_SIGNATURES.items():
        if sig & effects_present:
            return name
    return 'unknown'


def _pick_eye_bbox(rois):
    """The bbox to use for the camera/view-stability check: prefer a full eye box
    (left_eye / right_eye / *eye*) over the tiny fovea; fall back to any fovea/eye key."""
    keys = list(rois)
    for want in ('left_eye', 'right_eye'):
        if want in rois:
            return rois[want]
    eyes = [k for k in keys if 'eye' in k and 'fovea' not in k]
    if eyes:
        return rois[eyes[0]]
    fov = [k for k in keys if 'fovea' in k or 'eye' in k]
    return rois[fov[0]] if fov else None


def build_session(mouse_id, directory, write=True, verbose=True, check_camera=True):
    d = Path(directory).resolve()
    problems, warnings = [], []

    def need(cond, msg):
        if not cond:
            problems.append(msg)

    # --- required files ---------------------------------------------------------------
    log_path = d / 'log.json'
    need(log_path.exists(), f'missing log.json in {d}')
    roi_cands = sorted(d.glob('roi_config*.json'))
    need(bool(roi_cands), f'missing roi_config*.json in {d}')
    vids = _find_videos(d)
    need(vids['noreflection'] is not None, 'missing *noreflection*.mp4')
    if vids['original'] is None:
        warnings.append('no ORIGINAL (with-reflection) video found -> corneal-reflection '
                        'track for the off-glint eye gate will be unavailable')
    if problems:
        raise FileNotFoundError('session discovery failed:\n  - ' + '\n  - '.join(problems))

    log = json.load(open(log_path))
    roi_path = roi_cands[0]
    rois = {k: list(v['bbox']) for k, v in json.load(open(roi_path)).items()}

    # --- fps + frame-sync sanity ------------------------------------------------------
    ok_fps, fps_rep = fpsmod.check_fps(log, vids['noreflection'])
    if not ok_fps:
        warnings.append(f'fps/frame mismatch log vs video: {fps_rep}')
    vinfo = fpsmod.fps_from_video(vids['noreflection'])
    W, H = vinfo['width'], vinfo['height']

    # original <-> noreflection frame alignment
    if vids['original'] is not None:
        oi = fpsmod.fps_from_video(vids['original'])
        if oi['n_frames'] != vinfo['n_frames']:
            warnings.append(f"original ({oi['n_frames']}) and noreflection "
                            f"({vinfo['n_frames']}) frame counts differ -> not frame-aligned")
        if (oi['width'], oi['height']) != (W, H):
            warnings.append('original and noreflection resolutions differ')

    # --- ROI bounds check -------------------------------------------------------------
    for name, (x1, y1, x2, y2) in rois.items():
        if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
            warnings.append(f'ROI {name} bbox {rois[name]} out of frame {W}x{H}')

    # --- task scheme + world ----------------------------------------------------------
    collected = log.get('collected', [])
    effects_present = sorted({c.get('effect') for c in collected if c.get('effect')})
    task_type = _classify_task(set(effects_present))
    worlds = log.get('worlds', [])
    world_w = worlds[0].get('width') if worlds else None
    world_h = worlds[0].get('height') if worlds else None
    world_tex = worlds[0].get('world_texture_file') if worlds else None
    n_spawn_batches = len(sorted({s['time'] for s in log.get('spawns', [])}))

    if task_type == 'unknown':
        warnings.append(f'could not classify task from effects {effects_present}')

    # --- camera / view-stability sanity check (BEFORE any trials / optic-flow) ---------
    camera = None
    if check_camera:
        eye_bbox = _pick_eye_bbox(rois)
        if eye_bbox is None:
            warnings.append('no eye ROI found -> camera-move check skipped')
        else:
            try:
                camera = camera_check.detect_view_shift(
                    vids['noreflection'], eye_bbox, fps_rep['fps_log'], n_frames=vinfo['n_frames'])
                if camera['moved']:
                    warnings.append(
                        'CAMERA/VIEW MOVED: framing drifts at ~frame %s (%.1f min, %.0f%% through) '
                        'via %s (brightness %.1f->%.1f). Fixed ROIs stop covering their targets after '
                        'this point -> use the auto-tracking eye ROI and treat post-shift fixed-ROI '
                        'signals with care.' % (
                            camera['onset_frame'], (camera['onset_s'] or 0) / 60.0,
                            100 * (camera['onset_frac'] or 0), camera['moved_by'],
                            camera['brightness_early'], camera['brightness_late']))
            except Exception as e:                       # never let the check crash discovery
                warnings.append(f'camera-move check failed: {e}')

    sess = dict(
        mouse_id=mouse_id,
        session_dir=str(d),
        log=str(log_path),
        roi_config=str(roi_path),
        video_original=str(vids['original']) if vids['original'] else None,
        video_noreflection=str(vids['noreflection']),
        video_ui=str(vids['ui']) if vids['ui'] else None,
        video_viz=str(vids['viz']) if vids['viz'] else None,
        fps=round(fps_rep['fps_log'], 6),          # AUTHORITATIVE (log-derived)
        fps_video=round(fps_rep['fps_video'], 6),  # cross-check
        n_frames=vinfo['n_frames'],
        width=W, height=H,
        camera_name=fps_rep['camera_name'],
        duration_s=round(fps_rep['span_s'], 1),
        world_width=world_w, world_height=world_h, world_texture=world_tex,
        task_type=task_type,
        effects_present=effects_present,
        n_trials_from_spawns=n_spawn_batches,
        rois=rois,
        opticflow_dir=str(d / 'opticflow'),
        # camera / view-stability summary (full traces saved separately to camera_check.json)
        camera_moved=(camera['moved'] if camera else None),
        camera_move_onset_frame=(camera['onset_frame'] if camera else None),
        camera_move_onset_s=(camera['onset_s'] if camera else None),
        camera_move_onset_frac=(camera['onset_frac'] if camera else None),
        camera_check={k: v for k, v in camera.items() if k != '_traces'} if camera else None,
        warnings=warnings,
    )

    if verbose:
        print(f"=== session {mouse_id} ===")
        print(f"  task_type        : {task_type}   effects={effects_present}")
        print(f"  fps (log)        : {sess['fps']}   (video {sess['fps_video']}) "
              f"frames_match={fps_rep['frames_match']}")
        print(f"  frames / dur     : {sess['n_frames']} @ {W}x{H}  ({sess['duration_s']}s)")
        print(f"  world            : {world_w}x{world_h}")
        print(f"  trials (spawns)  : {n_spawn_batches}")
        if camera is not None:
            if camera['moved']:
                print(f"  camera/view      : MOVED at ~frame {camera['onset_frame']} "
                      f"({(camera['onset_s'] or 0)/60:.1f} min, {100*(camera['onset_frac'] or 0):.0f}%) "
                      f"by {camera['moved_by']}  [brightness {camera['brightness_early']}->{camera['brightness_late']}]")
            else:
                print(f"  camera/view      : STABLE (brightness {camera['brightness_early']}->"
                      f"{camera['brightness_late']}, pupil drift {camera['pos_shift_px']}px)")
        print(f"  original video   : {Path(sess['video_original']).name if sess['video_original'] else 'MISSING'}")
        print(f"  ROIs             : {list(rois)}")
        if warnings:
            print("  WARNINGS:")
            for w in warnings:
                print("    ! " + w)
        else:
            print("  no warnings -- clean")

    if write:
        outp = d / 'session.json'
        # PRESERVE MANUALLY-SET KEYS. session.json is both the derived contract AND where the
        # per-session tunables live (view_scale, iris_quality, ...). Re-running build_session used
        # to OVERWRITE the file and silently drop them -- `view_scale` vanished, and every
        # visibility-gated step downstream then failed (or, before the no-default guard, would have
        # scored the animal against the wrong chance baseline). Anything the builder does not itself
        # derive is carried over, so future tunables survive a rebuild automatically.
        if outp.exists():
            try:
                prev = json.load(open(outp))
            except (json.JSONDecodeError, OSError):
                prev = {}
            kept = {k: v for k, v in prev.items() if k not in sess}
            if kept:
                sess.update(kept)
                if verbose:
                    print(f"  kept manual key(s) from the existing session.json: "
                          f"{', '.join(sorted(kept))}")
        json.dump(sess, open(outp, 'w'), indent=2)
        if verbose:
            print(f"  wrote {outp}")
        if camera is not None:                           # full sampled traces for the sanity plot
            (d / 'opticflow').mkdir(exist_ok=True)
            json.dump(camera, open(d / 'opticflow' / 'camera_check.json', 'w'), indent=2)
    return sess


if __name__ == '__main__':
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    mouse = sys.argv[2] if len(sys.argv) > 2 else Path(directory).resolve().name
    build_session(mouse, directory)
