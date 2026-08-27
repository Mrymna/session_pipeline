"""
Grooming detector -- paw+whisker flow-high, gated on joystick stillness (port of
JPAS_0231/detect_grooming.py, generalised to this session's two whisker ROIs + common fps clock).

THE CONFOUND (from 231): "paw AND whisker |flow| both high" alone fires on active
LOCOMOTION / joystick-pushing, not grooming -- a forelimb pushing the joystick lights up the paw
ROI exactly like a grooming paw. THE FIX: real grooming needs the paw OFF the joystick, so the
joystick is (nearly) still. Gate the flow-high mask on per-frame joystick speed:

    groom = (paw_high & whisker_high) & (joystick_speed < JOY_STILL)

Reads opticflow/opticflow_{paw,whisker_left,whisker_right}_mag.npy (from compute_roi_flow.py) and
log.json joystick, resampled onto the eye-frame clock via fps.frame_to_ms.
Writes opticflow/groom_mask.npy (flow-high, ungated) + opticflow/groom_mask_clean.npy (gated).
Run:  python3 detect_grooming.py /home/maryam/repo/flow_test/JPAS_0168
"""
from pathlib import Path
import sys
import json
import numpy as np

_COMMON = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_COMMON))
import fps as fpsmod  # noqa: E402

PAW_PCT = 75           # paw |flow| above this percentile = "paw active"
WHISK_PCT = 75         # whisker |flow| above this percentile = "whisker active"
JOY_STILL = 0.5        # per-frame joystick speed below this = "still" (paw off the stick)
MERGE_GAP = 6          # frames; bridge gaps <= this (grooming strokes dip briefly between strokes)
MIN_BOUT = 12          # frames (0.4s @30fps); a real grooming bout is sustained, drop shorter


def per_frame_joystick_speed(log, nf):
    """|d(joystick)/dt| resampled onto the nf eye-frames via the camera-table ms clock."""
    frame_ms = fpsmod.frame_to_ms(log)[:nf]
    joy = np.asarray(log['joystick_t[ms]/x/y'], float)
    jx = np.interp(frame_ms, joy[:, 0], joy[:, 1])
    jy = np.interp(frame_ms, joy[:, 0], joy[:, 2])
    return np.hypot(np.gradient(jx), np.gradient(jy))


def _merge_and_filter(mask, merge_gap, min_bout, allow=None):
    """Bridge short OFF gaps (a grooming bout dips briefly between strokes) -- but ONLY over frames
    in `allow` (joystick-still), so bridging never re-includes joystick-active/locomotion frames --
    then drop bouts shorter than min_bout so only sustained episodes survive."""
    out = mask.copy()
    d = np.diff(out.astype(int))
    on = sorted(list(np.where(d == 1)[0] + 1) + ([0] if out[0] else []))
    off = sorted(list(np.where(d == -1)[0] + 1) + ([len(out)] if out[-1] else []))
    for k in range(len(off) - 1):
        gap = slice(off[k], on[k + 1])
        if on[k + 1] - off[k] <= merge_gap and (allow is None or allow[gap].all()):
            out[gap] = True
    d = np.diff(out.astype(int))
    on = sorted(list(np.where(d == 1)[0] + 1) + ([0] if out[0] else []))
    off = sorted(list(np.where(d == -1)[0] + 1) + ([len(out)] if out[-1] else []))
    for s, e in zip(on, off):
        if e - s < min_bout:
            out[s:e] = False
    return out


def _spans(mask):
    d = np.diff(mask.astype(int))
    starts = list(np.where(d == 1)[0] + 1) + ([0] if mask[0] else [])
    ends = list(np.where(d == -1)[0] + 1) + ([len(mask)] if mask[-1] else [])
    return list(zip(sorted(starts), sorted(ends)))


def run(session_dir, write=True, verbose=True):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    log = json.load(open(d / 'log.json'))
    od = d / 'opticflow'

    paw = np.load(od / 'opticflow_paw_mag.npy')
    wl = np.load(od / 'opticflow_whisker_left_mag.npy')
    wr = np.load(od / 'opticflow_whisker_right_mag.npy')
    whisk = np.maximum(wl, wr)              # either whisker pad active
    nf = len(paw)

    paw_high = paw > np.percentile(paw, PAW_PCT)
    whisk_high = whisk > np.percentile(whisk, WHISK_PCT)
    groom_raw = paw_high & whisk_high        # the ungated "both high" proxy

    jspeed = per_frame_joystick_speed(log, nf)
    still = jspeed < JOY_STILL
    groom_clean = _merge_and_filter(groom_raw & still, MERGE_GAP, MIN_BOUT, allow=still)

    if verbose:
        print(f"=== {sess['mouse_id']} grooming ===")
        print(f"  paw>p{PAW_PCT} & whisk>p{WHISK_PCT} (ungated): {groom_raw.sum()} frames "
              f"({100 * groom_raw.mean():.1f}%), {len(_spans(groom_raw))} bouts")
        print(f"  joystick still (<{JOY_STILL})               : {still.sum()} frames "
              f"({100 * still.mean():.1f}%)")
        print(f"  grooming (gated)                           : {groom_clean.sum()} frames "
              f"({100 * groom_clean.mean():.1f}%), {len(_spans(groom_clean))} bouts "
              f"[{100 * groom_clean.sum() / max(groom_raw.sum(), 1):.0f}% of ungated]")
        # sanity: gated grooming should be during LOW joystick motion by construction
        if groom_clean.any():
            print(f"  mean joy-speed  groom {jspeed[groom_clean].mean():.3f}  vs "
                  f"rest {jspeed[~groom_clean].mean():.3f}")

    if write:
        np.save(od / 'groom_mask.npy', groom_raw)
        np.save(od / 'groom_mask_clean.npy', groom_clean)
        if verbose:
            print(f"  wrote {od / 'groom_mask.npy'} + groom_mask_clean.npy")
    return dict(groom_raw=groom_raw, groom_clean=groom_clean, jspeed=jspeed)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
