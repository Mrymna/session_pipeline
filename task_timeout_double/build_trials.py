"""
PIPELINE 2 for `timeout_double` (JPAS_0231-class) -- log-only clean trial table.

Trial = one SPAWN BATCH -> the next (each collection triggers the next batch at the same ms),
the same definition the `banish_multiplier` fork uses. The batch's `current` icon list gives the
on-screen landscape for that trial.

Outcome model:
  - ends in single_reward / double_reward -> POSITIVE
  - ends in timeout                       -> NEGATIVE, and the mouse is FROZEN for ~14 s
  - no collection                         -> 'reshuffle' (mid-session board respawn) or
                                             'incomplete' (session-end tail); both analyze=False
No multiplier and no second world here (see task_spec.py).

*** THE TIMEOUT FREEZE IS HANDLED EXPLICITLY -- this is the main thing that differs from the
banish fork. *** After a timeout the game stops logging `coords` for ~14 s (verified: 45/46
timeouts, median 14.0 s). Interpolating across that blackout would invent a straight-line path,
so every geometry feature (efficiency, speed, time-in-corner, heading) is computed on the ACTIVE
part of the trial only, and the trial records:
    freeze_ms       total blacked-out ms inside the window
    active_s        duration MINUS the freeze -- use this, not dur_s, for any rate
    has_freeze      the window contains a freeze (i.e. it FOLLOWS a timeout)
A trial whose active part is too short to measure is marked degenerate.

Writes <session_dir>/df_trials_clean.pkl. No video decode.
Run: python3 build_trials.py /home/maryam/repo/flow_test/JPAS_0231
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

_COMMON = Path(__file__).resolve().parent.parent / 'common'
sys.path.insert(0, str(_COMMON))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geom            # noqa: E402
import fps as fpsmod   # noqa: E402
import task_spec as ts  # noqa: E402

MIN_ACTIVE_S = 0.5     # below this the trial cannot be measured -> degenerate


def _spawn_batches(log):
    """Ordered (t_ms, icons) batches, one per unique spawn timestamp; `icons` is the fullest
    `current` list logged at that timestamp."""
    batches = {}
    for s in sorted(log['spawns'], key=lambda s: s['time']):
        t = s['time']
        cur = s.get('current', []) or []
        if t not in batches or len(cur) > len(batches[t]):
            batches[t] = cur
    return sorted(batches.items())


def _freeze_spans(log, coll_t, coll_eff):
    """Coord-blackout spans that FOLLOW a timeout collection. Detected from the data (a gap in
    `coords_t` starting at a timeout), not assumed, so a session with a different penalty length
    is measured correctly rather than force-fitted to 14 s."""
    C = np.array(log['coords_t[ms]/x/y'], float)
    t = C[:, 0]
    gaps = np.diff(t)
    tos = coll_t[coll_eff == 'timeout']
    spans = []
    for i in np.where(gaps > ts.FREEZE_GAP_MIN_S * 1000)[0]:
        if len(tos) and np.abs(tos - t[i]).min() < 1500:
            spans.append((t[i], t[i + 1]))
    return spans


def _freeze_ms_in(spans, t0, t1):
    return float(sum(max(0.0, min(e, t1) - max(s, t0)) for s, e in spans))


def build(session_dir, write=True, verbose=True):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    log = json.load(open(d / 'log.json'))
    W, H = sess['world_width'], sess['world_height']
    if sess.get('task_type') not in (ts.TASK, None):
        print(f"  ! session.json says task_type={sess.get('task_type')!r}, but this builder is "
              f"for {ts.TASK!r} -- check you are running the right fork.")

    frame_ms = fpsmod.frame_to_ms(log)
    n_frames = len(frame_ms)

    def ms_to_frame(ms):
        return int(np.clip(np.searchsorted(frame_ms, ms), 0, n_frames - 1))

    t_xy, x_xy, y_xy = geom.load_coords(log)
    t_th, theta = geom.load_theta(log)

    collected = sorted(log['collected'], key=lambda c: c['time'])
    coll_t = np.array([c['time'] for c in collected], float)
    coll_eff = np.array([c.get('effect') for c in collected], object)
    freezes = _freeze_spans(log, coll_t, coll_eff)
    if verbose:
        tot = sum(e - s for s, e in freezes) / 1000
        print(f"  timeout freezes detected: {len(freezes)} spans, {tot:.0f}s total "
              f"(median {np.median([(e-s)/1000 for s, e in freezes]):.1f}s each)"
              if freezes else "  no timeout freezes detected")

    batches = _spawn_batches(log)
    batch_t = [t for t, _ in batches]
    session_end_ms = float(frame_ms[-1])

    rows = []
    prev_outcome = None
    for i, (t0, icons) in enumerate(batches):
        t1 = batch_t[i + 1] if i + 1 < len(batches) else session_end_ms

        outcome, tgt_x, tgt_y, tgt_loc = None, np.nan, np.nan, None
        j = np.searchsorted(coll_t, t1 - 1)
        if j < len(coll_t) and abs(coll_t[j] - t1) <= 50:
            c = collected[j]
            outcome = c.get('effect')
            tgt_x, tgt_y, tgt_loc = c.get('x', np.nan), c.get('y', np.nan), c.get('loc')
        if outcome is None:
            outcome = 'incomplete' if i == len(batches) - 1 else 'reshuffle'

        # the freeze belongs to the trial that FOLLOWS the timeout (it eats that trial's start)
        fz_ms = _freeze_ms_in(freezes, t0, t1)
        active_s = (t1 - t0 - fz_ms) / 1000.0
        has_freeze = fz_ms > 0
        degenerate = bool(active_s < MIN_ACTIVE_S)
        analyze = bool(outcome in ts.ANALYZE_OUTCOMES and not degenerate)

        s_f, e_f = ms_to_frame(t0), ms_to_frame(t1)
        tt, xx, yy = geom.slice_track(t_xy, x_xy, y_xy, t0, t1)
        # geometry on the ACTIVE part only: drop samples inside a freeze span so the blacked-out
        # stretch is never interpolated into a fake straight-line path
        if has_freeze and len(tt):
            keep = np.ones(len(tt), bool)
            for fs, fe in freezes:
                keep &= ~((tt > fs) & (tt < fe))
            tt, xx, yy = tt[keep], xx[keep], yy[keep]
        eff = geom.path_efficiency(xx, yy)
        corner = geom.time_in_corner(xx, yy, W, H)
        msp, ssp = geom.speed_stats(tt, xx, yy)
        _, align = geom.heading_to_target(t_th, theta, t_xy, x_xy, y_xy, (tgt_x, tgt_y), t0, t1)

        n_pos = sum(1 for ic in icons if ic.get('effect') in ts.POSITIVE)
        n_neg = sum(1 for ic in icons if ic.get('effect') in ts.NEGATIVE)

        rows.append(dict(
            trial=i, start_ms=t0, end_ms=t1, dur_s=(t1 - t0) / 1000.0,
            active_s=active_s, freeze_ms=fz_ms, has_freeze=has_freeze,
            start_frame=s_f, end_frame=e_f,
            world='NORMAL',
            outcome=outcome, is_positive=outcome in ts.POSITIVE,
            is_timeout=outcome == 'timeout',
            prev_outcome=prev_outcome, prev_was_timeout=prev_outcome == 'timeout',
            target_x=tgt_x, target_y=tgt_y, target_loc=tgt_loc,
            n_icons=len(icons), n_positive_icons=n_pos, n_negative_icons=n_neg,
            icons=[{k: ic.get(k) for k in ('effect', 'x', 'y', 'loc')} for ic in icons],
            path_efficiency=eff, time_in_corner=corner,
            mean_speed=msp, speed_std=ssp, heading_align=align,
            coord_t_ms=tt, coord_x=xx, coord_y=yy,
            degenerate=degenerate, analyze=analyze))
        prev_outcome = outcome

    df = pd.DataFrame(rows)
    if verbose:
        a = df[df.analyze]
        print(f"\n=== {sess['mouse_id']} trials ({ts.TASK}) ===")
        print(f"  {len(df)} spawn batches -> {len(a)} analyzable")
        for oc in ts.OUTCOME_ORDER + ['reshuffle', 'incomplete']:
            n = int((df.outcome == oc).sum())
            if n:
                print(f"    {oc:15s} {n:4d}")
        print(f"  trials containing a timeout freeze: {int(df.has_freeze.sum())} "
              f"(median freeze {df[df.has_freeze].freeze_ms.median()/1000:.1f}s)")
        print(f"  median duration {a.dur_s.median():.1f}s  ACTIVE {a.active_s.median():.1f}s")
        print(f"  path efficiency {a.path_efficiency.mean():.2f}  corner {a.time_in_corner.mean():.2f}")
    if write:
        df.to_pickle(d / 'df_trials_clean.pkl')
        if verbose:
            print(f"  wrote {d / 'df_trials_clean.pkl'}")
    return df


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else '.')
