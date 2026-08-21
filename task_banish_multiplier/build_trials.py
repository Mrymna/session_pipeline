"""
PIPELINE 2 (banish / unbanish + single_reward-with-multiplier) -- initial LOG-ONLY clean
trial table.  Trial = one SPAWN BATCH -> the next spawn batch (each collection triggers the
next batch at the same ms), which is Maryam's definition ("trial = spawn"). The batch's
`current` icon list gives the world + the on-screen targets for that trial directly.

Outcome model (validated against JPAS_0168):
  - trial ends in single_reward -> outcome='single_reward' (+ multiplier)
  - trial ends in banish        -> outcome='banish'   (mouse sent to the banishment world;
                                    the NEXT trial is the escape)
  - trial ends in unbanish      -> outcome='unbanish' (escaped)   AND that trial has
                                    in_banishment=True (its spawn batch = the two unbanish rings)
  - `in_banishment` is flagged so escape trials can be excluded / analysed separately, and so
    their geometry uses the unbanish rings, NOT the normal-world rewards.

Writes <session_dir>/df_trials_clean.pkl (the log-only skeleton; eye/lick/groom columns are
joined later by the enriched builder). No video decode.

Run: python3 build_trials.py /home/maryam/repo/flow_test/JPAS_0168
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

_COMMON = Path(__file__).resolve().parent.parent / 'common'
sys.path.insert(0, str(_COMMON))
import geom  # noqa: E402
import fps as fpsmod  # noqa: E402

REWARD_EFFECTS = {'single_reward'}
# outcomes that are a real collection -> a trial usable in analysis. A trial that ends
# WITHOUT a collection is either a mid-session board RESHUFFLE (icons replaced, nothing
# collected) or the session-end tail ('incomplete'); both are excluded from analysis
# (analyze=False) per Maryam: "label it reshuffle and don't use the data in any analysis".
ANALYZE_OUTCOMES = {'single_reward', 'banish', 'unbanish'}


def _spawn_batches(log):
    """Ordered list of (t_ms, icons) batches, one per unique spawn timestamp. `icons` is the
    fullest `current` list logged at that timestamp (all icons on screen for the trial)."""
    sp = sorted(log['spawns'], key=lambda s: s['time'])
    batches = {}
    for s in sp:
        t = s['time']
        cur = s.get('current', []) or []
        if t not in batches or len(cur) > len(batches[t]):
            batches[t] = cur
    return sorted(batches.items())


def build(session_dir, write=True, verbose=True):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    log = json.load(open(d / 'log.json'))
    W, H = sess['world_width'], sess['world_height']

    if sess.get('camera_moved'):
        print('  ! CAMERA/VIEW MOVED at ~frame %s (%.0f%% through) -- the log-only trial geometry '
              'here is unaffected, but VIDEO-derived (fixed-ROI) signals joined later are unreliable '
              'after this point. See session.json warnings.'
              % (sess.get('camera_move_onset_frame'), 100 * (sess.get('camera_move_onset_frac') or 0)))

    frame_ms = fpsmod.frame_to_ms(log)                 # per-frame ms (len = n_frames)
    n_frames = len(frame_ms)

    def ms_to_frame(ms):
        return int(np.clip(np.searchsorted(frame_ms, ms), 0, n_frames - 1))

    t_xy, x_xy, y_xy = geom.load_coords(log)
    t_th, theta = geom.load_theta(log)

    collected = sorted(log['collected'], key=lambda c: c['time'])
    coll_t = np.array([c['time'] for c in collected], float)

    batches = _spawn_batches(log)
    batch_t = [t for t, _ in batches]
    session_end_ms = float(frame_ms[-1])

    rows = []
    prev_outcome = None
    prev_multiplier = np.nan
    for i, (t0, icons) in enumerate(batches):
        t1 = batch_t[i + 1] if i + 1 < len(batches) else session_end_ms
        effs = {ic.get('effect') for ic in icons}
        in_banishment = 'unbanish' in effs
        world = 'BANISH_WORLD' if in_banishment else 'NORMAL'

        # the collection that ENDS this trial = the collected event at ~t1 (the next batch's
        # trigger). Match by nearest time within a small tolerance; last trial may have none.
        outcome, multiplier, tgt_x, tgt_y, tgt_loc = None, np.nan, np.nan, np.nan, None
        j = np.searchsorted(coll_t, t1 - 1)
        if j < len(coll_t) and abs(coll_t[j] - t1) <= 50:      # collection triggered this end
            c = collected[j]
            outcome = c.get('effect')
            multiplier = c.get('multiplier', np.nan)
            tgt_x, tgt_y, tgt_loc = c.get('x', np.nan), c.get('y', np.nan), c.get('loc')
        if outcome is None:                                    # no collection ended this trial
            outcome = 'incomplete' if i == len(batches) - 1 else 'reshuffle'
        degenerate = bool((t1 - t0) / 1000.0 < 0.5)
        analyze = bool(outcome in ANALYZE_OUTCOMES and not degenerate)

        s_f, e_f = ms_to_frame(t0), ms_to_frame(t1)
        tt, xx, yy = geom.slice_track(t_xy, x_xy, y_xy, t0, t1)
        eff = geom.path_efficiency(xx, yy)
        corner = geom.time_in_corner(xx, yy, W, H)
        msp, ssp = geom.speed_stats(tt, xx, yy)
        _, align = geom.heading_to_target(t_th, theta, t_xy, x_xy, y_xy, (tgt_x, tgt_y), t0, t1)

        reward_icons = [ic for ic in icons if ic.get('effect') in REWARD_EFFECTS]
        banish_icons = [ic for ic in icons if ic.get('effect') == 'banish']

        rows.append(dict(
            trial=i,
            start_ms=t0, end_ms=t1, dur_s=(t1 - t0) / 1000.0,
            start_frame=s_f, end_frame=e_f,
            world=world, in_banishment=in_banishment,
            outcome=outcome, multiplier=multiplier,
            prev_outcome=prev_outcome, prev_multiplier=prev_multiplier,
            # the target the mouse actually collected (goal of the trial)
            target_x=tgt_x, target_y=tgt_y, target_loc=tgt_loc,
            # on-screen context
            n_icons=len(icons), n_reward_icons=len(reward_icons), n_banish_icons=len(banish_icons),
            icons=[{k: ic.get(k) for k in ('effect', 'x', 'y', 'loc', 'multiplier')} for ic in icons],
            # path geometry (world units)
            path_efficiency=eff, time_in_corner=corner,
            mean_speed=msp, speed_std=ssp,
            heading_align=align,
            # coords slice for plotting / later resample
            coord_t_ms=tt, coord_x=xx, coord_y=yy,
            degenerate=degenerate,
            analyze=analyze,        # False for reshuffle / incomplete / degenerate -> exclude
        ))
        prev_outcome, prev_multiplier = outcome, multiplier

    df = pd.DataFrame(rows)

    if verbose:
        print(f"=== {sess['mouse_id']} log-only clean trials: {len(df)} ===")
        print("  outcome counts     :", df.outcome.value_counts(dropna=False).to_dict())
        print(f"  analyze / excluded : {int(df.analyze.sum())} / {int((~df.analyze).sum())} "
              f"(excluded = reshuffle {int((df.outcome == 'reshuffle').sum())} + "
              f"incomplete {int((df.outcome == 'incomplete').sum())} + "
              f"degenerate {int(df.degenerate.sum())})")
        print("  world counts       :", df.world.value_counts().to_dict())
        print(f"  in_banishment      : {int(df.in_banishment.sum())} (escape trials, analysed separately)")
        # every banish should be followed by an in_banishment escape trial
        bt = df.index[df.outcome == 'banish'].tolist()
        follow_ok = all((k + 1 < len(df)) and df.loc[k + 1, 'in_banishment'] for k in bt if k + 1 < len(df))
        print(f"  banish->escape rule: {'OK' if follow_ok else 'VIOLATED'} "
              f"({len(bt)} banish trials)")
        print(f"  multiplier (reward): {df[df.outcome == 'single_reward'].multiplier.value_counts().to_dict()}")
        print(f"  median dur         : {df.dur_s.median():.1f}s   degenerate(<0.5s): {int(df.degenerate.sum())}")

    if write:
        outp = d / 'df_trials_clean.pkl'
        df.to_pickle(outp)
        if verbose:
            print(f"  wrote {outp}")
    return df


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else '.')
