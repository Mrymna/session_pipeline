"""
Enrich df_trials_clean with per-trial EVENT fractions + motor/whisker FEATURES, from the detector
outputs, over each disjoint clean window (collection at t=0 = end_frame). The clean windows here are
already disjoint (trial = spawn batch -> next collection = reward-to-reward), unlike JPAS_0231's
3-deep overlap, so these per-trial stats are additive across trials.

Adds (per trial): frac_blink, frac_squint, frac_grooming, frac_whisking, n_blink, n_squint,
radius_mean/median (APPROXIMATE -- pupil size is not trustworthy for this mouse, kept for the report
only), whisk_sweep (mean combined sweep), whisk_cycle (median Hz), joy_fine (joyfine over the
window's joystick samples). Per-frame arrays are NOT stored -- the report slices the session-length
arrays by start_frame/end_frame.

Run:  python3 enrich_trials.py /home/maryam/repo/flow_test/JPAS_0168
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

_COMMON = Path(__file__).resolve().parent.parent / 'common'
sys.path.insert(0, str(_COMMON))
import fps as fpsmod      # noqa: E402
import joyfine            # noqa: E402


def _spans_count(mask):
    d = np.diff(mask.astype(int))
    return int((d == 1).sum() + (1 if mask[0] else 0))


def run(session_dir, write=True, verbose=True):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    log = json.load(open(d / 'log.json'))
    df = pd.read_pickle(d / 'df_trials_clean.pkl')
    od = d / 'opticflow'

    ev = np.load(od / 'eye_events.npz', allow_pickle=True)
    radius, blink, squint = ev['radius'].astype(float), ev['blink'], ev['squint']
    wz = np.load(od / 'whisker.npz', allow_pickle=True)
    sweep, cycle, whisking = wz['sweep'], wz['cycle_whisker_right'], wz['whisking']
    ms = np.load(od / 'mouth_state.npz', allow_pickle=True)['mouth_state']
    lick, groom = ms == 1, ms == 2       # licking-priority mouth state
    sacc = np.load(od / 'saccades.npz', allow_pickle=True)['saccade']
    sacc_mask = np.zeros(len(radius), bool); sacc_mask[sacc[sacc < len(radius)]] = True
    N = len(radius)

    joy = np.asarray(log['joystick_t[ms]/x/y'], float)   # [t_ms, x, y]

    def frac(mask, s, e):
        return float(mask[s:e].mean()) if e > s else np.nan

    rows = []
    for _, r in df.iterrows():
        s, e = int(r['start_frame']), int(min(r['end_frame'], N - 1))
        seg = slice(s, e)
        rad = radius[seg]
        # joystick fine over the window (by ms)
        m = (joy[:, 0] >= r['start_ms']) & (joy[:, 0] <= r['end_ms'])
        jf = joyfine.fine_scalar(joy[m, 1], joy[m, 2]) if m.sum() > 15 else np.nan
        rows.append(dict(
            frac_blink=frac(blink, s, e), frac_squint=frac(squint, s, e),
            frac_grooming=frac(groom, s, e), frac_whisking=frac(whisking, s, e),
            n_blink=_spans_count(blink[seg]), n_squint=_spans_count(squint[seg]),
            radius_mean=float(np.nanmean(rad)) if rad.size else np.nan,
            radius_median=float(np.nanmedian(rad)) if rad.size else np.nan,
            whisk_sweep=float(np.nanmean(sweep[seg])) if e > s else np.nan,
            whisk_cycle=float(np.nanmedian(cycle[seg])) if e > s else np.nan,
            joy_fine=jf,
        ))
    add = pd.DataFrame(rows, index=df.index)
    for c in add.columns:
        df[c] = add[c]

    if verbose:
        a = df[df.analyze]
        print(f"=== {sess['mouse_id']} enriched df_trials_clean ({len(df)} trials, {len(a)} analyzable) ===")
        print(f"  frac_whisking  mean {a.frac_whisking.mean():.2f}   frac_grooming mean {a.frac_grooming.mean():.3f}")
        print(f"  blink events   total {int(a.n_blink.sum())}   squint events total {int(a.n_squint.sum())}")
        print(f"  whisk_sweep    mean {a.whisk_sweep.mean():.3f}   whisk_cycle median {a.whisk_cycle.median():.1f}Hz")
        print(f"  joy_fine       mean {a.joy_fine.mean():.3f}   radius_mean {a.radius_mean.mean():.1f}px (APPROX)")
        print("  by outcome (frac_whisking / joy_fine):")
        for oc, g in a.groupby('outcome'):
            print(f"    {oc:14s} n={len(g):2d}  whisk {g.frac_whisking.mean():.2f}  joy_fine {g.joy_fine.mean():.3f}")

    if write:
        df.to_pickle(d / 'df_trials_clean.pkl')
        if verbose:
            print(f"  wrote {d / 'df_trials_clean.pkl'}")
    return df


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
