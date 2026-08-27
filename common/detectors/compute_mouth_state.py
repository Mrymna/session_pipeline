"""
Mouth-state axis (single, non-contradictory) from the RELIABLE flow signals -- NO pixel tongue_out.

The pixel-based tongue detector (231's detect_tongue) does NOT transfer to this mouse: the tongue is
too faint and the spout sits too close to the mouth, so no mouth-ROI pixel patch oscillates at the
lick rhythm (tested: dominant 0.4 Hz, not ~7 Hz). So mouth-state here is built only from signals that
work: LICKING from the flow-rhythm detector (opticflow/licking.npz, mouth fy 5-12 Hz, validated
7.7 Hz within bouts) and GROOMING from the paw+whisker joystick-still mask (groom_mask_clean.npy).

GROOMING is PRIORITISED over licking (corrected 2026-08-11): grooming REQUIRES the paw
raised to the face, and a raised paw sweeps over the mouth ROI rhythmically -> the flow lick-detector
FALSE-fires during grooming. Verified at f72912-72934 (debug/lick_groom_72920.png): the paw is clearly
UP the whole episode = grooming, not licking. So where the sustained paw-up grooming mask overlaps a
flow-lick, it is GROOMING; licking = flow bouts OUTSIDE grooming episodes (real spout-licking, paw down).
(The earlier licking-priority was backwards -- it rested on the wrong premise that grooming had been
mislabelled; the paw-up evidence + the already-validated grooming clip show grooming was correct.)

    mouth_state:  0 CLOSED   1 LICKING   2 GROOMING
Writes opticflow/mouth_state.npz (mouth_state, lick, groom, counts).
Run:  python3 compute_mouth_state.py <session_dir>
"""
from pathlib import Path
import sys
import json
import numpy as np


def _bouts(mask):
    d = np.diff(mask.astype(int))
    return int((d == 1).sum() + (1 if len(mask) and mask[0] else 0))


def run(session_dir, write=True, verbose=True):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    od = d / 'opticflow'
    lick = np.load(od / 'licking.npz', allow_pickle=True)['lick_bout'].astype(bool)
    groom_raw = np.load(od / 'groom_mask_clean.npy').astype(bool)
    N = len(lick)
    groom_raw = groom_raw[:N]

    lick = lick & ~groom_raw                        # grooming wins the overlap (paw up)
    groom = groom_raw
    reassigned = int((groom_raw & np.load(od / 'licking.npz', allow_pickle=True)['lick_bout']).sum())

    mouth_state = np.zeros(N, np.int8)
    mouth_state[lick] = 1
    mouth_state[groom] = 2                           # grooming last => priority

    counts = dict(closed=int((mouth_state == 0).sum()), licking=int((mouth_state == 1).sum()),
                  grooming=int((mouth_state == 2).sum()), groom_reassigned_to_lick=reassigned)

    if verbose:
        print(f"=== {sess['mouse_id']} mouth-state (flow-based; NO pixel tongue) ===")
        print(f"  CLOSED   {counts['closed']:6d} ({100*counts['closed']/N:.1f}%)")
        print(f"  LICKING  {counts['licking']:6d} ({100*counts['licking']/N:.1f}%)  {_bouts(lick)} bouts")
        print(f"  GROOMING {counts['grooming']:6d} ({100*counts['grooming']/N:.1f}%)  {_bouts(groom)} bouts")
        print(f"  lick-flow frames reassigned to grooming (paw up, overlap): {reassigned}")

    if write:
        out = od / 'mouth_state.npz'
        np.savez(out, mouth_state=mouth_state, lick=lick, groom=groom, counts=counts,
                 legend=['closed', 'licking', 'grooming'])
        if verbose:
            print(f"  wrote {out}")
    return dict(mouth_state=mouth_state, lick=lick, groom=groom, counts=counts)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
