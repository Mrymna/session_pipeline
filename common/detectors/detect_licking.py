"""
Lick detection from the mouth-ROI optic flow (port of JPAS_0231/detect_licking.py; no lick sensor
exists in this rig). Licking is a RHYTHMIC DOWNWARD mouth motion (tongue extending): mouth `fy`
(downward-positive) rises after reward while undirected |flow| barely does, and the motion has a
5-12 Hz rhythm (mouse lick frequency). We detect BOUTS from the 5-12 Hz envelope and individual
LICKS as downward peaks inside bouts.

Direction validated for THIS session (fy is the lick axis, same as 231): post-single_reward mouth
fy rises ~4.5x vs pre, |fx| only ~1.2x. If a future mouse's spout geometry makes licking non-vertical
this assumption must be re-checked (the code prints the post-reward fy rise as a self-check).

Sampling caveat: at 30 fps a 7.5 Hz rhythm is ~4 frames/cycle (near Nyquist) -> bouts reliable,
individual lick counts approximate. Reward frame = the collection frame (trial end).

Run:  python3 detect_licking.py <session_dir>   -> opticflow/licking.npz
"""
from pathlib import Path
import sys
import json
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from scipy.ndimage import gaussian_filter1d, binary_closing, binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fps as fpsmod  # noqa: E402

BAND = (5.0, 12.0)     # lick rhythm band (mouse lick ~7.5 Hz)
K_ENV = 4              # MAD multiplier on the bout envelope
MIN_BOUT_S = 0.20      # shortest accepted bout
PAD_S = 0.15           # widen bouts a little (onset/offset ramp)
MIN_LICK_S = 0.08      # min spacing between individual licks (<= ~12 Hz)
REWARD_EFFECTS = {'single_reward', 'double_reward'}   # collections that deliver reward


def robust_thr(x, k):
    m = np.median(x)
    return m + k * 1.4826 * np.median(np.abs(x - m))


def run(session_dir, write=True, verbose=True):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    fs = sess['fps']
    fy = np.load(d / 'opticflow' / 'opticflow_mouth_y.npy').astype(float)
    N = len(fy)

    b, a = butter(3, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)], btype='band')
    band = filtfilt(b, a, fy)
    env = gaussian_filter1d(np.abs(hilbert(band)), 2.0)

    thr = robust_thr(env, K_ENV)
    bout = env > thr
    bout = binary_closing(bout, np.ones(int(fs * 0.15), bool))
    bout = binary_dilation(bout, np.ones(int(fs * PAD_S), bool))

    min_len = int(fs * MIN_BOUT_S)
    spans, i = [], 0
    while i < N:
        if bout[i]:
            j = i
            while j < N and bout[j]:
                j += 1
            if j - i >= min_len:
                spans.append((i, j - 1))
            else:
                bout[i:j] = False
            i = j
        else:
            i += 1

    lick_sig = band.copy(); lick_sig[~bout] = 0
    licks, _ = find_peaks(lick_sig, height=robust_thr(band, 2),
                          distance=max(int(fs * MIN_LICK_S), 1))

    # self-check: does fy rise after reward? (the direction assumption)
    log = json.load(open(sess['log']))
    frame_ms = fpsmod.frame_to_ms(log)
    rew = [int(np.clip(np.searchsorted(frame_ms, c['time']), 0, N - 1))
           for c in log.get('collected', []) if c.get('effect') in REWARD_EFFECTS]
    post = np.nanmean([fy[f:f + int(fs)].mean() for f in rew if f + int(fs) < N]) if rew else np.nan
    pre = np.nanmean([fy[f - int(1.5 * fs):f - int(0.5 * fs)].mean() for f in rew
                      if f - int(1.5 * fs) >= 0]) if rew else np.nan

    if verbose:
        print(f"=== {sess['mouse_id']} licking ===")
        print(f"  fy rise post/pre reward : {post:.3f} / {pre:.3f}  ({abs(post)/abs(pre+1e-9):.1f}x) "
              f"[self-check: downward-lick direction]")
        print(f"  lick bouts   : {len(spans)}  covering {bout.mean()*100:.1f}% of the session")
        if spans:
            print(f"  median bout  : {np.median([b_ - a_ for a_, b_ in spans]) / fs:.2f} s")
        print(f"  licks        : {len(licks)}  ({len(licks)/(bout.sum()/fs+1e-9):.1f} Hz within bouts)")

    if write:
        out = d / 'opticflow' / 'licking.npz'
        np.savez(out, lick_bout=bout, lick_frames=licks,
                 bout_spans=np.array(spans, dtype=int).reshape(-1, 2),
                 envelope=env, band=band, thr=thr, fps=fs,
                 params=dict(band=BAND, k_env=K_ENV, min_bout_s=MIN_BOUT_S,
                             pad_s=PAD_S, min_lick_s=MIN_LICK_S))
        if verbose:
            print(f"  wrote {out}")
    return dict(lick_bout=bout, lick_frames=licks, spans=spans)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
