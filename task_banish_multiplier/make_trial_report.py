"""
Per-trial multi-signal PDF report for a banish_multiplier session (JPAS_0168-class) -- a FAITHFUL
match of JPAS_0231/make_trial_report.py's per-trial page ("exact same data per trial as
0231, plus the multiplier"), adapted for THIS task's outcomes + THIS session's detectors.

Per ANALYZABLE trial (one page, disjoint clean window = spawn batch -> collection, t=0=collection):
  PATH (left, full height): arena wall, plasma time-coloured trajectory, avatar FACING arrows at
    turns, START marker, the FULL on-screen ICON landscape for the trial (single_reward * / banish X
    / unbanish ring) with a ring on the COLLECTED one, and licking dots along the path.
  row0  per-ROI optic flow SEPARATED (whisker R / whisker L / nose / paw / mouth) + mouth-state bands
  row1  iris radius (APPROX - fit not clean) + eye events (blink / squint / saccade~)
  row2  whisker cycle + sweep
  row3  joystick RAW (x, y, |stick|)
  row4  joystick FINE (joyfine)
  row5  signals vs HEADING ERROR to the target (0 deg = facing it -> 180 = away), + saccade rug
Plus a summary page and a reward-MAGNITUDE (multiplier) page.

Reward structure: only single_reward pays out; multiplier = reward streak 1-4 (caps at 4);
banish captures the streak lost; unbanish = ESCAPE, no reward, resets to 1. Heading "target" = the
trial's collected icon (target_x/y): the reward for single_reward, else the banish/unbanish icon.

Run:  python3 make_trial_report.py <session_dir> [n_trials]   -> trial_report.pdf
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

_COMMON = Path(__file__).resolve().parent.parent / 'common'
sys.path.insert(0, str(_COMMON))
import joyfine     # noqa: E402
import heading     # noqa: E402
import fps as fpsmod   # noqa: E402
import viewport as vp  # noqa: E402
from scipy.stats import pearsonr, mannwhitneyu, t as tdist   # noqa: E402

OUTCOME_ORDER = ['single_reward', 'banish', 'unbanish']
# DISPLAY names. Every reward in this task carries a multiplier (streak 1-4) -- there is no plain
# "single" reward -- so the log's `single_reward` is shown as REWARD everywhere (2026-08-18).
# These are labels only; `outcome` in df_trials_clean keeps the log's values.
OUTCOME_DISPLAY = {'single_reward': 'reward', 'banish': 'banish', 'unbanish': 'unbanish (escape)'}
OUTCOME_COL = {'single_reward': '#27ae60', 'banish': '#c0392b', 'unbanish': '#2980b9'}
ICON_STYLE = {'single_reward': ('*', '#27ae60', 20, 'single reward'),
              'banish': ('X', '#c0392b', 14, 'banish (avoid)'),
              'unbanish': ('o', '#2980b9', 13, 'unbanish ring')}
LICK_COL, GROOM_COL = '#1f9e89', '#b8621b'
CONSUM_GAP_S, CONSUM_START_TOL_S = 2.0, 1.5

# ── cross-trial analysis pages (ported from JPAS_0231/make_trial_report.py) ──────────
# Grouping = the PATH-GEOMETRY KMeans clustering (cluster_paths.py -> df_trials_clean.cluster_name).
# Motor / whisker / heading / pupil / MULTIPLIER are NOT in that grouping, so testing them against
# it is non-circular by construction. Same colours as cluster_paths._cluster_color, so Direct /
# Corner-dwelling look identical on every page and figure.
G_DIRECT, G_CORNER = 'Direct', 'Corner-dwelling'
CAT_COL = {G_DIRECT: '#27ae60', G_CORNER: '#c0392b', 'Exploratory': '#9b59b6'}

CORR_COLS = ['whisker', 'whisker_l', 'nose', 'paw', 'mouth', 'sweep', 'joy_fine', 'radius', 'dist']
CORR_LAB = ['whiskerR\nflow', 'whiskerL\nflow', 'nose\nflow', 'paw\nflow', 'mouth\nflow',
            'whisk\nsweep', 'joy fine\nmove', 'iris\nradius', 'dist to\ntarget']
FULL_COLS = ['eff', 'tic', 'mean_speed', 'dur', 'joy_raw', 'joy_fine', 'paw',
             'whisker', 'whisker_l', 'sweep', 'cycle', 'nose', 'mouth',
             'radius', 'dist', 'heading_align', 'frac_lick', 'mult']
FULL_LAB = ['path\neff', 'corner\ntime', 'mean\nspeed', 'trial\nlength', 'joystick\nraw',
            'joystick\nfine', 'paw\nflow', 'whiskerR\nraw mag', 'whiskerL\nraw mag',
            'whisk\nsweep', 'whisk\ncycle', 'nose\nflow', 'mouth\nflow',
            'iris\nradius', 'dist to\ntarget', 'head\nalign', 'licking\nfrac', 'reward\nmult']

# ── the whisker<->paw hypothesis section (H1 / H2 / outcome valence) ─────────────────
# Three DIFFERENT questions about the same whisker-amplitude signal. They are easy to confuse, so
# the report states which is which before showing the figures.
HYPOTH_INTRO = (
    "THREE QUESTIONS ABOUT WHISKER AMPLITUDE - WHAT EACH ONE ACTUALLY ASKS\n"
    "\n"
    "  H1  CARRY-OVER        did the PREVIOUS trial (error / conflict) change THIS trial?\n"
    "  H2  MOVEMENT-LOCKED   does whisking reset around each PAW (joystick) re-steer?\n"
    "  VAL FEEDBACK-LOCKED   does amplitude differ hitting a POSITIVE vs a NEGATIVE icon?\n"
    "\n"
    "They are not interchangeable: H1 is between trials, H2 is locked to the MOVEMENT, and the\n"
    "valence test is locked to the COLLECTION INSTANT. H1 and H2 alone left the feedback moment\n"
    "untested - that gap is what the valence page fills.\n"
    "\n"
    "WHAT THE WHISKER SIGNAL CAN AND CANNOT SAY (this bounds all three):\n"
    "  The whisker measure is optic-FLOW motion energy (whisker.npz sweep/cycle: 3-13 Hz band-pass\n"
    "  + Hilbert). AMPLITUDE, FREQUENCY and TIMING are measurable; SET-POINT is NOT - flow is a\n"
    "  zero-mean velocity and the band-pass removes the DC term. The two whisker ROIs are also NOT\n"
    "  a contralateral pair (whisker_right at the snout, whisker_left on the cheek - one 3/4 view).\n"
    "  -> H3 (L-R set-point asymmetry) and H5 (error-type decoding) are PARKED by the SIGNAL, not\n"
    "     by effort. 30 fps -> 33 ms resolution. ONE session throughout.\n"
    "\n"
    "LICKING IS THE RECURRING CONFOUND: licking is ~7.7 Hz, INSIDE the 3-13 Hz whisker band, and\n"
    "rewarded trials are the lick-heaviest. Every test below excludes lick+groom frames - and where\n"
    "that exclusion leaves too few frames, the metric is DROPPED rather than reported (see below).\n"

)

HYPOTH_GLOSSARY = (
    "STATS TERMS USED ON THE NEXT THREE PAGES\n"
    "\n"
    "  BOOTSTRAP        re-draw the data with replacement many times, re-compute the number each\n"
    "                   time, and see how much it moves - 'would I get this again with a different\n"
    "                   sample?', without assuming the data are normal. We resample TRIALS, not\n"
    "                   single events: one trial contributes dozens of correlated re-steers, so\n"
    "                   resampling events would make the interval far too narrow (over-confident).\n"
    "  95% CONFIDENCE   the range holding the middle 95% of those re-draws. If it EXCLUDES 0 the\n"
    "  INTERVAL         effect is unlikely to be sampling noise; if it straddles 0 it is unresolved.\n"
    "  CIRCULAR-SHIFT   the chance level for a time-locked effect (H2). Slide the whisker trace by a\n"
    "  NULL             random time offset and re-run the same analysis: that destroys any real\n"
    "                   time-lock to the joystick but KEEPS the signal's own rhythm and drifts. A\n"
    "                   curve leaving that band is doing something a shifted signal cannot.\n"
    "  FIXED LAG        read every condition at ONE common moment (the all-events peak, +33 ms)\n"
    "                   instead of each condition's own maximum. A maximum is biased upward and the\n"
    "                   bias grows as n shrinks, so it would flatter the smallest group most.\n"
    "  MANN-WHITNEY     a rank test: does one group sit higher than the other? No normality\n"
    "                   assumption, and the conservative choice at small n.\n"
    "  n.s.             'not significant' - the data do NOT establish a difference. That is NOT the\n"
    "                   same as showing the two are equal, especially where n is small (see H1)."
)

HYPOTH_RESULTS = [
    ["H1  carry-over\n(post-error / post-conflict)",
     "NEGATIVE",
     "post-CONFLICT (the clean test): whisker p=0.58, joy_fine p=0.89;\n"
     "nothing survives controlling for current cluster (p=0.40 / 0.97).\n"
     "post-ERROR looks significant (0.37 vs 0.49, p=0.009) but is CONFOUNDED:\n"
     "banish->escape is deterministic, so ALL post-error trials ARE escape\n"
     "trials - the lowest-engagement state. Not an adjustment.\n"
     "DROPPED: the 'first whisk after the first movement' metric - on rewarded\n"
     "trials that window is consummatory LICKING, so ~0 clean cycles remain."],
    ["H2  movement-locked\n(whisker reset re paw)",
     "POSITIVE",
     "Whisker amplitude transient peaks +33 ms AFTER the joystick re-steer\n"
     "(it FOLLOWS the paw) and clears the circular-shift null; 1898 events.\n"
     "Magnitude SCALES with the reward multiplier: 0.041 / 0.120 / 0.120 /\n"
     "0.191 for streak 1->4, r=+0.95, trial-cluster bootstrap P(r<=0)=0.005.\n"
     "banish (0.19) ~ mult-4 reward (0.19) -> tracks AROUSAL, not error.\n"
     "Sign+latency robust to lick/groom handling; magnitude is lick-sensitive,\n"
     "so events are gated lick/groom-free within +-0.15 s of onset."],
    ["VAL feedback-locked\n(positive vs negative hit)",
     "SUGGESTIVE\n(underpowered)",
     "APPROACH window [-0.5, 0) s: reward +0.211 (n=34) vs banish -0.071\n"
     "(n=10), diff +0.283, bootstrap CI [+0.007, +0.586], Mann-Whitney\n"
     "p=0.166 (n.s.). CI and rank test DISAGREE -> read as NOT evidence of a\n"
     "valence effect, only a direction to re-test with more banish trials.\n"
     "POST-HIT IS NOT ESTIMABLE FOR REWARD: the window is ~96% licking\n"
     "(vs 35% post-banish) - a reward IS consummatory licking, so there are\n"
     "almost no clean whisker frames. Testing it would report the LICK\n"
     "detector, not the whiskers. Hence the approach-window-only test."],
]


CLASSIFY_METHOD = (
    "HOW THE PATH TYPES ARE DEFINED  (unsupervised KMeans - no hand labels)\n"
    "\n"
    "FEATURES (4) - PATH GEOMETRY only, selected from THIS session (NOT inherited from 231/299):\n"
    "   efficiency,  speed std,  time-in-corner,  effective duration.\n"
    " - candidate pool pruned for REDUNDANCY on this data. See the feature-selection page\n"
    "   (correlation + drop-one ablation) for the full justification.\n"
    " - MOTOR / WHISKER / HEADING / PUPIL are deliberately NOT features - they are tested as\n"
    "   INDEPENDENT OUTCOMES against the clusters, so any relationship is non-circular.\n"
    " - The reward MULTIPLIER is NOT a feature either (new for this task variant): it is a\n"
    "   reward-STATE variable, it partly re-encodes the outcome (escape trials are mult==1 by\n"
    "   construction), and it is a TESTED signal. See the multiplier page.\n"
    " - IRIS is held out for a second reason: the fit is only APPROXIMATE for this mouse.\n"
    "\n"
    "METHOD:\n"
    " 1. z-score each feature -> KMeans; k by SILHOUETTE over k=2,3; clusters <5 merged into the\n"
    "    nearest; each NAMED by RELATIVE efficiency rank. PCA(2) is only for the scatter below.\n"
    " 2. banish / unbanish trials ARE included (unlike 231's timeouts) - they are real navigation\n"
    "    and we want a path label on the error trials. Only reshuffle / incomplete / degenerate\n"
    "    (analyze == False) are excluded.\n"
    "Result: Direct (efficient, short) vs Corner-dwelling (low efficiency, dwells in corners).\n"
    "The motor / goal differences (outcome table + finding-6) are INDEPENDENT of this grouping."
)


class Report:
    def __init__(self, session_dir):
        self.d = d = Path(session_dir).resolve()
        self.sess = json.load(open(d / 'session.json'))
        self.log = json.load(open(d / 'log.json'))
        self.fps = self.sess['fps']
        self.W = self.sess['world_width'] or 2400
        self.H = self.sess['world_height'] or 2400
        self.df = pd.read_pickle(d / 'df_trials_clean.pkl')
        od = d / 'opticflow'
        ev = np.load(od / 'eye_events.npz', allow_pickle=True)
        self.radius, self.blink, self.squint = ev['radius'], ev['blink'], ev['squint']
        wz = np.load(od / 'whisker.npz', allow_pickle=True)
        self.wenv, self.wfreq = wz['sweep'], wz['cycle_whisker_right']
        ms = np.load(od / 'mouth_state.npz', allow_pickle=True)['mouth_state']
        self.mouth_state, self.lick, self.groom = ms, ms == 1, ms == 2
        sac = np.load(od / 'saccades.npz', allow_pickle=True)['saccade']
        self.sac = sac[sac < len(self.radius)]
        self.flow = {r: np.load(od / f'opticflow_{r}_mag.npy').astype(float)
                     for r in ('whisker_right', 'whisker_left', 'nose', 'paw', 'mouth')}
        self.COORD = np.array(self.log['coords_t[ms]/x/y'], float)
        self.JOY = np.array(self.log['joystick_t[ms]/x/y'], float)
        self.ANG = np.array(self.log['angles_t[ms]/theta'], float)
        self.ANG_UW = np.unwrap(self.ANG[:, 1])
        self.N = len(self.radius)
        self.VIEW_W, self.VIEW_H = vp.viewport(self.sess)

    # ---- geometry helpers ----------------------------------------------------------
    def _facing_change_arrows(self, s_ms, e_ms, min_turn_deg=30, min_gap_s=0.3):
        from scipy.ndimage import gaussian_filter1d
        m = (self.COORD[:, 0] >= s_ms) & (self.COORD[:, 0] <= e_ms)
        t, x, y = self.COORD[m, 0], self.COORD[m, 1], self.COORD[m, 2]
        if len(t) < 2:
            return (np.array([]),) * 5
        th = gaussian_filter1d(np.interp(t, self.ANG[:, 0], self.ANG_UW), 1.5)
        keep, anchor, last_t = [0], th[0], t[0]
        for i in range(1, len(t)):
            if abs(th[i] - anchor) >= np.radians(min_turn_deg) and (t[i] - last_t) >= min_gap_s * 1000:
                keep.append(i); anchor = th[i]; last_t = t[i]
        keep = np.array(keep); a = th[keep]
        return x[keep], y[keep], np.cos(a), np.sin(a), (t[keep] - e_ms) / 1000.0

    # viewport (world units) = sketch / get.scale, EXACT game camera (follows the avatar)
    # viewport comes from session.json via common/viewport.py -- NOT hard-coded (231 used 0.56,
    # this session 0.35; a wrong value silently biases every visibility-gated result).

    def _heading_error_visible(self, s_ms, e_ms, icons):
        """Per-theta-timestamp heading error to the NEAREST reward that is INSIDE the viewport
        (the reward the mouse is 'now seeing'); NaN where no reward is on screen. Rewards = the
        trial's goal icons (single_reward, or unbanish rings in the banishment world)."""
        goals = [(ic['x'], ic['y']) for ic in (icons or [])
                 if ic.get('effect') in ('single_reward', 'unbanish')
                 and np.isfinite(ic.get('x')) and np.isfinite(ic.get('y'))]
        m = (self.ANG[:, 0] >= s_ms) & (self.ANG[:, 0] <= e_ms)
        tt, th = self.ANG[m, 0], self.ANG[m, 1]
        if tt.size == 0 or not goals:
            return tt, np.full(tt.size, np.nan)
        ax = np.interp(tt, self.COORD[:, 0], self.COORD[:, 1])
        ay = np.interp(tt, self.COORD[:, 0], self.COORD[:, 2])
        hw, hh = self.VIEW_W / 2, self.VIEW_H / 2
        err = np.full(tt.size, np.nan)
        for k in range(tt.size):
            best = None
            for gx, gy in goals:
                if abs(gx - ax[k]) <= hw and abs(gy - ay[k]) <= hh:      # in viewport
                    e = abs(np.angle(np.exp(1j * (th[k] - np.arctan2(gy - ay[k], gx - ax[k])))))
                    best = e if best is None else min(best, e)
            if best is not None:
                err[k] = best
        return tt, err

    def _consummatory_end(self, lickmask):
        n = len(lickmask); st = int(round(CONSUM_START_TOL_S * self.fps))
        if n == 0 or not np.any(lickmask[:st]):
            return 0
        gap = int(round(CONSUM_GAP_S * self.fps)); run = 0
        for k in range(n):
            if lickmask[k]:
                run = 0
            else:
                run += 1
                if run >= gap:
                    return k - run + 1
        return n

    def _bands(self, ax, t, mask, color, label, y0=None, y1=None):
        d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        for k, (s, e) in enumerate(zip(np.where(d == 1)[0], np.where(d == -1)[0])):
            ax.axvspan(t[min(s, len(t) - 1)], t[min(e, len(t)) - 1], color=color, alpha=0.16,
                       label=label if k == 0 else None)

    def _overlay_events(self, ax, t, s, e):
        self._bands(ax, t, self.blink[s:e], '0.55', 'blink')
        self._bands(ax, t, self.squint[s:e], 'goldenrod', 'squint')
        si = self.sac[(self.sac >= s) & (self.sac < e)] - s
        if len(si):
            yl = ax.get_ylim()
            ax.plot(t[si], np.full(len(si), yl[0] + 0.02 * (yl[1] - yl[0])), '|',
                    color='#8e44ad', ms=9, label='saccade~')

    # ---- one trial page ------------------------------------------------------------
    def page(self, i):
        r = self.df.loc[i]
        s, e = int(r['start_frame']), int(min(r['end_frame'], self.N - 1))
        if e - s < 8:
            return None
        s_ms, e_ms = float(r['start_ms']), float(r['end_ms'])
        tgt = (r['target_x'], r['target_y'])
        tf = (np.arange(s, e) - e) / self.fps
        # per-frame heading error to the trial target
        htt, herr = heading.heading_error(self.ANG, self.COORD, tgt, s_ms, e_ms)
        R = heading.resultant_length(herr); align = heading.forward_alignment(herr)

        fig = plt.figure(figsize=(14, 16))
        gs = GridSpec(6, 3, width_ratios=[1.1, 2, 2], height_ratios=[1] * 6, hspace=0.6, wspace=0.36)
        oc = r['outcome']; ocd = OUTCOME_DISPLAY.get(oc, oc)
        mult = int(r['multiplier']) if np.isfinite(r['multiplier']) else None
        multlab = f"  x{mult}" if (oc == 'single_reward' and mult) else (
            f"  (streak lost {mult})" if (oc == 'banish' and mult) else "")
        prev = r['prev_outcome']; nsac = int(np.sum((self.sac >= s) & (self.sac < e)))
        fig.suptitle(f"{self.sess['mouse_id']}  Trial {int(r['trial'])}  |  {ocd.upper()}{multlab}  "
                     f"(prev: {prev})  |  eff={r['path_efficiency']:.2f} corner={r['time_in_corner']:.2f}  "
                     f"|  {nsac} saccades~"
                     f"{'   [BANISHMENT WORLD]' if r['in_banishment'] else ''}\n"
                     f"heading R={R:.2f}, align={align:+.2f} (+toward/-away)   "
                     f"[reward only for single_reward; banish=penalty, unbanish=escape]",
                     fontweight='bold', fontsize=10.5, color=OUTCOME_COL[oc])

        # --- PATH (col 0, all rows) ---
        axp = fig.add_subplot(gs[:, 0])
        axp.add_patch(Rectangle((0, 0), self.W, self.H, fill=False, ec='0.4', lw=1.2))
        x, y = np.asarray(r['coord_x'], float), np.asarray(r['coord_y'], float)
        ct = np.asarray(r['coord_t_ms'], float)
        if len(x) > 1:
            pts = np.array([x, y]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap='plasma', lw=2.4, zorder=3)
            lc.set_array((ct[:-1] - e_ms) / 1000.0); axp.add_collection(lc)
            cb = fig.colorbar(lc, ax=axp, location='bottom', fraction=0.05, pad=0.06, aspect=32)
            cb.set_label('time from collection (s)  (dark=start -> bright=t0)', fontsize=7)
            cb.ax.tick_params(labelsize=6)
            gx, gy, u, v, gt = self._facing_change_arrows(s_ms, e_ms)
            if len(gx):
                axp.quiver(gx, gy, u, v, gt, cmap='plasma', zorder=6, pivot='tail', scale=12,
                           width=0.013, headwidth=4.5, headlength=6, edgecolor='k', linewidth=0.5,
                           label='facing (at turns)')
                axp.plot(gx, gy, 'o', mfc='white', mec='k', ms=3.2, mew=0.6, zorder=6.5)
            axp.plot(x[0], y[0], 'o', mfc='k', mec='w', ms=9, mew=1.2, zorder=12, label='start')
        # full on-screen ICON landscape (ring the collected)
        seen = set()
        for ic in (r['icons'] or []):
            eff = ic.get('effect'); ox, oy = ic.get('x'), ic.get('y')
            if eff not in ICON_STYLE or not (np.isfinite(ox) and np.isfinite(oy)):
                continue
            mk, col, msz, lab = ICON_STYLE[eff]
            axp.plot(ox, oy, mk, mfc=col if mk != 'o' else 'none', mec='k', ms=msz, mew=1.4,
                     zorder=5, label=lab if eff not in seen else None); seen.add(eff)
            if ic.get('loc') == r['target_loc']:
                axp.plot(ox, oy, 'o', mfc='none', mec='k', ms=msz + 6, mew=1.8, zorder=6,
                         label='collected' if 'collected' not in seen else None); seen.add('collected')
        # fallback: session-start logging gap -- the collected icon (e.g. trial 0 loc 1) is
        # missing from the batch's `current` list (which is already the POST-collection board,
        # 2 reward + 1 banish), but the collection event still gave us its coordinates. Mark
        # WHERE it was collected with the ring ONLY -- do NOT add a filled icon, or the board
        # would show a spurious 3rd reward.
        if ('collected' not in seen and r['outcome'] in ICON_STYLE
                and np.isfinite(r['target_x']) and np.isfinite(r['target_y'])):
            msz = ICON_STYLE[r['outcome']][2]
            axp.plot(r['target_x'], r['target_y'], 'o', mfc='none', mec='k', ms=msz + 6, mew=1.8,
                     ls='--', zorder=6, label='collected (from log)'); seen.add('collected')
        # licking dots along the path
        d = np.diff(np.concatenate([[0], self.lick[s:e].astype(int), [0]]))
        mids = ((np.where(d == 1)[0] + np.where(d == -1)[0]) // 2 + s)
        if len(mids) and len(x) > 1:
            lms = s_ms + (mids - s) / max(e - s, 1) * (e_ms - s_ms)
            lx = np.interp(lms, ct, x); ly = np.interp(lms, ct, y)
            axp.plot(lx, ly, 'o', mfc=LICK_COL, mec='white', ms=6, mew=0.6, zorder=9, label='licking')
        M = 180; axp.set_xlim(-M, self.W + M); axp.set_ylim(self.H + M, -M); axp.set_aspect('equal')
        axp.legend(fontsize=6.6, loc='lower left', bbox_to_anchor=(-0.02, 1.01), ncol=2,
                   frameon=False, columnspacing=1.4, markerscale=0.8)

        # --- row 0: per-ROI flow separated + mouth-state bands ---
        ax0 = fig.add_subplot(gs[0, 1:])
        off = 0.0
        for name, roi, col in [('whisker R', 'whisker_right', '#2c7fb8'),
                               ('whisker L', 'whisker_left', '#8e9b1f'),
                               ('nose', 'nose', '#9b59b6'), ('paw', 'paw', '#e67e22'),
                               ('mouth', 'mouth', '#16a085')]:
            seg = self.flow[roi][s:e]
            norm = seg / (np.nanpercentile(seg, 98) + 1e-9)
            ax0.plot(tf, np.clip(norm, 0, 1.15) + off, color=col, lw=1.0)
            ax0.text(tf[0], off + 0.9, name, color=col, fontsize=7.5, va='center', fontweight='bold')
            off += 1.3
        ax0.set_yticks([]); ax0.set_ylim(-0.15, off)
        self._bands(ax0, tf, self.lick[s:e], LICK_COL, 'licking')
        self._bands(ax0, tf, self.groom[s:e], GROOM_COL, 'grooming')
        ax0.legend(fontsize=7, ncol=4, loc='upper left', frameon=False)
        ax0.set_title('per-ROI optic flow: whisker R/L, nose, paw, mouth (separated)', fontsize=9, loc='right')

        # --- row 1: iris radius + eye events ---
        axI = fig.add_subplot(gs[1, 1:])
        axI.plot(tf, self.radius[s:e], color='#333', lw=1.4)
        axI.set_ylabel('iris radius px\n(APPROX - not clean)', fontsize=8)
        self._overlay_events(axI, tf, s, e)
        if axI.get_legend_handles_labels()[0]:
            axI.legend(fontsize=6.5, ncol=4, loc='upper left')
        axI.set_title('iris radius + eye events   (iris fit NOT clean for this mouse)', fontsize=9, loc='right')

        # --- row 2: whisker cycle + sweep ---
        ax1 = fig.add_subplot(gs[2, 1:])
        ax1.plot(tf, self.wenv[s:e], color='#16a085', lw=1.4)
        ax1.set_ylabel('sweep', color='#16a085', fontsize=9)
        axf = ax1.twinx(); axf.plot(tf, self.wfreq[s:e], color='#7f8c8d', lw=1.0, alpha=0.8)
        axf.set_ylabel('cycle (Hz)', color='#7f8c8d', fontsize=9); axf.set_ylim(0, 15)
        ax1.set_title('whisker sweep + cycle  (Nyquist 15Hz)', fontsize=9, loc='right')

        # --- row 3: joystick raw ---
        jm = (self.JOY[:, 0] >= s_ms) & (self.JOY[:, 0] <= e_ms)
        jt = (self.JOY[jm, 0] - e_ms) / 1000.0
        ax2 = fig.add_subplot(gs[3, 1:])
        ax2.plot(jt, self.JOY[jm, 1], '#2980b9', lw=1.0, label='joy x')
        ax2.plot(jt, self.JOY[jm, 2], '#27ae60', lw=1.0, label='joy y')
        ax2.plot(jt, np.hypot(self.JOY[jm, 1], self.JOY[jm, 2]), '#7f8c8d', lw=1.0, alpha=0.7, label='|joystick|')
        ax2.set_ylabel('joystick', fontsize=9); ax2.legend(fontsize=7, loc='upper left')
        ax2.set_title('joystick raw movement', fontsize=9, loc='right')

        # --- row 4: joystick fine ---
        ax3 = fig.add_subplot(gs[4, 1:])
        if jm.sum() > 15:
            jf = joyfine.fine_trace(self.JOY[jm, 1], self.JOY[jm, 2])
            ax3.plot(jt, jf, '#c0392b', lw=1.4); ax3.fill_between(jt, 0, jf, color='#c0392b', alpha=0.12)
        ax3.set_ylabel('fine movement', fontsize=9); ax3.set_xlabel('time from collection (s)', fontsize=9)
        ax3.set_title('joystick fine movement (rolling std of low-passed |joystick|)', fontsize=9, loc='right')
        for ax in (ax0, axI, ax1, ax2, ax3):
            ax.axvline(0, color='r', ls='--', lw=1); ax.grid(alpha=0.2); ax.tick_params(labelsize=7)

        # --- row 5: HEADING ERROR over time — TWO lines: nearest-VISIBLE reward (attention) vs
        #     the COLLECTED reward (goal). 0 deg = facing it -> 180 = away. ---
        axH = fig.add_subplot(gs[5, 1:])
        # collected-reward error on theta timestamps
        tcol = (htt - e_ms) / 1000.0 if htt.size else np.array([])
        col_deg = np.abs(np.degrees(herr)) if herr.size else np.array([])
        # nearest-visible-reward error (viewport-gated)
        vtt, verr = self._heading_error_visible(s_ms, e_ms, r['icons'])
        tvis = (vtt - e_ms) / 1000.0 if vtt.size else np.array([])
        vis_deg = np.degrees(verr) if verr.size else np.array([])
        if col_deg.size:
            axH.plot(tcol, col_deg, '-', color='#e67e22', lw=1.3, label='to COLLECTED reward (goal)')
        if vis_deg.size:
            axH.plot(tvis, vis_deg, '-', color='#16a085', lw=1.5,
                     label='to nearest VISIBLE reward (attention)')
            offscreen = ~np.isfinite(verr)
            if offscreen.any():                       # shade where NO reward is on screen
                axH.fill_between(tvis, 0, 180, where=offscreen, color='0.85', step='mid',
                                 label='no reward on screen')
        si2 = self.sac[(self.sac >= s) & (self.sac < e)] - s
        if len(si2):
            axH.plot(tf[si2], np.full(len(si2), 173), '|', color='#8e44ad', ms=11, label='saccade~')
        axH.axhline(90, color='0.7', ls=':', lw=0.8)
        axH.set_ylim(0, 180); axH.set_ylabel('heading error (deg)\n0=facing / 180=away', fontsize=8)
        axH.set_xlabel('time from collection (s)', fontsize=9)
        axH.axvline(0, color='r', ls='--', lw=1)
        axH.set_title(f'heading error to reward over time  (collected: R={R:.2f}, align={align:+.2f})',
                      fontsize=9, loc='right')
        axH.legend(fontsize=6.3, loc='upper left', ncol=2); axH.grid(alpha=0.2); axH.tick_params(labelsize=7)
        return fig

    # ---- summary + reward-magnitude (unchanged style) ------------------------------
    def summary(self):
        a = self.df[self.df.analyze].copy()
        a['frac_licking'] = [self.lick[int(r.start_frame):int(r.end_frame)].mean() for r in a.itertuples()]
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 7))
        fig.suptitle(f"{self.sess['mouse_id']}  summary by outcome  (n_analyzable={len(a)})", fontsize=12)
        order = [o for o in OUTCOME_ORDER if o in a.outcome.unique()]
        for ax, (col, lab) in zip(axes.ravel(),
                                  [('frac_whisking', 'whisking'), ('whisk_sweep', 'whisk sweep'),
                                   ('joy_fine', 'joy_fine'), ('frac_licking', 'licking'),
                                   ('frac_grooming', 'grooming'), ('radius_mean', 'iris radius (NOT clean)')]):
            means = [a[a.outcome == o][col].mean() for o in order]
            sems = [a[a.outcome == o][col].sem() for o in order]
            ax.bar(range(len(order)), means, yerr=sems, color=[OUTCOME_COL[o] for o in order], alpha=0.85)
            ax.set_title(lab, fontsize=9); ax.set_xticks(range(len(order)))
            ax.set_xticklabels([OUTCOME_DISPLAY.get(o, o) for o in order], rotation=20, fontsize=7)
            ax.tick_params(labelsize=7)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def reward_magnitude(self):
        a = self.df[(self.df.analyze) & (self.df.outcome == 'single_reward')].copy()
        a['frac_licking'] = [self.lick[int(r.start_frame):int(r.end_frame)].mean() for r in a.itertuples()]
        a['whisk_mag'] = [np.nanmean(self.flow['whisker_right'][int(r.start_frame):int(r.end_frame)])
                          for r in a.itertuples()]
        mults = [1, 2, 3, 4]; total = int(a['multiplier'].sum())
        fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.6))
        fig.suptitle(f"{self.sess['mouse_id']}  reward MAGNITUDE (multiplier) — reward trials "
                     f"[total reward units = {total}]  (replicates JPAS_299 finding2: flow scales w/ reward)",
                     fontsize=10)
        fig.text(0.5, 0.90, 'HEADLINE: the IRIS detection for this mouse is NOT clean — read the iris '
                 'panel with caution; whisker/licking/motor are the reliable ones.',
                 ha='center', fontsize=9, color='#c0392b', weight='bold')
        for ax, (col, lab, color) in zip(axes, [
                ('whisk_sweep', 'whisk sweep (299: scales)', '#2e86c1'),
                ('whisk_mag', 'whisker |flow|', '#2e86c1'), ('frac_licking', 'licking', '#2e86c1'),
                ('joy_fine', 'joy_fine', '#2e86c1'), ('radius_mean', 'iris radius — NOT CLEAN', '#c0392b')]):
            means = [a[a.multiplier == mm][col].mean() for mm in mults]
            sems = [a[a.multiplier == mm][col].sem() for mm in mults]
            ns = [int((a.multiplier == mm).sum()) for mm in mults]
            ax.bar(mults, means, yerr=sems, color=color, alpha=0.85)
            ax.set_title(lab, fontsize=8.5, color=color if color == '#c0392b' else 'black')
            ax.set_xlabel('reward multiplier', fontsize=8); ax.set_xticks(mults)
            ax.set_xticklabels([f'{m}\n(n={nn})' for m, nn in zip(mults, ns)], fontsize=7)
            ax.tick_params(labelsize=7)
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        return fig

    # ---- cross-trial analysis pages (ported from 231) --------------------------------
    def _per_trial(self):
        """One pass over the clustered trials: per-trial means of every signal (for the
        correlation matrices / scatters / finding-6) plus pooled per-frame samples of
        iris radius and distance-to-target (for the radius-vs-distance panel).
        Trials with no path cluster (analyze == False) are dropped, so n matches the
        KMeans scatter on the classification page exactly."""
        if getattr(self, '_pt_cache', None) is not None:
            return self._pt_cache
        fm = fpsmod.frame_to_ms(self.log)
        rows, pooled = [], {'dist': [], 'radius': []}
        for i, r in self.df.iterrows():
            cat = r.get('cluster_name', 'excluded')
            if cat not in CAT_COL or not bool(r.get('analyze', False)):
                continue
            s, e = int(r['start_frame']), int(min(r['end_frame'], self.N))
            if e - s < 8:
                continue
            t_ms = fm[s:e]
            ax_ = np.interp(t_ms, self.COORD[:, 0], self.COORD[:, 1])
            ay_ = np.interp(t_ms, self.COORD[:, 0], self.COORD[:, 2])
            dist = np.hypot(ax_ - r['target_x'], ay_ - r['target_y'])
            rad = self.radius[s:e].astype(float).copy()
            rad[self.blink[s:e] | self.squint[s:e]] = np.nan     # never average over a shut eye
            jm = (self.JOY[:, 0] >= r['start_ms']) & (self.JOY[:, 0] <= r['end_ms'])
            joy_raw = (float(np.nanmean(np.hypot(self.JOY[jm, 1], self.JOY[jm, 2])))
                       if jm.sum() else np.nan)
            with np.errstate(invalid='ignore'):
                rows.append(dict(
                    idx=i, cat=cat, outcome=r['outcome'],
                    mult=r['multiplier'] if np.isfinite(r['multiplier']) else np.nan,
                    eff=r['path_efficiency'], tic=r['time_in_corner'],
                    mean_speed=r['mean_speed'], dur=r['dur_s'],
                    joy_raw=joy_raw, joy_fine=r['joy_fine'],
                    whisker=float(np.nanmean(self.flow['whisker_right'][s:e])),
                    whisker_l=float(np.nanmean(self.flow['whisker_left'][s:e])),
                    nose=float(np.nanmean(self.flow['nose'][s:e])),
                    paw=float(np.nanmean(self.flow['paw'][s:e])),
                    mouth=float(np.nanmean(self.flow['mouth'][s:e])),
                    sweep=float(np.nanmean(self.wenv[s:e])),
                    cycle=float(np.nanmean(self.wfreq[s:e])),
                    radius=float(np.nanmean(rad)), dist=float(np.nanmean(dist)),
                    heading_align=r['heading_align'],
                    frac_lick=float(self.lick[s:e].mean()),
                    frac_groom=float(self.groom[s:e].mean()),
                    sacc_rate=float(np.sum((self.sac >= s) & (self.sac < e)) / ((e - s) / self.fps / 60))))
            pooled['dist'].append(dist); pooled['radius'].append(rad)
        dfa = pd.DataFrame(rows).set_index('idx')
        pooled = {k: np.concatenate(v) if v else np.array([]) for k, v in pooled.items()}
        self._pt_cache = (dfa, pooled)
        return self._pt_cache

    def _embed_png(self, name, figsize=(16, 7.5)):
        """Wrap a diagnostic PNG written by another pipeline step as a full report page
        (the embedded figure carries its own title)."""
        png = self.d / 'debug' / name
        if not png.exists():
            print(f'  (skipping page: {png} not found -- run the step that writes it)')
            return None
        import matplotlib.image as mpimg
        fig = plt.figure(figsize=figsize)
        a = fig.add_axes([0.01, 0.01, 0.98, 0.98]); a.axis('off')
        a.imshow(mpimg.imread(png))
        return fig

    def _outcome_rows(self):
        """Direct vs Corner-dwelling means + one-sided Mann-Whitney p for the OUTCOME signals
        tested against the path-geometry clusters. None of these was used to form the clusters."""
        a = self.df[self.df.analyze]
        out = []
        for col, lab, signed in [('joy_fine', 'joystick fine movement', False),
                                 ('whisk_sweep', 'whisker sweep', False),
                                 ('heading_align', 'heading alignment (+toward/-away)', True),
                                 ('frac_whisking', 'whisking fraction', False),
                                 ('multiplier', 'reward multiplier (streak)', False)]:
            d = a[a.cluster_name == G_DIRECT][col].dropna()
            c = a[a.cluster_name == G_CORNER][col].dropna()
            if len(d) < 3 or len(c) < 3:
                continue
            pv = mannwhitneyu(d, c, alternative='greater').pvalue
            f = '{:+.3f}' if signed else '{:.3f}'
            out.append([lab, f.format(d.mean()), f.format(c.mean()),
                        '<0.001' if pv < 0.001 else f'{pv:.3f}'])
        return out

    def clustering_page(self):
        """CLASSIFICATION FIRST: the method behind the Direct / Corner-dwelling grouping used on
        every later page, its independent outcome table, and the KMeans diagnostic."""
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(f"{self.sess['mouse_id']}  Path / behaviour classification  -  the method behind "
                     'the "Direct vs Corner-dwelling" grouping used on every page',
                     fontweight='bold', fontsize=13)
        axt = fig.add_axes([0.03, 0.55, 0.57, 0.40]); axt.axis('off')
        axt.text(0.0, 1.0, CLASSIFY_METHOD, va='top', ha='left', fontsize=8.2, family='monospace',
                 linespacing=1.22,
                 bbox=dict(boxstyle='round', fc='#f7f7f7', ec='0.55', alpha=0.97))
        rows = self._outcome_rows()
        if rows:
            a = self.df[self.df.analyze]
            nD = int((a.cluster_name == G_DIRECT).sum()); nC = int((a.cluster_name == G_CORNER).sum())
            axtab = fig.add_axes([0.62, 0.63, 0.35, 0.27]); axtab.axis('off')
            axtab.set_title('Do the path clusters carry motor / goal signals?\nINDEPENDENT test - these '
                            'signals were NOT used to form the clusters', fontsize=9.8, fontweight='bold')
            tab = axtab.table(cellText=rows,
                              colLabels=['outcome (Direct> one-sided)', f'Direct\nn={nD}',
                                         f'Corner\nn={nC}', 'p'],
                              colWidths=[0.52, 0.16, 0.16, 0.16], loc='center', cellLoc='center')
            tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.7)
            for (rr, cc), cell in tab.get_celld().items():
                if rr == 0:
                    cell.set_facecolor('#34495e'); cell.set_text_props(color='w', fontweight='bold')
                elif cc == 0:
                    cell.set_text_props(ha='left')
                cell.set_edgecolor('#bbb')
            axtab.text(0.5, -0.10, 'The clustering is PATH-GEOMETRY only, so these are independent evidence.\n'
                       'The reward MULTIPLIER is in this table as an OUTCOME and comes out FLAT -> path\n'
                       'type and reward streak are INDEPENDENT axes in this session. IRIS is shown as an\n'
                       'outcome on the iris row / correlation pages but is NOT a feature and is NOT here:\n'
                       'the fit is only APPROXIMATE for this mouse.',
                       transform=axtab.transAxes, ha='center', va='top', fontsize=7.6, style='italic',
                       color='#444')
        png = self.d / 'debug' / 'path_clustering.png'
        if png.exists():
            import matplotlib.image as mpimg
            axi = fig.add_axes([0.02, 0.02, 0.96, 0.50]); axi.axis('off')
            axi.imshow(mpimg.imread(png))
        return fig

    def summary_by_cluster(self):
        """Hypothesis at a glance: do Direct paths carry more fine motor control, whisking,
        heading alignment and saccades than Corner-dwelling ones?"""
        dfa, _ = self._per_trial()
        order = [c for c in (G_DIRECT, G_CORNER) if (dfa.cat == c).any()]
        panels = [('joy_fine', 'joystick fine movement'), ('sweep', 'mean whisker sweep'),
                  ('heading_align', 'heading alignment to target\n(+ toward / - away)'),
                  ('sacc_rate', 'saccades per min (APPROX)')]
        fig, ax = plt.subplots(1, 4, figsize=(18, 5))
        for a, (col, lab) in zip(ax, panels):
            vals = [dfa[dfa.cat == c][col].dropna() for c in order]
            a.bar(range(len(order)), [v.mean() for v in vals],
                  yerr=[v.std() / max(np.sqrt(len(v)), 1) for v in vals],
                  color=[CAT_COL[c] for c in order], alpha=0.85, capsize=4)
            for xi, v in enumerate(vals):
                a.scatter(np.full(len(v), xi) + np.random.uniform(-.12, .12, len(v)),
                          v, s=10, color='k', alpha=0.35, zorder=3)
            pv = (mannwhitneyu(vals[0], vals[1], alternative='two-sided').pvalue
                  if len(order) == 2 and min(len(v) for v in vals) >= 3 else np.nan)
            ps = '<0.001' if pv < 0.001 else f'p={pv:.3f}'
            a.set_xticks(range(len(order)))
            a.set_xticklabels([f'{c}\n(n={len(v)})' for c, v in zip(order, vals)], fontsize=8)
            a.set_title(f'{lab}\n{ps}', fontsize=10.5, fontweight='bold'); a.grid(alpha=0.2, axis='y')
            if col == 'heading_align':
                a.axhline(0, color='k', lw=1)
                a.text(0.98, 0.96, 'toward', transform=a.transAxes, ha='right', va='top',
                       fontsize=8, color='#2c7')
                a.text(0.98, 0.04, 'away', transform=a.transAxes, ha='right', va='bottom',
                       fontsize=8, color='#c33')
        fig.suptitle(f"{self.sess['mouse_id']}  Hypothesis check (KMeans path clusters): do Direct paths "
                     'show more joystick fine movement, whisker sweep, heading alignment and saccades?',
                     fontweight='bold', fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        return fig

    @staticmethod
    def _pcorr(x, y, Z):
        """Partial correlation of x,y controlling for covariates Z, with a df-adjusted p
        (residualise both on [1|Z], then correlate the residuals)."""
        x, y, Z = np.asarray(x, float), np.asarray(y, float), np.atleast_2d(Z)
        if Z.shape[0] != len(x):
            Z = Z.T
        Z1 = np.column_stack([np.ones(len(x)), Z])
        rx = x - Z1 @ np.linalg.lstsq(Z1, x, rcond=None)[0]
        ry = y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]
        r = np.corrcoef(rx, ry)[0, 1]
        dfree = len(x) - 2 - Z.shape[1]
        tt = r * np.sqrt(dfree / max(1 - r * r, 1e-12))
        return r, 2 * tdist.sf(abs(tt), dfree)

    @staticmethod
    def _pcorr_matrix(dfa, cols):
        """Full partial-correlation matrix (each pair controlling for ALL other cols) from the
        precision matrix. whisk SWEEP is excluded on purpose - it is the envelope of the same
        whisker signal, so controlling for it would absorb whisker's own variance."""
        X = dfa[cols].dropna().to_numpy()
        P = np.linalg.pinv(np.corrcoef(X, rowvar=False))
        d = np.sqrt(np.diag(P))
        PC = -P / np.outer(d, d)
        np.fill_diagonal(PC, 1.0)
        return PC, len(X)

    def full_corr_page(self):
        """Big 'see everything' correlation matrix over every per-trial signal - path geometry,
        motor, all five ROI flows, whisking, iris, distance, heading and the reward multiplier.
        Purpose: eyeball every correlation as CONTEXT for the clusters defined above."""
        dfa, _ = self._per_trial()
        cols = [c for c in FULL_COLS if c in dfa.columns]
        labs = [FULL_LAB[FULL_COLS.index(c)] for c in cols]
        C = dfa[cols].corr().to_numpy(); n = len(cols)
        fig, ax = plt.subplots(figsize=(13.5, 12))
        im = ax.imshow(C, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(n)); ax.set_xticklabels(labs, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(range(n)); ax.set_yticklabels(labs, fontsize=7)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center', fontsize=6,
                        color='k' if abs(C[i, j]) < 0.6 else 'w')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{self.sess['mouse_id']}  ALL per-trial correlations (n={len(dfa)} clustered trials)\n"
                     'how the per-trial signals relate - CONTEXT for the clusters defined above\n'
                     'whisk SWEEP = the whisking feature (principal-axis envelope)\n'
                     'head align = +1 toward / -1 away the target;  reward mult = the streak (an OUTCOME)',
                     fontsize=10.5, fontweight='bold')
        plt.tight_layout()
        return fig

    def correlation_page(self):
        """Is the Direct/Corner split real, is whisker flow an INDEPENDENT second motor axis from
        the paw/joystick, and how does the (APPROX) iris radius track distance-to-target?"""
        dfa, pooled = self._per_trial()
        fig, ax = plt.subplots(2, 3, figsize=(18, 11))

        def _heat(a, M, labels, title):
            im = a.imshow(M, vmin=-1, vmax=1, cmap='RdBu_r')
            a.set_xticks(range(len(labels))); a.set_xticklabels(labels, fontsize=6.5)
            a.set_yticks(range(len(labels))); a.set_yticklabels(labels, fontsize=6.5)
            for ii in range(len(labels)):
                for jj in range(len(labels)):
                    a.text(jj, ii, f'{M[ii, jj]:.2f}', ha='center', va='center', fontsize=6.5,
                           color='k' if abs(M[ii, jj]) < 0.6 else 'w')
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
            a.set_title(title, fontsize=9.5, fontweight='bold')

        # (1) raw per-trial-mean correlation matrix
        raw_cols = CORR_COLS + ['heading_align']
        raw_lab = CORR_LAB + ['head\nalign']
        _heat(ax[0, 0], dfa[raw_cols].corr().to_numpy(), raw_lab,
              f'RAW correlation (per-trial means, n={len(dfa)})')

        # (2) PARTIAL correlation - each pair controlling for all others (sweep excluded)
        pc_cols = ['whisker', 'nose', 'paw', 'joy_fine', 'radius', 'dist']
        pc_lab = ['whisker\nflow', 'nose\nflow', 'paw\nflow', 'joy fine\nmove',
                  'iris\nradius', 'dist to\ntarget']
        PC, npc = self._pcorr_matrix(dfa, pc_cols)
        _heat(ax[0, 1], PC, pc_lab, f'PARTIAL correlation (control all others, n={npc})')

        # (3) joy fine vs whisker flow, with the paw-controlled partial r
        a = ax[0, 2]
        for c in [G_DIRECT, G_CORNER]:
            d = dfa[dfa.cat == c]
            a.scatter(d.whisker, d.joy_fine, s=22, alpha=0.7, color=CAT_COL[c], label=f'{c} (n={len(d)})')
        m = dfa[['whisker', 'joy_fine', 'paw']].dropna()
        rr, pp = pearsonr(m.whisker, m.joy_fine)
        rpw, ppw = self._pcorr(m.whisker, m.joy_fine, m.paw.to_numpy())
        a.set_xlabel('whisker R flow (px/fr)'); a.set_ylabel('joystick fine movement')
        a.set_title(f'joy fine vs whisker flow\nraw r={rr:.2f} (p={pp:.1e})   |   '
                    f'partial r={rpw:.2f} (p={ppw:.2f}), paw removed', fontsize=9.5, fontweight='bold')
        a.legend(fontsize=7); a.grid(alpha=0.2)

        # (4) iris radius vs distance-to-target (pooled per-frame, binned median +/- IQR)
        a = ax[1, 0]
        dd, rd = pooled['dist'], pooled['radius']
        ok = np.isfinite(dd) & np.isfinite(rd); dd, rd = dd[ok], rd[ok]
        if dd.size > 50:
            bins = np.linspace(0, np.nanpercentile(dd, 98), 16); bc = 0.5 * (bins[:-1] + bins[1:])
            idx = np.digitize(dd, bins) - 1
            q = lambda f: np.array([f(rd[idx == k]) if np.any(idx == k) else np.nan for k in range(len(bc))])
            a.fill_between(bc, q(lambda v: np.percentile(v, 25)), q(lambda v: np.percentile(v, 75)),
                           color='#c0392b', alpha=0.2)
            a.plot(bc, q(np.median), color='#c0392b', lw=2, marker='o', ms=4)
            rr2, pp2 = pearsonr(dd, rd)
            a.set_title(f'iris radius vs distance to target (pooled frames)\nPearson r={rr2:.2f}, '
                        f'p={pp2:.1e}   [iris APPROX for this mouse]', fontsize=9.5, fontweight='bold',
                        color='#c0392b')
        a.set_xlabel('distance to target (world units)'); a.set_ylabel('iris radius (px, APPROX)')
        a.grid(alpha=0.2)

        # (5) the KMeans clusters in the eff/corner plane
        a = ax[1, 1]
        for c in [G_DIRECT, G_CORNER]:
            d = dfa[dfa.cat == c]
            a.scatter(d.eff, d.tic, s=24, alpha=0.75, color=CAT_COL[c], label=f'{c} (n={len(d)})')
        a.set_xlabel('path efficiency (straight/actual)'); a.set_ylabel('time in corner (frac)')
        a.set_title('KMeans clusters in the eff/corner plane\n(4 path-geometry features; '
                    'banish/unbanish included)', fontsize=9.5, fontweight='bold')
        a.legend(fontsize=7); a.grid(alpha=0.2)

        # (6) does each motor signal mark DIRECTness INDEPENDENTLY?
        a = ax[1, 2]
        sub = dfa[dfa.cat.isin([G_DIRECT, G_CORNER])].dropna(subset=['whisker', 'paw', 'joy_fine'])
        yb = (sub.cat == G_DIRECT).to_numpy(float)
        w, pw, jf = sub.whisker.to_numpy(), sub.paw.to_numpy(), sub.joy_fine.to_numpy()
        specs = [('whisker\n(ctrl paw)', pearsonr(w, yb), self._pcorr(w, yb, pw)),
                 ('paw\n(ctrl whisker)', pearsonr(pw, yb), self._pcorr(pw, yb, w)),
                 ('joy_fine\n(ctrl paw)', pearsonr(jf, yb), self._pcorr(jf, yb, pw))]
        xx = np.arange(len(specs))
        a.bar(xx - 0.19, [abs(sp[1][0]) for sp in specs], 0.36, color='#95a5a6', label='raw |r|')
        a.bar(xx + 0.19, [abs(sp[2][0]) for sp in specs], 0.36, color='#2c3e50', label='partial |r|')
        for k, sp in enumerate(specs):
            a.text(k + 0.19, abs(sp[2][0]) + 0.01, f'p={sp[2][1]:.2f}', ha='center', fontsize=7)
        a.set_xticks(xx); a.set_xticklabels([sp[0] for sp in specs], fontsize=8)
        a.set_ylabel('|correlation| with DIRECT indicator')
        a.set_title(f'independent marker of DIRECTness?\n(Direct vs Corner-dwelling, n={len(sub)})',
                    fontsize=9.5, fontweight='bold')
        a.legend(fontsize=7); a.grid(alpha=0.2, axis='y')
        print(f'  [correlation] whisker|joy_fine ctrl paw r={rpw:+.2f} p={ppw:.3f}  |  '
              f'whisker->DIRECT ctrl paw r={specs[0][2][0]:+.2f} p={specs[0][2][1]:.3f}  |  '
              f'paw->DIRECT ctrl whisker r={specs[1][2][0]:+.2f} p={specs[1][2][1]:.3f}')

        fig.suptitle(f"{self.sess['mouse_id']}  Cross-trial correlations: is the DIRECT/CORNER split "
                     'real, and is whisker flow an INDEPENDENT second motor axis (not just the paw)?',
                     fontweight='bold', fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def finding6_page(self):
        """Direct vs Corner-dwelling for EACH parameter (mean +/- SEM, per-trial dots, two-sided
        Mann-Whitney). All are INDEPENDENT of the clustering EXCEPT trial length, which IS a
        clustering feature (greyed, descriptive only)."""
        dfa, _ = self._per_trial()
        params = [('sweep', 'whisker AMPLITUDE\n(sweep)', False, False),
                  ('cycle', 'whisker CYCLE\n(frequency, Hz)', False, False),
                  ('joy_fine', 'joystick fine movement', False, False),
                  ('heading_align', 'heading alignment\n(+ toward / - away)', True, False),
                  ('radius', 'iris radius (px)\n[APPROX - NOT CLEAN]', False, False),
                  ('dur', 'trial length (s)\n[CLUSTERING FEATURE]', False, True)]
        order = [c for c in (G_DIRECT, G_CORNER) if (dfa.cat == c).any()]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        line = []
        for a, (key, lab, signed, feat) in zip(axes.ravel(), params):
            vals = [dfa[dfa.cat == c][key].dropna().values for c in order]
            means = [v.mean() if len(v) else np.nan for v in vals]
            sems = [v.std() / max(np.sqrt(len(v)), 1) if len(v) else 0 for v in vals]
            a.bar(range(len(order)), means, yerr=sems, capsize=5, alpha=0.85,
                  color=[CAT_COL[c] for c in order])
            for xi, v in enumerate(vals):
                a.scatter(np.full(len(v), xi) + np.random.uniform(-.12, .12, len(v)),
                          v, s=11, color='k', alpha=0.35, zorder=3)
            pv = (mannwhitneyu(vals[0], vals[1], alternative='two-sided').pvalue
                  if len(order) == 2 and min(len(v) for v in vals) >= 3 else np.nan)
            ps = 'p<0.001' if pv < 0.001 else f'p={pv:.3f}'
            star = ' *' if (np.isfinite(pv) and pv < 0.05) else ' (n.s.)'
            fmt = '{:+.2f}' if signed else '{:.2f}'
            a.set_title(f'{lab}\nDirect {fmt.format(means[0])} vs Corner {fmt.format(means[1])}  '
                        f'{ps}{star}', fontsize=10, fontweight='bold',
                        color='#555' if feat else ('#c0392b' if 'APPROX' in lab else '#111'))
            a.set_xticks(range(len(order)))
            a.set_xticklabels([f'{c}\n(n={len(v)})' for c, v in zip(order, vals)], fontsize=8)
            a.grid(alpha=0.2, axis='y')
            if signed:
                a.axhline(0, color='k', lw=0.8)
                a.text(0.98, 0.96, 'toward', transform=a.transAxes, ha='right', va='top',
                       fontsize=8, color='#2c7')
                a.text(0.98, 0.04, 'away', transform=a.transAxes, ha='right', va='bottom',
                       fontsize=8, color='#c33')
            line.append(f'{key} {means[0]:.2f}/{means[1]:.2f}')
        nd = int((dfa.cat == G_DIRECT).sum()); nc = int((dfa.cat == G_CORNER).sum())
        fig.suptitle(f"{self.sess['mouse_id']}  Finding-6: motor / goal parameters vs the PATH-GEOMETRY "
                     f'clusters - Direct (n={nd}) vs Corner-dwelling (n={nc})\n'
                     'mean +/- SEM, dots = trials, two-sided Mann-Whitney. The clustering used PATH '
                     'geometry ONLY, so these are INDEPENDENT evidence (except trial length, a feature).',
                     fontweight='bold', fontsize=10.5)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        print('  [finding-6 D vs C] ' + ' | '.join(line))
        return fig

    def hypotheses_page(self):
        """Overview of the whisker<->paw hypothesis section: what H1 / H2 / the outcome-valence test
        each ASK (they are easy to confuse), the signal limits that bound all three, and a
        result-with-caveat table -- shown BEFORE the three figures it introduces."""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"{self.sess['mouse_id']}  Whisker <-> paw hypotheses: H1 (carry-over), "
                     'H2 (movement-locked) and OUTCOME VALENCE (feedback-locked)',
                     fontweight='bold', fontsize=13)
        # two columns: what each test ASKS (left) and the stats vocabulary (right), so neither
        # overflows the page.
        axt = fig.add_axes([0.025, 0.575, 0.47, 0.36]); axt.axis('off')
        axt.text(0.0, 1.0, HYPOTH_INTRO, va='top', ha='left', fontsize=7.6, family='monospace',
                 linespacing=1.32,
                 bbox=dict(boxstyle='round', fc='#f7f7f7', ec='0.55', alpha=0.97))
        axg = fig.add_axes([0.508, 0.575, 0.47, 0.36]); axg.axis('off')
        axg.text(0.0, 1.0, HYPOTH_GLOSSARY, va='top', ha='left', fontsize=7.6, family='monospace',
                 linespacing=1.32,
                 bbox=dict(boxstyle='round', fc='#eef4fa', ec='#8fb4d6', alpha=0.97))
        # NOT a matplotlib table: table cells do not grow to fit multi-line text, so the caveat
        # text spilled across neighbouring rows. One panel per hypothesis instead.
        fig.text(0.025, 0.545, 'Result of each test, with the caveat that qualifies it',
                 fontsize=11.5, fontweight='bold', va='top')
        verdict_col = {'POSITIVE': '#27ae60', 'NEGATIVE': '#c0392b'}
        top, gap = 0.515, 0.012
        h = (top - 0.03 - gap * (len(HYPOTH_RESULTS) - 1)) / len(HYPOTH_RESULTS)
        for k, (q, verdict, body) in enumerate(HYPOTH_RESULTS):
            y = top - (k + 1) * h - k * gap
            col = verdict_col.get(verdict.split('\n')[0], '#b7791f')
            a = fig.add_axes([0.03, y, 0.94, h]); a.axis('off')
            a.add_patch(Rectangle((0, 0), 1, 1, transform=a.transAxes, facecolor='#fbfbfb',
                                  edgecolor='0.75', lw=0.8, zorder=0))
            a.add_patch(Rectangle((0, 0), 0.006, 1, transform=a.transAxes, facecolor=col,
                                  edgecolor='none', zorder=1))
            a.text(0.018, 0.80, q, fontsize=9.5, fontweight='bold', va='top', transform=a.transAxes)
            a.text(0.018, 0.30, verdict, fontsize=10, fontweight='bold', color=col, va='top',
                   transform=a.transAxes)
            a.text(0.20, 0.90, body, fontsize=8.4, family='monospace', va='top', linespacing=1.32,
                   transform=a.transAxes)
        return fig

    def build(self, n=None, out='trial_report.pdf'):
        a = self.df[self.df.analyze].copy()
        a['_ord'] = a.outcome.map({o: k for k, o in enumerate(OUTCOME_ORDER)})
        idx = list(a.sort_values(['_ord', 'trial']).index)[:n] if n else list(a.sort_values(['_ord', 'trial']).index)
        outp = self.d / out
        # SUMMARY/ANALYSIS PAGES FIRST, in 231's order: CLASSIFICATION first, then everything that
        # relies on it. The multiplier page is the addition for this task variant (231 had no
        # multiplier) and sits with the feature story: it documents why the multiplier is an
        # OUTCOME, not a clustering feature.
        pages = [
            ('classification', self.clustering_page),                        # 1. method + outcome table + diagnostic
            ('feature selection', lambda: self._embed_png('feature_selection.png')),   # 2. how the features were chosen
            ('multiplier handling', lambda: self._embed_png('multiplier_outcome.png', (16, 5.2))),  # 3. NEW for this task
            ('summary by cluster', self.summary_by_cluster),                 # 4. hypothesis at a glance
            ('summary by outcome', self.summary),                            # 5. task outcomes
            ('reward magnitude', self.reward_magnitude),                     # 6. multiplier scaling
            ('full correlation', self.full_corr_page),                       # 7. all per-trial correlations
            ('correlation + partial', self.correlation_page),                # 8. is whisker independent of paw
            ('finding-6', self.finding6_page),                               # 9. outcomes vs the clusters
            # 10-13. the whisker<->paw hypothesis section: the explainer FIRST (what each test asks
            # and why they differ), then H1, H2 and the outcome-valence figures.
            ('hypotheses intro', self.hypotheses_page),
            ('H1 post-error/conflict', lambda: self._embed_png('h1_post_error.png', (16, 8.5))),
            ('H2 whisker reset', lambda: self._embed_png('h2_whisker_reset.png', (16, 7.8))),
            ('outcome valence', lambda: self._embed_png('collection_valence.png', (16, 5.0))),
            # 14. whisker sweep by CONTEXT: session half, and reward-on-screen vs not
            ('whisker context', lambda: self._embed_png('whisker_context.png', (16, 4.6))),
            # 15-17. the performance block, in the same order as the standalone
            # performance_report.pdf: define the metric, show the result, then the ordering
            # information D is blind to.
            ('what he can see', lambda: self._embed_png('visibility_census.png', (16, 5.0))),
            ('chance explainer', lambda: self._embed_png('visibility_chance_explainer.png', (16, 9.8))),
            ('performance', lambda: self._embed_png('performance.png', (16, 8.6))),
            ('choice: when available', lambda: self._embed_png('choice_a.png', (16, 9.0))),
            ('choice: decision point', lambda: self._embed_png('choice_b.png', (16, 9.0))),
            ('choice: avoidance', lambda: self._embed_png('choice_c.png', (16, 9.0))),
            ('streaks / combo', lambda: self._embed_png('streaks.png', (16, 10.3))),
        ]
        n_head = 0
        pageno = [0]                                    # list so the closure can mutate it

        def _stamp(fig, label=''):
            """Bottom-right page number on EVERY page, so a page can be referred to by number
            (2026-08-18). The optional label names the page in the same stamp."""
            pageno[0] += 1
            fig.text(0.995, 0.006, f'{label}   p. {pageno[0]}'.strip(), ha='right', va='bottom',
                     fontsize=8, color='#555')
            return fig

        with PdfPages(outp) as pdf:
            for name, fn in pages:
                try:
                    fig = fn()
                except Exception as ex:                 # one bad analysis page must not lose the report
                    print(f"  !! skipped '{name}' page: {type(ex).__name__}: {ex}")
                    continue
                if fig is None:
                    continue
                _stamp(fig, name)
                pdf.savefig(fig); plt.close(fig); n_head += 1
                print(f"  + p.{pageno[0]}  {name}")
            for k, i in enumerate(idx):
                fig = self.page(i)
                if fig is not None:
                    _stamp(fig, f"trial {int(self.df.loc[i, 'trial'])}")
                    pdf.savefig(fig); plt.close(fig)
                if k % 20 == 0:
                    print(f'  trial pages {k}/{len(idx)}', flush=True)
        print(f"wrote {outp}  ({len(idx)} trial pages + {n_head} summary/analysis pages)")
        return outp


if __name__ == '__main__':
    sd = sys.argv[1] if len(sys.argv) > 1 else '.'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    Report(sd).build(n=n, out='trial_report_test.pdf' if n else 'trial_report.pdf')
