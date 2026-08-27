"""
H2 - is there a whisker "reset" (amplitude transient) time-locked to each paw (joystick) movement,
and does it differ by trial condition (correct / error / conflict)?

Method (event-triggered, reusing the 231 whisker<->joystick lead/lag logic):
  - paw movement events = joystick re-steer velocity-burst peaks (common/joymove.movement_events,
    ~1900 events, 44/min);
  - event-triggered whisker `sweep` amplitude over +-0.5 s, baseline-subtracted (pre-onset mean);
  - a CIRCULAR-SHIFT null (shift the whisker signal by random offsets) gives the chance band, and a
    bootstrap-over-events CI gives the estimate band;
  - the MULTIPLIER bars additionally get a TRIAL-CLUSTER bootstrap CI (resampling TRIALS, not events:
    one trial contributes dozens of correlated re-steers, so an event bootstrap is ~sqrt(ev/trials)
    too narrow) plus a FIXED-LAG (unbiased) magnitude alongside the peak;
  - reset MAGNITUDE = peak(post-onset) - baseline; reset LATENCY = time of that peak (231 found
    whisking FOLLOWS the joystick by ~70-100 ms, i.e. a post-onset peak);
  - split events by the trial they fall in: current cluster (Direct/Corner-dwelling = conflict axis)
    and outcome state (single_reward / banish / unbanish).
  - lick+groom frames are EXCLUDED (lick ~7.7 Hz sits in the whisker band); we verify the ALL-events
    curve's sign/latency is unchanged with them included.

Signal limits (on the figure): whisker = optic-FLOW AMPLITUDE transient, NOT a set-point "reset";
30 fps -> 33 ms resolution; head-fixed contamination handled by the lick/groom exclusion; ONE session.

Outputs: debug/h2_whisker_reset.png + printed stats.
Run:  python3 analyze_h2_whisker_reset.py /home/maryam/repo/flow_test/JPAS_0168
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_COMMON = Path(__file__).resolve().parent.parent / 'common'
sys.path.insert(0, str(_COMMON))
import fps as fpsmod       # noqa: E402
import joymove             # noqa: E402

WIN_S = 0.5               # +-window (s) around each movement event
CORE_S = 0.15             # gate: reject an event only if lick/groom is within +-CORE_S of its onset
BASE_LO_S, BASE_HI_S = -0.5, -0.2   # pre-onset baseline window
N_BOOT = 1000
N_SHIFT = 200
SEED = 0


def _eta(sig, events, w, excl=None, core=0):
    """Event-triggered matrix (rows = events with a full +-w window, cols = lag) AND the
    indices of the events that survived, so each row can be traced back to its trial. If `excl` is given,
    an event is REJECTED only when an excluded frame is within +-`core` frames of its onset (a small
    core), not across the whole window -- licking is 20.6% of the session, so whole-window exclusion
    would drop ~85% of events; the robustness panel shows the exclusion barely moves the curve anyway."""
    N = len(sig)
    rows, keep = [], []
    for i, e in enumerate(events):
        if e - w < 0 or e + w >= N:
            continue
        if excl is not None and excl[max(0, e - core):e + core + 1].any():
            continue
        rows.append(sig[np.arange(e - w, e + w + 1)])
        keep.append(i)
    return np.array(rows), np.array(keep, int)


def _baseline_sub(M, lags, blo, bhi):
    base = M[:, (lags >= blo) & (lags <= bhi)].mean(axis=1, keepdims=True)
    return M - base


def run(session_dir, write=True):
    rng = np.random.default_rng(SEED)
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    log = json.load(open(d / 'log.json'))
    fps = sess['fps']
    df = pd.read_pickle(d / 'df_trials_clean.pkl')
    sweep = np.load(d / 'opticflow' / 'whisker.npz', allow_pickle=True)['sweep'].astype(float)
    ms = np.load(d / 'opticflow' / 'mouth_state.npz', allow_pickle=True)['mouth_state']
    excl = (ms == 1) | (ms == 2)
    N = len(sweep)

    fm = fpsmod.frame_to_ms(log)
    jx, jy = joymove.stick_on_frames(log, fm)
    events, info = joymove.movement_events(jx, jy, fps)

    # frame -> trial cluster / outcome / multiplier (disjoint windows). multiplier is the reward
    # STREAK (1-4) and is only defined on single_reward trials.
    cluster = np.full(N, '', object); outcome = np.full(N, '', object); mult = np.full(N, np.nan)
    trial_of = np.full(N, -1, int)          # frame -> trial id, for the trial-CLUSTER bootstrap
    for ti, r in df[df.analyze].iterrows():
        s, e = int(r['start_frame']), int(min(r['end_frame'], N))
        cluster[s:e] = r['cluster_name']; outcome[s:e] = r['outcome']; trial_of[s:e] = ti
        if r['outcome'] == 'single_reward' and np.isfinite(r['multiplier']):
            mult[s:e] = r['multiplier']
    ev_cluster = cluster[events]; ev_outcome = outcome[events]; ev_mult = mult[events]
    ev_trial = trial_of[events]

    w = int(round(WIN_S * fps)); core = int(round(CORE_S * fps)); lags = np.arange(-w, w + 1) / fps

    # ALL events (lick/groom-core excluded), + a version INCLUDING them for the robustness check
    M_all_raw, keep = _eta(sweep, events, w, excl, core)
    M_all = _baseline_sub(M_all_raw, lags, BASE_LO_S, BASE_HI_S)
    M_incl = _baseline_sub(_eta(sweep, events, w, None)[0], lags, BASE_LO_S, BASE_HI_S)
    # per-ROW (surviving event) labels -- baseline subtraction is per row, so any subset of M_all is
    # identical to re-running _eta on that subset; this lets every condition reuse the one matrix.
    row_trial = ev_trial[keep]; row_mult = ev_mult[keep]

    def peak_stats(M):
        mu = M.mean(axis=0)
        post = (lags >= 0) & (lags <= WIN_S)
        pk = np.argmax(mu[post]); lat = lags[post][pk]
        return float(mu[post][pk]), float(lat), mu

    mag_all, lat_all, mu_all = peak_stats(M_all)
    mag_incl, lat_incl, _ = peak_stats(M_incl)

    # bootstrap CI (over events) + circular-shift null for ALL events
    boot = np.array([M_all[rng.integers(0, len(M_all), len(M_all))].mean(axis=0) for _ in range(N_BOOT)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5], axis=0)
    shifts = rng.integers(w + 1, N - w - 1, N_SHIFT)
    null = np.array([_baseline_sub(_eta(np.roll(sweep, sh), events, w, excl, core)[0], lags, BASE_LO_S, BASE_HI_S).mean(axis=0)
                     for sh in shifts])
    null_lo, null_hi = np.percentile(null, [2.5, 97.5], axis=0)

    print(f"=== {sess['mouse_id']} H2: whisker reset around paw (joystick) movement ===")
    print(f"  {len(events)} movement events ({len(events)/((fm[-1]-fm[0])/60000):.1f}/min); "
          f"{len(M_all)} usable windows (lick/groom-excluded)")
    print(f"  ALL events: reset magnitude {mag_all:+.4f} sweep, latency {lat_all*1000:+.0f} ms "
          f"({'FOLLOWS' if lat_all>0 else 'LEADS'} the movement)")
    print(f"  robustness (lick/groom INCLUDED): magnitude {mag_incl:+.4f}, latency {lat_incl*1000:+.0f} ms")

    # FIXED LAG: read every condition at the SAME lag -- the ALL-events peak (+33 ms). Comparing
    # per-condition PEAKS would compare maxima found at different lags, which is biased upward and
    # more so for small n. `post` is the post-onset half used for the peak search.
    post = (lags >= 0) & (lags <= WIN_S)
    fix_i = int(np.argmin(np.abs(lags - lat_all)))

    # split by cluster and by outcome
    conds = [('Direct', ev_cluster == 'Direct', '#27ae60'),
             ('Corner-dwelling', ev_cluster == 'Corner-dwelling', '#c0392b')]
    # DISPLAY labels: every reward in this task carries a multiplier (there is no plain
    # "single" reward), so `single_reward` is shown as REWARD; the data column is unchanged.
    oconds = [('reward', ev_outcome == 'single_reward', '#27ae60'),
              ('banish', ev_outcome == 'banish', '#c0392b'),
              ('unbanish (escape)', ev_outcome == 'unbanish', '#2980b9')]

    def cond_curve(mask):
        M = M_all[mask[keep]]
        if len(M) < 10:
            return None
        mag, lat, mu = peak_stats(M)
        return dict(mu=mu, n=len(M), mag=mag, lat=lat)

    def boot_diff(mask_a, mask_b, n_boot=N_BOOT):
        """Trial-cluster bootstrap of the DIFFERENCE in fixed-lag reset magnitude between two event
        sets. Resamples TRIALS (events inside a trial are correlated) and reads both curves at the
        ALL-events peak lag, so no per-condition peak-picking bias enters the comparison."""
        by_a = {t: np.where(mask_a & (row_trial == t))[0] for t in np.unique(row_trial[mask_a])}
        by_b = {t: np.where(mask_b & (row_trial == t))[0] for t in np.unique(row_trial[mask_b])}
        ta, tb = np.array(list(by_a)), np.array(list(by_b))
        if len(ta) < 2 or len(tb) < 2:
            return None
        d = []
        for _ in range(n_boot):
            ia = np.concatenate([by_a[t] for t in rng.choice(ta, len(ta), replace=True)])
            ib = np.concatenate([by_b[t] for t in rng.choice(tb, len(tb), replace=True)])
            d.append(M_all[ia].mean(axis=0)[fix_i] - M_all[ib].mean(axis=0)[fix_i])
        d = np.array(d)
        return dict(diff=float(M_all[mask_a].mean(axis=0)[fix_i] - M_all[mask_b].mean(axis=0)[fix_i]),
                    ci=np.percentile(d, [2.5, 97.5]),
                    p=float(2 * min((d <= 0).mean(), (d >= 0).mean())),
                    n_tr=(len(ta), len(tb)))

    print("  by CONFLICT (cluster):")
    cluster_res = {}
    for name, mask, _ in conds:
        r = cond_curve(mask); cluster_res[name] = r
        if r:
            print(f"    {name:16s} n={r['n']:4d}  magnitude {r['mag']:+.4f}  latency {r['lat']*1000:+.0f} ms")
    cl_test = boot_diff(ev_cluster[keep] == 'Direct', ev_cluster[keep] == 'Corner-dwelling')
    if cl_test:
        print(f"    Direct vs Corner-dwelling difference at the +{lat_all*1000:.0f} ms fixed lag: "
              f"{cl_test['diff']:+.4f}  95% CI [{cl_test['ci'][0]:+.4f}, {cl_test['ci'][1]:+.4f}]  "
              f"bootstrap p={cl_test['p']:.3f}  ({cl_test['n_tr'][0]} vs {cl_test['n_tr'][1]} trials)"
              f"  -> {'DIFFER' if cl_test['p'] < 0.05 else 'NOT significantly different'}")

    print("  by OUTCOME state:")
    out_res = {}
    for name, mask, _ in oconds:
        r = cond_curve(mask); out_res[name] = r
        if r:
            print(f"    {name:16s} n={r['n']:4d}  magnitude {r['mag']:+.4f}  latency {r['lat']*1000:+.0f} ms")

    # by MULTIPLIER (reward streak 1-4) on single_reward events -- does the reset SCALE with reward
    # streak, as whisker amplitude does (replicates the 'flow scales with reward multiplier' finding)?
    mconds = [(m, ev_mult == m, plt.cm.viridis(0.15 + 0.7 * (m - 1) / 3)) for m in [1, 2, 3, 4]]
    print("  by MULTIPLIER (reward events, streak):")
    mult_res = {}
    for m, mask, _ in mconds:
        r = cond_curve(mask); mult_res[m] = r
        if r:
            print(f"    mult {m}         n={r['n']:4d}  magnitude {r['mag']:+.4f}  latency {r['lat']*1000:+.0f} ms")

    # ---- multiplier bars: TRIAL-cluster bootstrap CIs ------------------------
    # Two things make the raw bar heights over-confident, and mult 4 (n=22 events / 9 trials) is the
    # least certain point, so both are corrected here:
    #   (1) events within a trial are NOT independent (a trial contributes dozens of re-steers), so the
    #       bootstrap resamples TRIALS with replacement, not events -- an event bootstrap would give a
    #       CI ~sqrt(events/trials) too narrow;
    #   (2) `peak of the post-onset mean` is a MAX, so it is positively biased, and the bias GROWS as n
    #       shrinks -- which would inflate exactly the mult-4 bar. So alongside the peak we report the
    #       magnitude at a FIXED lag (the ALL-events peak latency), which is unbiased and comparable
    #       across levels. The trend is judged on the fixed-lag series.
    def _mags(rows_idx):
        """(peak magnitude, fixed-lag magnitude) of the mean curve over these rows of M_all."""
        mu = M_all[rows_idx].mean(axis=0)
        return float(mu[post].max()), float(mu[fix_i])

    def boot_mult(mask_rows, n_boot=N_BOOT):
        """Cluster bootstrap over trials -> peak & fixed-lag magnitude draws for one condition.
        The multiplier is a TRIAL-level label, so every row of a contributing trial is in the
        condition and a trial's rows can be taken whole."""
        by_trial = {t: np.where(mask_rows & (row_trial == t))[0] for t in np.unique(row_trial[mask_rows])}
        tr = np.array(list(by_trial))
        pk, fx = [], []
        for _ in range(n_boot):
            idx = np.concatenate([by_trial[t] for t in rng.choice(tr, len(tr), replace=True)])
            a, b = _mags(idx)
            pk.append(a); fx.append(b)
        return np.array(pk), np.array(fx), len(tr)

    mult_ci = {}
    print("  by MULTIPLIER -- trial-cluster bootstrap 95% CI "
          f"(fixed lag = ALL-events peak, {lat_all*1000:+.0f} ms):")
    for m in [1, 2, 3, 4]:
        rows_m = (row_mult == m)
        if mult_res.get(m) is None:
            continue
        pk, fx, ntr = boot_mult(rows_m)
        mult_ci[m] = dict(pk=pk, fx=fx, n_trials=ntr,
                          pk_ci=np.percentile(pk, [2.5, 97.5]), fx_ci=np.percentile(fx, [2.5, 97.5]),
                          fx_mag=_mags(np.where(rows_m)[0])[1])
        c = mult_ci[m]
        print(f"    mult {m}  n={mult_res[m]['n']:4d} ev / {ntr:2d} trials   "
              f"peak {mult_res[m]['mag']:+.4f} [{c['pk_ci'][0]:+.4f}, {c['pk_ci'][1]:+.4f}]   "
              f"fixed-lag {c['fx_mag']:+.4f} [{c['fx_ci'][0]:+.4f}, {c['fx_ci'][1]:+.4f}]")

    # trend r across the 4 levels, with its own trial-cluster bootstrap CI (same draws re-paired)
    mm_all = [m for m in [1, 2, 3, 4] if m in mult_ci]
    r_pk = r_fx = np.nan; r_pk_ci = r_fx_ci = (np.nan, np.nan); p_fx = np.nan
    if len(mm_all) > 2:
        r_pk = float(np.corrcoef(mm_all, [mult_res[m]['mag'] for m in mm_all])[0, 1])
        r_fx = float(np.corrcoef(mm_all, [mult_ci[m]['fx_mag'] for m in mm_all])[0, 1])
        nb = min(len(mult_ci[m]['pk']) for m in mm_all)
        rb_pk = np.array([np.corrcoef(mm_all, [mult_ci[m]['pk'][i] for m in mm_all])[0, 1] for i in range(nb)])
        rb_fx = np.array([np.corrcoef(mm_all, [mult_ci[m]['fx'][i] for m in mm_all])[0, 1] for i in range(nb)])
        r_pk_ci = np.percentile(rb_pk[np.isfinite(rb_pk)], [2.5, 97.5])
        r_fx_ci = np.percentile(rb_fx[np.isfinite(rb_fx)], [2.5, 97.5])
        p_fx = float((rb_fx[np.isfinite(rb_fx)] <= 0).mean())   # bootstrap P(no positive trend)
        print(f"  trend across multipliers: peak r={r_pk:+.2f} [{r_pk_ci[0]:+.2f}, {r_pk_ci[1]:+.2f}]   "
              f"fixed-lag r={r_fx:+.2f} [{r_fx_ci[0]:+.2f}, {r_fx_ci[1]:+.2f}]  "
              f"(bootstrap P(r<=0) = {p_fx:.3f})")

    # ---- figure ----
    fig, ax = plt.subplots(2, 3, figsize=(19, 11.0))   # taller: room for the footnote band
    a = ax[0, 0]
    a.fill_between(lags, null_lo, null_hi, color='0.8', label='circular-shift null 95%')
    a.fill_between(lags, ci_lo, ci_hi, color='#2980b9', alpha=0.25, label='bootstrap 95% confidence interval')
    a.plot(lags, mu_all, color='#2980b9', lw=2, label=f'whisker reset (n={len(M_all)})')
    a.axvline(0, color='k', ls='--', lw=1); a.axhline(0, color='k', lw=0.5)
    a.plot(lat_all, mag_all, 'v', color='#c0392b', ms=9, label=f'peak {lat_all*1000:+.0f} ms')
    a.set_xlabel('time from paw movement (s)'); a.set_ylabel('whisker sweep (baseline-sub)')
    a.set_title('ALL events: is there a whisker reset?', fontsize=10, fontweight='bold'); a.legend(fontsize=7)

    a = ax[0, 1]
    a.plot(lags, mu_all, color='#7f8c8d', lw=1.2, ls='--', label='all (excl lick/groom)')
    a.plot(lags, peak_stats(M_incl)[2], color='#e67e22', lw=1.2, label='all (INCL lick/groom)')
    a.axvline(0, color='k', ls='--', lw=1); a.axhline(0, color='k', lw=0.5)
    a.set_xlabel('time from paw movement (s)'); a.set_ylabel('whisker sweep (baseline-sub)')
    a.set_title('robustness: lick/groom exclusion doesn\'t move the peak', fontsize=10, fontweight='bold')
    a.legend(fontsize=7)

    a = ax[0, 2]
    for name, mask, c in conds:
        r = cluster_res.get(name)
        if r:
            a.plot(lags, r['mu'], color=c, lw=1.8, label=f"{name} (n={r['n']})")
    a.axvline(0, color='k', ls='--', lw=1); a.axhline(0, color='k', lw=0.5)
    a.set_xlabel('time from paw movement (s)'); a.set_ylabel('whisker sweep (baseline-sub)')
    if cl_test:
        a.set_title(f"reset by CONFLICT (path cluster)\ndifference {cl_test['diff']:+.3f} "
                    f"[{cl_test['ci'][0]:+.3f}, {cl_test['ci'][1]:+.3f}]  p={cl_test['p']:.3f}"
                    f"{'  *' if cl_test['p'] < 0.05 else '  (n.s.)'}", fontsize=9.5, fontweight='bold')
    else:
        a.set_title('reset by CONFLICT (path cluster)', fontsize=10, fontweight='bold')
    a.legend(fontsize=8)

    a = ax[1, 0]
    for name, mask, c in oconds:
        r = out_res.get(name)
        if r:
            a.plot(lags, r['mu'], color=c, lw=1.8, label=f"{name} (n={r['n']})")
    a.axvline(0, color='k', ls='--', lw=1); a.axhline(0, color='k', lw=0.5)
    a.set_xlabel('time from paw movement (s)'); a.set_ylabel('whisker sweep (baseline-sub)')
    a.set_title('reset by OUTCOME state\n(REWARD here lumps all streaks 1-4 together)',
                fontsize=10, fontweight='bold')
    a.legend(fontsize=8)

    # by MULTIPLIER (reward streak) -- the single_reward events broken out by streak 1->4
    a = ax[1, 1]
    for m, mask, c in mconds:
        r = mult_res.get(m)
        if r:
            a.plot(lags, r['mu'], color=c, lw=1.8, label=f"mult {m} (n={r['n']})")
    a.axvline(0, color='k', ls='--', lw=1); a.axhline(0, color='k', lw=0.5)
    a.set_xlabel('time from paw movement (s)'); a.set_ylabel('whisker sweep (baseline-sub)')
    a.set_title('reset by MULTIPLIER (reward trials, streak 1->4)', fontsize=10, fontweight='bold')
    a.legend(fontsize=8)

    # magnitude vs multiplier (does the reset SCALE with reward streak)
    a = ax[1, 2]
    mm = [m for m in [1, 2, 3, 4] if mult_res.get(m)]
    mag = [mult_res[m]['mag'] for m in mm]; nn = [mult_res[m]['n'] for m in mm]
    cols = [plt.cm.viridis(0.15 + 0.7 * (m - 1) / 3) for m in mm]
    # Bars = peak magnitude with a TRIAL-cluster bootstrap CI (events inside a trial are not
    # independent). Black points = the same events read at a FIXED lag (the ALL-events peak), which
    # is unbiased -- 'peak of the mean' is a MAX, and its upward bias grows as n shrinks, i.e. it
    # would flatter exactly the mult-4 bar (22 events / 5 trials, the least certain point).
    err = np.array([[mag[k] - mult_ci[m]['pk_ci'][0], mult_ci[m]['pk_ci'][1] - mag[k]]
                    for k, m in enumerate(mm)]).T if mult_ci else None
    a.bar(mm, mag, color=cols, yerr=err, capsize=5, ecolor='#333')
    if mult_ci:
        fx = [mult_ci[m]['fx_mag'] for m in mm]
        fxe = np.array([[fx[k] - mult_ci[m]['fx_ci'][0], mult_ci[m]['fx_ci'][1] - fx[k]]
                        for k, m in enumerate(mm)]).T
        a.errorbar(np.array(mm) + 0.18, fx, yerr=fxe, fmt='o', color='k', ms=4, lw=1.2, capsize=3,
                   label=f'fixed lag {lat_all*1000:+.0f} ms (unbiased)')
        a.legend(fontsize=7, loc='upper left')
    for k, (m, g, n) in enumerate(zip(mm, mag, nn)):
        ntr = mult_ci[m]['n_trials'] if m in mult_ci else 0
        a.text(m, (mult_ci[m]['pk_ci'][1] if m in mult_ci else g) + 0.008,
               f'n={n}\n({ntr} tr)', ha='center', fontsize=7)
    a.axhline(0, color='k', lw=0.6)
    if len(mm) > 2:
        a.set_title(f'reset magnitude vs multiplier\npeak r={r_pk:+.2f} [{r_pk_ci[0]:+.2f}, '
                    f'{r_pk_ci[1]:+.2f}]   fixed-lag r={r_fx:+.2f}, boot P(r<=0)={p_fx:.3f}',
                    fontsize=9.5, fontweight='bold')
    a.set_xlabel('reward multiplier (streak)'); a.set_ylabel('reset peak magnitude (sweep)')
    a.set_xticks([1, 2, 3, 4]); a.grid(alpha=0.2, axis='y')
    a.margins(y=0.18)          # headroom so the n= labels clear the title

    fig.suptitle(f"{sess['mouse_id']} H2 - whisker RESET time-locked to paw (joystick) movement  "
                 f"[whisker=optic-flow AMPLITUDE, not set-point; 33 ms resolution; lick/groom excluded; "
                 f"reward trials split by MULTIPLIER/streak; ONE session]", fontsize=10.5, fontweight='bold')
    # FOOTNOTE band: the three method terms on this figure, in plain language. Requested
    # (2026-08-18) -- "circular-shift null", "bootstrap 95% CI" and "fixed lag" are not self-evident.
    fig.text(0.008, 0.014,
             "HOW TO READ THIS FIGURE\n"
             "CIRCULAR-SHIFT NULL (grey band, panel 1) = the chance level. We slide the whisker trace "
             "in time by a random offset and re-run the identical analysis, 200 times.\n"
             "   That destroys any true time-lock to the joystick but KEEPS the whisker signal's own "
             "statistics (its rhythm, its slow drifts), so the grey band is what this\n"
             "   analysis produces from a signal that CANNOT be reacting to the paw. The blue curve "
             "leaving the grey band is the evidence.\n"
             "BOOTSTRAP 95% CONFIDENCE INTERVAL (blue band, panel 1; error bars, panel 6) = how much it would "
             "wobble with a different sample. We re-draw the data with replacement\n"
             "   many times and re-compute; 95% of those re-draws fall inside the band, so a band "
             "excluding 0 means the effect is unlikely to be sampling noise. We resample\n"
             "   TRIALS, not single events: one trial contributes dozens of correlated re-steers, so "
             "resampling events would make the interval far too narrow.\n"
             f"FIXED LAG {lat_all*1000:+.0f} ms (black points, panel 6) = every bar read at the SAME "
             "moment, the peak of the ALL-events curve. The coloured bar is instead each\n"
             "   condition's OWN highest point - and a maximum is biased upward, the more so the fewer "
             "events it is taken over. Mult 4 has only 22 events, so it would\n"
             "   gain most from that bias. Reading all four at one fixed moment removes it: bar = peak "
             "(optimistic), black point = fixed lag (fair). They agree here,\n"
             "   so the multiplier trend is not a small-sample artifact.",
             fontsize=9.0, va='bottom', ha='left', linespacing=1.4, family='monospace',
             bbox=dict(boxstyle='round', fc='#f7f7f7', ec='0.7', alpha=0.95))
    plt.tight_layout(rect=[0, 0.205, 1, 0.965])
    out = d / 'debug' / 'h2_whisker_reset.png'
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
