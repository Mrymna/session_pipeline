"""
mixed_protocol -- shared helpers for the RANDOM timeout/banishment task (JPAS_473-style).

This protocol puts ONE punishment icon on the board that is randomly a **banishment** or a **timeout**
(the effect's `options` field lists both, e.g. [['banish','fountains'],['timeout','target']]), alongside
the reward(s). It is log-only: everything here is derived from `log.json`.

The single-session notebook (`single_session_performance_0.ipynb`) builds the per-collection dataframe
and the plots inline; THIS module holds the same computations so the multi-session notebook
(`2_all_sessions_performance.ipynb`) can pool them across sessions without re-deriving anything. The
per-collection df and the summary mirror the notebook's sections 1-2 exactly.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

_COMMON = None
for _base in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    for _rel in ('common', 'session_pipeline/common'):
        if (_base / _rel / 'perf_from_log.py').exists():
            _COMMON = _base / _rel
            break
    if _COMMON:
        break
if _COMMON is None:
    raise ImportError('cannot locate common/ from mixed_perf.py')
sys.path.insert(0, str(_COMMON))
import perf_from_log as pfl        # noqa: E402
import geom                        # noqa: E402
import heading                     # noqa: E402
import joyfine                     # noqa: E402

POS, NEG, NEU = pfl.POSITIVE_EFFECTS, pfl.NEGATIVE_EFFECTS, pfl.NEUTRAL_EFFECTS
DEFAULT_VIEW_SCALE = getattr(pfl, 'DEFAULT_VIEW_SCALE', 0.35)
# Okabe-Ito colour-blind-safe palette
COLR = {'single_reward': '#009E73', 'banish': '#0072B2', 'timeout': '#D55E00', 'unbanish': '#999999'}
ELABEL = {'single_reward': 'reward', 'banish': 'banishment', 'timeout': 'timeout', 'unbanish': 'escape'}
TYPES = [('single_reward', 'reward'), ('banish', 'banishment'), ('timeout', 'timeout')]


def valence(e):
    return 'positive' if e in POS else 'negative' if e in NEG else 'neutral' if e in NEU else 'other'


# ── protocol detection ─────────────────────────────────────────────────────────────────
def _opt_effects(entry):
    o = entry.get('options')
    return {x[0] for x in o} if o else set()


def is_mixed_protocol(log):
    """True if the board's punishment slot is randomly banish OR timeout -- i.e. some effect's
    `options` list names BOTH banish and timeout (the signature of this task)."""
    opt = set()
    for c in log.get('collected', []):
        opt |= _opt_effects(c)
    for s in log.get('spawns', []):
        for ic in (s.get('current') or []):
            opt |= _opt_effects(ic)
    return {'banish', 'timeout'} <= opt


def load_log(p):
    return json.load(open(p))


def arrays(log):
    """(t_xy, ax, ay, t_th, theta, theta_t[Nx2], coords[Mx3]) for one log."""
    t_xy, ax, ay = geom.load_coords(log)
    t_th, theta = geom.load_theta(log)
    return t_xy, ax, ay, t_th, theta, np.column_stack([t_th, theta]), np.column_stack([t_xy, ax, ay])


# ── the per-collection dataframe (mirrors notebook section 1) ───────────────────────────
def build_session_df(log, view_scale=None, session='', mouse=''):
    view_scale = view_scale or DEFAULT_VIEW_SCALE
    W = float(log['worlds'][0].get('width', 2400)); H = float(log['worlds'][0].get('height', 2400))
    t_xy, ax, ay, t_th, theta, _, _ = arrays(log)
    coll = log['collected']; rows = []; prev_t = 0.0
    for i, c in enumerate(coll):
        tc = float(c['time']); s_ms, e_ms = prev_t, tc
        tt, xx, yy = geom.slice_track(t_xy, ax, ay, s_ms, e_ms)
        msp, _ = geom.speed_stats(tt, xx, yy)
        _, align = geom.heading_to_target(t_th, theta, t_xy, ax, ay, (c['x'], c['y']), s_ms, e_ms)
        dist_prev = float(np.hypot(c['x'] - coll[i-1]['x'], c['y'] - coll[i-1]['y'])) if i > 0 else np.nan
        rows.append(dict(
            session=session, mouse=mouse, idx=i, effect=c['effect'], valence=valence(c['effect']),
            texture=c.get('texture'), x=float(c['x']), y=float(c['y']), loc=c.get('loc'), time_ms=tc,
            dt_prev_ms=tc - prev_t, dist_prev=dist_prev, multiplier=c.get('multiplier'),
            duration=c.get('duration'), random_punish=c.get('options') is not None,
            path_efficiency=geom.path_efficiency(xx, yy),
            time_in_corner=geom.time_in_corner(xx, yy, W, H, t=tt),
            mean_speed=msp, heading_align=align, start_ms=s_ms, end_ms=e_ms))
        prev_t = tc
    return pd.DataFrame(rows)


# ── generated (spawned) vs collected punishments + randomness of the draw ────────────────
def _runs_test(binseq):
    """Two-sided Wald-Wolfowitz runs test on a 0/1 sequence -> p. A truly random order gives the
    expected number of runs; too FEW runs = clustered (streaky), too MANY = over-alternating
    (a balanced pseudo-random shuffle tends this way). nan if a class is empty or n<2."""
    x = np.asarray(binseq, int)
    n1 = int((x == 1).sum()); n2 = int((x == 0).sum()); n = n1 + n2
    if n1 == 0 or n2 == 0 or n < 2:
        return np.nan, (1 if n else 0)
    runs = 1 + int((x[1:] != x[:-1]).sum())
    mu = 1 + 2 * n1 * n2 / n
    var = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))
    if var <= 0:
        return np.nan, runs
    from scipy.stats import norm
    z = (runs - mu) / np.sqrt(var)
    return float(2 * norm.sf(abs(z))), runs


def generated_punishments(log):
    """Punishment icons GENERATED (spawned) vs COLLECTED, and whether the banish/timeout draw looks
    random. Each spawn batch puts one punishment on the board that is randomly a banishment OR a
    timeout; this walks the spawn stream in time order, records each punishment icon the FIRST time
    its ID appears (so a lingering icon is counted once), and asks two things of the generated order:
      - p_balance: binomial test of timeout fraction vs 0.5  (is the draw biased to one type?)
      - p_runs:    runs test of the B/T order                (random draw vs a balanced/streaky
                   pseudo-random generator)
    Returns generated counts, the timeout fraction, the collected counts, the B/T string, and both p's.
    """
    from collections import Counter, defaultdict
    seen, order = {}, []
    for s in sorted(log.get('spawns', []), key=lambda s: s.get('time', 0)):
        for ic in (s.get('current') or []):
            e, i = ic.get('effect'), ic.get('ID')
            if e in ('banish', 'timeout') and i is not None and i not in seen:
                seen[i] = e; order.append(e)
    gb = order.count('banish'); gt = order.count('timeout'); n = gb + gt

    # per-TRIAL (spawn-batch) board presence, independent of collection: how many batches carried
    # each punishment. A punishment icon PERSISTS across the reward-respawn batches until it is hit,
    # so these are >= the distinct-draw counts above (one drawn icon spans several batches).
    batches = defaultdict(list)
    for s in log.get('spawns', []):
        batches[s.get('time')].append(s)
    tb = tt = nsh = 0
    for t in sorted(batches):
        cur = []
        for s in batches[t]:
            cur = s.get('current') or cur
        effs = {ic.get('effect') for ic in cur}
        tb += 'banish' in effs; tt += 'timeout' in effs
        nsh += (not ({'banish', 'timeout'} & effs)) and ('unbanish' in effs)
    n_batches = len(batches)

    cc = Counter(c.get('effect') for c in log.get('collected', []))
    frac_t = (gt / n) if n else np.nan
    p_balance = np.nan
    if n:
        from scipy.stats import binomtest
        p_balance = float(binomtest(gt, n, 0.5, alternative='two-sided').pvalue)
    p_runs, n_runs = _runs_test([1 if e == 'timeout' else 0 for e in order])
    return dict(
        gen_banish=gb, gen_timeout=gt, gen_n=n, gen_timeout_frac=frac_t, n_runs=n_runs,
        trial_banish=int(tb), trial_timeout=int(tt), n_batches=int(n_batches), n_shadow=int(nsh),
        col_banish=int(cc.get('banish', 0)), col_timeout=int(cc.get('timeout', 0)),
        gen_sequence=''.join('B' if e == 'banish' else 'T' for e in order),
        p_balance=p_balance, p_runs=p_runs)


def punishment_lifetimes(log):
    """For every punishment icon (unique ID), how many TRIALS (spawn batches) it stayed on the board
    before it was collected/removed -- i.e. how long each negative icon lingers before it is hit. Returns
    {'banish': [n_trials, ...], 'timeout': [...]}. A larger value = the animal takes more trials to
    reach that negative icon (avoids it longer); a value of 1 = hit on the trial it appeared."""
    from collections import defaultdict
    batches = defaultdict(list)
    for s in log.get('spawns', []):
        batches[s.get('time')].append(s)
    life, kind = defaultdict(set), {}
    for t in sorted(batches):
        cur = []
        for s in batches[t]:
            cur = s.get('current') or cur
        for ic in cur:
            e, i = ic.get('effect'), ic.get('ID')
            if e in ('banish', 'timeout') and i is not None:
                life[i].add(t); kind[i] = e
    out = {'banish': [], 'timeout': []}
    for i, ts in life.items():
        out[kind[i]].append(len(ts))
    return out


# ── per-session summary (mirrors notebook sections 2 + 4c/4d) ───────────────────────────
def _drops(seq, cap=4):
    tot, m = 0, 1
    for x in seq:
        if x:
            tot += m; m = min(m + 1, cap)
        else:
            m = 1
    return tot


def session_summary(log, df):
    ben, det, chance = pfl.world_opportunity(log)
    npos = int((df.valence == 'positive').sum())
    ntime = int((df.effect == 'timeout').sum()); nban = int((df.effect == 'banish').sum())
    nesc = int((df.effect == 'unbanish').sum()); nneg = ntime + nban

    def sp(good, bad):
        x = good + bad
        return pfl.prob_at_least(good, x, chance) if x else np.nan

    seq = df[df.valence.isin(['positive', 'negative'])].valence.eq('positive').values
    base = float(seq.mean()) if len(seq) else np.nan
    ws = ab = earned = captured = np.nan
    if len(seq) > 1:
        cur, nxt = seq[:-1], seq[1:]
        ws = float(nxt[cur].mean()) if cur.any() else np.nan
        ab = float(nxt[~cur].mean()) if (~cur).any() else np.nan
        earned = _drops(seq)
        flat = int(seq.sum()); best = _drops(np.ones(int(seq.sum()), bool))
        captured = (earned - flat) / (best - flat) if best > flat else np.nan

    def pemed(e):
        v = df[df.effect == e]['path_efficiency'].dropna()
        return float(v.median()) if len(v) else np.nan

    gen = generated_punishments(log)
    lif = punishment_lifetimes(log)
    _med = lambda v: float(np.median(v)) if v else np.nan
    _mn = lambda v: float(np.mean(v)) if v else np.nan

    return dict(
        mouse=mouse_of(df), session=session_of(df), n_coll=len(df),
        n_reward=npos, n_timeout=ntime, n_banish=nban, n_escape=nesc,
        chance=chance, reward_rate=(npos / (npos + nneg) if (npos + nneg) else np.nan),
        p_all=sp(npos, nneg), p_timeout=sp(npos, ntime), p_banish=sp(npos, nban),
        win_stay=ws, p_reward_after_bad=ab, drops_earned=earned, bonus_captured=captured,
        mult_mean=float(df.loc[df.valence == 'positive', 'multiplier'].dropna().mean()),
        mult_max=float(df.loc[df.valence == 'positive', 'multiplier'].dropna().max()) if npos else np.nan,
        pe_reward=pemed('single_reward'), pe_banish=pemed('banish'), pe_timeout=pemed('timeout'),
        life_banish_med=_med(lif['banish']), life_timeout_med=_med(lif['timeout']),
        life_banish_mean=_mn(lif['banish']), life_timeout_mean=_mn(lif['timeout']),
        **gen)


def mouse_of(df):
    return df['mouse'].iloc[0] if len(df) else ''


def session_of(df):
    return df['session'].iloc[0] if len(df) else ''


# ── board + geometry extractors for the pooled plots ────────────────────────────────────
def board_at_fn(log):
    spawns = sorted(log.get('spawns', []), key=lambda s: s['time'])

    def f(tc):
        cur = []
        for sp in spawns:
            if sp['time'] <= tc:
                cur = sp.get('current') or cur
            else:
                break
        return [ic for ic in cur if ic.get('effect') in (POS | NEG)]
    return f


def collection_radius(log, df):
    """Median centre-to-centre distance to the collected icon AT collection = the collection radius."""
    t_xy, ax, ay, *_ = arrays(log)
    d = [np.hypot(r.x - np.interp(r.end_ms, t_xy, ax), r.y - np.interp(r.end_ms, t_xy, ay))
         for _, r in df.iterrows()]
    return float(np.median(d)) if d else np.nan


def collection_offsets(log, df, effect, window_s=3.0):
    """Icon-centred avatar offsets (ox, oy) + the ms each sample stands for (dt), for every collection
    of `effect`. Pool across sessions for the occupancy maps (count) or weight by dt (ms)."""
    t_xy, ax, ay, *_ = arrays(log)
    ox, oy, dt = [], [], []
    for _, r in df[df.effect == effect].iterrows():
        m = (t_xy >= r.time_ms - window_s * 1000) & (t_xy <= r.time_ms + window_s * 1000)
        if m.sum() < 2:
            continue
        tt = t_xy[m]; d = np.diff(tt); d = np.append(d, np.median(d))
        ox.append(ax[m] - r.x); oy.append(ay[m] - r.y); dt.append(d)
    if not ox:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(ox), np.concatenate(oy), np.concatenate(dt)


def collected_curves_time(log, df, effect, window_s=8.0):
    """[(t_rel_s, err_deg), ...] toward the COLLECTED icon for each collection of `effect`."""
    *_, theta_t, coords = arrays(log)
    curves = []
    for i in range(len(df)):
        r = df.iloc[i]
        if r.effect != effect:
            continue
        e_ms = r.end_ms; s_ms = max(r.start_ms, e_ms - window_s * 1000)
        tt, err = heading.heading_error(theta_t, coords, (r.x, r.y), s_ms, e_ms)
        if tt.size:
            curves.append(((tt - e_ms) / 1000, np.degrees(np.abs(err))))
    return curves


def collected_joyfine_curves_time(log, df, effect, window_s=8.0):
    """[(t_rel_s, joy_fine_value), ...] over the approach to each collection of `effect` -- the joystick
    FINE-movement trace (joyfine.fine_trace = low-passed rolling-std of |joystick|), analogous to
    collected_curves_time for heading. Empty if the log has no joystick stream."""
    joy = np.asarray(log.get('joystick_t[ms]/x/y', []), float)
    if joy.ndim != 2 or joy.shape[0] < 2:
        return []
    jt, jx, jy = joy[:, 0], joy[:, 1], joy[:, 2]
    fine = joyfine.fine_trace(jx, jy)          # per-sample trace aligned to jt
    curves = []
    for i in range(len(df)):
        r = df.iloc[i]
        if r.effect != effect:
            continue
        e_ms = r.end_ms; s_ms = max(r.start_ms, e_ms - window_s * 1000)
        m = (jt >= s_ms) & (jt <= e_ms)
        if m.sum() >= 2:
            curves.append(((jt[m] - e_ms) / 1000, fine[m]))
    return curves


def collected_samples_dist(log, df, effect, window_s=8.0):
    """(distance, err_deg) samples toward the COLLECTED icon for each collection of `effect`."""
    *_, theta_t, coords = arrays(log)
    dd, ee = [], []
    for i in range(len(df)):
        r = df.iloc[i]
        if r.effect != effect:
            continue
        e_ms = r.end_ms; s_ms = max(r.start_ms, e_ms - window_s * 1000)
        m = (theta_t[:, 0] >= s_ms) & (theta_t[:, 0] <= e_ms)
        if not m.any():
            continue
        tt, th = theta_t[m, 0], theta_t[m, 1]
        axp = np.interp(tt, coords[:, 0], coords[:, 1]); ayp = np.interp(tt, coords[:, 0], coords[:, 2])
        bearing = np.arctan2(r.y - ayp, r.x - axp)
        dd.append(np.hypot(r.x - axp, r.y - ayp))
        ee.append(np.degrees(np.abs(np.angle(np.exp(1j * (th - bearing))))))
    return (np.concatenate(dd), np.concatenate(ee)) if dd else (np.array([]), np.array([]))


# ── discovery ───────────────────────────────────────────────────────────────────────────
def _progress(seq, desc='', enabled=True):
    """Wrap an iterable with a progress bar (tqdm in a notebook, else a carriage-return percentage)."""
    seq = list(seq)
    if not enabled or not seq:
        return iter(seq)
    try:
        from tqdm.auto import tqdm
        return tqdm(seq, desc=desc)
    except Exception:
        def gen():
            n = len(seq)
            for i, x in enumerate(seq, 1):
                print(f'\r{desc}: {i}/{n} ({100 * i // n}%)', end='', flush=True)
                yield x
            print()
        return gen()


def find_sessions(main_dir, pattern='*/*/log.json', progress=True):
    """Every log under `main_dir` whose board is the mixed random-punishment protocol.

    The real server layout is `MAIN_DIR/<mouse>/<session>/log.json`, so the default pattern is
    `*/*/log.json` (mouse folder / session folder / log). If that matches nothing (a different depth,
    or MAIN_DIR is already a mouse or a session folder) it falls back to a recursive search, and to
    MAIN_DIR/log.json for a single session. Returns [(path, log), ...] sorted by path.

    Logs are large (~90 MB each), so this shows a **progress bar** and does a cheap text pre-filter --
    a log with no 'timeout' or no 'banish' anywhere in it cannot be this protocol, so it is skipped
    WITHOUT the expensive JSON parse. Pass `progress=False` to silence the bar.
    """
    main_dir = Path(main_dir)
    paths = sorted(main_dir.glob(pattern))
    if not paths:
        paths = sorted(main_dir.glob('**/log.json'))          # any depth
    if not paths and (main_dir / 'log.json').exists():
        paths = [main_dir / 'log.json']                       # MAIN_DIR is itself one session
    if progress:
        print(f'scanning {len(paths)} log file(s) under {main_dir} ...')
    out = []
    for p in _progress(paths, 'checking logs', progress):
        try:
            raw = p.read_text()
        except Exception:
            continue
        if 'timeout' not in raw or 'banish' not in raw:       # cheap reject before parsing 90 MB of JSON
            continue
        try:
            log = json.loads(raw)
        except Exception:
            continue
        if is_mixed_protocol(log):
            out.append((p, log))
    if progress:
        print(f'  -> {len(out)} mixed-protocol session(s)')
    return out


def session_label(path, log, main_dir=None):
    """(mouse, session, mouse_folder) identity for a log, for the layout MAIN_DIR/<mouse>/<session>/.

    session  = the session FOLDER name (the leaf directory holding log.json).
    mouse    = experiment_data.ID with its digits normalised to JPAS_XXXX when it has digits, else the
               mouse FOLDER name.
    mouse_folder = the folder name one level up from the session (what the directory says the mouse is),
               returned separately so a mismatch with the log's ID is visible rather than hidden.
    """
    p = Path(path)
    session = p.parent.name or p.stem
    mouse_folder = p.parent.parent.name if p.parent.parent != p.parent else ''
    if main_dir is not None:
        try:
            parts = p.relative_to(main_dir).parts
            if len(parts) >= 2:
                mouse_folder = parts[0]
        except ValueError:
            pass
    ed = log.get('experiment_data', {})
    digits = ''.join(ch for ch in str(ed.get('ID', '') or '') if ch.isdigit())
    mouse = f'JPAS_{int(digits):04d}' if digits else (mouse_folder or 'unknown')
    return mouse, session, mouse_folder
