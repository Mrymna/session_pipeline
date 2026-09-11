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

    return dict(
        mouse=mouse_of(df), session=session_of(df), n_coll=len(df),
        n_reward=npos, n_timeout=ntime, n_banish=nban, n_escape=nesc,
        chance=chance, reward_rate=(npos / (npos + nneg) if (npos + nneg) else np.nan),
        p_all=sp(npos, nneg), p_timeout=sp(npos, ntime), p_banish=sp(npos, nban),
        win_stay=ws, p_reward_after_bad=ab, drops_earned=earned, bonus_captured=captured,
        mult_mean=float(df.loc[df.valence == 'positive', 'multiplier'].dropna().mean()),
        mult_max=float(df.loc[df.valence == 'positive', 'multiplier'].dropna().max()) if npos else np.nan,
        pe_reward=pemed('single_reward'), pe_banish=pemed('banish'), pe_timeout=pemed('timeout'))


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
