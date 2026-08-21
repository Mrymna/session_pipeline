"""Discover and classify every session of ONE animal, from the logs alone.

Point `discover()` at a folder holding one sub-folder per recording day and it returns a table of
what is there: animal, date, task variant, view_scale and whether the session can be scored. It
reads only `log.json` (plus `session.json` if present), so it is instant and needs no video.

WHY THIS EXISTS
---------------
`multi_day_performance.ipynb` used to take a hand-written list of `(log, view_scale, label)` rows.
That made it easy to put two DIFFERENT ANIMALS on DIFFERENT PROTOCOLS into one learning curve --
which is meaningless, because D is a within-animal measure and the protocol decides what "positive"
even means. Discovery removes the hand-editing and adds the guards.

THREE THINGS THIS REFUSES TO GUESS
----------------------------------
1. THE ANIMAL. `experiment_data.ID` is written inconsistently by the rig (`'mice 168'`,
   `'JPASS_0231'`), so it is normalised to the digits (168, 231). `discover()` reports every
   distinct animal it found and `require_single_animal()` raises if a set spans more than one.
2. THE TASK. Taken from the collection effects actually present in the log (`classify_task`),
   never from the folder name -- a folder can be named anything.
3. THE VIEW SCALE. It is a property of the WORLD, is not in the log, and differs per session, so a
   silent default would re-score the animal against another world's zoom and every visibility-gated
   number would be wrong with nothing looking broken. Resolution order is
   `session.json` -> explicit `view_scales=` override -> the known-values table -> **give up and
   mark the session unscorable**. It is never invented.

A session that cannot be scored is kept in the table with `use=False` and a `note` saying why --
it is never dropped silently, because a session vanishing from a learning curve is exactly the kind
of thing nobody notices.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

import perf_from_log as pfl


def normalise_mouse(raw):
    """'mice 168' / 'JPASS_0231' / 'JPAS_0168' -> a canonical 'JPAS_0168'.

    The rig writes the ID free-form, so the only stable part is the digits. Anything without
    digits is returned stripped, so it at least groups with itself.
    """
    if raw is None:
        return None
    digits = re.findall(r'\d+', str(raw))
    if not digits:
        return str(raw).strip() or None
    # FIRST group, not last: names are commonly suffixed with a date
    # ('JPASS_0168_2026-08-07'), and the last group would then be the day.
    return f'JPAS_{int(digits[0]):04d}'


# view_scale is a property of the WORLD, and the log DOES identify the world (size + icon width +
# texture file) even though it does not record the scale itself. Keying the known values on that
# signature means a scale is stated ONCE per world and then applies automatically to every session
# that uses it -- and a session on a NEW world is flagged instead of silently inheriting.
KNOWN_VIEW_SCALE_BY_WORLD = {
    '2400x2400/icon500/worldLowContrastLowSaturationRed.png': 0.35,   # JPAS_0168 (confirmed)
    '2000x2000/icon450/worldLowContrastLowSaturationRed.png': 0.56,   # JPAS_0231 (game code)
}


def world_signature(L):
    """A stable id for the world a log was recorded on: 'WxH/iconN/texture.png'.

    Everything in it comes from the log. Sessions sharing a signature share a view_scale.
    """
    try:
        w = (L.get('worlds') or [{}])[0]
        tex = str(w.get('world_texture_file', '')).split('/')[-1]
        return f"{int(w.get('width', 0))}x{int(w.get('height', 0))}/icon{int(w.get('icon_w', 0))}/{tex}"
    except Exception:
        return ''


def _read_view_scale(d, log_mouse, overrides, wsig=''):
    """(value, source) -- or (None, reason it could not be resolved)."""
    if overrides:
        for key in (d.name, log_mouse, str(d), wsig):
            if key and key in overrides:
                return float(overrides[key]), 'override'
    sj = d / 'session.json'
    if sj.exists():
        try:
            v = json.load(open(sj)).get('view_scale')
            if v is not None:
                return float(v), 'session.json'
        except Exception:
            pass
    if wsig in KNOWN_VIEW_SCALE_BY_WORLD:
        return float(KNOWN_VIEW_SCALE_BY_WORLD[wsig]), 'known world'
    # deliberately NO by-mouse fallback. The scale belongs to the WORLD, so the same animal on a
    # NEW world would silently inherit the old zoom -- the exact failure this is built to prevent
    # (caught in testing: a 3000x3000 world quietly took JPAS_0168's 0.35 and looked fine).
    return None, 'NOT FOUND'


def discover(root, view_scales=None, pattern='*', task=None):
    """Scan `root` for session folders and describe each one.

    root        folder holding one sub-folder per session (a folder containing log.json
                directly is also accepted, so a single session works)
    view_scales optional {folder-name or mouse-id or path: scale} override map
    pattern     glob for which sub-folders to consider
    task        keep only this task variant ('banish_multiplier' / 'timeout_double'). Sessions of
                another task are still LISTED, with use=False and a note -- so a stray session of
                the wrong protocol is visible rather than silently absent.

    Returns a DataFrame sorted by session datetime, one row per session.
    """
    root = Path(root).resolve()
    dirs = [root] if (root / 'log.json').exists() else \
           sorted(p for p in root.glob(pattern) if (p / 'log.json').exists())

    rows = []
    for d in dirs:
        rec = dict(dir=str(d), name=d.name, log=str(d / 'log.json'),
                   mouse=None, mouse_raw=None, datetime=pd.NaT, task='unreadable',
                   view_scale=None, vs_src='', n_collected=0, world='', effects='',
                   has_banish=False, has_multiplier=False, max_multiplier=0,
                   use=False, note='', warn='')
        try:
            L = json.load(open(d / 'log.json'))
        except Exception as ex:
            rec['note'] = f'log unreadable: {type(ex).__name__}'
            rows.append(rec); continue

        ed = L.get('experiment_data', {}) or {}
        rec['mouse_raw'] = ed.get('ID')
        rec['mouse'] = normalise_mouse(ed.get('ID')) or normalise_mouse(d.name)
        try:
            rec['datetime'] = pd.to_datetime(ed.get('datetime'))
        except Exception:
            pass

        collected = L.get('collected', []) or []
        rec['n_collected'] = len(collected)
        effects = sorted({c.get('effect') for c in collected if c.get('effect')})
        rec['effects'] = '+'.join(effects)
        rec['task'] = pfl.classify_task(set(effects))
        # evidence for the grouping, read from the log rather than assumed from the task name
        mults = [c.get('multiplier') for c in collected if c.get('multiplier') is not None]
        rec['has_banish'] = 'banish' in effects
        rec['has_multiplier'] = bool(mults) or ('multiplier' in L)
        rec['max_multiplier'] = int(max(mults)) if mults else 0
        rec['world'] = world_signature(L)

        vs, src = _read_view_scale(d, rec['mouse'], view_scales, rec['world'])
        rec['view_scale'], rec['vs_src'] = vs, src

        # a session.json flagged by the camera check is not analysable -- keep it visible
        cam_moved = False
        sj = d / 'session.json'
        if sj.exists():
            try:
                cam_moved = bool(json.load(open(sj)).get('camera_moved'))
            except Exception:
                pass

        notes = []
        if rec['task'] == 'unknown':
            notes.append('task not recognised from the collection effects')
        if vs is None:
            notes.append(f'no view_scale for world {rec["world"]!r} '
                         f'(add view_scales={{{rec["world"]!r}: <the scale for THIS world>}})')
        if cam_moved:
            notes.append('camera MOVED mid-session (session.json)')
        if rec['n_collected'] == 0:
            notes.append('no collections in the log')
        if task and rec['task'] != task and rec['task'] != 'unreadable':
            notes.append(f"task is {rec['task']}, not the requested {task}")
        # WARNINGS do not disqualify a session -- they are things to look at. Folder names follow
        # <animal>_<date>_<HH;MM;SS>; anything extra (e.g. a trailing '_i') usually means the
        # experimenter marked it, so it is surfaced but NOT excluded -- that call is Maryam's.
        warns = []
        if not re.fullmatch(r'.+_\d{4}-\d{2}-\d{2}_\d{2};\d{2};\d{2}', d.name):
            warns.append('unusual folder-name suffix -- check what it marks')
        rec['warn'] = '; '.join(warns)
        rec['note'] = '; '.join(notes)
        rec['use'] = not notes
        rows.append(rec)

    S = pd.DataFrame(rows)
    if not len(S):
        return S
    S = S.sort_values('datetime', na_position='last').reset_index(drop=True)
    S['day'] = S.datetime.dt.strftime('%Y-%m-%d')
    S['time'] = S.datetime.dt.strftime('%H:%M')
    # a day can hold MORE THAN ONE session (seen in the real directory: two on 2026-07-22), so the
    # date alone is not a unique label -- repeats get an a/b suffix, in time order.
    S['label'] = [r.day if isinstance(r.day, str) else r['name'] for _, r in S.iterrows()]
    for day, g in S.groupby('day'):
        if len(g) > 1:
            for k, i in enumerate(g.index):
                S.at[i, 'label'] = f'{day} {chr(97 + k)}'

    # SAME animal + SAME timestamp = the same recording reached by two paths (a copy, a symlinked
    # working dir). Scoring both would count that day twice in every pooled block, so all but the
    # first are marked unusable -- visibly, with the twin named, so it can be overridden.
    seen = {}
    for i, r in S.iterrows():
        if pd.isna(r.datetime):
            continue
        key = (r.mouse, r.datetime)
        if key in seen:
            note = f'duplicate of {seen[key]} (same animal + timestamp)'
            S.at[i, 'note'] = '; '.join(x for x in [S.at[i, 'note'], note] if x)
            S.at[i, 'use'] = False
        else:
            seen[key] = r['name']
    return S


def list_animals(main_dir, pattern='*'):
    """List the animal directories inside the MAIN data directory.

    Just the folders that contain at least one session (a sub-folder with a `log.json`), so an
    animal is distinguishable from an unrelated directory sitting alongside them.
    """
    main_dir = Path(main_dir).expanduser()
    if not main_dir.exists():
        raise FileNotFoundError(f'MAIN directory does not exist (is the server mounted?): {main_dir}')
    rows = []
    for p_ in sorted(main_dir.glob(pattern)):
        if not p_.is_dir():
            continue
        n = sum(1 for q in p_.glob('*') if q.is_dir() and (q / 'log.json').exists())
        n += int((p_ / 'log.json').exists())
        if n:
            rows.append(dict(animal=p_.name, path=str(p_), n_sessions=n))
    A = pd.DataFrame(rows, columns=['animal', 'path', 'n_sessions'])
    print(f'{main_dir}\n  {len(A)} animal folder(s) containing sessions')
    if not len(A):
        print('  !! none found -- check the path, or that the mount has finished syncing')
    return A


def list_sessions(root, pattern='*'):
    """STEP 1 -- what is actually in the animal's directory, before any log is parsed.

    Deliberately dumb: it lists every sub-folder and says whether it holds a `log.json`. Separating
    this from `discover()` means a mount that is empty, mis-typed or still syncing looks obviously
    wrong here instead of surfacing later as "0 sessions scored".
    """
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f'directory does not exist (is the server mounted?): {root}')
    if not root.is_dir():
        raise NotADirectoryError(root)

    print(f'{root}')
    # the directory may BE a single session rather than a folder of them; its sub-folders are then
    # the session's own output dirs (debug/, opticflow/, ...) and listing them as "no log.json"
    # is noise, not information.
    if (root / 'log.json').exists():
        print('  this directory IS a single session (it holds log.json directly)')
        return pd.DataFrame([dict(name=root.name, path=str(root), has_log=True,
                                  n_files=sum(1 for _ in root.glob('*')))])

    rows = [dict(name=p_.name, path=str(p_), has_log=(p_ / 'log.json').exists(),
                 n_files=sum(1 for _ in p_.glob('*')))
            for p_ in sorted(root.glob(pattern)) if p_.is_dir()]
    L = pd.DataFrame(rows, columns=['name', 'path', 'has_log', 'n_files'])
    print(f'  {len(L)} sub-folder(s); {int(L.has_log.sum()) if len(L) else 0} contain a log.json')
    if len(L) and not L.has_log.all():
        miss = list(L.loc[~L.has_log, 'name'])
        print(f'  no log.json in: {", ".join(miss[:12])}{" ..." if len(miss) > 12 else ""}')
    elif not len(L):
        print('  !! nothing here -- is the server mounted and the path right?')
    return L


def group_report(S, group_task='banish_multiplier', world_id=None):
    """STEP 3 -- split the discovered sessions into the group to analyse and everything else.

    The group is defined by EVIDENCE READ FROM THE LOG (a `banish` effect, a multiplier stream),
    not by the folder name, and every session outside it is printed with the effects that put it
    there -- so a new protocol variant shows up as its own line instead of being silently binned.

    Returns (group, rest).
    """
    if not len(S):
        print('no sessions'); return S, S
    sel = S.task == group_task
    if world_id is not None:
        wanted = {world_id} if isinstance(world_id, str) else set(world_id)
        sel &= S.world_id.isin(wanted)
    G, rest = S[sel], S[~sel]

    print(f'GROUP TO ANALYSE -- {group_task}:  {len(G)} session(s)')
    if len(G):
        for _, r in G.iterrows():
            flag = '' if r.use else f'   [EXCLUDED: {r.note}]'
            if r.use and r.get('warn'):
                flag = f'   [warn: {r.warn}]'
            print(f'   {r.label:<14} {r["name"]:<32} world {r.world.split("/")[0]:<10} '
                  f'mult<={r.max_multiplier}{flag}')
        print(f'   -> {int(G.use.sum())} of {len(G)} usable')
        worlds = sorted(G.world.unique())
        if len(worlds) > 1:
            print(f'   !! this group spans {len(worlds)} DIFFERENT worlds: {worlds}')
            print('      the world sets the viewport and therefore the chance baseline -- check '
                  'each has its own view_scale before pooling.')
    if len(rest):
        # summarised, not listed: at 28+ sessions a full dump buries the group we care about
        print(f'\nNOT IN THE GROUP:  {len(rest)} session(s)')
        for task, g in rest.groupby('task'):
            print(f'   {task:<20} {len(g):>3} session(s)   {g.day.min()} .. {g.day.max()}')
            print(f'   {"":<20}     effects: {" | ".join(sorted(g.effects.unique()))}')
    warned = S[(S.warn != '') & S.use] if 'warn' in S else S.iloc[:0]
    if len(warned):
        print(f'\nWARNINGS ({len(warned)} session(s) -- still included, but look at these):')
        for _, r in warned.iterrows():
            print(f'   {r["name"]:<34} {r.warn}')
    return G, rest


def world_report(S):
    """Print the worlds found and where each one's view_scale came from.

    This replaces printing a hard-coded table of scales: what matters is not which values are known
    in the abstract, but whether THESE sessions resolved one, and from where.
    """
    if not len(S):
        print('no sessions'); return
    print('worlds found in these sessions (view_scale is a property of the WORLD):')
    for wsig, g in S.groupby('world'):
        vs = g.view_scale.dropna().unique()
        srcs = ', '.join(sorted(g.vs_src.unique()))
        if len(vs) == 0:
            print(f'  {wsig}\n      {len(g)} session(s)  ->  NO view_scale. Add:'
                  f'  view_scales={{{wsig!r}: <value>}}')
        elif len(vs) > 1:
            print(f'  {wsig}\n      {len(g)} session(s)  ->  !! CONFLICTING scales {list(vs)} '
                  f'({srcs}) -- same world must have one scale')
        else:
            print(f'  {wsig}\n      {len(g)} session(s)  ->  view_scale {vs[0]}  (from {srcs})')


def require_single_animal(S):
    """Raise if the discovered set spans more than one animal.

    D is a WITHIN-animal learning measure; pooling animals produces a curve that means nothing.
    This is a hard stop rather than a warning because the resulting plot looks perfectly fine.
    """
    animals = sorted({m for m in S.mouse.dropna().unique()})
    if len(animals) > 1:
        by = S.groupby('mouse').size().to_dict()
        raise ValueError(
            f'these sessions span {len(animals)} animals: {by}. A learning curve is a WITHIN-animal '
            f'measure -- point discover() at ONE animal\'s folder, or filter with '
            f'S[S.mouse == "{animals[0]}"].')
    return animals[0] if animals else None


def score(S, verbose=True):
    """Score every `use=True` row with perf_from_log. Returns a DataFrame, one row per session.

    Sessions of different TASKS are scored side by side (D is defined the same way for both), but
    the `task` column is carried through so downstream plots can keep them visually separate --
    a protocol switch is a break in the curve, not a step along it.
    """
    out = []
    for _, r in S.iterrows():
        if not r.use:
            if verbose:
                print(f'  -- skipped {r["name"]}: {r.note}')
            continue
        try:
            rec = pfl.score_log(r.log, view_scale=r.view_scale, mouse_id=r.mouse, label=r.label)
            rec['dir'] = r.dir
            rec['datetime'] = r.datetime
            out.append(rec)
        except Exception as ex:
            if verbose:
                print(f'  !! failed {r["name"]}: {type(ex).__name__}: {ex}')
    R = pd.DataFrame(out)
    if len(R):
        R = R.sort_values('datetime', na_position='last').reset_index(drop=True)
    return R


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    S = discover(root)
    cols = ['name', 'mouse', 'day', 'task', 'view_scale', 'vs_src', 'n_collected', 'use', 'note']
    print(S[cols].to_string(index=False) if len(S) else 'no sessions found')


def blocks_of(R, size=3):
    """Pool consecutive sessions into blocks of `size` for the conflict-trial criterion.

    WHY POOL: one session yields only ~15 conflict trials, so its interval is ~+-0.25 -- too wide to
    read. Pooling ~3 sessions puts ~45 trials in a block, which is the smallest unit that can show a
    change. The blocks are consecutive in TIME so the sequence stays a learning curve.

    Returns a DataFrame with one row per block: pooled conflict and control rates + Wilson intervals.
    """
    rows = []
    for b0 in range(0, len(R), size):
        g = R.iloc[b0:b0 + size]
        kc, nc = int(g.k_conflict.sum()), int(g.n_conflict.sum())
        ka, na = int((g.agree_p * g.n_agree).round().sum()), int(g.n_agree.sum())
        clo, chi = pfl._wilson(kc, nc)
        alo, ahi = pfl._wilson(ka, na)
        rows.append(dict(
            block=f'{b0 // size + 1}', label=f'{g.label.iloc[0]}\n..{g.label.iloc[-1]}',
            n_sessions=len(g),
            k_conflict=kc, n_conflict=nc, conflict_p=kc / nc if nc else np.nan,
            conflict_lo=clo, conflict_hi=chi,
            k_control=ka, n_control=na, control_p=ka / na if na else np.nan,
            control_lo=alo, control_hi=ahi,
            D=g.D.mean(), acc=g.acc.mean()))
    return pd.DataFrame(rows)


def protocol_census(S):
    """STEP 2b -- WHICH WORLDS and WHICH PROTOCOLS are in this animal's directory?

    Answers "what is actually here?" BEFORE anything is chosen to analyse. Worlds are numbered
    W1, W2, ... in order of first appearance, so they can be referred to by number; the number is
    a label for THIS listing, not an id from the log (the log records the world's size, icon width
    and texture, not an index).

    Returns the census DataFrame; also adds a `world_id` column to S in place.
    """
    if not len(S):
        print('no sessions'); return S

    order = (S.dropna(subset=['datetime']).sort_values('datetime')
             .drop_duplicates('world').world.tolist())
    order += [w for w in S.world.unique() if w not in order]
    wid = {w: f'W{i + 1}' for i, w in enumerate(order)}
    S['world_id'] = S.world.map(wid)

    print('WORLDS in this directory (numbered by first appearance):')
    for w in order:
        g = S[S.world == w]
        vs = g.view_scale.dropna().unique()
        vs_s = f'{vs[0]}' if len(vs) == 1 else ('NONE -- must be supplied' if not len(vs)
                                                else f'CONFLICTING {list(vs)}')
        print(f'  {wid[w]}  {w}')
        print(f'       {len(g):>3} session(s)   {g.day.min()} .. {g.day.max()}   '
              f'view_scale {vs_s}')

    print('\nPROTOCOLS in this directory:')
    for task, g in S.groupby('task'):
        print(f'  {task:<20} {len(g):>3} session(s)   {g.day.min()} .. {g.day.max()}')
        print(f'  {"":<20}     effects: {" | ".join(sorted(g.effects.unique()))}')

    print('\nWORLD x PROTOCOL  (session counts):')
    ct = S.pivot_table(index='world_id', columns='task', values='name', aggfunc='count',
                       fill_value=0)
    print(ct.to_string())

    rows = []
    for (w, task), g in S.groupby(['world_id', 'task']):
        rows.append(dict(world_id=w, world=g.world.iloc[0], task=task, n=len(g),
                         first=g.day.min(), last=g.day.max(),
                         view_scale=(g.view_scale.dropna().iloc[0]
                                     if g.view_scale.notna().any() else None),
                         usable=int(g.use.sum())))
    return pd.DataFrame(rows).sort_values(['task', 'first']).reset_index(drop=True)
