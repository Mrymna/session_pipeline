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

TWO THINGS THIS REFUSES TO GUESS
--------------------------------
1. THE ANIMAL. `experiment_data.ID` is written inconsistently by the rig (`'mice 168'`,
   `'JPASS_0231'`), so it is normalised to the digits (168, 231). `discover()` reports every
   distinct animal it found and `require_single_animal()` raises if a set spans more than one.
2. THE TASK. Decided from the FIRST 10 TRIALS' boards (`pfl.first_trials_task`) -- the MAJORITY
   protocol of those trials, so a clean trial-0 timeout->banishment switch still counts as the task
   it settles into. A session with no clear majority is flagged `task_stable=False` and shown, not
   trusted. Never the folder name. The whole-log union is kept as `task_union` for comparison.

`task_protocol(S)` is the one function to summarise a directory (tasks first, worlds only as a
secondary `texture_id` label). It replaced `protocol_census`; that old name no longer exists.

THE VIEW SCALE is 0.35 for all animals and all worlds in the current setup -- only the world SIZE
changes across the curriculum, not the zoom -- so it DEFAULTS to `DEFAULT_VIEW_SCALE` (0.35).
Resolution order is `session.json` -> explicit `view_scales=` override -> `KNOWN_VIEW_SCALE_BY_WORLD`
(per-world exceptions, e.g. JPAS_0231's old 0.56) -> the 0.35 default. A session that fell back to
the default carries `vs_src='default'`, so if a future world ever uses a different zoom the fallback
is visible and auditable rather than silently wrong.

A session that still cannot be scored (e.g. an unreadable log) is kept in the table with `use=False`
and a `note` saying why -- it is never dropped silently, because a session vanishing from a learning
curve is exactly the kind of thing nobody notices.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

import perf_from_log as pfl


def normalise_mouse(raw):
    """'mice 168' / 'JPASS_0231' / 'JPAS_0168' -> a canonical 'JPAS_0168'.

    The rig writes the ID free-form, so the only stable part is the DIGITS.

    *** No digits => None (no identity evidence), NOT the stripped string. ***
    An earlier version returned the stripped text so that an id-less session "at least groups with
    itself". On the real server that backfired: 28 sessions carry `experiment_data.ID` == '_', so
    they all normalised to the single animal '_' and `require_single_animal` reported the folder as
    spanning two mice. That is worse than useless -- '_' is not an animal, it is a rig field nobody
    filled in, and treating it as one both invented an animal and masked the real one.

    Returning None instead lets the caller fall back to other evidence (the FOLDER NAME does carry
    the animal: `JPAS_168_2026-07-22_10;46;59`). See `discover`, which records WHICH source the id
    came from in `mouse_src` so the fallback is visible rather than silent.
    """
    if raw is None:
        return None
    digits = re.findall(r'\d+', str(raw))
    if not digits:
        return None
    # FIRST group, not last: names are commonly suffixed with a date
    # ('JPASS_0168_2026-08-07'), and the last group would then be the day.
    return f'JPAS_{int(digits[0]):04d}'


# The log records NO world number. `task_protocol` groups sessions by TEXTURE (the size/icon/texture
# signature, paired with the task) and labels them `Text1..Textn` -- deliberately NOT `W1..Wn`,
# because those texture groups are not the lab's region-based worlds and the labels must not be
# mistaken for them (2026-08-26). They are ordered by task stage, then size. Put the lab's
# own names here -- keyed on the `texture_key` ('<task>  ||  <signature>') or on the signature alone
# -- and they replace the invented `Text` labels everywhere.
WORLD_NAMES = {
    # '2400x2400/icon500/worldLowContrastLowSaturationRed.png': 'timeout_world',
}

# The training stages, in order. The world NUMBER follows this (then size), because the same texture
# is reused across tasks -- so numbering by the image put one physical world under several protocols
# and split one task across several numbers. The task is the reliable, board-classified axis.
TASK_STAGE_RANK = {'reward_only': 0, 'timeout_multiplier': 1, 'banish_multiplier': 2}
_OTHER_STAGE = 90     # timeout_double (the OLD 0231 experiment) and any unknown task sort last

# The view scale is 0.35 for all animals / all worlds in the current setup (2026-08-26); the
# world SIZE changes across the curriculum, the zoom does not. So this is the DEFAULT, used when no
# more specific source applies -- change it here in one place. It is the LAST resort: session.json, a
# view_scales= override, and KNOWN_VIEW_SCALE_BY_WORLD all take precedence, so a genuine exception can
# still be pinned per world.
DEFAULT_VIEW_SCALE = 0.35

# per-world exceptions to the default. JPAS_0231's 2000x2000 world was 0.56 in the old rig (game
# code); it stays correct here. Add a row only for a world whose zoom is NOT the 0.35 default.
KNOWN_VIEW_SCALE_BY_WORLD = {
    '2000x2000/icon450/worldLowContrastLowSaturationRed.png': 0.56,   # JPAS_0231 (old rig, game code)
}


def _world_sig_of(w):
    tex = str(w.get('world_texture_file', '')).split('/')[-1]
    return f"{int(w.get('width', 0))}x{int(w.get('height', 0))}/icon{int(w.get('icon_w', 0))}/{tex}"


def _world_size(sig):
    """(width, height) parsed from a 'WxH/iconN/texture' signature; (0, 0) if unparseable."""
    try:
        w, h = sig.split('/', 1)[0].lower().split('x')
        return int(w), int(h)
    except Exception:
        return 0, 0


def world_config(L, i=0):
    """What the world CONTAINS, from the log: its icon set and how many of each are active.

    *** THE LOG HAS NO WORLD NUMBER. ***
    Nothing in `log['worlds'][i]` is an id or a name -- there is only the texture file, the size,
    the icon size, the spawn locations and the effect definitions. So `W1..Wn` in the census are
    labels THIS CODE invents, numbered by first appearance, and they need not match the numbering
    used in the lab.

    They also used to be too COARSE. The signature is size + icon width + texture, which is what
    fixes the viewport (and therefore the view_scale) -- but two genuinely different worlds can
    share all three and differ only in which icons they define. A timeout world and a banishment
    world of the same size on the same background collapsed into one entry, which is why a single
    "world" appeared to host two protocols.

    This returns the icon configuration so the two can be told apart. `texture_id` is numbered on
    the PAIR (signature, config), so such worlds now get separate numbers.
    """
    ws = L.get('worlds') or [{}]
    if not isinstance(ws, list):
        ws = [ws]
    if i >= len(ws):
        return ''
    w = ws[i] or {}
    eff = sorted({e.get('effect') for e in (w.get('effects') or []) if e.get('effect')})
    alt = sorted({e.get('effect') for e in (w.get('alt_effects') or []) if e.get('effect')})
    parts = ['+'.join(eff) or '-']
    if alt:
        parts.append('alt:' + '+'.join(alt))
    if w.get('active_benefits') is not None:
        parts.append(f"{w.get('active_benefits')}good/{w.get('active_detriments')}bad")
    parts.append(f"{len(w.get('spawn_locations') or [])}loc")
    return ' '.join(parts)


def world_signatures(L):
    """Every DISTINCT world signature in a log.

    `log['worlds']` is a LIST and can hold more than one entry -- JPAS_0168 has two. In every
    session seen so far the entries are identical (the banishment task's swap between the normal
    world and the shadow realm is NOT recorded here; both entries name the same red texture), so
    taking the first is correct today. But it is only correct BECAUSE they agree, and a session
    where they did not would be silently mis-labelled -- and the signature decides the view_scale,
    hence every visibility-gated number. So the entries are compared rather than assumed.

    NOTE this reads only the world's SIZE, ICON WIDTH and TEXTURE FILENAME from the log. The
    texture image itself is never opened anywhere in this pipeline; the filename is part of the
    world's identity, not a pointer to pixels that get loaded.
    """
    ws = L.get('worlds') or [{}]
    if not isinstance(ws, list):
        ws = [ws]
    out = []
    for w in ws:
        try:
            sig = _world_sig_of(w)
        except Exception:
            sig = ''
        if sig not in out:
            out.append(sig)
    return out


def world_signature(L):
    """A stable id for the world a log was recorded on: 'WxH/iconN/texture.png'.

    The FIRST distinct signature. Use `world_signatures()` to see them all; `discover` warns when
    a log carries more than one.
    """
    sigs = world_signatures(L)
    return sigs[0] if sigs else ''


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
    # DEFAULT (2026-08-26): the view scale is 0.35 for all animals and all worlds in the
    # current setup; only the world SIZE changes, not the zoom. So an unlisted world resolves to the
    # default rather than being marked unscorable. It is returned with vs_src='default' so a session
    # that fell back to it is still VISIBLE -- if a future world ever uses a different zoom, add it to
    # KNOWN_VIEW_SCALE_BY_WORLD (which still wins) or pass view_scales=, and change DEFAULT_VIEW_SCALE
    # in one place. The old 0.56 (JPAS_0231) stays a per-world exception above.
    return float(DEFAULT_VIEW_SCALE), 'default'


def _progress(seq, desc, enabled=True, labels=None):
    """Wrap an iterable in a progress bar.

    Reading a directory of logs is the slow step of discovery -- each `log.json` carries the full
    coords/joystick streams, so a 40-session animal takes long enough that a silent cell looks
    hung. The bar names the session currently being read, which also makes it obvious WHICH log is
    slow (or which one the traceback came from) if one of them is malformed.

    tqdm is imported lazily and falls back to a plain carriage-return counter, so this module still
    imports on a machine without it, and a non-notebook caller still gets feedback.

    `labels` names each item. It is passed explicitly rather than read off the item, because the
    two callers hold different things: discovery iterates Paths (where `.name` is the folder) but
    scoring iterates pandas Series (where `.name` is the ROW INDEX, so the bar would count 0,1,2).
    """
    seq = list(seq)
    labels = list(labels) if labels is not None else [getattr(x, 'name', '') for x in seq]
    if not enabled or len(seq) < 2:
        return seq
    try:
        from tqdm.auto import tqdm
        return tqdm(seq, desc=desc, unit='session')
    except Exception:
        def gen():
            n = len(seq)
            for i, (x, lb) in enumerate(zip(seq, labels), 1):
                print(f'\r{desc}: {i}/{n} ({100 * i // n}%)  {str(lb or "")[:40]:40s}',
                      end='', flush=True)
                yield x
            print()
        return gen()


def discover(root, view_scales=None, pattern='*', task=None, progress=True):
    """Scan `root` for session folders and describe each one.

    root        folder holding one sub-folder per session (a folder containing log.json
                directly is also accepted, so a single session works)
    view_scales optional {folder-name or mouse-id or path: scale} override map
    pattern     glob for which sub-folders to consider
    task        keep only this task variant ('banish_multiplier' / 'timeout_double'). Sessions of
                another task are still LISTED, with use=False and a note -- so a stray session of
                the wrong protocol is visible rather than silently absent.
    progress    show a progress bar while the logs are read (they are large; a silent cell over a
                40-session animal looks hung). Set False for scripted/batch use.

    Returns a DataFrame sorted by session datetime, one row per session.
    """
    root = Path(root).resolve()
    dirs = [root] if (root / 'log.json').exists() else \
           sorted(p for p in root.glob(pattern) if (p / 'log.json').exists())

    rows = []
    bar = _progress(dirs, f'reading {len(dirs)} log(s)', enabled=progress,
                    labels=[re.sub(r'_\\d{2};\\d{2};\\d{2}$', '', d.name) for d in dirs])
    for d in bar:
        if hasattr(bar, 'set_postfix_str'):
            bar.set_postfix_str(re.sub(r'_\\d{2};\\d{2};\\d{2}$', '', d.name)[:32])
        rec = dict(dir=str(d), name=d.name, log=str(d / 'log.json'),
                   mouse=None, mouse_raw=None, mouse_src='', datetime=pd.NaT, task='unreadable',
                   view_scale=None, vs_src='', n_collected=0, world='', world_cfg='',
                   n_worlds_in_log=0, effects='',
                   effects_offered='', never_collected='',
                   task_stable=False, task_seq='', task_union='',
                   camera_stable=None, camera_move_frame=np.nan, camera_move_ms=np.nan,
                   has_banish=False, has_multiplier=False, max_multiplier=0,
                   use=False, note='', warn='')
        try:
            L = json.load(open(d / 'log.json'))
        except Exception as ex:
            rec['note'] = f'log unreadable: {type(ex).__name__}'
            rows.append(rec); continue

        ed = L.get('experiment_data', {}) or {}
        rec['mouse_raw'] = ed.get('ID')
        # the animal comes from the log's ID when that field carries digits, otherwise from the
        # FOLDER NAME (which does: JPAS_168_2026-07-22_10;46;59). Which source was used is recorded,
        # because a folder name is weaker evidence than the log -- a mis-filed folder would rename
        # the animal silently, and `mouse_src == 'folder'` is what makes that checkable.
        m_log, m_dir = normalise_mouse(ed.get('ID')), normalise_mouse(d.name)
        rec['mouse'] = m_log or m_dir
        rec['mouse_src'] = 'log' if m_log else ('folder' if m_dir else '')
        try:
            rec['datetime'] = pd.to_datetime(ed.get('datetime'))
        except Exception:
            pass

        collected = L.get('collected', []) or []
        rec['n_collected'] = len(collected)
        # what he COLLECTED (behaviour) vs what the protocol OFFERED (the board). The task is
        # classified from the board -- see pfl.log_effects for why using behaviour drops exactly
        # the sessions where the animal avoided the punishment.
        effects = sorted({c.get('effect') for c in collected if c.get('effect')})
        offered = sorted(pfl.log_effects(L))
        rec['effects'] = '+'.join(effects)
        rec['effects_offered'] = '+'.join(offered)
        rec['never_collected'] = '+'.join(sorted(set(offered) - set(effects)))
        # TASK from the FIRST 10 TRIALS' boards: stable across them -> that task; not
        # stable -> keep the settled guess but flag task_stable=False so the session is SHOWN, not
        # trusted. The whole-log union (task_union) is kept alongside as a cross-check -- when it
        # disagrees with the first-trials task, the union saw a later board the first trials did not.
        ft = pfl.first_trials_task(L, n=10)
        rec['task'] = ft['task']
        rec['task_stable'] = ft['stable']
        rec['task_seq'] = '>'.join(ft['seq'])
        rec['task_union'] = pfl.classify_task(set(offered))
        # evidence for the grouping, read from the log rather than assumed from the task name
        mults = [c.get('multiplier') for c in collected if c.get('multiplier') is not None]
        rec['has_banish'] = 'banish' in effects
        rec['has_multiplier'] = bool(mults) or ('multiplier' in L)
        rec['max_multiplier'] = int(max(mults)) if mults else 0
        _sigs = world_signatures(L)
        rec['world'] = _sigs[0] if _sigs else ''
        rec['world_cfg'] = world_config(L)
        rec['n_worlds_in_log'] = len(_sigs)

        vs, src = _read_view_scale(d, rec['mouse'], view_scales, rec['world'])
        rec['view_scale'], rec['vs_src'] = vs, src

        # CAMERA CHECK. This is a VIDEO-pipeline result (detect_view_shift samples video frames), so
        # it is NOT in log.json -- it only exists once tracking has run and written session.json /
        # opticflow/camera_check.json. From the log alone it is UNKNOWN, so camera_stable is a
        # TRI-STATE: True = checked and stable, False = camera MOVED, None = not checked yet.
        #   camera_move_frame / camera_move_ms = where the shift began (NaN if stable or unchecked).
        # NB this is unrelated to `switch_ms` below, which is the PROTOCOL (board) switch, not the
        # camera -- two different 'switches'.
        cam_moved = False
        sj = d / 'session.json'
        cc = d / 'opticflow' / 'camera_check.json'
        if sj.exists():
            try:
                _sj = json.load(open(sj))
                if 'camera_moved' in _sj:
                    cam_moved = bool(_sj.get('camera_moved'))
                    rec['camera_stable'] = not cam_moved
                    if _sj.get('camera_move_onset_frame') is not None:
                        rec['camera_move_frame'] = float(_sj['camera_move_onset_frame'])
            except Exception:
                pass
        if cc.exists():
            try:
                _cc = json.load(open(cc))               # the fuller record (frame AND seconds)
                cam_moved = bool(_cc.get('moved'))
                rec['camera_stable'] = not cam_moved
                if _cc.get('onset_frame') is not None:
                    rec['camera_move_frame'] = float(_cc['onset_frame'])
                if _cc.get('onset_s') is not None:
                    rec['camera_move_ms'] = float(_cc['onset_s']) * 1000.0
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
        # experimenter marked it, so it is surfaced but NOT excluded -- that is a manual call.
        warns = []
        if not re.fullmatch(r'.+_\d{4}-\d{2}-\d{2}_\d{2};\d{2};\d{2}', d.name):
            warns.append('unusual folder-name suffix -- check what it marks')
        # both sources named an animal and they disagree -- one of them is wrong, and which one
        # matters more than the mismatch: it decides whose learning curve this session joins.
        if m_log and m_dir and m_log != m_dir:
            warns.append(f'animal MISMATCH: log says {m_log}, folder says {m_dir} (using {m_log})')
        if not m_log:
            warns.append(f'log ID {ed.get("ID")!r} carries no animal number -- '
                         f'took {rec["mouse"]} from the folder name')
        # the signature decides the view_scale, so an ambiguous one must not pass quietly
        if len(_sigs) > 1:
            warns.append(f'log lists {len(_sigs)} DIFFERENT worlds {_sigs}; using the first -- '
                         f'check which one the viewport should follow')
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
    # SHORT DISPLAY NAME: the folder name minus the trailing _HH;MM;SS, e.g.
    # 'JPAS_168_2026-06-19_10;35;09' -> 'JPAS_168_2026-06-19'.
    # `name` stays the true folder name -- it is how a session is found on disk and how duplicates
    # are reported -- but it is too long to read in a 40-row table, and the clock time is the part
    # that carries no information for a per-DAY analysis. Derived from the folder rather than
    # rebuilt from mouse+day so it keeps whatever the folder actually calls the animal
    # ('JPAS_168', not the normalised 'JPAS_0168') and still matches what is on disk.
    # The clock time is stripped from ANYWHERE in the name, not just the end: a folder can carry a
    # marker after it ('JPAS_168_2026-07-01_11;35;09_i'), and anchoring to the end left those names
    # with the full timestamp still in them. Stripping mid-name keeps the marker, which is the part
    # that means something: -> 'JPAS_168_2026-07-01_i'.
    _stripped = [re.sub(r'_\d{2};\d{2};\d{2}', '', str(n)) for n in S['name']]
    # An experimenter marker can trail the timestamp ('..._11;35;09_i'). It is kept in its OWN
    # column rather than in the name: it still needs to be visible (it is why `warn` fires), but
    # inside the name it makes one day's label differ in shape from every other day's, which is
    # exactly the inconsistency these short names exist to remove.
    S['marker'] = [m.group(1) if (m := re.search(r'_(\D{1,3})$', v)) else '' for v in _stripped]
    S['session_base'] = [re.sub(r'_\D{1,3}$', '', v) for v in _stripped]
    S['session'] = S['session_base']
    for day, g in S.groupby('day'):
        if len(g) > 1:
            for k, i in enumerate(g.index):
                S.at[i, 'label'] = f'{day} {chr(97 + k)}'
                # while BOTH sessions of a day are present they need distinct names, so an a/b
                # suffix is appended. `one_per_day` removes it again from the survivor -- once a day
                # holds one session the suffix is noise, and worse, it makes the name inconsistent
                # with every other day.
                if (S['session_base'] == S.at[i, 'session_base']).sum() > 1:
                    S.at[i, 'session'] = f'{S.at[i, "session_base"]}{chr(97 + k)}'

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


def group_report(S, group_task='banish_multiplier', texture_id=None):
    """STEP 3 -- split the discovered sessions into the group to analyse and everything else.

    The group is defined by EVIDENCE READ FROM THE LOG (a `banish` effect, a multiplier stream),
    not by the folder name, and every session outside it is printed with the effects that put it
    there -- so a new protocol variant shows up as its own line instead of being silently binned.

    Returns (group, rest).
    """
    if not len(S):
        print('no sessions'); return S, S
    sel = S.task == group_task
    if texture_id is not None:
        wanted = {texture_id} if isinstance(texture_id, str) else set(texture_id)
        sel &= S.texture_id.isin(wanted)
    G, rest = S[sel], S[~sel]

    print(f'GROUP TO ANALYSE -- {group_task}:  {len(G)} session(s)')
    if len(G):
        for _, r in G.iterrows():
            flag = '' if r.use else f'   [EXCLUDED: {r.note}]'
            if r.use and r.get('warn'):
                flag = f'   [warn: {r.warn}]'
            print(f'   {r.label:<14} {r["session"]:<26} world {r.world.split("/")[0]:<10} '
                  f'mult<={r.max_multiplier}{flag}')
        print(f'   -> {int(G.use.sum())} of {len(G)} usable')
        # A protocol NAME can cover more than one board. JPAS_0168 ran banishment sessions both
        # with and without a timeout icon (a shaping step), and both are correctly named
        # banish_multiplier -- but they are not the same experiment: the timeout variant gives the
        # animal a third icon type and a second way to be wrong, which moves the chance baseline.
        # Pooling them into one learning curve may be right, but it must be a decision, so the
        # distinct effect sets in the group are always listed.
        if 'effects_offered' in G.columns and G.effects_offered.nunique() > 1:
            print(f'\n   ! this group spans {G.effects_offered.nunique()} different ICON SETS -- '
                  f'same protocol name, different boards:')
            for eff, gg in G.groupby('effects_offered'):
                print(f'       {len(gg):>3} session(s)  {gg.day.min()} .. {gg.day.max()}   {eff}')
            print('     Pooling them assumes the extra icon type does not change the task. To keep')
            print('     only one board:  G = G[G.effects_offered == "<the set you want>"]')
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


def one_per_day(S, keep='last', verbose=True, mark_only=False):
    """Collapse days that hold more than one session down to a single session per day.

    The real server directory does record two sessions on one date (seen on 2026-07-22), and for a
    LEARNING CURVE that is a problem: the x-axis is meant to be "training day", so a day with two
    recordings contributes two points and is weighted twice against every other day. The rule
    for this animal is to KEEP THE LAST session of such a day (the later one is the one that ran to
    completion; an early short recording is usually a restart).

    keep='last' (default) or 'first'.

    The de-selected sessions are NOT deleted -- they stay in the table with `use=False` and a note
    naming the session that superseded them, the same convention the duplicate and no-view_scale
    checks use. Nothing about the set should be invisible; the row is still there to be overridden
    by setting `use=True` by hand.

    Only sessions that are still usable compete: if the last recording of a day is unusable (no
    view_scale, camera moved, wrong protocol), the rule falls through to the latest one that IS
    usable rather than throwing the whole day away.
    """
    if not len(S) or 'day' not in S:
        return S
    S = S.copy()
    # `mark_only` flags the losers instead of clearing `use`. The dataset build wants EVERY session
    # kept as a row with the duplicate visible as a column, so that filtering is an explicit choice
    # made at analysis time rather than a deletion baked into the build.
    if 'is_dup_day' not in S.columns:
        S['is_dup_day'] = False
        S['keep_of_day'] = True
    dropped = []
    for day, g in S[S.use & S.day.notna()].groupby('day'):
        if len(g) < 2:
            continue
        g = g.sort_values('datetime', na_position='first')
        winner = g.index[-1] if keep == 'last' else g.index[0]
        for i in g.index:
            if i == winner:
                continue
            note = (f"{'later' if keep == 'last' else 'earlier'} session "
                    f"{S.at[winner, 'session']} kept for {day} (one session per day)")
            S.at[i, 'note'] = '; '.join(x for x in [S.at[i, 'note'], note] if x)
            S.at[i, 'is_dup_day'] = True
            S.at[i, 'keep_of_day'] = False
            if not mark_only:
                S.at[i, 'use'] = False
            dropped.append((day, S.at[i, 'session'], S.at[i, 'time'], S.at[winner, 'session']))
        # the day now holds ONE usable session, so the a/b disambiguator is no longer needed --
        # drop it so this day's name matches the form every other day uses.
        if 'session_base' in S.columns:
            S.at[winner, 'session'] = S.at[winner, 'session_base']
        S.at[winner, 'is_dup_day'] = True
        S.at[winner, 'keep_of_day'] = True
    if verbose:
        if dropped:
            print(f'ONE SESSION PER DAY (keep={keep}): de-selected {len(dropped)} session(s); '
                  f'{int(S.use.sum())} remain usable')
            for day, name, t, win in dropped:
                print(f'    {day}  dropped {name} ({t})  ->  kept {win}')
        else:
            print(f'ONE SESSION PER DAY (keep={keep}): no day held more than one usable session')
    return S


def world_protocol_audit(S, verbose=True):
    """Cross-check the CLASSIFIED task against the PHYSICAL world, and explain every mismatch.

    *** Grouped by the physical world (the `world` signature), NOT by `texture_id`. ***
    `texture_id` is now the TASK-stage label, so grouping on it would put one task per group and the
    check would be trivially self-consistent. The useful question is the other way round: does one
    physical TEXTURE host more than one task? On this rig it does -- the same red texture is reused
    across the reward / timeout / banishment stages -- so a texture carrying several tasks is
    EXPECTED (it is the training progression), not a bug. What is worth a human look is a single
    session whose classified task is the minority on its texture: the printout lists it with the
    evidence that decided it (the effects the log OFFERED from the spawn board, what was COLLECTED,
    the spawn-batch count), so a genuine mis-read is told apart from a real earlier-stage session.
    """
    if 'texture_id' not in S.columns:
        task_protocol(S, verbose=False)
    rows = []
    for sig, g in S.groupby('world'):
        tasks = g.task.value_counts()
        maj = tasks.idxmax()
        odd = g[g.task != maj]
        if verbose:
            wids = ', '.join(sorted(g.texture_id.dropna().unique()))
            print(f'\nphysical world {sig or "(unknown)"}   -> {wids}')
            print(f'   {len(g)} session(s)   tasks on this texture: {tasks.to_dict()}')
            if len(tasks) > 1:
                print(f'   (one texture, {len(tasks)} tasks -- expected if it spans training stages)')
            if len(odd):
                print(f'   {len(odd)} session(s) are the MINORITY task on this texture:')
                for _, r in odd.iterrows():
                    off = r.get('effects_offered', '') or '(none)'
                    print(f"      {r['session']:<26} task={r.task:<18} offered={off}")
                    print(f"      {'':<26} collected={r.effects or '(none)'}  "
                          f"n_collected={r.n_collected}")
        for _, r in odd.iterrows():
            rows.append(dict(world=sig, texture_id=r.get('texture_id'), session=r['session'],
                             task=r.task, majority=maj,
                             effects_offered=r.get('effects_offered', ''),
                             effects=r.effects, n_collected=r.n_collected))
    A = pd.DataFrame(rows)
    if verbose and not len(A):
        print('\nevery texture hosts a single task.')
    return A


def require_single_animal(S):
    """Raise if the discovered set spans more than one animal.

    D is a WITHIN-animal learning measure; pooling animals produces a curve that means nothing.
    This is a hard stop rather than a warning because the resulting plot looks perfectly fine.
    """
    animals = sorted({m for m in S.mouse.dropna().unique()})
    if len(animals) > 1:
        by = S.groupby('mouse').size().to_dict()
        hint = ''
        if 'mouse_src' in S and (S.mouse_src == 'folder').any():
            n = int((S.mouse_src == 'folder').sum())
            hint = (f'\n  NOTE: {n} of these took the animal from the FOLDER NAME because the log\'s '
                    f'experiment_data.ID had no number in it. Check S[["name","mouse","mouse_raw",'
                    f'"mouse_src"]] -- if a folder is mis-named, the split is spurious.')
        raise ValueError(
            f'these sessions span {len(animals)} animals: {by}. A learning curve is a WITHIN-animal '
            f'measure -- point discover() at ONE animal\'s folder, or filter with '
            f'S[S.mouse == "{animals[0]}"].{hint}')
    return animals[0] if animals else None


def score(S, verbose=True, progress=True):
    """Score every `use=True` row with perf_from_log. Returns a DataFrame, one row per session.

    Sessions of different TASKS are scored side by side (D is defined the same way for both), but
    the `task` column is carried through so downstream plots can keep them visually separate --
    a protocol switch is a break in the curve, not a step along it.

    `progress` shows a bar: this re-reads every log AND does the visibility geometry, so it is the
    slowest cell in the notebook -- slower than discovery, which only parses.
    """
    out = []
    todo = [r for _, r in S.iterrows()]
    bar = _progress(todo, f'scoring {int(S.use.sum())} of {len(todo)} session(s)',
                    enabled=progress, labels=[str(r.get('session', r['name'])) for r in todo])
    for r in bar:
        if hasattr(bar, 'set_postfix_str'):
            bar.set_postfix_str(str(r.get('session', r['name']))[:32])
        if not r.use:
            if verbose:
                print(f'  -- skipped {r.get("session", r["name"])}: {r.note}')
            continue
        try:
            rec = pfl.score_log(r.log, view_scale=r.view_scale, mouse_id=r.mouse, label=r.label)
            rec['dir'] = r.dir
            rec['datetime'] = r.datetime
            out.append(rec)
        except Exception as ex:
            if verbose:
                print(f'  !! failed {r.get("session", r["name"])}: {type(ex).__name__}: {ex}')
    R = pd.DataFrame(out)
    if len(R):
        R = R.sort_values('datetime', na_position='last').reset_index(drop=True)
    return R


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    S = discover(root)
    cols = ['session', 'mouse', 'day', 'task', 'view_scale', 'vs_src', 'n_collected', 'use', 'note']
    print(S[cols].to_string(index=False) if len(S) else 'no sessions found')


def blocks_of(R, size=None):
    """Pool consecutive sessions into blocks for the conflict-trial criterion.

    *** size=None pools EVERY session into ONE block. ***
    That is the default because it is the question to ask first: what does the whole world say?
    Binning is a way to TEST a pattern once you have seen one, and choosing a bin width before
    looking is how a pattern gets manufactured -- so the un-binned answer comes first, and a
    specific `size` is a deliberate second step.

    WHY POOL: one session yields only ~15 conflict trials, so its interval is ~+-0.25 -- too wide to
    read. Pooling ~3 sessions puts ~45 trials in a block, which is the smallest unit that can show a
    change. The blocks are consecutive in TIME so the sequence stays a learning curve.

    Returns a DataFrame with one row per block: pooled conflict and control rates + Wilson intervals.
    """
    # POOLING ACROSS PROTOCOLS IS A MISTAKE, and a silent one: the block would average a combo
    # task against a multiplier task, whose reward units and chance baselines are not the same
    # quantity. It is refused rather than warned about, because the output looks perfectly normal.
    if 'task' in R.columns and R.task.nunique() > 1:
        raise ValueError(
            f'these sessions span {R.task.nunique()} protocols ({sorted(R.task.unique())}) -- '
            f'{pfl.NEVER_POOL}. Select one first, e.g. R[R.task == "{sorted(R.task.unique())[0]}"].')
    rows = []
    size = len(R) if size is None else size
    size = max(1, int(size))
    for b0 in range(0, len(R), size):
        g = R.iloc[b0:b0 + size]
        d = _block_stats(g)
        d['block'] = 'all' if size >= len(R) else f'{b0 // size + 1}'
        d['label'] = (f'ALL {len(g)} sessions\n{g.label.iloc[0]} .. {g.label.iloc[-1]}'
                      if size >= len(R) else f'{g.label.iloc[0]}\n..{g.label.iloc[-1]}')
        rows.append(d)
    return pd.DataFrame(rows)


def _block_stats(g):
    """Pooled conflict/control criterion + the pooled STOCHASTIC (world-design) baseline for one
    group of sessions. Used by both blocks_of (fixed session count) and week_blocks (calendar week).

    The stochastic p is POOLED, not averaged: p-values do not average, so pos/neg are summed across
    the block and one binomial test is run against the world's good:bad ratio. This is what lets the
    opportunity p be read next to D / conflict_p / control_p block by block.
    """
    kc, nc = int(g.k_conflict.sum()), int(g.n_conflict.sum())
    ka, na = int((g.agree_p * g.n_agree).round().sum()), int(g.n_agree.sum())
    clo, chi = pfl._wilson(kc, nc)
    alo, ahi = pfl._wilson(ka, na)
    d = dict(n_sessions=len(g),
             k_conflict=kc, n_conflict=nc, conflict_p=kc / nc if nc else np.nan,
             conflict_lo=clo, conflict_hi=chi,
             k_control=ka, n_control=na, control_p=ka / na if na else np.nan,
             control_lo=alo, control_hi=ahi,
             D=g.D.mean(), acc=g.acc.mean())
    if {'pos', 'neg', 'good_opportunity_ratio'} <= set(g.columns):
        pos, neg = int(g.pos.sum()), int(g.neg.sum())
        ratio = (float(g.good_opportunity_ratio.dropna().mean())
                 if g.good_opportunity_ratio.notna().any() else np.nan)
        accp = pos / (pos + neg) if (pos + neg) else np.nan
        d.update(pos=pos, neg=neg, good_opportunity_ratio=ratio, acc_pooled=accp)
        if np.isfinite(ratio) and (pos + neg) > 0 and ratio < 1:
            from scipy.stats import binomtest
            d['p_opportunity'] = binomtest(pos, pos + neg, ratio, alternative='greater').pvalue
            d['D_opportunity'] = (accp - ratio) / (1 - ratio)
        else:
            d['p_opportunity'] = d['D_opportunity'] = np.nan
    return d


def week_blocks(R, full_week=5, fallback_size=5):
    """Pool sessions by ISO CALENDAR WEEK (Mon-Sun), falling back to consecutive sessions.

    Each block is one week of training, labelled by the dates of the sessions in it (e.g.
    `2026-08-24 .. 2026-08-28`), so a block is "the week of the 24th-28th" rather than "sessions
    6-10". The week comes from the `day` column (the session date).

    *** WHEN A WEEK IS INCOMPLETE, fall back to `fallback_size` consecutive sessions. ***
    A week with at least `full_week` sessions stands on its own. Weeks with fewer (a couple of
    scattered days, a holiday-shortened week) carry too little data to read as a block, so
    consecutive incomplete-week sessions are pooled in chronological order into blocks of about
    `fallback_size` (default 5, a working week). So full weeks keep their calendar identity and
    sparse stretches are grouped by count instead of leaving tiny one- or two-session blocks.

    Same pooled criterion + stochastic baseline as blocks_of (via `_block_stats`).
    """
    if 'task' in R.columns and R.task.nunique() > 1:
        raise ValueError(
            f'these sessions span {R.task.nunique()} protocols ({sorted(R.task.unique())}) -- '
            f'{pfl.NEVER_POOL}. Select one first.')
    R = R.sort_values(['day', 'time'] if 'time' in R.columns else 'day').copy()
    iso = pd.to_datetime(R['day'], errors='coerce').dt.isocalendar()
    R['_wkkey'] = [f'{int(y)}-W{int(w):02d}' if np.isfinite(y) else 'NA'
                   for y, w in zip(iso['year'], iso['week'])]
    n_per_week = R['_wkkey'].value_counts()

    rows, buf = [], []                               # buf = list of single-session rows (DataFrames)

    def _md(x):                                      # 'YYYY-MM-DD' -> 'MM-DD' (full date overlaps on the axis)
        x = str(x)
        return x[5:] if len(x) >= 10 and x[4] == '-' else x

    def _emit(g, block, partial):
        d = _block_stats(g)
        d['block'] = block
        lo, hi = g.day.min(), g.day.max()
        rng = f'{_md(lo)} .. {_md(hi)}' if lo != hi else _md(lo)
        d['label'] = rng + (f'\n({len(g)} sess, partial)' if partial else '')
        d['is_week'] = not partial
        rows.append(d)

    def _flush_full():                               # emit full fallback_size chunks from the buffer
        while len(buf) >= fallback_size:
            g = pd.concat(buf[:fallback_size]); del buf[:fallback_size]
            _emit(g, f'{g.day.min()}+{len(g)}s', partial=True)

    def _flush_all():                                # emit whatever is left, even if short
        if buf:
            g = pd.concat(buf); buf.clear()
            _emit(g, f'{g.day.min()}+{len(g)}s', partial=True)

    for wk in dict.fromkeys(R['_wkkey']):            # weeks in chronological order
        g = R[R['_wkkey'] == wk]
        if wk != 'NA' and n_per_week[wk] >= full_week:
            _flush_all()                             # close any partial run before a full week
            _emit(g, wk, partial=False)
        else:
            buf.extend(g.iloc[[i]] for i in range(len(g)))   # add this week's sessions
            _flush_full()
    _flush_all()
    return pd.DataFrame(rows).reset_index(drop=True)


def task_protocol(S, verbose=True):
    """STEP 2b -- WHICH TASKS (and, secondarily, worlds) are in this animal's directory?

    Answers "what is actually here?" BEFORE anything is chosen to analyse. Worlds are numbered
    W1, W2, ... to follow the TRAINING CURRICULUM -- by TASK first (`TASK_STAGE_RANK`), then by
    world SIZE within a task -- so reward_only is W1(+W2 if two sizes), the timeout world next, and
    banishment last. This replaces the old first-appearance-by-texture numbering, which put one
    reused texture under several protocols (2026-08-26). The number is a LABEL for this
    listing, not an id from the log; the physical world (the `world` signature) still fixes
    view_scale, and analysis groups by TASK, never by this number.

    Returns the census DataFrame; also adds a `texture_id` column to S in place.
    """
    if not len(S):
        if verbose:
            print('no sessions')
        return S

    # texture_key = (TASK, physical signature). The task is the board-classified stage; keying on it
    # means the number follows the task, not the image. Two tasks on one texture -> two keys (they
    # were wrongly one before); two sizes of one task -> two keys, ordered small->large.
    if 'world_cfg' not in S.columns:
        S['world_cfg'] = ''
    if 'task' not in S.columns:
        S['task'] = 'unknown'
    S['texture_key'] = S.task.fillna('unknown') + '  ||  ' + S.world.fillna('')

    def _key_sort(k):
        task, _, sig = k.partition('  ||  ')
        w, h = _world_size(sig)
        return (TASK_STAGE_RANK.get(task, _OTHER_STAGE), w, h, sig)

    order = sorted(S.texture_key.unique(), key=_key_sort)
    wid = {w: f'Text{i + 1}' for i, w in enumerate(order)}
    # WORLD_NAMES (keyed on the task||sig key OR the signature alone) overrides the invented labels.
    S['texture_id'] = S.texture_key.map(
        lambda k: WORLD_NAMES.get(k, WORLD_NAMES.get(k.split('  ||  ')[-1], wid.get(k))))

    if verbose:
        print('TEXTURE GROUPS in this directory (Text1.., by task stage then size).')
        print('  NOTE these are TEXTURE groups, not the lab\'s region-worlds -- the labels are')
        print('  invented here. Put your own names in session_index.WORLD_NAMES to replace them.')
        for w in order:
            g = S[S.texture_key == w]
            task, _, sig = w.partition('  ||  ')
            cfg = g.world_cfg.dropna().iloc[0] if g.world_cfg.notna().any() else ''
            vs = g.view_scale.dropna().unique()
            vs_s = f'{vs[0]}' if len(vs) == 1 else ('NONE -- must be supplied' if not len(vs)
                                                    else f'CONFLICTING {list(vs)}')
            print(f'  {(g.texture_id.iloc[0] if len(g) else wid[w])}  {task:<20} {sig}')
            print(f'       icons: {cfg}')
            print(f'       {len(g):>3} session(s)   {g.day.min()} .. {g.day.max()}   '
                  f'view_scale {vs_s}')

        print('\nPROTOCOLS in this directory:')
        for task, g in S.groupby('task'):
            print(f'  {task:<20} {len(g):>3} session(s)   {g.day.min()} .. {g.day.max()}')
            print(f'  {"":<20}     effects: {" | ".join(sorted(g.effects.unique()))}')

        # sessions with no clear task majority in the first 10 trials -- surfaced for a look
        if 'task_stable' in S.columns:
            uns = S[~S.task_stable.fillna(False)]
            print(f'\nFIRST-10-TRIALS: {len(S) - len(uns)}/{len(S)} sessions have a clear task '
                  f'majority.')
            if len(uns):
                print(f'  {len(uns)} session(s) with no clear majority -- inspect these:')
                for _, r in uns.iterrows():
                    print(f"    {r['name']:<30} task={r.task:<18} "
                          f"trials: {r.get('task_seq', '')}")

    rows = []
    for (w, task), g in S.groupby(['texture_id', 'task']):
        rows.append(dict(texture_id=w, world=g.world.iloc[0], task=task, n=len(g),
                         first=g.day.min(), last=g.day.max(),
                         view_scale=(g.view_scale.dropna().iloc[0]
                                     if g.view_scale.notna().any() else None),
                         usable=int(g.use.sum())))
    return pd.DataFrame(rows).sort_values(['task', 'first']).reset_index(drop=True)

