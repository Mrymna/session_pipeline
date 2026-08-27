"""
LOG-ONLY performance + discrimination score, for tracking ONE ANIMAL ACROSS DAYS.

Why this exists: the full per-session pipeline decodes video (optic flow ~30 min, pupil ~4 min), but
**the performance measures need none of it**. Everything below comes from `log.json` alone --
`collected` (what he took), `spawns[].current` (what was on the board), `coords_t[ms]/x/y` (where he
was) and `worlds` (world size). So a whole week of sessions can be scored in seconds, which is what
makes a day-by-day learning curve practical.

WHAT IS COMPUTED (identical definitions to the per-session scorecards, so the numbers are directly
comparable to `analyze_performance.py`):
  - counts of each outcome, reward DROPS (banish task: sum of the streak multiplier; timeout task:
    1 x single + 2 x double -- both verified against the rigs' own delivery counters)
  - P(collect positive | collected)
  - the VISIBILITY-WEIGHTED chance baseline: how often each icon TYPE was inside the viewport,
    since the animal can only choose among what is on his screen
  - DISCRIMINATION SCORE  D = (observed - chance) / (1 - chance), with a binomial p and a
    Wilson 95% CI mapped through D -- reported for TWO memory assumptions: ZERO memory (an icon
    counts only while on screen) and PERFECT memory (it counts once he has seen it this trial).
    Real memory lies between, so D is a RANGE; the two are the ends of an assumption, not a right
    and a wrong baseline.
  - an EXOGENOUS cross-check baseline (nearest icon at spawn), which his behaviour cannot move
  - *** THE LEARNING CRITERION *** `conflict_p` = P(positive | BOTH types on screen AND the negative
    one was NEARER). These are the trials where proximity and value disagree, so he must override the
    pull of the closer icon. This is the number to track across days: it must RISE if he is learning
    to avoid the hazard, while `agree_p` (the control) stays high. ~15 such trials per session, so
    pool ~3 sessions per block -- one session alone gives a +-0.25 interval.
  - throughput (collections/min, drops/min) and, for the timeout task, ACTIVE minutes with the
    ~14 s post-timeout coord blackout removed

AGREEMENT WITH THE PER-SESSION PIPELINE: `chance` lands within ~0.015 of `analyze_performance.py`
(0168 0.729 vs 0.713; 0231 0.717 vs 0.709), which never changes a conclusion here. The residual is
methodological, not a bug: the per-session version restricts to `analyze==True` (and, for the banish
task, NORMAL-world) trials and samples the avatar on the uniform video-frame clock, while this one
scores every collectable trial straight from the coord stream. For a CROSS-DAY curve what matters is
that the same definition is applied to every day -- do not mix the two sources within one figure.

⚠️ `view_scale` IS NOT IN THE LOG and differs per session (it is a property of the WORLD). It must be
supplied per session -- `VIEW_SCALE_BY_MOUSE` below is only a convenience for the sessions already
established; anything else must be passed explicitly or the visibility baseline is silently wrong.
"""
from pathlib import Path
import json
import numpy as np

SKETCH_W, SKETCH_H = 800, 600

# *** VALENCE IS A PROPERTY OF THE EFFECT, NOT OF THE TASK. ***
# The per-task POSITIVE/NEGATIVE sets this replaces could not describe a MIXED protocol, and mixed
# protocols exist: JPAS_0168 ran sessions offering banish+unbanish AND timeout together (a shaping
# step between the two tasks). Under the old per-task sets those sessions scored with
# NEGATIVE={'banish'}, so every timeout the animal hit was silently dropped from the hit rate --
# the D score was computed on an incomplete definition of "wrong". Keying valence on the EFFECT
# makes any combination score correctly, including combinations nobody has run yet.
POSITIVE_EFFECTS = {'single_reward', 'double_reward'}
NEGATIVE_EFFECTS = {'timeout', 'banish'}
NEUTRAL_EFFECTS = {'unbanish'}          # escape: a collection, but pays nothing and is not a hazard
# fixed drops per collection, where the effect pays a fixed amount. `single_reward` is absent on
# purpose: in the banishment task it pays the streak MULTIPLIER, which is read per collection.
FIXED_DROPS = {'single_reward': 1, 'double_reward': 2, 'timeout': 0, 'banish': 0, 'unbanish': 0}

# BENEFIT / DETRIMENT effect classes for the WORLD-DESIGN "opportunity" baseline.
# `money` (benefit) and `debt` (detriment) do NOT appear in the current sessions -- they are kept
# here only so a FUTURE task generation that uses them is counted the same way, with no code change.
# On today's data they never match, so including them changes nothing.
BENEFIT_EFFECTS = {'single_reward', 'double_reward', 'money'}
DETRIMENT_EFFECTS = {'timeout', 'banish', 'debt'}


def world_opportunity(log):
    """(active_benefits, active_detriments, good_opportunity_ratio) from the WORLD's effect defs.

    active_benefits / active_detriments = how many GOOD / BAD icons the world puts on the board
    (Maryam's exact effect lists: benefits {single_reward, double_reward, money}, detriments
    {timeout, banish, debt}). Uses the counts the log stores where present (newer sessions log
    `active_benefits` / `active_detriments`), else counts `world['effects']` by effect name (older
    logs, e.g. JPAS_0231).

    good_opportunity_ratio = active_benefits / (active_benefits + active_detriments) -- the
    STOCHASTIC baseline: the hit rate expected if the animal collected icons in proportion to how
    many good vs bad the world offers, with no discrimination at all. It is the world-DESIGN chance
    level, a further baseline alongside the visibility-weighted `chance` and the exogenous `chance_exo`.

    NB the log's own `good_ratio` field is the ODDS (benefits / detriments, e.g. 2.0), NOT this
    probability -- do not use it directly as a chance level.
    """
    w = log.get('worlds') or [{}]
    w = w[0] if isinstance(w, list) else w
    ab, ad = w.get('active_benefits'), w.get('active_detriments')
    if ab is None or ad is None:
        eff = w.get('effects') or []
        ab = sum(1 for e in eff if isinstance(e, dict) and e.get('effect') in BENEFIT_EFFECTS)
        ad = sum(1 for e in eff if isinstance(e, dict) and e.get('effect') in DETRIMENT_EFFECTS)
    ab, ad = int(ab), int(ad)
    ratio = ab / (ab + ad) if (ab + ad) else np.nan
    return ab, ad, ratio


def prob_at_least(y, x, p):
    """P(>= y successes in x Bernoulli(p) trials) -- the one-sided binomial upper tail. This is the
    `prob_at_least_y_in_x` from Maryam's snippet: is he collecting MORE positives than the world's
    opportunity ratio would give by chance? Returns nan when there are no trials or no ratio."""
    if x is None or x <= 0 or p is None or not np.isfinite(p):
        return np.nan
    from scipy.stats import binomtest
    return binomtest(int(y), int(x), float(p), alternative='greater').pvalue

# Protocol names, in PRECEDENCE order. Each entry is (name, required, forbidden).
# Precedence matters because protocols overlap during shaping; `banish`/`unbanish` are the
# unambiguous marker of the banishment task and win even when a timeout is also present.
# Each rule: `all` must be present, `any` needs at least one, `none` must be absent.
# `any` exists for the banishment task specifically: the two escape rings live INSIDE the
# banishment world, so a snapshot of the normal-world board shows `banish` without `unbanish`.
# Requiring both would leave every such board unclassified -- which is what happened when this was
# first applied to within-session board segments rather than to whole logs.
TASK_RULES = [
    dict(name='banish_multiplier', all=set(), any={'banish', 'unbanish'}, none=set()),
    dict(name='timeout_double',    all={'timeout', 'double_reward'}, any=set(), none=set()),
    dict(name='timeout_multiplier', all={'timeout', 'single_reward'}, any=set(),
         none={'double_reward', 'banish', 'unbanish'}),
    dict(name='reward_only',       all={'single_reward'}, any=set(),
         none={'timeout', 'banish', 'double_reward', 'unbanish'}),
]

# What separates the three protocols, in one place (Maryam, 2026-08-25):
#   banish_multiplier   reward(+multiplier) + banish/unbanish -> SHADOW REALM, the animal CAN move
#   timeout_multiplier  reward(+multiplier) + timeout         -> FROZEN, the animal CANNOT move
#   timeout_double      reward + double_reward + timeout, NO multiplier  (the old JPAS_0231 task)
# The first two differ ONLY in what the punishment does; both carry the streak multiplier. The name
# `reward_timeout` used earlier hid that, which is why it was wrong.
TASK_DESCRIPTION = {
    'banish_multiplier':
        'CURRENT. reward (+multiplier) + banish/unbanish; punished into the shadow realm, CAN move',
    'timeout_multiplier':
        'CURRENT. reward (+multiplier) + timeout; punished by a freeze, CANNOT move',
    'timeout_double':
        'OLD (JPAS_0231 only). COMBO (double reward) + timeout, NO multiplier. '
        'A DIFFERENT EXPERIMENT from timeout_multiplier -- never pool the two',
    'reward_only': 'reward only, no punishment',
}

# Protocols that must not be pooled into one analysis. `timeout_double` and `timeout_multiplier`
# both contain a timeout and are one word apart, which makes them exactly the pair most likely to
# be combined by accident -- but 0231's task pays a COMBO (a double-reward icon) and carries no
# multiplier, so its trials, its reward units and its chance baseline all mean something different.
# The two CURRENT tasks differ only in the punishment (shadow realm vs freeze) and are likewise
# separate experiments.
NEVER_POOL = "each protocol is a different experiment: different icons, different reward units, " \
             "different chance baseline"


# retained for callers that still read them; derived from the effect-level sets above
POSITIVE = {'timeout_double': POSITIVE_EFFECTS, 'banish_multiplier': {'single_reward'},
            'timeout_multiplier': {'single_reward'}, 'reward_only': {'single_reward'}}
NEGATIVE = {'timeout_double': {'timeout'}, 'banish_multiplier': {'banish'},
            'timeout_multiplier': {'timeout'}, 'reward_only': set()}
DROPS = {'timeout_double': FIXED_DROPS, 'timeout_multiplier': FIXED_DROPS,
         'reward_only': FIXED_DROPS}

# known per-session view scales (see common/viewport.py). NOT a default for new animals.
VIEW_SCALE_BY_MOUSE = {'JPAS_0231': 0.56, 'JPAS_0168': 0.35}

FREEZE_GAP_MIN_S = 2.0     # a coord gap this long starting at a timeout = the penalty blackout


def log_effects(log):
    """Every effect this session's protocol OFFERED, from the board and not from behaviour.

    *** Read the SPAWNS, not just `collected`. ***
    Classifying from `collected` alone makes the protocol label depend on how the animal
    PERFORMED: a banish_multiplier session in which he never actually got banished has collected
    effects {single_reward}, matches no signature, and is labelled 'unknown' -- so it is dropped
    from the group. That is backwards (the protocol is a property of the session, not of the
    outcome) and it biases the sample in the worst possible direction, removing exactly the
    sessions where the animal avoided the hazard most successfully.

    The spawn stream records every icon ever put on the board, so it carries the full vocabulary
    whether or not he touched each type.
    """
    eff = {c.get('effect') for c in (log.get('collected') or [])}
    eff |= {s.get('effect') for s in (log.get('spawns') or [])}
    eff |= {i.get('effect') for s in (log.get('spawns') or [])
            for i in (s.get('current') or [])}
    return {e for e in eff if e}


def board_segments(log, min_batches=3):
    """Split a session where the BOARD CHANGES PART-WAY THROUGH.

    A session is not always one protocol. On JPAS_0168 the shaping sessions begin under the old
    task and switch to the new one mid-recording -- the first trials offer a timeout, and only
    later do the banishment icons appear. Scoring such a session as one thing mixes two different
    experiments, with different icon counts and therefore different chance baselines, into a
    single number.

    The board is read from each spawn batch's `current` list (what was actually on screen), so
    the switch is detected from the data rather than assumed from the date. A segment is a run of
    consecutive batches offering the SAME set of effects.

    `min_batches` drops runs too short to be a real protocol phase -- a single batch that happens
    to be missing an icon (because the animal had just collected it and the replacement had not
    spawned yet) is a gap in the board, not a change of task. Without this the board looks like it
    changes constantly.

    Returns a list of dicts: t0, t1, effects (frozenset), n_batches, task.
    """
    sp = sorted(log.get('spawns') or [], key=lambda x: x.get('time', 0))
    batches = {}
    for b in sp:
        t = b.get('time')
        if t is None:
            continue
        cur = b.get('current') or []
        eff = {i.get('effect') for i in cur if i.get('effect')}
        if not eff and b.get('effect'):
            eff = {b['effect']}
        # keep the FULLEST list logged at a timestamp, as elsewhere in the pipeline
        if t not in batches or len(eff) > len(batches[t]):
            batches[t] = eff
    if not batches:
        return []
    items = sorted(batches.items())

    runs = []
    for t, eff in items:
        if runs and runs[-1][2] == eff:
            runs[-1][1] = t
            runs[-1][3] += 1
        else:
            runs.append([t, t, eff, 1])
    # Absorb runs too short to be a phase into the PRECEDING one -- a batch briefly missing an
    # icon (the animal has just collected it, the replacement has not spawned) is a gap in the
    # board, not a change of task.
    #
    # A short run at the very START is deliberately NOT absorbed: there is nothing before it to
    # absorb into, and a session that opens with one different batch is exactly the shaping case
    # (trial 0 under the old task, then the switch). Keeping it is what lets that first trial be
    # excluded, which is the whole point of the switch detection.
    merged = []
    for r in runs:
        if merged and r[3] < min_batches:
            merged[-1][1] = r[1]
            merged[-1][3] += r[3]
        else:
            merged.append(r)
    # re-merge neighbours that became identical after absorbing
    out = []
    for r in merged:
        if out and out[-1][2] == r[2]:
            out[-1][1] = r[1]; out[-1][3] += r[3]
        else:
            out.append(r)
    return [dict(t0=r[0], t1=r[1], effects=frozenset(r[2]), n_batches=r[3],
                 task=classify_task(r[2])) for r in out]


def _first_batches(log, n=None):
    """The first n spawn batches as sorted (t_ms, effect_set) -- one per unique spawn timestamp,
    keeping the fullest `current` board logged at that timestamp. n=None returns all."""
    batches = {}
    for b in sorted(log.get('spawns') or [], key=lambda x: x.get('time', 0)):
        t = b.get('time')
        if t is None:
            continue
        eff = {i.get('effect') for i in (b.get('current') or []) if i.get('effect')}
        if not eff and b.get('effect'):
            eff = {b['effect']}
        if t not in batches or len(eff) > len(batches[t]):
            batches[t] = eff
    items = sorted(batches.items())
    return items[:n] if n is not None else items


def first_trials_task(log, n=10):
    """Classify the session's task from its FIRST n trials by MAJORITY, and flag ties.

    Maryam (2026-08-26): to decide banishment vs timeout, read the task from each of the first ~10
    trials' boards, not from the whole-log union of effects (which folds a trial-0 lead-in into the
    label). A trial = one spawn batch; its task is classified from that batch's `current` icon set,
    i.e. the icons actually on the board -- the same thing visible in the video at that time.

    THE RULE: the task is the protocol that is the MAJORITY of the first n trials. A session can open
    under the old task and switch once (timeout -> banishment), and that switch can take a couple of
    trials to show on the board -- so a session with e.g. 2 timeout then 8 banishment is banishment
    (8 of 10). `stable` is True when that majority is STRICT (more than half); it is False only when
    no protocol has a clear majority (an even split / oscillation), which is what warrants a look.
    `seq`, `times` and `counts` come back so a flagged session can be lined up against the video (see
    `show_first_trials`).

    Returns dict(task, stable, seq, times, counts, n_switch).
    """
    items = _first_batches(log, n)
    seq = [classify_task(eff) for _, eff in items]
    times = [t for t, _ in items]
    if not seq:
        return dict(task='unknown', stable=False, seq=[], times=[], counts={}, n_switch=0)
    counts = {t: seq.count(t) for t in set(seq)}
    task = max(counts, key=counts.get)                 # the majority protocol of the first n trials
    stable = counts[task] > len(seq) / 2               # a STRICT majority (more than half)
    n_switch = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    return dict(task=task, stable=bool(stable), seq=seq, times=times, counts=counts,
                n_switch=n_switch)


def show_first_trials(log_path, n=12):
    """Print the first n spawn batches -- time, on-board effects, classified task -- so a session's
    task can be lined up against the video. `log_path` is a path or an already-loaded log dict."""
    log = log_path if isinstance(log_path, dict) else json.load(open(log_path))
    items = _first_batches(log, n)
    print(f'first {len(items)} spawn batches  (board on screen -> task):')
    for i, (t, eff) in enumerate(items):
        print(f'  trial {i:2d}   t={t/1000:8.2f}s   {str(sorted(eff)):<48} -> {classify_task(eff)}')
    ft = first_trials_task(log, n=max(n, 10))
    print(f'\n  => task = {ft["task"]}   clean = {ft["stable"]}   board changes = {ft["n_switch"]}')


def protocol_switch(log, min_batches=3, max_lead_batches=1):
    """(switch_time_ms, segments) for a session that opens under a different board.

    *** THE SWITCH IS ONLY HONOURED AT THE VERY START OF THE SESSION. ***
    Maryam: on this animal the leftover board can only appear in TRIAL 0, never later. So a
    change detected mid-session is not a protocol switch -- it is a gap in the board or a logging
    artefact -- and truncating there would silently throw away most of a good session while
    looking perfectly healthy in the output.

    Accordingly a switch is returned only when the pre-switch part is at most `max_lead_batches`
    trials long (default 1 = trial 0) AND everything after it is a single protocol. Anything else
    returns None, so the session is scored WHOLE and the oddity is left visible in `segments` for
    a human to look at, rather than acted on automatically.
    """
    segs = board_segments(log, min_batches=min_batches)
    tasks = [g['task'] for g in segs]
    if len(segs) < 2 or len(set(tasks)) < 2:
        return None, segs
    lead, rest = segs[0], segs[1:]
    if lead['n_batches'] > max_lead_batches:
        return None, segs          # a long opening phase is not the trial-0 case; score it whole
    if len({g['task'] for g in rest}) > 1:
        return None, segs          # the board changes more than once; not the trial-0 case
    return rest[0]['t0'], segs


def log_has_multiplier(log):
    """Does this session carry the reward STREAK MULTIPLIER?

    The distinguishing fact between the two timeout protocols: the current task pays a multiplier,
    the old JPAS_0231 one paid a fixed 1 or 2 and never logged the field.
    """
    return any(c.get('multiplier') is not None for c in (log.get('collected') or []))


def classify_task(effects, has_multiplier=None):
    """Name the protocol from the effects the session OFFERED.

    Matching requires ALL of a rule's effects and NONE of its forbidden ones, in precedence
    order. The previous version tested `signature & effects` -- ANY overlap -- which meant a
    lone `timeout` matched `timeout_double` even with no `double_reward` in the session, and
    because the first match won, a banishment session that also contained a timeout was labelled
    timeout_double. Both misreadings were seen in JPAS_0168's real directory.
    """
    eff = set(effects)
    # `double_reward` already separates timeout_double from timeout_multiplier, but the multiplier
    # is the more direct evidence and is used when the caller supplies it: a timeout session that
    # logs a multiplier is the CURRENT task whatever else it contains.
    if has_multiplier is True and 'timeout' in eff and not (eff & {'banish', 'unbanish'}):
        return 'timeout_multiplier'
    for r in TASK_RULES:
        if (r['all'] <= eff
                and (not r['any'] or (r['any'] & eff))
                and not (r['none'] & eff)):
            return r['name']
    return 'unknown'


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def _spawn_batches(log):
    batches = {}
    for s in sorted(log['spawns'], key=lambda s: s['time']):
        t = s['time']
        cur = s.get('current', []) or []
        if t not in batches or len(cur) > len(batches[t]):
            batches[t] = cur
    return sorted(batches.items())


def score_log(log_path, view_scale=None, mouse_id=None, label=None, after_switch=True):
    """Score one session from its log.json. Returns a flat dict, one row per session.

    `after_switch` (default True): if the BOARD CHANGES PROTOCOL part-way through the session,
    score only the final segment. Shaping sessions open under the old task and switch to the new
    one mid-recording, so the opening trials are a different experiment -- different icon types,
    a different number of ways to be wrong, and therefore a different chance baseline. Averaging
    across the switch produces a number that describes neither half.

    The trials before the switch are not deleted, they are reported: `switch_ms` says when the
    change happened and `n_before_switch` how many collections were dropped, so a session scored
    on a fraction of its length is visible rather than silently short. Pass after_switch=False to
    score the whole recording regardless.
    """
    log_path = Path(log_path)
    log = json.load(open(log_path))
    mouse_id = mouse_id or log_path.parent.name
    collected = sorted(log['collected'], key=lambda c: c['time'])

    switch_ms, _segs = protocol_switch(log)
    n_before = 0
    if after_switch and switch_ms is not None:
        # Count the dropped TRIALS (spawn batches), not the dropped collections. A trial that
        # started under the old board is finished by a collection that lands at or after the
        # switch -- the boundaries are the same instant -- so counting collections reports 0 while
        # a trial really was removed, which reads as "nothing was excluded" when something was.
        # unique batch TIMES, not raw spawn entries: the log writes one entry per ICON, so
        # counting entries reports 3 for a single 3-icon trial.
        n_before = len({b.get('time', 0) for b in (log.get('spawns') or [])
                        if b.get('time', 0) < switch_ms})
        collected = [c for c in collected if c['time'] >= switch_ms]
        log = dict(log, collected=collected,
                   spawns=[b for b in (log.get('spawns') or [])
                           if b.get('time', 0) >= switch_ms])
    task = classify_task(log_effects(log), has_multiplier=log_has_multiplier(log))
    if task == 'unknown':
        raise ValueError(f'{log_path}: cannot classify task from effects '
                         f'{sorted({c.get("effect") for c in collected})}')
    pos_set, neg_set = POSITIVE_EFFECTS, NEGATIVE_EFFECTS

    if view_scale is None:
        view_scale = VIEW_SCALE_BY_MOUSE.get(mouse_id)
    if not view_scale:
        raise ValueError(
            f"{log_path}: no view_scale for {mouse_id!r}. It is a property of the WORLD, is not in "
            f"the log, and differs per session -- pass view_scale=... explicitly. Known: "
            f"{VIEW_SCALE_BY_MOUSE}")
    vw, vh = SKETCH_W / view_scale, SKETCH_H / view_scale

    w0 = log['worlds'][0] if isinstance(log['worlds'], list) else log['worlds']
    C = np.array(log['coords_t[ms]/x/y'], float)

    # --- timeout penalty blackouts (timeout task only) -----------------------------------
    freezes = []
    if task == 'timeout_double':
        t = C[:, 0]
        tos = np.array([c['time'] for c in collected if c['effect'] == 'timeout'], float)
        for i in np.where(np.diff(t) > FREEZE_GAP_MIN_S * 1000)[0]:
            if len(tos) and np.abs(tos - t[i]).min() < 1500:
                freezes.append((t[i], t[i + 1]))

    # --- trials = spawn batches ------------------------------------------------------------
    batches = _spawn_batches(log)
    bt = [b[0] for b in batches]
    end_ms = float(C[-1, 0])
    coll_t = np.array([c['time'] for c in collected], float)

    rows = []
    for i, (t0, icons) in enumerate(batches):
        t1 = bt[i + 1] if i + 1 < len(batches) else end_ms
        outcome, mult = None, np.nan
        j = np.searchsorted(coll_t, t1 - 1)
        if j < len(coll_t) and abs(coll_t[j] - t1) <= 50:
            outcome = collected[j].get('effect')
            mult = collected[j].get('multiplier', np.nan)
        if outcome not in pos_set | neg_set:
            continue                                  # reshuffle / incomplete / escape-only
        fz = sum(max(0.0, min(e, t1) - max(s, t0)) for s, e in freezes)
        rows.append(dict(t0=t0, t1=t1, outcome=outcome, mult=mult, icons=icons, freeze_ms=fz))

    if not rows:
        raise ValueError(f'{log_path}: no scorable trials')

    pos = sum(r['outcome'] in pos_set for r in rows)
    neg = sum(r['outcome'] in neg_set for r in rows)
    # per collection: the streak multiplier where one was logged, else the effect's fixed pay.
    # Chosen per ROW rather than per task so a mixed protocol (multiplier rewards alongside
    # timeouts) is paid correctly instead of falling into one task's rule.
    drops = int(np.nansum([
        r['mult'] if (r['outcome'] == 'single_reward' and r.get('mult') is not None
                      and not (isinstance(r['mult'], float) and np.isnan(r['mult'])))
        else FIXED_DROPS.get(r['outcome'], 0)
        for r in rows]))

    # elapsed_min = real clock time from the first scored trial to the last (was `wall_min`).
    # It splits into active_min + freeze_min: active = time the animal could actually move,
    # freeze = the ~14 s post-timeout blackout when the game ignores the joystick.
    elapsed_min = (rows[-1]['t1'] - rows[0]['t0']) / 60000
    freeze_min = sum(r['freeze_ms'] for r in rows) / 60000
    active_min = elapsed_min - freeze_min

    # --- visibility-weighted chance ---------------------------------------------------------
    vpos = vneg = 0.0      # ZERO-memory: time-weighted fraction on screen
    mpos = mneg = 0.0      # PERFECT-memory: ever on screen this trial
    n_use = 0
    for r in rows:
        m = (C[:, 0] >= r['t0']) & (C[:, 0] <= r['t1'])
        if m.sum() < 5:
            continue
        tt, ax, ay = C[m, 0], C[m, 1], C[m, 2]
        # TIME-weight each sample. `coords` is logged only while the avatar MOVES, so a plain mean
        # over samples over-weights moving periods and biases the visibility estimate. Weighting by
        # the time each sample represents makes this match the per-session pipeline, which samples
        # on the uniform 30 Hz video-frame clock.
        wgt = np.diff(np.concatenate([tt, [r['t1']]]))
        wgt = np.clip(wgt, 0, None)
        if wgt.sum() <= 0:
            continue
        got = False
        for ic in r['icons']:
            if ic.get('effect') not in pos_set | neg_set:
                continue
            vis = (np.abs(ic['x'] - ax) <= vw / 2) & (np.abs(ic['y'] - ay) <= vh / 2)
            v = float(np.average(vis, weights=wgt))
            if ic['effect'] in pos_set:
                vpos += v; mpos += float(vis.any())
            else:
                vneg += v; mneg += float(vis.any())
            got = True
        n_use += got
    vpos, vneg = (vpos / n_use, vneg / n_use) if n_use else (np.nan, np.nan)
    mpos, mneg = (mpos / n_use, mneg / n_use) if n_use else (np.nan, np.nan)
    chance = vpos / (vpos + vneg) if np.isfinite(vpos + vneg) and (vpos + vneg) > 0 else np.nan
    # PERFECT-memory bound: he also knows icons he has already seen this trial (see the module
    # docstring). The two are the ends of a memory assumption, not a right and a wrong baseline.
    chance_mem = mpos / (mpos + mneg) if np.isfinite(mpos + mneg) and (mpos + mneg) > 0 else np.nan

    # --- *** THE LEARNING CRITERION *** ------------------------------------------------------
    # On the trials where BOTH types were on screen together, split by which was NEARER at that
    # first joint moment. The CONFLICT trials (negative icon closer) are where he has to override
    # proximity, so P(positive | conflict) is the number that must RISE if he learns to avoid the
    # hazard. The agree-trials are the CONTROL: if BOTH columns move he has just stopped following
    # proximity, which is not the same as recognising the icon.
    n_conf = k_conf = n_agree = k_agree = 0
    for r in rows:
        m = (C[:, 0] >= r['t0']) & (C[:, 0] <= r['t1'])
        if m.sum() < 5:
            continue
        ax, ay = C[m, 1], C[m, 2]
        P_ = [ic for ic in r['icons'] if ic.get('effect') in pos_set]
        N_ = [ic for ic in r['icons'] if ic.get('effect') in neg_set]
        if not P_ or not N_:
            continue
        vP = np.zeros(m.sum(), bool); vN = np.zeros(m.sum(), bool)
        for ic in P_:
            vP |= (np.abs(ic['x'] - ax) <= vw / 2) & (np.abs(ic['y'] - ay) <= vh / 2)
        for ic in N_:
            vN |= (np.abs(ic['x'] - ax) <= vw / 2) & (np.abs(ic['y'] - ay) <= vh / 2)
        both = vP & vN
        if not both.any():
            continue
        i = int(np.argmax(both))
        dP = min(np.hypot(ic['x'] - ax[i], ic['y'] - ay[i]) for ic in P_)
        dN = min(np.hypot(ic['x'] - ax[i], ic['y'] - ay[i]) for ic in N_)
        got = r['outcome'] in pos_set
        if dN < dP:
            n_conf += 1; k_conf += got
        else:
            n_agree += 1; k_agree += got
    conflict_p = k_conf / n_conf if n_conf else np.nan
    conflict_lo, conflict_hi = _wilson(k_conf, n_conf)
    agree_p = k_agree / n_agree if n_agree else np.nan

    # --- exogenous cross-check: is the NEAREST icon at spawn positive? -----------------------
    near_pos = near_tot = 0
    for r in rows:
        ic = [i for i in r['icons'] if i.get('effect') in pos_set | neg_set]
        if not ic:
            continue
        ax = float(np.interp(r['t0'], C[:, 0], C[:, 1]))
        ay = float(np.interp(r['t0'], C[:, 0], C[:, 2]))
        nearest = min(ic, key=lambda i: np.hypot(i['x'] - ax, i['y'] - ay))
        near_pos += nearest['effect'] in pos_set
        near_tot += 1
    chance_exo = near_pos / near_tot if near_tot else np.nan

    from scipy.stats import binomtest
    acc = pos / (pos + neg) if (pos + neg) else np.nan
    def toD(p0, v=None):
        # D is the fraction of the headroom ABOVE chance that was captured, so it is undefined when
        # there is no headroom: chance == 1 means every icon he could see was positive, and no
        # behaviour could have scored better. Return nan rather than dividing by zero -- a session
        # like that is unscorable, not infinitely good, and it must not take the whole run down.
        if p0 is None or not np.isfinite(p0) or p0 >= 1:
            return np.nan
        return ((acc if v is None else v) - p0) / (1 - p0)
    lo, hi = _wilson(pos, pos + neg)
    # world-DESIGN opportunity baseline (Maryam's active_benefits / active_detriments)
    ab, ad, opp = world_opportunity(log)
    _switch_info = dict(switch_ms=switch_ms, n_before_switch=n_before)
    return dict(**_switch_info,
        mouse=mouse_id, label=label or mouse_id, task=task, log=str(log_path),
        view_scale=view_scale, n_trials=len(rows),
        pos=pos, neg=neg, drops=drops, acc=acc,
        vis_pos=vpos, vis_neg=vneg, chance=chance,
        mem_pos=mpos, mem_neg=mneg, chance_mem=chance_mem,
        D_mem=toD(chance_mem), p_mem=binomtest(pos, pos + neg, chance_mem).pvalue
        if np.isfinite(chance_mem) else np.nan,
        D=toD(chance), D_lo=toD(chance, lo), D_hi=toD(chance, hi),
        p=binomtest(pos, pos + neg, chance).pvalue if np.isfinite(chance) else np.nan,
        n_both=n_conf + n_agree,
        n_conflict=n_conf, k_conflict=k_conf, conflict_p=conflict_p,
        conflict_lo=conflict_lo, conflict_hi=conflict_hi,
        n_agree=n_agree, agree_p=agree_p,
        chance_exo=chance_exo, D_exo=toD(chance_exo),
        p_exo=binomtest(pos, pos + neg, chance_exo).pvalue if np.isfinite(chance_exo) else np.nan,
        active_benefits=ab, active_detriments=ad, good_opportunity_ratio=opp,
        p_opportunity=prob_at_least(pos, pos + neg, opp),
        elapsed_min=elapsed_min, active_min=active_min, freeze_min=freeze_min,
        coll_per_min=pos / elapsed_min, drops_per_min=drops / elapsed_min,
        coll_per_active_min=pos / max(active_min, 1e-9),
        world_w=w0.get('width'), world_h=w0.get('height'))


def score_many(entries):
    """entries: list of (log_path, view_scale, label) or dicts -> list of score rows."""
    out = []
    for e in entries:
        if isinstance(e, dict):
            out.append(score_log(**e))
        else:
            lp, vs, lab = (list(e) + [None, None])[:3]
            out.append(score_log(lp, view_scale=vs, label=lab))
    return out


def banishment_evidence(log_path):
    """Raw, unprocessed counts of every place a banishment could show up in one log.

    For settling "is this really a banishment session?" without trusting any
    classification in between. Each number is a direct count of a log field:
      collected   what the animal actually took
      spawn.effect        the icon each spawn event names
      spawn.current[]     the icons listed as being on the board
    A session with zeros across all three contains no banishment to find.
    """
    log = json.load(open(log_path))
    coll, sp_eff, cur_eff = {}, {}, {}
    for c in (log.get('collected') or []):
        coll[c.get('effect')] = coll.get(c.get('effect'), 0) + 1
    for b in (log.get('spawns') or []):
        sp_eff[b.get('effect')] = sp_eff.get(b.get('effect'), 0) + 1
        for i in (b.get('current') or []):
            cur_eff[i.get('effect')] = cur_eff.get(i.get('effect'), 0) + 1
    w = (log.get('worlds') or [{}])[0]
    return dict(log=str(log_path),
                datetime=(log.get('experiment_data') or {}).get('datetime'),
                world=f"{w.get('width')}x{w.get('height')}/icon{w.get('icon_w')}/"
                      f"{str(w.get('world_texture_file','')).split('/')[-1]}",
                collected=coll, spawn_effect=sp_eff, on_board=cur_eff,
                n_spawn_batches=len({b.get('time') for b in (log.get('spawns') or [])}),
                task=classify_task(log_effects(log)))
