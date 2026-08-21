"""
TASK SPEC for `timeout_double` (JPAS_0231-class) -- the vocabulary that makes this task
different from `banish_multiplier`, in one place so the fork's scripts stay thin.

THE TASK. Three icons are on the board at once and the mouse drives to one of them:
    single_reward  droplet      GOOD   one reward drop
    double_reward  glowCubes    GOOD   two reward drops
    timeout        target       BAD    a 14 s penalty
There is no multiplier and no second world -- that is the whole difference from
`banish_multiplier`, plus the timeout FREEZE below.

*** THE TIMEOUT FREEZE IS REAL AND VERIFIED (measured here, 2026-08-18). ***
JPAS_0231's `build_session_frames.py` ASSUMED a 14 s no-joystick window after each target hit.
It is stronger than assumed and worth stating precisely: after a timeout the game stops logging
`coords` ENTIRELY for ~14 s. Measured on JPAS_0231: 45 of 46 timeouts are followed by a coord
blackout of median 14.0 s (min 14.0, max 14.3); no reward collection is ever followed by one.
Consequences the pipeline MUST respect:
  - the avatar does not move, so any np.interp across the blackout invents a straight-line path;
  - path efficiency / speed / heading over a window containing a freeze are meaningless unless
    the frozen span is excluded -> `build_trials` records `freeze_ms` per trial and geometry is
    computed on the ACTIVE part only;
  - the joystick keeps being logged during the freeze (the mouse still pushes the stick, the game
    just ignores it), so a joystick-based "is he moving" test does NOT detect the freeze -- only
    the coord blackout does.

POSITIVE vs NEGATIVE for the performance score: both reward types are POSITIVE, timeout is
NEGATIVE. That makes the icon-count baseline 2:1 exactly as in `banish_multiplier` -- and just as
there, it is the WRONG baseline; use the visibility-weighted one (see analyze_performance).
"""

TASK = 'timeout_double'

# outcome -> (display label, is_positive, colour)
OUTCOMES = {
    'single_reward': ('single reward', True,  '#27ae60'),
    'double_reward': ('double reward', True,  '#16a085'),
    'timeout':       ('timeout',       False, '#c0392b'),
}
OUTCOME_ORDER = ['single_reward', 'double_reward', 'timeout']
POSITIVE = [k for k, v in OUTCOMES.items() if v[1]]        # single_reward, double_reward
NEGATIVE = [k for k, v in OUTCOMES.items() if not v[1]]    # timeout
ANALYZE_OUTCOMES = set(OUTCOMES)

OUTCOME_DISPLAY = {k: v[0] for k, v in OUTCOMES.items()}
OUTCOME_COL = {k: v[2] for k, v in OUTCOMES.items()}

# log `texture` name -> the asset the game actually draws.
# (In banish_multiplier the log's texture string did NOT match the filename -- "circles" was
# really circles2.png. Here they do match, but the mapping stays explicit so a future mismatch
# is a one-line fix rather than a silent wrong icon.)
ICON_TEX = {'single_reward': 'dropletB.png',
            'double_reward': 'glowCubes.png',
            'timeout': 'target.png'}

BACKGROUND = {'NORMAL': 'worldLowContrastLowSaturationRed.png'}   # ONE world, no shadow realm

HAS_MULTIPLIER = False
HAS_SECOND_WORLD = False

FREEZE_S = 14.0            # the penalty window after a timeout (verified above)
FREEZE_GAP_MIN_S = 2.0     # a coord gap this long starting at a timeout counts as the freeze
