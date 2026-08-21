"""
The game camera / viewport -- ONE definition, shared by every consumer.

The game draws with `translate(centre); scale(s); translate(-x,-y)` on an 800x600 sketch, so the
mouse sees `SKETCH / s` world units centred on the avatar, and anything past the world edge is BLACK
(that black is part of what reaches the eye -- it is why the screen dims when he corners).

*** `s` IS A PROPERTY OF THE WORLD, NOT A CONSTANT OF THE RIG (Maryam, 2026-08-18). ***
Different worlds are shown at different zoom, so a session's scale depends on which world it ran.
It is NOT recorded in log.json. Known values:

    JPAS_0231   s = 0.56   (read from the game code)   -> view 1428.6 x 1071.4 wu, world 2000x2000
    JPAS_0168   s = 0.35   (CONFIRMED by Maryam)       -> view 2285.7 x 1714.3 wu, world 2400x2400

Because it is per-world, there is deliberately **NO usable default**: `viewport()` RAISES if
`session.json` has no `view_scale`. A silent default is the dangerous option here -- it would let a
new session quietly inherit another world's zoom, and every "could the mouse SEE this icon?" result
(icon visibility, the attention gating, the performance chance-baseline) would be wrong without
anything looking broken. Set it per session:

    session.json  ->  "view_scale": 0.35, "view_scale_source": "confirmed by Maryam 2026-08-18"
"""
SKETCH_W, SKETCH_H = 800, 600


class MissingViewScale(KeyError):
    pass


def view_scale(sess):
    s = sess.get('view_scale')
    if not s:
        raise MissingViewScale(
            f"session.json for {sess.get('mouse_id', '?')} has no 'view_scale'. It is a property of "
            "the WORLD (it differs per session) and is not in log.json, so it must be set explicitly "
            "-- there is no safe default. Known: JPAS_0231 = 0.56, JPAS_0168 = 0.35.")
    return float(s)


def viewport(sess):
    """(width, height) of the visible window in WORLD units, for this session."""
    s = view_scale(sess)
    return SKETCH_W / s, SKETCH_H / s


def visible(sess, ax, ay, ix, iy):
    """Is the icon at (ix, iy) inside the viewport centred on the avatar at (ax, ay)?
    Accepts scalars or arrays."""
    w, h = viewport(sess)
    return (abs(ix - ax) <= w / 2) & (abs(iy - ay) <= h / 2)


def spawn_geometry_chance(sub, log, positive, negative):
    """SECOND, EXOGENOUS chance baseline: at each trial's START, which icon is NEAREST the avatar?

    Why this exists. The visibility-weighted baseline is ENDOGENOUS -- if the animal steers away
    from the hazard, the hazard is on screen less BECAUSE of that skill, so the baseline can absorb
    the very thing we are trying to measure. This one is computed from the spawn configuration and
    the avatar's position at the moment the batch appears, before he has acted on it, so his
    behaviour cannot move it. It is cruder (it assumes proximity-driven choice, and these animals
    collect the nearest icon only ~36-44% of the time), so the two baselines BRACKET the answer
    rather than either being definitive -- report both, and trust a conclusion only if it survives
    under each.
    """
    import numpy as np
    C = np.array(log['coords_t[ms]/x/y'], float)
    near_pos = tot = 0
    for _, r in sub.iterrows():
        ic = [i for i in (r['icons'] or []) if i['effect'] in set(positive) | set(negative)]
        if not ic:
            continue
        ax = np.interp(r['start_ms'], C[:, 0], C[:, 1])
        ay = np.interp(r['start_ms'], C[:, 0], C[:, 2])
        nearest = min(ic, key=lambda i: np.hypot(i['x'] - ax, i['y'] - ay))
        near_pos += nearest['effect'] in set(positive)
        tot += 1
    return (near_pos / tot) if tot else float('nan')
