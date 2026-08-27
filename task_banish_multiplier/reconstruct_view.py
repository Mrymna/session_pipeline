"""
MOUSE MONITOR VIEW for a `banish_multiplier` session -- reconstruct, frame by frame, the image the
mouse actually saw on its screen, and render it as a video.

Port of JPAS_0231/reconstruct_view.py to this task. What is DIFFERENT here is the banishment:

  NORMAL trials      background = worldLowContrastLowSaturationRed.png  (red -- nearly invisible to
                     a mouse), icons = 2x `circles` (single_reward, GREEN) + 1x `fountains`
                     (banish, BLUE).
  BANISHMENT trials  the mouse has hit the banish icon and is thrown into the SHADOW REALM:
                     background = shadowRealmHex.png, and the only icons are the TWO `ring`
                     (unbanish) escape portals. No reward is available in there.

Which world a frame is in comes from `df_trials_clean.world` (NORMAL / BANISH_WORLD) -- the log's
`worlds` list does NOT record the swap (both entries name the red texture), so the trial table is
the authority. The per-trial `icons` list gives the on-screen landscape, so every frame draws
exactly the icons that were really on the mouse's screen at that moment.

Geometry, all from the log where possible:
  - world 2400x2400, icons drawn at `icon_w` = 500 world units (this session RECORDS it; JPAS_0231
    had to guess 450).
  - camera follows the avatar: `translate(centre); scale(s); translate(-x,-y)`, so the view
    spans SKETCH/s world units and the world edge shows as BLACK border when the mouse
    corners -- that black is part of what reaches the eye and is NOT cropped away.
    THE WORLD VIEW = a box of  SKETCH / view_scale  world units, centred on the avatar
    (SKETCH = 800 x 600 px). Worked (JPAS_0168): view_scale 0.35 -> 800/0.35 x 600/0.35
    = 2285.7 x 1714.3 wu on the 2400x2400 world. This is the SAME box the performance code uses to
    decide if an icon is on screen: an icon is visible at a sample when its CENTRE is inside the
    box, |icon.x - avatar.x| <= vw/2 AND |icon.y - avatar.y| <= vh/2 (vw, vh = SKETCH /
    view_scale). Only the ratio SKETCH/view_scale matters (the viewport is that many world units
    across).
  - avatar = the GREEN breathing pair (heroArGreenBr-50 / heroBrGreenBr-50) alternated every 0.5 s
    and mirrored L/R by travel direction, drawn upright (the sprite never rotates); the heading
    arrow is a green triangle OUTLINE rotated by theta, drawn only while moving.
  - the game's post-draw `fill(0,50)` overlay = a global x0.804 dim on the whole frame.

The view scale is a property of the WORLD, so it DIFFERS PER SESSION (Maryam, 2026-08-18) and is
not in the log -- it comes from `session.json["view_scale"]` via `common/viewport.py`, which RAISES
rather than defaulting (a silent default would let a new session inherit another world's zoom and
quietly corrupt every visibility-gated result). JPAS_0168 = **0.35** (confirmed), JPAS_0231 = 0.56.
It sets how much world is visible and therefore how much black border appears; it does NOT change
the avatar's on-screen size, a fixed 130 px of the 800 px sketch either way. `--scale` overrides.

`--vision mouse` renders what the MOUSE sees rather than what a human sees: the red channel is
dropped, because a mouse is nearly blind to display red (M-cone lmax 508 nm, melanopsin 480 nm vs
display red ~615 nm). The session dashboard does exactly this -- its game panel has R == 0 in every
pixel. Under mouse vision the red BACKGROUND nearly vanishes while the GREEN reward and BLUE banish
icons stay bright, which is the point: the icons, not the terrain, are what the mouse can see.

Run:
    python3 reconstruct_view.py <session_dir> --test              # short clip over a banishment
    python3 reconstruct_view.py <session_dir> --trial 5           # one trial
    python3 reconstruct_view.py <session_dir> --full              # the whole session
    python3 reconstruct_view.py <session_dir> --lo 1000 --hi 2000 [--vision mouse] [--scale 0.35]
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import cv2

_COMMON = Path(__file__).resolve().parent.parent / 'common'
sys.path.insert(0, str(_COMMON))
import fps as fpsmod          # noqa: E402
import viewport as vp         # noqa: E402

SKETCH_W, SKETCH_H = 800, 600      # the game sketch, as in JPAS_0231
GLOBAL_DIM = 0.804                 # the game's post-draw fill(0,50) black overlay
CHAR_PX = 130                      # avatar is a fixed 130 px of the sketch
BREATH_S = 0.5                     # sprite alternates every 0.5 s (animation_period 1.0)
MOVE_EPS = 0.5                     # world units/sample below which the avatar counts as still
ARROW_COL = (127, 192, 127)        # BGR green heading triangle
ARROW_VERTS = np.array([(100, -65), (200, 0), (100, 65)], float)   # world units, from 231
ARROW_STROKE = 15

BACKGROUND = {'NORMAL': 'worldLowContrastLowSaturationRed.png',
              'BANISH_WORLD': 'shadowRealmHex.png'}
# ⚠️ The log's `texture` field is NOT the filename. `log['worlds'][*]['effects']` names the reward
# texture "circles", but the asset the game actually draws is **circles2.png** (the faceted green
# gem), NOT circles.png (a flat dotted disc) -- confirmed by Maryam, 2026-08-18. Do not "correct"
# this back to circles.png by matching the log string. banish/unbanish do match their log names.
ICON_TEX = {'single_reward': 'circles2.png',    # GREEN  reward  (log calls it "circles")
            'banish': 'fountains.png',          # BLUE   hazard -> throws him into the shadow realm
            'unbanish': 'ring.png'}             # TEAL   escape portal, only in the shadow realm


class MouseView:
    def __init__(self, session_dir, scale=None, vision='rgb'):
        self.d = d = Path(session_dir).resolve()
        self.sess = json.load(open(d / 'session.json'))
        self.log = json.load(open(d / 'log.json'))
        self.df = pd.read_pickle(d / 'df_trials_clean.pkl')
        self.assets = d / 'task' / 'assets_game'
        if not self.assets.exists():
            sys.exit(f'no asset folder at {self.assets}')
        self.scale = float(scale if scale is not None else vp.view_scale(self.sess))
        self.vision = vision
        self.W = float(self.sess['world_width']); self.H = float(self.sess['world_height'])
        self.view_w = SKETCH_W / self.scale       # world units visible across the sketch
        self.view_h = SKETCH_H / self.scale
        self.char_world = CHAR_PX / self.scale    # avatar size in world units
        w0 = self.log['worlds'][0]
        self.icon_w = float(w0.get('icon_w', 500))
        self.fps = self.sess['fps']

        self.frame_ms = fpsmod.frame_to_ms(self.log)
        self.COORD = np.array(self.log['coords_t[ms]/x/y'], float)
        self.ANG = np.array(self.log['angles_t[ms]/theta'], float)
        self.bg = {k: self._load(v, alpha=False) for k, v in BACKGROUND.items()}
        self.icons = {k: self._load(v, alpha=True) for k, v in ICON_TEX.items()}
        self.hero = [self._load('heroArGreenBr-50.png', alpha=True),
                     self._load('heroBrGreenBr-50.png', alpha=True)]
        # frame -> (world, icons) via the disjoint trial windows
        self._index_trials()

    def _load(self, name, alpha):
        p = self.assets / name
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            sys.exit(f'missing asset {p}')
        if alpha and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        if not alpha and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _index_trials(self):
        n = int(self.sess['n_frames'])
        self.world_of = np.array(['NORMAL'] * n, object)
        self.icons_of = [[] for _ in range(n)]
        self.trial_of = np.full(n, -1, int)
        for _, r in self.df.iterrows():
            s, e = int(r['start_frame']), int(min(r['end_frame'], n))
            self.world_of[s:e] = r['world'] if r['world'] in BACKGROUND else 'NORMAL'
            self.trial_of[s:e] = int(r['trial'])
            for k in range(s, e):
                self.icons_of[k] = r['icons'] or []

    # ---- geometry -------------------------------------------------------------------
    def avatar_at(self, f):
        t = self.frame_ms[f]
        x = float(np.interp(t, self.COORD[:, 0], self.COORD[:, 1]))
        y = float(np.interp(t, self.COORD[:, 0], self.COORD[:, 2]))
        th = float(np.interp(t, self.ANG[:, 0], self.ANG[:, 1]))
        tp = self.frame_ms[max(f - 1, 0)]
        px = float(np.interp(tp, self.COORD[:, 0], self.COORD[:, 1]))
        py = float(np.interp(tp, self.COORD[:, 0], self.COORD[:, 2]))
        return x, y, th, (x - px), (y - py)

    def _w2s(self, wx, wy, ax, ay):
        """World -> sketch pixels for a camera centred on the avatar."""
        return (int(round((wx - ax) * self.scale + SKETCH_W / 2)),
                int(round((wy - ay) * self.scale + SKETCH_H / 2)))

    def _blit(self, dst, src_bgra, cx, cy, size_px):
        """Alpha-composite a sprite centred at (cx, cy), scaled to size_px."""
        if size_px < 1:
            return
        s = cv2.resize(src_bgra, (size_px, size_px), interpolation=cv2.INTER_AREA)
        x0, y0 = cx - size_px // 2, cy - size_px // 2
        x1, y1 = x0 + size_px, y0 + size_px
        sx0, sy0 = max(0, -x0), max(0, -y0)
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(dst.shape[1], x1), min(dst.shape[0], y1)
        if x1c <= x0c or y1c <= y0c:
            return
        patch = s[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
        a = (patch[:, :, 3:4].astype(float) / 255.0)
        dst[y0c:y1c, x0c:x1c] = (patch[:, :, :3] * a +
                                 dst[y0c:y1c, x0c:x1c] * (1 - a)).astype(np.uint8)

    # ---- the frame ------------------------------------------------------------------
    def render(self, f, show_arrow=True, dim=True):
        ax, ay, th, dx, dy = self.avatar_at(f)
        world = self.world_of[f]
        bg = self.bg[world]

        # 1. background: sample the world texture over the visible window; OUTSIDE the world is
        #    BLACK, and that black is a real part of what the eye receives (it is what makes the
        #    screen dim when the mouse corners), so it is kept, never cropped.
        canvas = np.zeros((SKETCH_H, SKETCH_W, 3), np.uint8)
        xs = np.linspace(ax - self.view_w / 2, ax + self.view_w / 2, SKETCH_W)
        ys = np.linspace(ay - self.view_h / 2, ay + self.view_h / 2, SKETCH_H)
        inx = (xs >= 0) & (xs < self.W)
        iny = (ys >= 0) & (ys < self.H)
        if inx.any() and iny.any():
            u = np.clip((xs[inx] / self.W * bg.shape[1]).astype(int), 0, bg.shape[1] - 1)
            v = np.clip((ys[iny] / self.H * bg.shape[0]).astype(int), 0, bg.shape[0] - 1)
            canvas[np.ix_(iny, inx)] = bg[np.ix_(v, u)]

        # 2. the icons actually on screen this trial
        isz = int(round(self.icon_w * self.scale))
        for ic in self.icons_of[f]:
            tex = self.icons.get(ic.get('effect'))
            if tex is None:
                continue
            sx, sy = self._w2s(ic['x'], ic['y'], ax, ay)
            if -isz < sx < SKETCH_W + isz and -isz < sy < SKETCH_H + isz:
                self._blit(canvas, tex, sx, sy, isz)

        # 3. avatar: breathing pair, mirrored by travel direction, never rotated
        sprite = self.hero[int((self.frame_ms[f] / 1000.0) // BREATH_S) % 2]
        if dx < 0:
            sprite = np.flip(sprite, axis=1)
        csz = int(round(self.char_world * self.scale))          # == CHAR_PX by construction
        self._blit(canvas, sprite, SKETCH_W // 2, SKETCH_H // 2, csz)

        # 4. heading arrow -- only while moving (that is when the game draws it)
        if show_arrow and (dx * dx + dy * dy) ** 0.5 > MOVE_EPS:
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]])
            pts = (ARROW_VERTS @ R.T) * self.scale + np.array([SKETCH_W / 2, SKETCH_H / 2])
            cv2.polylines(canvas, [pts.astype(np.int32)], True, ARROW_COL,
                          max(1, int(ARROW_STROKE * self.scale)), cv2.LINE_AA)

        if dim:
            canvas = (canvas * GLOBAL_DIM).astype(np.uint8)
        if self.vision == 'mouse':
            # a mouse is nearly blind to display red -- drop it, as the rig dashboard does
            canvas[:, :, 2] = 0
        return canvas

    # ---- video ----------------------------------------------------------------------
    def render_clip(self, lo, hi, out=None, label=True):
        lo, hi = int(lo), int(min(hi, len(self.frame_ms)))
        outdir = self.d / 'test'; outdir.mkdir(exist_ok=True)
        out = Path(out) if out else outdir / f'mouse_view_{self.vision}_{lo}_{hi}.mp4'
        wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                             (SKETCH_W, SKETCH_H))
        n_ban = 0
        for f in range(lo, hi):
            img = self.render(f)
            if self.world_of[f] == 'BANISH_WORLD':
                n_ban += 1
            if label:
                w = self.world_of[f]
                col = (200, 200, 90) if w == 'BANISH_WORLD' else (170, 170, 170)
                cv2.putText(img, f'frame {f}  trial {self.trial_of[f]}  {w}', (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            wr.write(img)
        wr.release()
        print(f'wrote {out}   ({hi - lo} frames, {n_ban} in the SHADOW REALM)')
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('session_dir')
    ap.add_argument('--lo', type=int); ap.add_argument('--hi', type=int)
    ap.add_argument('--trial', type=int, help='render exactly this trial')
    ap.add_argument('--test', action='store_true',
                    help='short clip spanning a banish -> escape pair (both worlds)')
    ap.add_argument('--full', action='store_true', help='the whole session')
    ap.add_argument('--scale', type=float, default=None,
                    help='override session.json view_scale')
    ap.add_argument('--vision', choices=['rgb', 'mouse'], default='rgb')
    ap.add_argument('--still', type=int, help='write a single frame as a PNG instead of a video')
    a = ap.parse_args()

    mv = MouseView(a.session_dir, scale=a.scale, vision=a.vision)
    print(f'  view {mv.view_w:.0f} x {mv.view_h:.0f} world units (scale {mv.scale}), '
          f'icons {mv.icon_w:.0f} wu, world {mv.W:.0f}x{mv.H:.0f}, vision={mv.vision}')

    if a.still is not None:
        p = mv.d / 'debug' / f'mouse_view_f{a.still}_{mv.vision}.png'
        cv2.imwrite(str(p), mv.render(a.still)); print(f'wrote {p}')
        return
    if a.trial is not None:
        r = mv.df[mv.df.trial == a.trial].iloc[0]
        lo, hi = int(r['start_frame']), int(r['end_frame'])
    elif a.test:
        # the SHORT TEST: the last normal trial before a banish, through the escape trial, so the
        # clip contains the world swap, the fountains icon AND the two rings in one go.
        esc = mv.df[mv.df.in_banishment].iloc[0]
        prev = mv.df[mv.df.trial == esc['trial'] - 1]
        lo = int(prev.iloc[0]['start_frame']) if len(prev) else int(esc['start_frame'])
        hi = int(esc['end_frame'])
        lo = max(lo, hi - 900)                     # cap at ~30 s
        print(f"  test clip: trial {esc['trial'] - 1} (NORMAL) -> trial {esc['trial']} (SHADOW REALM)")
    elif a.full:
        lo, hi = 0, int(mv.sess['n_frames'])
    else:
        lo, hi = a.lo or 0, a.hi or 600
    mv.render_clip(lo, hi)


if __name__ == '__main__':
    main()
