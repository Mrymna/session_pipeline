"""
Teaching figure: WHY the chance baseline must be VISIBILITY-WEIGHTED, and what the discrimination
score D means. Dark theme, grounded in the real JPAS_0168 numbers (same convention as the JPAS_0231
make_*_explainer.py figures).

    python3 make_visibility_chance_explainer.py <session_dir>
        -> debug/visibility_chance_explainer.png
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

_COMMON = Path(__file__).resolve().parent.parent / 'common'
sys.path.insert(0, str(_COMMON))
import viewport as vp                    # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_performance as ap         # noqa: E402

BG = '#12151c'
FG = '#e8e8e8'
MUT = '#9aa4b2'
GREEN = '#27ae60'
RED = '#c0392b'
GREY = '#95a5a6'
AMBER = '#f39c12'


def run(session_dir, write=True, return_fig=False):
    d = Path(session_dir).resolve()
    sess = json.load(open(d / 'session.json'))
    blocks = ap.run(str(d), write=False)
    B = blocks['ALL']
    vw, vh = vp.viewport(sess)
    W = float(sess['world_width']); H = float(sess['world_height'])

    plt.rcParams.update({'text.color': FG, 'axes.labelcolor': FG, 'xtick.color': MUT,
                         'ytick.color': MUT, 'axes.edgecolor': '#3a4050'})
    fig = plt.figure(figsize=(19, 11.6), facecolor=BG)
    fig.suptitle('What is the chance level when the animal cannot see the whole board?',
                 fontsize=17, fontweight='bold', color=FG, y=0.975)
    fig.text(0.5, 0.933, f"{sess['mouse_id']}  -  why the discrimination score D is measured against "
             "a MOUSE-VIEW baseline (chance from what is on the mouse's screen), and why it is "
             'reported as a RANGE',
             ha='center', fontsize=11.5, color=MUT)

    # ── 1. the board vs what he can see ────────────────────────────────────────────
    a = fig.add_subplot(2, 3, 1, facecolor='#181c25')
    a.set_title('1.  the board vs the screen', fontsize=12, fontweight='bold', color=FG, pad=10)
    a.add_patch(Rectangle((0, 0), W, H, fc='#20252f', ec='#3a4050', lw=1.5))
    ax_, ay_ = 0.58 * W, 0.26 * H          # high in the world, so the lower band is off-screen
    a.add_patch(Rectangle((ax_ - vw / 2, ay_ - vh / 2), vw, vh, fc='#2b3550', ec=AMBER,
                          lw=2, alpha=0.55))
    # draw the REAL game assets, not stand-in circles, so the panel matches what he actually sees
    assets = d / 'task' / 'assets_game'

    def _sprite(name, cx, cy, size, alpha=1.0):
        img = plt.imread(str(assets / name))
        if img.dtype != np.float32 and img.max() > 1:
            img = img / 255.0
        a.imshow(img, extent=(cx - size / 2, cx + size / 2, cy + size / 2, cy - size / 2),
                 origin='upper', alpha=alpha, zorder=4, interpolation='bilinear')

    # NOTE the y-axis is INVERTED (world y grows downward), so a POSITIVE label offset puts the
    # text visually BELOW its icon. Each label's side is set explicitly -- an automatic rule put
    # the banish caption outside the axes and stacked the reward captions on each other.
    ICON_PX = 430
    icons = [(0.86 * W, 0.44 * H, 'circles2.png', 'reward', +1),
             (0.22 * W, 0.11 * H, 'circles2.png', 'reward', +1),
             (0.32 * W, 0.88 * H, 'fountains.png', 'banish', -1)]   # -1 = caption ABOVE the icon
    for ix, iy, tex, lab, side in icons:
        inside = abs(ix - ax_) <= vw / 2 and abs(iy - ay_) <= vh / 2
        _sprite(tex, ix, iy, ICON_PX, alpha=1.0 if inside else 0.30)
        a.text(ix, iy + side * (ICON_PX / 2 + 70),
               lab + ('\n(ON SCREEN)' if inside else '\n(OFF screen)'),
               ha='center', va='top' if side > 0 else 'bottom', fontsize=9,
               color=FG if inside else '#6b7280', fontweight='bold' if inside else 'normal')
    _sprite('heroArGreenBr-50.png', ax_, ay_, 300)
    a.text(ax_ + 230, ay_, 'the mouse', ha='left', va='center', fontsize=9.5, color=AMBER,
           fontweight='bold')
    # anchor to the viewport's BOTTOM edge: its top edge sits above the world here, so a
    # top-anchored label lands outside the axes and collides with the panel title.
    a.text(ax_ - vw / 2 + 60, ay_ + vh / 2 - 90, 'VIEWPORT', fontsize=9.5, color=AMBER,
           fontweight='bold', ha='left')
    a.set_xlim(-60, W + 60); a.set_ylim(H + 60, -60); a.set_xticks([]); a.set_yticks([])
    a.text(0.5, -0.07, f'world {W:.0f}x{H:.0f} wu   |   he sees {vw:.0f}x{vh:.0f} wu\n'
           'orange box = his VIEWPORT: the board has 2 rewards + 1 banish,\nbut the SCREEN usually does not',
           transform=a.transAxes, ha='center', va='top', fontsize=9, color=MUT)

    # ── 2. how often each type is actually on screen ───────────────────────────────
    a = fig.add_subplot(2, 3, 2, facecolor='#181c25')
    a.set_title('2.  measured: they are NOT on screen equally often',
                fontsize=12, fontweight='bold', color=FG, pad=10)
    a.barh([1, 0], [B['vis_r'], B['vis_b']], color=[GREEN, RED], height=0.55, alpha=0.92)
    for y, v in [(1, B['vis_r']), (0, B['vis_b'])]:
        a.text(v + 0.02, y, f'{v:.2f}', va='center', fontsize=13, fontweight='bold', color=FG)
    a.set_yticks([1, 0]); a.set_yticklabels(['reward icons', 'banish icons'], fontsize=11)
    a.set_xlabel('mean icons of that type on screen, per frame', fontsize=10)
    a.set_xlim(0, max(B['vis_r'], B['vis_b']) * 1.35)
    a.grid(alpha=0.12, axis='x')
    a.text(0.5, -0.30, f'a reward icon is on screen ~{B["vis_r"]/B["vis_b"]:.1f}x as often as a\n'
           'banish icon -- so bumping around at random already\nfinds rewards more than banishments',
           transform=a.transAxes, ha='center', va='top', fontsize=9.5, color=AMBER)

    # ── 3. the two candidate baselines ─────────────────────────────────────────────
    a = fig.add_subplot(2, 3, 3, facecolor='#181c25')
    a.set_title('3.  two ends of a MEMORY assumption', fontsize=12, fontweight='bold', color=FG, pad=10)
    a.bar([0, 1], [B['chance_mem'], B['chance']], color=[GREY, AMBER], width=0.55, alpha=0.92)
    a.axhline(B['acc'], color=GREEN, lw=2.5, ls='--', label=f"observed {B['acc']:.3f}")
    for x, v in [(0, B['chance_mem']), (1, B['chance'])]:
        a.text(x, v + 0.02, f'{v:.3f}', ha='center', fontsize=13, fontweight='bold', color=FG)
    a.set_xticks([0, 1])
    a.set_xticklabels(['perfect memory\n(seen once this trial)', 'zero memory\n(on screen NOW)'],
                      fontsize=9.5)
    a.set_ylabel('P(collect positive) with no skill', fontsize=10)
    a.set_ylim(0, 1); a.legend(fontsize=9, facecolor='#20252f', edgecolor='#3a4050', labelcolor=FG)
    a.grid(alpha=0.12, axis='y')
    a.text(0.5, -0.30, 'LEFT assumes he remembers every icon he has seen;\nRIGHT assumes he knows only '
           'what is on screen NOW.\nThe truth is between -- so D is a RANGE.',
           transform=a.transAxes, ha='center', va='top', fontsize=9.5, color=MUT)

    # ── 4. the formula ─────────────────────────────────────────────────────────────
    a = fig.add_subplot(2, 3, 4, facecolor='#181c25'); a.axis('off')
    a.set_title('4.  the discrimination score D', fontsize=12, fontweight='bold', color=FG, pad=10)
    a.text(0.5, 0.99, r'$D=\dfrac{\mathrm{observed}-\mathrm{chance}}{1-\mathrm{chance}}$',
           transform=a.transAxes, ha='center', va='top', fontsize=15, color=FG)
    # the two sides of D are counted in DIFFERENT UNITS -- spell both out, because that is the
    # first thing anyone asks and it is not obvious from the formula.
    a.text(0.0, 0.74,
           'OBSERVED  -  unit = one COLLECTION\n'
           f'   {B["pos"]} reward  +  {B["neg"]} banishment  =  {B["pos"]+B["neg"]} choices\n'
           f'   ({B["esc"]} unbanish excluded: an escape trial\n'
           '    offers no reward-vs-banishment choice)\n'
           f'   observed = {B["pos"]}/{B["pos"]+B["neg"]} = {B["acc"]:.3f}',
           transform=a.transAxes, va='top', ha='left', fontsize=8.2, color='#8fd19e',
           family='monospace', linespacing=1.32)
    a.text(0.0, 0.50,
           'CHANCE (zero memory)  -  unit = one FRAME\n'
           '   over ALL frames of the trial (frames where he\n'
           '   sees nothing add 0 to both and cancel out):\n'
           f'   REWARD {B["vis_r"]:.2f}      BANISHMENT {B["vis_b"]:.2f}   icons on screen\n'
           f'   chance = {B["vis_r"]:.2f}/({B["vis_r"]:.2f}+{B["vis_b"]:.2f}) = {B["chance"]:.3f}',
           transform=a.transAxes, va='top', ha='left', fontsize=8.2, color='#f0c06a',
           family='monospace', linespacing=1.32)
    a.text(0.0, 0.26,
           'CHANCE (perfect memory)  -  unit = ICONS SEEN\n'
           '   count an icon once he has seen it AT ALL this\n'
           f'   trial ({B["mem_r"]/2*100:.0f}% of rewards, {B["mem_b"]*100:.0f}% of banishments):\n'
           f'   REWARD {B["mem_r"]:.2f}      BANISHMENT {B["mem_b"]:.2f}   seen per trial\n'
           f'   chance = {B["mem_r"]:.2f}/({B["mem_r"]:.2f}+{B["mem_b"]:.2f}) = {B["chance_mem"]:.3f}\n'
           '   (close to 2/3 only because he sees nearly all)',
           transform=a.transAxes, va='top', ha='left', fontsize=8.2, color='#9ab8e0',
           family='monospace', linespacing=1.32)
    # one compact line, not a mathtext fraction -- the tall fraction collided with the CHANCE block
    a.text(0.0, -0.02, f"D = ({B['acc']:.3f} - {B['chance']:.3f}) / (1 - {B['chance']:.3f}) "
           f"= {B['D']:+.3f}", transform=a.transAxes, ha='left', va='top', fontsize=10,
           color=AMBER, family='monospace', fontweight='bold')

    # ── 5. the D scale ─────────────────────────────────────────────────────────────
    a = fig.add_subplot(2, 3, 5, facecolor='#181c25')
    a.set_title('5.  how to read D', fontsize=12, fontweight='bold', color=FG, pad=10)
    a.axhspan(0, 1, color=GREEN, alpha=0.10); a.axhspan(-0.6, 0, color=RED, alpha=0.10)
    a.axhline(0, color=FG, lw=2); a.axhline(1, color=GREEN, lw=1.5, ls=':')
    for y, txt, c in [(1.0, 'D = 1   perfect: never takes the banishment', GREEN),
                      (0.0, 'D = 0   CHANCE: collects in proportion to what is on screen', FG),
                      (-0.45, 'D < 0   worse than chance: drawn TO the banishment', RED)]:
        a.text(0.5, y + 0.05, txt, ha='center', fontsize=10, color=c, fontweight='bold')
    a.plot([0.5], [B['D']], marker='D', ms=15, color=AMBER, zorder=5)
    a.annotate(f"this session\nD = {B['D']:+.3f}\n(p = {B['p']:.2f})", xy=(0.5, B['D']),
               xytext=(0.78, 0.62), fontsize=10.5, color=AMBER, fontweight='bold', ha='center',
               arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.6))
    a.set_xlim(0, 1); a.set_ylim(-0.6, 1.15); a.set_xticks([])
    a.set_ylabel('D', fontsize=12); a.grid(alpha=0.12, axis='y')

    # ── 6. the punchline ───────────────────────────────────────────────────────────
    a = fig.add_subplot(2, 3, 6, facecolor='#181c25'); a.axis('off')
    a.set_title('6.  why it matters here', fontsize=12, fontweight='bold', color=FG, pad=10)
    a.text(0.02, 0.92, 'Assuming PERFECT memory (knows what he saw):', fontsize=11, color=MUT)
    a.text(0.06, 0.80, f'D = {B["D_mem"]:+.3f}   (p = {B["p_mem"]:.2f}, n.s.)',
           fontsize=13.5, color=GREEN, fontweight='bold')
    a.text(0.02, 0.62, 'Assuming ZERO memory (only what is on screen now):', fontsize=11, color=MUT)
    a.text(0.06, 0.50, f'D = {B["D"]:+.3f}   (p = {B["p"]:.2f}, n.s.)',
           fontsize=13.5, color=RED, fontweight='bold')
    a.text(0.02, 0.30,
           'Same animal, same data. The two numbers differ only in\n'
           'what we assume he REMEMBERS of what he has seen.',
           fontsize=10.5, color=FG, linespacing=1.6, va='top')
    a.text(0.02, 0.10,
           'Both ends are non-significant, so the verdict is the same\n'
           'either way: he is AT CHANCE -- no evidence yet that he\n'
           'tells reward from banishment.',
           fontsize=10.5, color=AMBER, linespacing=1.6, va='top')

    plt.tight_layout(rect=[0, 0.02, 1, 0.915])
    out = d / 'debug' / 'visibility_chance_explainer.png'
    fig.savefig(out, dpi=110, facecolor=BG)
    if return_fig:
        plt.rcParams.update(plt.rcParamsDefault)
        return fig
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print(f'wrote {out}')
    return out


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '.')
