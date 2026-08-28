"""
Wrap every results figure into ONE multi-page PDF -- the deliverable for notebook 4.

Reads only the saved tables (`df_sessions`, `df_trials`); reruns no logs. It calls the same plotting
functions the interactive notebooks use, so the PDF and the notebooks never diverge. Each section is
wrapped in try/except: a section that cannot be drawn (e.g. a task with no clustered trials) prints a
short note page instead of killing the whole report.
"""
from pathlib import Path
import sys
import traceback

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'common'))
import session_index as sidx          # noqa: E402
import plot_patterns as pp            # noqa: E402
import plot_clusters as pc            # noqa: E402

TASKS = ('banish_multiplier', 'timeout_multiplier')


def _text_page(pdf, lines, title=''):
    fig = plt.figure(figsize=(11, 8.5)); fig.patch.set_facecolor('white')
    y = 0.92
    if title:
        fig.text(0.08, y, title, fontsize=20, fontweight='bold'); y -= 0.08
    for ln in lines:
        fig.text(0.08, y, ln, fontsize=11, family='monospace'); y -= 0.035
    plt.axis('off'); pdf.savefig(fig); plt.close(fig)


def _add(pdf, fn, *a, **k):
    """Call a plotting fn that returns a fig, save it, close it. Never raises."""
    try:
        fig = fn(*a, **k)
        if fig is not None:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        return True
    except Exception as ex:
        _text_page(pdf, [f'{fn.__name__}: {type(ex).__name__}: {ex}',
                         *traceback.format_exc().splitlines()[-4:]],
                   title='(section skipped)')
        plt.close('all')
        return False


def _usable(df_sessions):
    X = df_sessions
    for c in ('use', 'keep_of_day'):
        if c in X.columns:
            X = X[X[c].fillna(False)]
    if 'perf_error' in X.columns:
        X = X[X.perf_error.fillna('').eq('')]
    sort = [c for c in ('day', 'time') if c in X.columns]
    return X.sort_values(sort).reset_index(drop=True) if sort else X.reset_index(drop=True)


def build_pdf(df_sessions, df_trials, out_path, tasks=TASKS):
    """Assemble the full report to `out_path`. Returns the Path written."""
    out_path = Path(out_path)
    U = _usable(df_sessions)
    animals = sorted(U.mouse.unique()) if 'mouse' in U.columns else []
    present = [t for t in tasks if 'task' in U.columns and (U.task == t).any()]

    with PdfPages(out_path) as pdf:
        # ---- title ----
        _text_page(pdf, [
            f'sessions (usable) : {len(U)}',
            f'animals           : {", ".join(animals)}',
            f'protocols         : {", ".join(present)}',
            f'trials            : {len(df_trials)}',
            '',
            'Contents:',
            '  1  what is in the dataset',
            '  2  per task -- per animal + across-animal criterion',
            '  3  per-session pattern + across-mice average',
            '  4  first vs second half',
            '  5  path clusters (counts, feature space, examples, share)',
            '',
            'Baselines: D (mouse-view) / D (nearest-icon) / D (game-design).',
        ], title='Results report')

        # ---- 1. dataset overview ----
        _text_page(pdf, ['SECTION 1 -- what is in the dataset'])
        _add(pdf, pp.overview, U, df_trials)

        # ---- 2. per task: per animal + grouped ----
        _text_page(pdf, ['SECTION 2 -- each task, per animal then across animals'])
        for t in present:
            Xt = U[U.task == t]
            _text_page(pdf, [f'task: {t}', f'{len(Xt)} sessions, {Xt.mouse.nunique()} animal(s)'],
                       title=t)
            try:
                A = pp.by_animal_pooled(Xt, sidx)
                for _, r in A.iterrows():
                    B = sidx.blocks_of(Xt[Xt.mouse == r.mouse], size=None)
                    _add(pdf, pp.pooled_criterion, B, title=f'{r.mouse} - {t}')
                if len(A) > 1:
                    _add(pdf, pp.grouped_by_animal, A, title=f'{t}: every animal side by side')
            except Exception as ex:
                _text_page(pdf, [f'{type(ex).__name__}: {ex}'], title=f'{t} (skipped)')

        # ---- 3. per-session pattern ----
        _text_page(pdf, ['SECTION 3 -- per-session pattern (per mouse) + across-mice average'])
        for m in animals:
            _add(pdf, pp.session_lines, U[U.mouse == m], title=f'{m} - per-session pattern')
        if len(animals) > 1:
            _add(pdf, pp.mouse_summary, U, title='per mouse, then averaged across mice')

        # ---- 4. halves ----
        _text_page(pdf, ['SECTION 4 -- first vs second half'])
        for t in present:
            Xt = U[U.task == t]
            Tt = df_trials[df_trials.session.isin(Xt.session)] if 'session' in df_trials.columns else df_trials
            try:
                fig, _, _ = pp.half_split(Tt, Xt)
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
            except Exception as ex:
                _text_page(pdf, [f'{type(ex).__name__}: {ex}'], title=f'{t} halves (skipped)')

        # ---- 5. path clusters ----
        _text_page(pdf, ['SECTION 5 -- path clusters'])
        Cs = {}
        for t in present:
            Tt = df_trials[df_trials.task == t] if 'task' in df_trials.columns else df_trials
            Cs[t] = pc.clustered(Tt)
        Cs = {t: c for t, c in Cs.items() if len(c)}
        if Cs:
            _add(pdf, pc.cluster_overview_by_task, Cs, title='cluster counts + features by task')
            _add(pdf, pc.cluster_scatter_by_task, Cs, title='cluster feature space by task')
            for t, C in Cs.items():
                _add(pdf, pc.example_paths, C, title=f'{t} - example paths')
                _add(pdf, pc.cluster_share_by_animal, C, title=f'{t} - cluster share per animal')
        else:
            _text_page(pdf, ['no clustered trials in the dataset'])

    print(f'wrote {out_path}')
    return out_path
