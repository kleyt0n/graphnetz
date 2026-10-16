"""The figure palette used in the paper, and the LaTeX colours that mirror it.

The paper is set in four steps of the project's neutral ramp
(``graphnetz.plotting.BRAND_COLORS``), so the figures, the shaded table cells and
the TikZ diagrams in ``main.tex`` all read from one family.  The LaTeX side of
that palette is declared at the top of ``main.tex`` as ``paperink``,
``paperaccent``, ``winnerc`` and ``tiec``; :data:`PAPER_COLORS` is the matplotlib
side of the same steps, and the two must be edited together or a figure and the
table beside it stop matching.

Call :func:`install_paper_style` once, before building any figure.
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt

PAPER_COLORS: tuple[str, ...] = (
    "#212529",  # Carbon Black  14.6:1  -- LaTeX \paperink
    "#6c757d",  # Slate Grey     4.5:1
    "#343a40",  # Gunmetal      10.9:1
    "#495057",  # Iron Grey      7.8:1  -- LaTeX \paperaccent
)
r"""Series colours, cycled by model index.

Four steps of the neutral ramp, all clearing 4.5:1 against the page.  The light
steps are deliberately absent: a series colour also tints the model labels in the
critical-difference diagram, and at 2.0:1 or less those labels cannot be read at
figure scale -- the first draft of this palette put GCN and the Graph Transformer
in `pale_slate` and both became invisible.  The light steps are used for fills
that carry no text (`CREAM`, `\winnerc`, the TikZ heatmap) instead.
"""

CREAM = "#e9ecef"
"""Platinum -- LaTeX ``\\tiec``; table shading for a statistical tie, not a mark.

The lightest step that is still distinguishable from the page, so a shaded run of
tied cells reads as shaded without competing with the winner's darker fill.
"""


def install_paper_style() -> None:
    """Apply the graphnetz rcParams, then swap in the paper palette.

    ``graphnetz.benchmark._report_plots`` binds ``NATURE_COLORS`` by value at
    import time, so the name is rebound in both modules; ``set_plot_style``
    is called afterwards because it reads the (now patched) module global to
    build the axes property cycle.
    """
    from graphnetz import plotting
    from graphnetz.benchmark import _report_plots

    plotting.NATURE_COLORS = PAPER_COLORS
    _report_plots.NATURE_COLORS = PAPER_COLORS
    plotting.set_plot_style()


# --------------------------------------------------------------------------- #
# Critical-difference diagram
# --------------------------------------------------------------------------- #

CD_CMAP = "RdYlGn"
"""Diverging colormap for CD-diagram ranks: best rank green, worst red."""

# Contrast window for the CD ramp, best rank first. Sampling ``RdYlGn`` directly
# is not usable here: its midpoint is near-white (about 1.1:1 against the page)
# and these colours tint the model labels, not only the markers. Hue and
# saturation therefore come from the colormap while *contrast* is imposed on a
# monotonic schedule.
#
# Contrast, not HLS lightness, is the control variable: at equal lightness yellow
# is far brighter than green or red, so a lightness schedule collapses -- an
# earlier version put the first three of five ranks at 4.7, 4.7 and 4.6:1, which
# is no ordering at all. Solving for contrast keeps the ramp monotonic whatever
# the hue, so rank stays recoverable from luminance alone, in greyscale print and
# under red-green colour-vision deficiency, where the hues collapse.
_CD_CONTRAST_DARKEST = 11.0  # best rank: heaviest mark, draws the eye first
_CD_CONTRAST_LIGHTEST = 4.6  # worst rank: still clears AA for text
_MIN_SATURATION = 0.45


# The dark scheme's page colour, so the same ramp can be solved against it.
DARK_PAGE = "#212529"


def _luminance(rgb: tuple[float, float, float]) -> float:
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(rgb: tuple[float, float, float], background: tuple[float, float, float]) -> float:
    """WCAG contrast ratio between a colour and its background."""
    a, b = sorted((_luminance(rgb), _luminance(background)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def _contrast_on_white(rgb: tuple[float, float, float]) -> float:
    """Contrast against the printed page. Kept for the palette tests."""
    return _contrast(rgb, (1.0, 1.0, 1.0))


def _at_contrast(
    hue: float,
    sat: float,
    target: float,
    background: tuple[float, float, float],
    *,
    lighten: bool,
) -> tuple[float, float, float]:
    """The (hue, sat) colour whose contrast against ``background`` is ``target``.

    Bisects on HLS lightness, which is monotone in luminance at fixed hue.
    ``lighten`` says which side of the background to search: on a white page the
    ramp runs darker, on a dark page it runs lighter, and only one direction
    reaches a given ratio.
    """
    import colorsys

    lo, hi = 0.02, 0.98
    rgb = colorsys.hls_to_rgb(hue, 0.5, sat)
    for _ in range(40):
        mid = (lo + hi) / 2
        rgb = colorsys.hls_to_rgb(hue, mid, sat)
        # ``bool(...)``, and ``!=`` rather than ``is not``: matplotlib colormaps
        # hand back numpy scalars, and ``numpy.False_ is not False`` is True, so
        # an identity test here silently took the same branch every iteration and
        # the bisection converged on the background colour.
        too_far = bool(_contrast(rgb, background) > target)
        if too_far != lighten:
            lo = mid
        else:
            hi = mid
    return rgb


def cd_colors(n: int, *, on_dark: bool = False) -> list[str]:
    """``n`` rank colours, best first: ``RdYlGn`` hues on a monotonic contrast ramp.

    Green marks the lowest mean rank and red the highest, so the diagram reads
    the way its axis does, and the best rank is also the darkest mark so it draws
    the eye first.  Every colour clears 4.5:1 against the page because these
    tint the model labels as well as the markers.

    Colour encodes mean rank and nothing else.  Whether two ranks *differ* is
    what the clique bar says, and that bar stays in ink so it remains the
    dominant mark -- on this benchmark it spans every model, and the figure
    should not suggest otherwise.
    """
    import colorsys

    import matplotlib as mpl

    cmap = mpl.colormaps[CD_CMAP]
    background = mpl.colors.to_rgb(DARK_PAGE) if on_dark else (1.0, 1.0, 1.0)
    out: list[str] = []
    for idx in range(n):
        span = max(n - 1, 1)
        # Best rank at the green end of the colormap, worst at the red end.
        r, g, b = (float(c) for c in cmap(1.0 - idx / span)[:3])
        hue, _, sat = colorsys.rgb_to_hls(r, g, b)
        target = _CD_CONTRAST_DARKEST + (_CD_CONTRAST_LIGHTEST - _CD_CONTRAST_DARKEST) * (idx / span)
        rgb = _at_contrast(hue, max(sat, _MIN_SATURATION), target, background, lighten=on_dark)
        out.append(mpl.colors.to_hex(rgb))
    return out


# Outcome classes for the epoch-selection figure, best to worst. The same ramp
# as the CD diagram, used for a categorical judgement rather than a rank: a
# verdict that survives both selection rules, one that was never significant,
# and one that held only at the fixed final epoch -- which is the finding the
# figure exists to show, so it should be visible without reading the annotations.
EPOCH_OUTCOMES = ("robust", "never", "inflated")

EPOCH_OUTCOME_LABELS = {
    "robust": "significant under both rules",
    "never": "significant under neither",
    "inflated": "significant only at the fixed epoch",
}


def epoch_outcome_colors(*, on_dark: bool = False) -> dict[str, str]:
    """Colour per outcome class, green for robust through red for inflated."""
    return dict(zip(EPOCH_OUTCOMES, cd_colors(len(EPOCH_OUTCOMES), on_dark=on_dark), strict=True))


def classify_epoch_outcome(significant_final: bool, significant_best: bool) -> str:
    """Which outcome class a comparison falls in.

    A comparison significant only under *best-validation* selection is the
    mirror image of an inflated one and does not occur in this sweep, but it is
    mapped to ``"robust"`` rather than silently to ``"inflated"``: it would mean
    the fixed-epoch rule was hiding a difference, not manufacturing one, and
    colouring it red would invert the figure's meaning.
    """
    if significant_final and significant_best:
        return "robust"
    if significant_final and not significant_best:
        return "inflated"
    if significant_best and not significant_final:
        return "robust"
    return "never"


# ``GraphSAGE (2.10)`` -- the model labels drawn by ``plot_critical_difference``.
_CD_LABEL = re.compile(r"^(?P<name>\S.*) \(\d+\.\d+\)$")


def style_cd_axes(ax: plt.Axes, *, on_dark: bool = False) -> list[str]:
    """Recolour a CD diagram's rank markers and labels by rank; returns the colours.

    ``BenchmarkReport.plot_critical_difference`` draws the markers from the
    series palette and the labels in neutral grey, neither of which says anything
    about rank.  Both are restyled here from :func:`cd_colors`.

    Markers and labels are created in ascending-mean-rank order, so index order
    is rank order for both; the markers are the single-point ``o`` lines, which
    distinguishes them from the leader lines and the clique and CD bars.
    """
    labels = [t for t in ax.texts if _CD_LABEL.match(t.get_text())]
    markers = [ln for ln in ax.lines if ln.get_marker() == "o" and len(ln.get_xdata()) == 1]
    colors = cd_colors(max(len(labels), len(markers), 1), on_dark=on_dark)
    for idx, text in enumerate(labels):
        text.set_color(colors[idx])
    for idx, line in enumerate(markers):
        line.set_color(colors[idx])
        line.set_markeredgecolor(colors[idx])
    return colors


def color_cd_labels(ax: plt.Axes) -> list[str]:
    """Deprecated alias for :func:`style_cd_axes`, which also styles the markers."""
    return style_cd_axes(ax)
