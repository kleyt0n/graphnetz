"""Pin the figure palette's invariants.

The palette is a nine-step neutral ramp, so a series colour carries no hue --
only luminance -- and every distinction a reader makes depends on which steps are
chosen and in what order. Two mistakes are easy and were both made while
adopting it:

* using a light step for a series, which is fine for a bar fill but illegible
  once the same colour tints a 4 pt marker or a model label (an early draft put
  GCN and the Graph Transformer in ``pale_slate`` and both vanished);
* letting the paper's LaTeX colours drift from the matplotlib ones, so a figure
  and the table beside it stop matching.

These tests are the guard rail for both.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

from graphnetz.plotting import BRAND_COLORS, NATURE_COLORS

RAMP = {value.lower() for value in BRAND_COLORS.values()}

# Series colours double as marker and label colours, so they are text.
WCAG_AA_TEXT = 4.5

# How many leading steps must be text-safe: the number of architectures a
# benchmark row usually compares, which is what lands in a CD diagram.
TEXT_SAFE_PREFIX = 4


def _relative_luminance(hex_colour: str) -> float:
    """WCAG relative luminance of ``#rrggbb``."""
    channels = [int(hex_colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_on_white(hex_colour: str) -> float:
    return 1.05 / (_relative_luminance(hex_colour) + 0.05)


def test_series_colours_are_all_steps_of_the_identity_ramp() -> None:
    assert {c.lower() for c in NATURE_COLORS} <= RAMP


def test_series_colours_are_unique() -> None:
    """A repeated step would silently merge two series."""
    assert len({c.lower() for c in NATURE_COLORS}) == len(NATURE_COLORS)


def test_the_leading_series_colours_can_carry_text() -> None:
    """The first four steps tint CD-diagram labels, so they must be readable."""
    for colour in NATURE_COLORS[:TEXT_SAFE_PREFIX]:
        ratio = _contrast_on_white(colour)
        assert ratio >= WCAG_AA_TEXT, f"{colour} is {ratio:.1f}:1, below {WCAG_AA_TEXT}:1"


def test_the_leading_series_colours_are_mutually_distinguishable() -> None:
    """Without hue, only a luminance gap separates two series."""
    lums = sorted(_relative_luminance(c) for c in NATURE_COLORS[:TEXT_SAFE_PREFIX])
    for dark, light in itertools.pairwise(lums):
        assert light / dark >= 1.5, f"luminance ratio {light / dark:.2f} is too close to call"


def test_bars_are_stroked_in_ink_not_white() -> None:
    """A white stroke erases a light fill instead of bounding it."""
    source = Path(__file__).resolve().parents[1] / "src" / "graphnetz" / "plotting.py"
    body = source.read_text()
    start = body.index("ax.bar(")
    bar_call = body[start : body.index("\n        )", start)]
    assert 'edgecolor="white"' not in bar_call
    assert "edgecolor=_INK" in bar_call
    # and nowhere else in the module either
    assert 'edgecolor="white"' not in body


# --------------------------------------------------------------------------- #
# The paper's palette must not drift from the library's
# --------------------------------------------------------------------------- #

_PAPER = Path(__file__).resolve().parents[1] / "paper"


@pytest.mark.skipif(not _PAPER.exists(), reason="manuscript not present in this checkout")
def test_paper_figure_palette_is_within_the_ramp() -> None:
    from paper.experiments._style import CREAM, PAPER_COLORS

    assert {c.lower() for c in PAPER_COLORS} <= RAMP
    assert CREAM.lower() in RAMP
    for colour in PAPER_COLORS:
        assert _contrast_on_white(colour) >= WCAG_AA_TEXT, colour


@pytest.mark.skipif(not (_PAPER / "main.tex").exists(), reason="manuscript not present")
def test_latex_colours_are_within_the_ramp() -> None:
    """``main.tex`` declares the LaTeX side of the same palette.

    Reviewer-markup colours are excluded deliberately: they exist to make each
    reviewer's changes visible at a glance, which a monochrome ramp cannot do.
    """
    markup = {"revAcol", "revBcol", "revCcol", "revAllcol"}
    declared = re.findall(
        r"\\definecolor\{(\w+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}",
        (_PAPER / "main.tex").read_text(),
    )
    offenders = {name: value for name, value in declared if name not in markup and f"#{value.lower()}" not in RAMP}
    assert not offenders, f"colours outside the identity ramp: {offenders}"


@pytest.mark.skipif(not (_PAPER / "main.tex").exists(), reason="manuscript not present")
def test_table_captions_do_not_name_a_hue_the_palette_lacks() -> None:
    """Captions describe the shading's role; naming a hue goes stale on restyle."""
    text = (_PAPER / "main.tex").read_text().lower()
    for stale in ("khaki", "almond", "green cell"):
        assert stale not in text, f"caption still refers to {stale!r}"


# --------------------------------------------------------------------------- #
# Critical-difference diagram ramp
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _PAPER.exists(), reason="manuscript not present in this checkout")
@pytest.mark.parametrize("on_dark", [False, True])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8])
def test_cd_ramp_is_legible_and_ordered(n: int, on_dark: bool) -> None:
    """Rank colours must be readable as text and ordered by luminance.

    They tint the model labels, not only the markers, so ``RdYlGn`` cannot be
    sampled directly -- its midpoint is about 1.1:1 against the page. And the
    ordering has to be recoverable without hue, because red and green are the
    pair most often confused and because the paper may be printed in greyscale.
    """
    import matplotlib.colors as mcolors
    from paper.experiments._style import DARK_PAGE, _contrast, cd_colors

    background = mcolors.to_rgb(DARK_PAGE) if on_dark else (1.0, 1.0, 1.0)
    colours = cd_colors(n, on_dark=on_dark)
    assert len(colours) == n

    ratios = [_contrast(mcolors.to_rgb(c), background) for c in colours]
    for colour, ratio in zip(colours, ratios, strict=True):
        assert ratio >= WCAG_AA_TEXT, f"{colour} is {ratio:.1f}:1 on {'dark' if on_dark else 'light'}"
    # Best rank is the heaviest mark, and every step is strictly lighter.
    assert ratios == sorted(ratios, reverse=True)
    if n > 1:
        assert len(set(colours)) == n, "two ranks share a colour"


@pytest.mark.skipif(not _PAPER.exists(), reason="manuscript not present in this checkout")
def test_cd_ramp_does_not_collapse_onto_the_background() -> None:
    """Regression: the solver once returned the background colour itself.

    ``matplotlib`` colormaps hand back numpy scalars, so the bisection's
    ``too_far is not lighten`` test compared a ``numpy.bool_`` by identity --
    always true, because ``numpy.False_ is not False``. Every rank converged on
    near-white at about 1.0:1.
    """
    import matplotlib.colors as mcolors
    from paper.experiments._style import cd_colors

    for colour in cd_colors(4):
        r, g, b = mcolors.to_rgb(colour)
        assert min(r, g, b) < 0.9, f"{colour} has collapsed onto the page"


@pytest.mark.skipif(not _PAPER.exists(), reason="manuscript not present in this checkout")
def test_cd_ramp_runs_green_to_red() -> None:
    """Best rank green, worst red -- the diagram should read like its axis."""
    import matplotlib.colors as mcolors
    from paper.experiments._style import cd_colors

    best, worst = cd_colors(4)[0], cd_colors(4)[-1]
    br, bg, bb = mcolors.to_rgb(best)
    wr, wg, wb = mcolors.to_rgb(worst)
    assert bg > br, f"best rank {best} is not green-dominant"
    assert bg > bb, f"best rank {best} is not green-dominant"
    assert wr > wg, f"worst rank {worst} is not red-dominant"
    assert wr > wb, f"worst rank {worst} is not red-dominant"


@pytest.mark.skipif(not _PAPER.exists(), reason="manuscript not present in this checkout")
def test_epoch_outcome_classes_are_distinct_and_legible() -> None:
    """Figure 5's three outcome classes share the CD ramp, so the same rules apply."""
    import matplotlib.colors as mcolors
    from paper.experiments._style import EPOCH_OUTCOMES, _contrast, epoch_outcome_colors

    palette = epoch_outcome_colors()
    assert tuple(palette) == EPOCH_OUTCOMES
    assert len(set(palette.values())) == len(EPOCH_OUTCOMES)
    for name, colour in palette.items():
        ratio = _contrast(mcolors.to_rgb(colour), (1.0, 1.0, 1.0))
        assert ratio >= WCAG_AA_TEXT, f"{name} ({colour}) is {ratio:.1f}:1"


@pytest.mark.skipif(not _PAPER.exists(), reason="manuscript not present in this checkout")
@pytest.mark.parametrize(
    ("final", "best", "expected"),
    [
        (True, True, "robust"),
        (True, False, "inflated"),
        (False, False, "never"),
        # The mirror case: best-val finds a difference the fixed epoch missed.
        # That is the fixed rule *hiding* an effect, not inventing one, so it
        # must not be coloured red.
        (False, True, "robust"),
    ],
)
def test_epoch_outcome_classification(final: bool, best: bool, expected: str) -> None:
    from paper.experiments._style import classify_epoch_outcome

    assert classify_epoch_outcome(final, best) == expected
