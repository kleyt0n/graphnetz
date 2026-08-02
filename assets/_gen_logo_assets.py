"""Derive every logo asset from one geometric definition.

The mark is a four-node graph on a strict grid: a diamond of three hollow
nodes plus one solid node, joined by five hairline edges.  It is defined here
as geometry rather than traced from a bitmap, so every output is a true vector
at heart and the whole identity is one edit away from changing.

Run:  uv run python assets/_gen_logo_assets.py

Outputs (all regenerated, never hand-edited):

    assets/logo.svg            Carbon Black ink, the canonical mark
    assets/logo-light.svg      Bright Snow ink, for dark grounds
    assets/logo.png            Carbon Black, transparent, 1024 px
    assets/logo-light.png      Bright Snow, transparent, 1024 px
    assets/logo-banner.svg     mark + wordmark, light scheme
    assets/logo-banner-dark.svg
    docs/logo.png              Bright Snow ink, for dark header/hero surfaces
    docs/logo-ink.png          Carbon Black ink, for light header/hero surfaces
    docs/favicon.png           Carbon Black on a Bright Snow plate

The docs need both inks because the header and hero follow the colour scheme:
a single white mark vanishes on a light header, and a single dark one vanishes
on a dark header. The stylesheet picks between them per scheme.

Design constraints that the numbers below encode:

* **One stroke weight.**  Edges and node rings share ``STROKE``.  A single
  weight is what makes a small mark read as drawn rather than assembled.
* **Edges stop short of nodes.**  Each segment is trimmed by the node's outer
  radius plus ``GAP``, so the nodes stay legible as discrete vertices instead
  of fusing into the lines at small sizes.
* **The solid node matches the rings' outer diameter.**  Filled and hollow
  vertices then sit on the same optical grid.

``assets/logo-source.png`` (the old traced bitmap) is no longer read by this
script and can be deleted.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

# graphnetz.plotting.BRAND_COLORS
SNOW = "#f8f9fa"
CARBON = "#212529"

# ------------------------------------------------------------------ #
# Mark geometry, in a 120x120 design box.                             #
# ------------------------------------------------------------------ #
BOX = 120
NODES: dict[str, tuple[float, float]] = {
    "top": (60.0, 16.0),
    "left": (16.0, 60.0),
    "right": (104.0, 60.0),
    "bottom": (60.0, 104.0),
}
EDGES = [
    ("top", "left"),
    ("top", "right"),
    ("left", "bottom"),
    ("right", "bottom"),
    ("left", "right"),
]
SOLID = "left"  # the one filled vertex

RING = 11.0  # radius of a hollow node's stroke centreline
STROKE = 5.0  # the single weight, shared by edges and rings
DISC = RING + STROKE / 2  # 13.5 — outer radius, and the solid node's radius
GAP = 3.0  # clear space between a node's outer edge and an incident edge
# A round cap projects STROKE/2 past the segment's endpoint, so the endpoint
# must be pulled back by that much again or the drawn line lands on the node.
TRIM = DISC + GAP + STROKE / 2
# Optical correction: a filled disc reads larger than a ring of the same outer
# diameter, because the eye weighs area, not extent. Drawn at the geometric
# radius the solid vertex looks like a blot next to the other three.
SOLID_R = DISC * 0.92


def _hex(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    return tuple(int(v[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def segments() -> list[tuple[float, float, float, float]]:
    """The five edges, each trimmed back to clear both of its endpoints."""
    out = []
    for a, b in EDGES:
        (x1, y1), (x2, y2) = NODES[a], NODES[b]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        out.append(
            (
                round(x1 + ux * TRIM, 2),
                round(y1 + uy * TRIM, 2),
                round(x2 - ux * TRIM, 2),
                round(y2 - uy * TRIM, 2),
            )
        )
    return out


# ------------------------------------------------------------------ #
# SVG                                                                 #
# ------------------------------------------------------------------ #
def mark_svg_body(ink: str, indent: str = "  ") -> str:
    """The mark's SVG elements, in the 120x120 design box."""
    lines = [
        f'{indent}<g fill="none" stroke="{ink}" stroke-width="{STROKE}" stroke-linecap="round">',
    ]
    for x1, y1, x2, y2 in segments():
        lines.append(f'{indent}  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>')
    for name, (cx, cy) in NODES.items():
        if name != SOLID:
            lines.append(f'{indent}  <circle cx="{cx}" cy="{cy}" r="{RING}"/>')
    lines.append(f"{indent}</g>")
    sx, sy = NODES[SOLID]
    lines.append(f'{indent}<circle cx="{sx}" cy="{sy}" r="{SOLID_R:.2f}" fill="{ink}"/>')
    return "\n".join(lines)


def write_mark_svg(ink: str, path: Path) -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {BOX} {BOX}" width="{BOX}" height="{BOX}" role="img" aria-label="GraphNetz">
  <title>GraphNetz</title>
  <!-- Generated by assets/_gen_logo_assets.py; do not edit by hand. -->
{mark_svg_body(ink)}
</svg>
"""
    path.write_text(svg)


# Banner layout. The mark is scaled into a 64 px square at the left, the
# wordmark sits on the mark's optical centreline.
BANNER_H = 88
MARK_PX = 64
MARK_XY = (8, 12)
TEXT_X = 88
TEXT_SIZE = 42
TEXT_BASELINE = 59
TEXT_TRACKING = -0.6
# Mean glyph advance for "GraphNetz", in em, measured by rasterising this
# banner. Deliberately measured against the *fallback* face (Helvetica/Arial),
# not Google Sans Flex: GitHub renders the SVG in an <img> sandbox where a
# webfont cannot load, so the fallback is what actually sets the width.
ADVANCE = 0.5407
BANNER_W = round(TEXT_X + len("GraphNetz") * TEXT_SIZE * ADVANCE + 12)


def write_banner_svg(ink: str, path: Path) -> None:
    """Mark + wordmark, as pure vector.

    The old banner embedded the mark as a base64 PNG because GitHub renders an
    SVG inside an ``<img>`` sandbox where external references do not load. A
    self-contained vector needs no such workaround, and costs ~1 KB instead
    of ~17 KB.
    """
    scale = MARK_PX / BOX
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {BANNER_W} {BANNER_H}" width="{BANNER_W}" height="{BANNER_H}" role="img" aria-label="GraphNetz">
  <title>GraphNetz</title>
  <!-- Generated by assets/_gen_logo_assets.py; do not edit by hand.
       Ink {ink} from graphnetz.plotting.BRAND_COLORS. -->
  <g transform="translate({MARK_XY[0]},{MARK_XY[1]}) scale({scale:.6f})">
{mark_svg_body(ink, indent="    ")}
  </g>
  <text x="{TEXT_X}" y="{TEXT_BASELINE}" font-family="'Google Sans Flex', 'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="{TEXT_SIZE}" font-weight="600" fill="{ink}" letter-spacing="{TEXT_TRACKING}">GraphNetz</text>
</svg>
"""
    path.write_text(svg)


# ------------------------------------------------------------------ #
# PNG                                                                 #
# ------------------------------------------------------------------ #
SS = 4  # supersample factor; PIL's draw primitives do not antialias


def _round_line(draw: ImageDraw.ImageDraw, seg, ink, width: float) -> None:
    """A line with round caps, which ImageDraw does not provide natively."""
    x1, y1, x2, y2 = seg
    draw.line([(x1, y1), (x2, y2)], fill=ink, width=round(width))
    r = width / 2
    for cx, cy in ((x1, y1), (x2, y2)):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ink)


def render(colour: str, size: int) -> Image.Image:
    """The mark drawn in ``colour`` on transparency, at ``size`` px square."""
    ink = (*_hex(colour), 255)
    big = size * SS
    img = Image.new("RGBA", (big, big), (*_hex(colour), 0))
    draw = ImageDraw.Draw(img)
    k = big / BOX

    for x1, y1, x2, y2 in segments():
        _round_line(draw, (x1 * k, y1 * k, x2 * k, y2 * k), ink, STROKE * k)
    for name, (cx, cy) in NODES.items():
        r = RING * k
        if name == SOLID:
            r = SOLID_R * k
            draw.ellipse([cx * k - r, cy * k - r, cx * k + r, cy * k + r], fill=ink)
        else:
            draw.ellipse(
                [cx * k - r, cy * k - r, cx * k + r, cy * k + r],
                outline=ink,
                width=round(STROKE * k),
            )
    return img.resize((size, size), Image.LANCZOS)


def plated(ink: str, plate: str, size: int) -> Image.Image:
    """The mark on an opaque rounded plate, for favicon legibility.

    A transparent favicon disappears against a tab bar that happens to match
    the ink, so the icon carries its own background.
    """
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    radius = int(size * 0.18)
    ImageDraw.Draw(out).rounded_rectangle([0, 0, size - 1, size - 1], radius, fill=(*_hex(plate), 255))
    mark = render(ink, int(size * 0.78))
    off = (size - mark.width) // 2
    out.alpha_composite(mark, (off, off))
    return out


def main() -> None:
    svgs = [
        (ROOT / "assets" / "logo.svg", CARBON, write_mark_svg),
        (ROOT / "assets" / "logo-light.svg", SNOW, write_mark_svg),
        (ROOT / "assets" / "logo-banner.svg", CARBON, write_banner_svg),
        (ROOT / "assets" / "logo-banner-dark.svg", SNOW, write_banner_svg),
    ]
    for path, ink, writer in svgs:
        writer(ink, path)
        print(f"  wrote {path.relative_to(ROOT)}  {path.stat().st_size} B")

    pngs = [
        (ROOT / "assets" / "logo.png", render(CARBON, 1024)),
        (ROOT / "assets" / "logo-light.png", render(SNOW, 1024)),
        (ROOT / "docs" / "logo.png", render(SNOW, 512)),
        (ROOT / "docs" / "logo-ink.png", render(CARBON, 512)),
        (ROOT / "docs" / "favicon.png", plated(CARBON, SNOW, 512)),
    ]
    for path, img in pngs:
        img.save(path, format="PNG", optimize=True)
        print(f"  wrote {path.relative_to(ROOT)}  {img.size[0]}px  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
