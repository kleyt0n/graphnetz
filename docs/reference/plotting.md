# `graphnetz.plotting`

Figure helpers and the two palettes:

- [`BRAND_COLORS`][graphnetz.plotting.BRAND_COLORS] is the project's identity
  palette — a nine-step neutral ramp, light to dark — used by this site, the
  logo, and now the figures.
- [`NATURE_COLORS`][graphnetz.plotting.NATURE_COLORS] selects and **orders**
  seven of those steps for **plot series**.

The two used to be independent, so that restyling the site could not change a
published figure. They now share the ramp, which means the guarantee has to come
from somewhere else: `NATURE_COLORS` names its steps explicitly rather than
slicing `BRAND_COLORS` programmatically, so adding or reordering an identity
colour cannot silently repaint a figure — a figure's colours only change when
this tuple is edited.

Ordering carries the weight that hue used to. The first four steps all clear
4.5:1 against the page, because a series colour also tints markers and, in a
critical-difference diagram, the model labels themselves; the three lightest
steps come last and are reached only by a plot with five or more series, where
they appear as bar fills stroked in ink. Luminance is a weaker cue than hue, so
prefer a marker or a direct label over colour alone when a series must be
identified at a glance.

## Palettes

::: graphnetz.plotting
    options:
      members:
        - BRAND_COLORS
        - NATURE_COLORS
        - NATURE_RC
        - COLUMN_INCHES

## Figure helpers

::: graphnetz.plotting
    options:
      members:
        - set_plot_style
        - figure
        - save_figure
        - panel_label
        - pretty_metric

## Plot builders

::: graphnetz.plotting
    options:
      members:
        - plot_history
        - plot_grouped_bars
