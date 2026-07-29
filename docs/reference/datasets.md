# `graphnetz.datasets`

The catalogue: loader functions organised by research category and task type.
Every loader returns a PyTorch Geometric dataset, so anything in the catalogue
can be handed straight to [`run_benchmark`][graphnetz.benchmark.run_benchmark].

!!! info "Three different counts"

    The catalogue **contains** 62 loaders (57 without the optional `ogb`
    extra); [`validate_loaders`][graphnetz.datasets.validate_loaders] reports
    how many currently **load end to end**; and a subset is promoted to
    **curated benchmark tasks** in
    [`BENCHMARK_TASKS`][graphnetz.benchmark.BENCHMARK_TASKS]. See
    [Dataset taxonomy](../guide/datasets.md).

## Registry

::: graphnetz.datasets
    options:
      members:
        - LOADER_REGISTRY
        - CATEGORIES
        - list_datasets

## Auditing the catalogue

::: graphnetz.datasets
    options:
      members:
        - validate_loaders

## Netzschleuder access

::: graphnetz.datasets
    options:
      members:
        - Netz
        - download_all_networks_netz

## Per-category loaders

::: graphnetz.datasets.combinatorial

::: graphnetz.datasets.biology

::: graphnetz.datasets.social

::: graphnetz.datasets.knowledge

::: graphnetz.datasets.infrastructure

::: graphnetz.datasets.finance

::: graphnetz.datasets.computing

::: graphnetz.datasets.vision

::: graphnetz.datasets.physics

::: graphnetz.datasets.security
