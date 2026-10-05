# `graphnetz.benchmark`

The benchmark layer: a curated task catalogue, a model registry, the runner
that executes every compatible *(task, model, seed)* triple, and the report
those runs return.

!!! tip "The report *is* the return type"

    `run_benchmark` does not return an accuracy table. It returns a
    [`BenchmarkReport`][graphnetz.benchmark.BenchmarkReport] whose methods are
    the statistical layer: per-cell intervals, corrected pairwise tests, the
    quantities that say when those tests are uninformative, and rank
    aggregation across tasks. See [Reading the report](../guides/report.md).

## Overview

::: graphnetz.benchmark
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

## Running a benchmark

::: graphnetz.benchmark.run_benchmark

::: graphnetz.benchmark.SearchSpace

## The report

The plotting methods (`plot`, `plot_forest`, `plot_pairwise`,
`plot_critical_difference`, `plot_learning_curves`) are mixed in from a
separate module to keep the statistics and the figure code apart; they are
part of the public surface and are documented here alongside the rest.

::: graphnetz.benchmark.BenchmarkReport
    options:
      inherited_members: true

## Tasks and models

::: graphnetz.benchmark.Task

::: graphnetz.benchmark.ModelSpec

::: graphnetz.benchmark.BENCHMARK_TASKS

::: graphnetz.benchmark.TASK_TYPES

::: graphnetz.benchmark.iter_benchmark_tasks

::: graphnetz.benchmark.task_from_dataset

::: graphnetz.benchmark.register_task

::: graphnetz.benchmark.unregister_task

::: graphnetz.benchmark.register_model

## Convenience plotting

::: graphnetz.benchmark.plot_benchmark
