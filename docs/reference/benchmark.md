# `graphnetz.benchmark`

The benchmark layer: a curated task catalogue, a model registry, the runner
that executes every compatible *(task, model, seed)* triple, and the report
those runs return.

!!! tip "The report *is* the return type"

    `run_benchmark` does not return an accuracy table. It returns a
    [`BenchmarkReport`][graphnetz.benchmark.BenchmarkReport] whose methods are
    the statistical layer: per-cell intervals, corrected pairwise tests, the
    quantities that say when those tests are uninformative, and rank
    aggregation across tasks. See [Reading the report](../guide/report.md).

## Running a benchmark

::: graphnetz.benchmark
    options:
      members:
        - run_benchmark
        - SearchSpace

## The report

The plotting methods (`plot`, `plot_forest`, `plot_pairwise`,
`plot_critical_difference`, `plot_learning_curves`) are mixed in from a
separate module to keep the statistics and the figure code apart; they are
part of the public surface and are documented here alongside the rest.

::: graphnetz.benchmark
    options:
      members:
        - BenchmarkReport
      inherited_members: true

## Tasks and models

::: graphnetz.benchmark
    options:
      members:
        - Task
        - ModelSpec
        - BENCHMARK_TASKS
        - TASK_TYPES
        - iter_benchmark_tasks
        - task_from_dataset
        - register_task
        - unregister_task
        - register_model

## Convenience plotting

::: graphnetz.benchmark
    options:
      members:
        - plot_benchmark
