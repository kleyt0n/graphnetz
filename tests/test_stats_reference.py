"""Cross-check the statistical layer against independent references.

A framework whose selling point is that it does the statistics for you owes
the reader evidence that it does them *correctly*.  Every quantity the report
emits is checked here against either an established implementation (scipy,
statsmodels) or a published worked example, so "trust the integration" is a
testable claim rather than an assertion.
"""

from __future__ import annotations

import itertools
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from graphnetz.benchmark._power import (
    equivalence_verdict,
    minimum_detectable_effect,
    observed_power,
    paired_tost,
    seeds_for_effect,
)
from graphnetz.benchmark._stats import (
    _bootstrap_ci_half_width,
    _ci_half_width,
    _friedman_chi2,
    _holm_correction,
    _nemenyi_cd,
    _paired_pvalue,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240501)


# --------------------------------------------------------------------------- #
# Holm-Bonferroni
# --------------------------------------------------------------------------- #


def test_holm_matches_statsmodels(rng: np.random.Generator) -> None:
    """Our step-down Holm must equal statsmodels' ``holm`` adjustment."""
    pytest.importorskip("statsmodels", reason="optional cross-check dependency")
    from statsmodels.stats.multitest import multipletests

    for _ in range(20):
        p = rng.uniform(0, 1, size=rng.integers(2, 15))
        ours = _holm_correction(p)
        theirs = multipletests(p, method="holm")[1]
        np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)


def test_holm_excludes_nan_from_the_family() -> None:
    """NaN tests are undefined, not significant, and must not inflate ``m``."""
    p = np.array([0.01, np.nan, 0.02])
    adj = _holm_correction(p)
    assert np.isnan(adj[1])
    # Family size is 2 (not 3), so the smallest p is multiplied by 2.
    assert adj[0] == pytest.approx(0.02)


def test_holm_is_monotone(rng: np.random.Generator) -> None:
    p = np.sort(rng.uniform(0, 1, 10))
    adj = _holm_correction(p)
    assert np.all(np.diff(adj) >= -1e-15)
    assert np.all(adj <= 1.0)


# --------------------------------------------------------------------------- #
# Paired tests
# --------------------------------------------------------------------------- #


def test_paired_t_matches_scipy(rng: np.random.Generator) -> None:
    a, b = rng.normal(0.8, 0.02, 10), rng.normal(0.79, 0.02, 10)
    assert _paired_pvalue(a, b, "t") == pytest.approx(float(stats.ttest_rel(a, b).pvalue))


def test_wilcoxon_matches_scipy(rng: np.random.Generator) -> None:
    a, b = rng.normal(0.8, 0.02, 10), rng.normal(0.79, 0.02, 10)
    expected = float(stats.wilcoxon(a - b, zero_method="wilcox").pvalue)
    assert _paired_pvalue(a, b, "wilcoxon") == pytest.approx(expected)


def test_wilcoxon_exact_floor_at_ten_seeds() -> None:
    """Section 3.5's claim: the smallest exact two-sided p at S=10 is 2/2^10."""
    d = np.arange(1.0, 11.0)  # all differences share a sign
    p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    assert p == pytest.approx(2.0 * 2.0**-10)
    assert p == pytest.approx(0.001953125)


def test_wilcoxon_all_zero_differences_is_undefined() -> None:
    a = np.full(10, 0.5)
    assert np.isnan(_paired_pvalue(a, a, "wilcoxon"))


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #


def test_t_ci_matches_closed_form(rng: np.random.Generator) -> None:
    x = rng.normal(0.7, 0.05, 12)
    expected = float(stats.sem(x) * stats.t.ppf(0.975, x.size - 1))
    assert _ci_half_width(x, 0.95) == pytest.approx(expected)


def test_t_ci_matches_scipy_interval(rng: np.random.Generator) -> None:
    x = rng.normal(0.7, 0.05, 12)
    lo, hi = stats.t.interval(0.95, x.size - 1, loc=x.mean(), scale=stats.sem(x))
    assert _ci_half_width(x, 0.95) == pytest.approx((hi - lo) / 2)


def test_bootstrap_ci_is_seed_reproducible(rng: np.random.Generator) -> None:
    x = rng.normal(0.7, 0.05, 12)
    a = _bootstrap_ci_half_width(x, 0.95, 2000, 7)
    b = _bootstrap_ci_half_width(x, 0.95, 2000, 7)
    assert a == b


def test_bootstrap_and_t_agree_for_gaussian_samples(rng: np.random.Generator) -> None:
    """At n=40 and Gaussian data the two envelopes should be close."""
    x = rng.normal(0.7, 0.05, 40)
    t_half = _ci_half_width(x, 0.95)
    boot = _bootstrap_ci_half_width(x, 0.95, 20000, 0)
    assert boot == pytest.approx(t_half, rel=0.15)


# --------------------------------------------------------------------------- #
# Friedman and Nemenyi
# --------------------------------------------------------------------------- #


def test_friedman_chi2_matches_scipy(rng: np.random.Generator) -> None:
    """Our statistic, computed from ranks, must equal scipy's on the raw data.

    scipy ranks internally, so feeding it a table whose rows are already the
    per-task orderings gives the same statistic our rank table produces.
    """
    scores = rng.normal(0, 1, size=(8, 4))
    ranks = np.array([stats.rankdata(row, method="average") for row in scores])
    expected = float(stats.friedmanchisquare(*scores.T).statistic)
    assert _friedman_chi2(ranks) == pytest.approx(expected)


def test_friedman_scales_linearly_in_n() -> None:
    """chi2 is linear in N for a fixed mean-rank profile (used for the N-vs-CD table)."""
    base = np.array([[1.0, 2.0, 3.0, 4.0]])
    one = _friedman_chi2(base)
    ten = _friedman_chi2(np.repeat(base, 10, axis=0))
    assert ten == pytest.approx(10 * one)


def test_nemenyi_cd_matches_demsar_worked_example() -> None:
    """Demšar (2006) Section 3.2.2: k=5 classifiers on N=14 datasets.

    The paper's Table 5 gives q_0.05 = 2.728 for five classifiers, so
    CD = 2.728 * sqrt(5*6/(6*14)) = 1.630 rank units.
    """
    cd = _nemenyi_cd(5, 14, 0.05)
    assert cd == pytest.approx(1.630, abs=0.005)


def test_nemenyi_q_matches_demsar_table() -> None:
    """The q_alpha column of Demšar (2006) Table 5, for alpha = 0.05."""
    published = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 10: 3.164}
    for k, q in published.items():
        # CD = q sqrt(k(k+1)/6N)  =>  q = CD / sqrt(k(k+1)/6N)
        recovered = _nemenyi_cd(k, 10, 0.05) / np.sqrt(k * (k + 1) / 60)
        assert recovered == pytest.approx(q, abs=0.003), f"k={k}"


def test_nemenyi_cd_shrinks_as_one_over_sqrt_n() -> None:
    assert _nemenyi_cd(4, 40, 0.05) == pytest.approx(_nemenyi_cd(4, 10, 0.05) / 2.0)


# --------------------------------------------------------------------------- #
# Power and minimum detectable effect
# --------------------------------------------------------------------------- #


def test_mde_matches_closed_form() -> None:
    sigma, n = 0.0138, 10
    expected = (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * sigma / np.sqrt(n)
    assert minimum_detectable_effect(sigma, n) == pytest.approx(expected)


def test_mde_shrinks_with_more_seeds() -> None:
    mdes = [minimum_detectable_effect(0.02, n) for n in (5, 10, 20, 50)]
    assert all(a > b for a, b in itertools.pairwise(mdes))


def test_mde_is_nan_without_a_variance_estimate() -> None:
    assert np.isnan(minimum_detectable_effect(0.02, 1))


def test_power_at_the_mde_is_the_target() -> None:
    """By construction, an effect exactly at the MDE has ~the target power."""
    sigma, n = 0.02, 12
    mde = minimum_detectable_effect(sigma, n, 0.05, 0.80)
    assert observed_power(mde, sigma, n, 0.05) == pytest.approx(0.80, abs=0.02)


def test_power_is_monotone_in_effect_and_n() -> None:
    assert observed_power(0.01, 0.02, 10) < observed_power(0.03, 0.02, 10)
    assert observed_power(0.02, 0.02, 8) < observed_power(0.02, 0.02, 30)


def test_power_matches_statsmodels_ttest_power() -> None:
    pytest.importorskip("statsmodels", reason="optional cross-check dependency")
    from statsmodels.stats.power import TTestPower

    effect, sigma, n = 0.02, 0.03, 15
    expected = float(TTestPower().power(effect_size=effect / sigma, nobs=n, alpha=0.05, alternative="two-sided"))
    assert observed_power(effect, sigma, n, 0.05) == pytest.approx(expected, abs=0.01)


def test_seeds_for_effect_inverts_the_mde() -> None:
    sigma, target = 0.0138, 0.005
    n = seeds_for_effect(target, sigma)
    assert minimum_detectable_effect(sigma, n) <= target
    assert minimum_detectable_effect(sigma, n - 1) > target


# --------------------------------------------------------------------------- #
# Equivalence (TOST)
# --------------------------------------------------------------------------- #


def test_tost_matches_statsmodels(rng: np.random.Generator) -> None:
    pytest.importorskip("statsmodels", reason="optional cross-check dependency")
    from statsmodels.stats.weightstats import ttost_paired

    a, b = rng.normal(0.80, 0.01, 12), rng.normal(0.802, 0.01, 12)
    expected = float(ttost_paired(a, b, -0.02, 0.02)[0])
    assert paired_tost(a, b, 0.02) == pytest.approx(expected, rel=1e-9)


def test_tost_declares_identical_samples_equivalent(rng: np.random.Generator) -> None:
    a = rng.normal(0.8, 0.01, 10)
    assert paired_tost(a, a, 0.01) < 0.05


def test_tost_refuses_a_large_separation(rng: np.random.Generator) -> None:
    a = rng.normal(0.8, 0.005, 10)
    assert paired_tost(a, a + 0.10, 0.01) > 0.05


def test_tost_rejects_a_nonpositive_margin() -> None:
    with pytest.raises(ValueError, match="margin must be positive"):
        paired_tost(np.zeros(5), np.ones(5), 0.0)


def test_tost_is_underpowered_at_wide_variance(rng: np.random.Generator) -> None:
    """A noisy cell should be 'undetermined': neither test rejects."""
    a = rng.normal(0.8, 0.20, 10)
    b = a + rng.normal(0.0, 0.20, 10)
    p_diff = _paired_pvalue(a, b, "t")
    p_eq = paired_tost(a, b, 0.01)
    assert p_diff > 0.05
    assert p_eq > 0.05
    assert equivalence_verdict(p_diff, p_eq) == "undetermined"


@pytest.mark.parametrize(
    ("p_diff", "p_eq", "expected"),
    [
        (0.001, 0.900, "different"),
        (0.900, 0.001, "equivalent"),
        (0.900, 0.900, "undetermined"),
        (0.001, 0.001, "trivial"),
        (np.nan, 0.001, "undetermined"),
    ],
)
def test_equivalence_verdict_table(p_diff: float, p_eq: float, expected: str) -> None:
    assert equivalence_verdict(p_diff, p_eq) == expected


# --------------------------------------------------------------------------- #
# Loader-audit classification
# --------------------------------------------------------------------------- #


def test_truncated_download_is_not_reported_as_a_broken_loader(tmp_path) -> None:
    """A zero-byte raw archive is a transport failure, not a catalogue defect.

    The audit table in the paper is evidence about which loaders work, so a
    misclassification here would become a false claim. Regression test for a
    real case: an interrupted download left a 0-byte ``roman_empire.npz`` and
    the resulting ``EOFError`` was first reported as ``error``.
    """
    from graphnetz.datasets._validate import _classify, _download_incomplete

    raw = tmp_path / "roman_empire" / "raw"
    raw.mkdir(parents=True)
    (raw / "roman_empire.npz").write_bytes(b"")

    assert _download_incomplete(str(tmp_path))
    assert _classify(EOFError("No data left in file"), str(tmp_path)) == "download-failed"
    # Even an uninformative exception is caught by the zero-byte probe.
    assert _classify(RuntimeError("something odd"), str(tmp_path)) == "download-failed"


def test_a_stalled_download_is_not_reported_as_a_broken_loader(tmp_path) -> None:
    """A fetch that timed out leaves an empty directory, not a defect.

    Second regression case from the catalogue audit: ``elliptic_bitcoin``
    raised ``FSTimeoutError`` after five minutes having downloaded nothing, and
    was first reported as ``error``. The zero-byte probe missed it because
    there were no files at all.
    """
    from graphnetz.datasets._validate import _classify, _download_incomplete

    empty = tmp_path / "elliptic_bitcoin"
    empty.mkdir()

    assert _download_incomplete(str(empty))
    assert _classify(TimeoutError("read timed out"), str(empty)) == "download-failed"
    # And from the message alone, without the filesystem probe.
    assert _classify(RuntimeError("FSTimeoutError: "), None) == "download-failed"
    assert _classify(RuntimeError("Connection broken: IncompleteRead"), None) == "download-failed"


def test_missing_optional_dependency_is_not_an_error() -> None:
    from graphnetz.datasets._validate import _classify

    assert _classify(ImportError("No module named 'ogb'")) == "needs-extra"


def test_a_genuine_defect_is_still_reported_as_an_error(tmp_path) -> None:
    """With intact files on disk, a real exception must not be excused."""
    from graphnetz.datasets._validate import _classify

    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "data.bin").write_bytes(b"intact")
    assert _classify(ValueError("bad shape"), str(tmp_path)) == "error"


# --------------------------------------------------------------------------- #
# Loaders the audit found broken
# --------------------------------------------------------------------------- #


def test_a_transient_download_failure_is_retried_once(tmp_path) -> None:
    """One dropped connection is not evidence that a loader is broken."""
    import graphnetz.datasets._validate as validate

    calls = {"n": 0}

    def flaky(_root: str) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionResetError(54, "Connection reset by peer")
        return object()

    registry = {"toy": {"node_cls": [("flaky", flaky)]}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("graphnetz.datasets.LOADER_REGISTRY", registry, raising=False)
        frame = validate.validate_loaders(["toy"], root=str(tmp_path / "validate"))

    assert calls["n"] == 2, "the loader was not retried"
    assert frame.iloc[0]["status"] == "ok"
    assert frame.iloc[0]["error"] == "", "a recovered loader must not carry a stale error"


def test_a_real_defect_is_not_retried(tmp_path) -> None:
    """Retrying a genuine bug just fails again; spend the time downloading."""
    import graphnetz.datasets._validate as validate

    calls = {"n": 0}

    def broken(_root: str) -> object:
        calls["n"] += 1
        raise ValueError("bad shape")

    target = tmp_path / "validate" / "toy" / "broken"
    target.mkdir(parents=True)
    (target / "data.bin").write_bytes(b"intact")

    registry = {"toy": {"node_cls": [("broken", broken)]}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("graphnetz.datasets.LOADER_REGISTRY", registry, raising=False)
        frame = validate.validate_loaders(["toy"], root=str(tmp_path / "validate"))

    assert calls["n"] == 1
    assert frame.iloc[0]["status"] == "error"


def test_a_dropped_connection_is_recognised_by_type_not_wording(tmp_path) -> None:
    """``BrokenPipeError`` says "[Errno 32] Broken pipe" and nothing more.

    Its message names neither a connection nor a download, so matching on text
    alone filed ModelNet10 -- whose transfer died 374 MB into a 451 MB archive --
    as a catalogue defect. Every ``ConnectionError`` subclass is transport.
    """
    from graphnetz.datasets._validate import _classify

    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "partial.zip").write_bytes(b"substantial but incomplete")

    for exc in (BrokenPipeError(32, "Broken pipe"), ConnectionResetError(54, "Connection reset by peer")):
        assert _classify(exc, str(tmp_path)) == "download-failed", exc


def test_a_truncated_archive_is_cleared_so_a_retry_can_succeed(tmp_path) -> None:
    """A poisoned cache must not make a loader permanently unusable.

    OGB's downloader logs ``Using exist file`` and reuses whatever archive is on
    disk without checking it, so one interrupted transfer made ``ogbn_products``
    fail identically forever. Clearing the unreadable archive is what turns that
    into a retryable failure.
    """
    from graphnetz.datasets._validate import _purge_partial_archives

    (tmp_path / "products.zip").write_bytes(b"not really a zip")
    (tmp_path / "empty.npz").write_bytes(b"")
    (tmp_path / "processed.pt").write_bytes(b"keep me")

    cleared = _purge_partial_archives(str(tmp_path), "BadZipFile: File is not a zip file")

    assert sorted(cleared) == ["empty.npz", "products.zip"]
    assert not (tmp_path / "products.zip").exists()
    # Non-archive payloads are never touched.
    assert (tmp_path / "processed.pt").read_bytes() == b"keep me"


def test_intact_archives_survive_an_unrelated_failure(tmp_path) -> None:
    """Only a *corrupt-archive* symptom licenses deleting an archive."""
    from graphnetz.datasets._validate import _purge_partial_archives

    (tmp_path / "data.zip").write_bytes(b"PK\x03\x04 plausibly fine")
    cleared = _purge_partial_archives(str(tmp_path), "FSTimeoutError: read timed out")
    assert cleared == []
    assert (tmp_path / "data.zip").exists()


def test_a_library_incompatibility_is_not_excused_as_a_download_failure() -> None:
    """A complete download that we then fail to read is our defect, not theirs.

    ``UnpicklingError`` had been treated as a truncated-archive symptom, so when
    PyTorch 2.6 flipped ``torch.load(weights_only=...)`` to ``True`` and every
    ``ogbg-*`` loader stopped reading its own cache, the audit reported the
    breakage as a failed download. That is the classifier's central error in
    reverse: it published our bug as somebody else's dead mirror.
    """
    import pickle

    from graphnetz.datasets._validate import _classify

    weights_only = pickle.UnpicklingError(
        "Weights only load failed. In PyTorch 2.6, we changed the default value "
        "of the `weights_only` argument in `torch.load` from `False` to `True`."
    )
    assert _classify(weights_only) == "error"
    # ...even with no files on disk, where the incomplete-download probe would
    # otherwise excuse it.
    assert _classify(weights_only, "/nonexistent/root") == "error"

    # A genuinely truncated archive is still excused.
    assert _classify(pickle.UnpicklingError("Ran out of input")) == "download-failed"


def test_validate_loaders_reports_each_row_as_it_finishes() -> None:
    """Progress must reach the caller during the walk, not only at the end.

    The audit downloads tens of gigabytes over hours, so it is normally
    interrupted rather than completed. Returning results only on completion
    discarded every status the run had established, even though the archives
    were already cached on disk.
    """
    import graphnetz.datasets._validate as validate

    seen: list[str] = []
    calls = {"n": 0}

    def loader(_root: str) -> object:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("second loader is broken")
        return object()

    registry = {"toy": {"node_cls": [(f"loader_{i}", loader) for i in range(3)]}}
    root = Path(tempfile.mkdtemp()) / "validate"
    # Give the failing loader an intact cache, so its exception is judged a real
    # defect rather than excused as an incomplete download.
    (root / "toy" / "loader_1").mkdir(parents=True)
    (root / "toy" / "loader_1" / "data.bin").write_bytes(b"intact")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("graphnetz.datasets.LOADER_REGISTRY", registry, raising=False)
        frame = validate.validate_loaders(
            ["toy"],
            root=str(root),
            on_row=lambda row: seen.append(f"{row['loader']}:{row['status']}"),
        )

    # One callback per loader, delivered before the call returned, and the
    # failure is reported through the hook rather than aborting the walk.
    assert len(seen) == len(frame) == 3
    assert any(s.endswith(":error") for s in seen), seen


def test_product_space_requests_a_network_that_exists() -> None:
    """``product_space`` holds two networks, and neither is called that.

    The loader asked Netzschleuder for a network named ``product_space`` inside
    the dataset ``product_space``; the catalogue publishes ``SITC`` and ``HS``,
    so every call returned HTTP 404. This pins the URL shape without touching
    the network.
    """
    from graphnetz.datasets import finance

    for network, expected in (("SITC", "SITC"), ("HS", "HS")):
        url = _product_space_url(finance, network)
        assert url.endswith(f"/net/product_space/files/{expected}.csv.zip")

    # The default must be one of the two real networks, never the dataset name.
    assert _product_space_url(finance).endswith("/SITC.csv.zip")


def _product_space_url(finance, network: str | None = None) -> str:
    """Build the URL ``product_space`` would fetch, without constructing it."""
    import inspect

    from graphnetz.datasets._netz import NETZ_FILES

    default = inspect.signature(finance.product_space).parameters["network_name"].default
    return f"{NETZ_FILES}/product_space/files/{network or default}.csv.zip"


def test_ogb_download_confirmation_never_reads_stdin(monkeypatch) -> None:
    """OGB's >1 GB prompt must not reach ``input()`` from inside a loader.

    ``ogb.utils.url.decide_download`` calls ``input()``, and the caller treats a
    negative answer as ``exit(-1)``. Under a batch audit stdin is closed, so the
    prompt raised ``EOFError`` and ``ogbn_products`` was recorded as a download
    failure it never attempted.
    """
    pytest.importorskip("ogb")
    import ogb.nodeproppred.dataset as node_ds

    from graphnetz.datasets._ogb import _download_without_prompting

    def explode(*_args, **_kwargs):
        raise AssertionError("decide_download consulted stdin")

    monkeypatch.setattr("builtins.input", explode)
    monkeypatch.setattr("graphnetz.datasets._ogb._download_size_gb", lambda _url: 1.5)

    original = node_ds.decide_download
    with _download_without_prompting():
        assert node_ds.decide_download("https://example.invalid/huge.zip") is True
    assert node_ds.decide_download is original


def test_max_seconds_bounds_the_whole_audit_not_one_loader(monkeypatch) -> None:
    """``max_seconds`` is a budget for the call, not a per-loader timeout.

    The budget was measured against a clock each loader reset for its own
    timing, so the check only ever fired if a *single* loader outran the whole
    budget. An audit asked for two minutes would run for hours.
    """
    import graphnetz.datasets._validate as validate

    registry = {
        "toy": {"node_cls": [(f"loader_{i}", lambda root: object()) for i in range(10)]},
    }
    monkeypatch.setattr("graphnetz.datasets.LOADER_REGISTRY", registry, raising=False)

    # Each loader "takes" 1s of a 3s budget, so the walk must stop after ~3.
    clock = itertools.count(0.0, 1.0)
    monkeypatch.setattr(validate.time, "perf_counter", lambda: next(clock))

    frame = validate.validate_loaders(["toy"], root=str(tmp_root()), max_seconds=3.0)
    assert 0 < len(frame) < 10, f"budget ignored: audited {len(frame)} of 10 loaders"


def tmp_root() -> Path:
    """A path that does not exist, so no loader is skipped for being uncached."""
    return Path(tempfile.mkdtemp()) / "validate"


def test_ogb_prompt_is_restored_when_the_download_raises(monkeypatch) -> None:
    """The patch is scoped: a failing download must not leave OGB monkeyed."""
    pytest.importorskip("ogb")
    import ogb.nodeproppred.dataset as node_ds

    from graphnetz.datasets._ogb import _download_without_prompting

    original = node_ds.decide_download
    with pytest.raises(RuntimeError), _download_without_prompting():
        raise RuntimeError("download blew up")
    assert node_ds.decide_download is original
