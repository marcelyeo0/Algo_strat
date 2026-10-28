import json

import numpy as np

from common.utils.meta import config_hash, write_meta
from common.utils.paths import repo_root
from common.utils.seeds import set_global_seed


def test_repo_root_contains_pyproject():
    assert (repo_root() / "pyproject.toml").is_file()


def test_seed_reproducible():
    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_write_meta(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGO_RESULTS_DIR", str(tmp_path))
    run_dir = write_meta("unit_test", {"x": 1}, {"numpy": 1})
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["config"] == {"x": 1}
    assert meta["seeds"] == {"numpy": 1}
    assert "git_commit" in meta and "packages" in meta


def test_meta_header_shows_cost_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGO_RESULTS_DIR", str(tmp_path))
    run_dir = write_meta("unit_test", {"x": 1}, {"numpy": 1}, cost_profile="estimates")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert list(meta)[:2] == ["banner", "cost_profile"]
    assert meta["banner"].endswith("Résultats provisoires — coûts estimés")


def test_holdout_refused_with_estimates_even_with_confirmation():
    from typer.testing import CliRunner

    from common.cli import app

    res = CliRunner().invoke(app, ["holdout", "--strat", "gold_f2f", "--config", "x.yaml",
                                   "--cost-profile", "estimates", "--i-confirm"])
    assert res.exit_code == 2
    assert "Refused" in res.output
