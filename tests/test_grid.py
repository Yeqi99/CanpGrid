from __future__ import annotations

import pytest

from canpgrid.grid import DENSITY_CONFIGS, suggest_grid_size


def test_suggest_grid_size_landscape_medium() -> None:
    cols, rows = suggest_grid_size(1920, 1080, "medium")

    assert cols > rows


def test_suggest_grid_size_portrait_medium() -> None:
    cols, rows = suggest_grid_size(1080, 2400, "medium")

    assert rows > cols


def test_suggest_grid_size_rejects_unknown_density() -> None:
    with pytest.raises(ValueError, match="unknown density"):
        suggest_grid_size(1920, 1080, "dense")


def test_suggest_grid_size_clamps_to_density_limits() -> None:
    for density, config in DENSITY_CONFIGS.items():
        cols, rows = suggest_grid_size(20_000, 20_000, density)

        assert cols <= config["max_cols"]
        assert rows <= config["max_rows"]

