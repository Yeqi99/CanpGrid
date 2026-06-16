from __future__ import annotations

import pytest

from canpgrid import resolve_region


def test_cell_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="cell x must be"):
        resolve_region(
            image_size=(1600, 1000),
            levels=[{"grid_size": [4, 4], "cell": [4, 0]}],
        )


def test_grid_size_less_than_one_raises() -> None:
    with pytest.raises(ValueError, match="grid_size cols and rows must be >= 1"):
        resolve_region(
            image_size=(1600, 1000),
            levels=[{"grid_size": [0, 4], "cell": [0, 0]}],
        )

