from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from .schemas import GridSize

Density = Literal["coarse", "medium", "fine"]

DENSITY_CONFIGS: dict[str, dict[str, int]] = {
    "coarse": {
        "target_cell_size": 260,
        "min_cols": 3,
        "max_cols": 16,
        "min_rows": 3,
        "max_rows": 12,
    },
    "medium": {
        "target_cell_size": 160,
        "min_cols": 4,
        "max_cols": 24,
        "min_rows": 4,
        "max_rows": 18,
    },
    "fine": {
        "target_cell_size": 90,
        "min_cols": 6,
        "max_cols": 32,
        "min_rows": 6,
        "max_rows": 24,
    },
}


def suggest_grid_size(width: int, height: int, density: str = "medium") -> list[int]:
    if width < 1 or height < 1:
        raise ValueError("width and height must be >= 1")
    if density not in DENSITY_CONFIGS:
        raise ValueError(f"unknown density: {density!r}; expected one of {sorted(DENSITY_CONFIGS)}")

    config = DENSITY_CONFIGS[density]
    target = config["target_cell_size"]
    cols = _clamp(round(width / target), config["min_cols"], config["max_cols"])
    rows = _clamp(round(height / target), config["min_rows"], config["max_rows"])
    return [cols, rows]


def validate_grid_size(grid_size: Sequence[int]) -> GridSize:
    if isinstance(grid_size, (str, bytes)) or len(grid_size) != 2:
        raise ValueError("grid_size must be a two-item sequence [cols, rows]")
    cols, rows = grid_size
    if not isinstance(cols, int) or isinstance(cols, bool):
        raise ValueError("grid_size cols must be an integer")
    if not isinstance(rows, int) or isinstance(rows, bool):
        raise ValueError("grid_size rows must be an integer")
    if cols < 1 or rows < 1:
        raise ValueError("grid_size cols and rows must be >= 1")
    return cols, rows


def validate_cell(cell: Sequence[int], grid_size: Sequence[int]) -> tuple[int, int]:
    if isinstance(cell, (str, bytes)) or len(cell) != 2:
        raise ValueError("cell must be a two-item sequence [x, y]")
    cols, rows = validate_grid_size(grid_size)
    cell_x, cell_y = cell
    if not isinstance(cell_x, int) or isinstance(cell_x, bool):
        raise ValueError("cell x must be an integer")
    if not isinstance(cell_y, int) or isinstance(cell_y, bool):
        raise ValueError("cell y must be an integer")
    if cell_x < 0 or cell_x >= cols:
        raise ValueError(f"cell x must be in [0, {cols - 1}]")
    if cell_y < 0 or cell_y >= rows:
        raise ValueError(f"cell y must be in [0, {rows - 1}]")
    return cell_x, cell_y


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))

