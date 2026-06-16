from __future__ import annotations

import pytest

from canpgrid import resolve_region

LEVELS = [
    {"grid_size": [4, 4], "cell": [0, 0]},
    {"grid_size": [4, 4], "cell": [2, 1]},
    {"grid_size": [4, 4], "cell": [1, 3]},
]


def test_resolve_region_recursive_bbox() -> None:
    result = resolve_region(image_size=(1600, 1000), levels=LEVELS)
    bbox = result["bbox_on_original"]

    assert bbox["x1"] == pytest.approx(225)
    assert bbox["y1"] == pytest.approx(109.375)
    assert bbox["x2"] == pytest.approx(250)
    assert bbox["y2"] == pytest.approx(125)
    assert bbox["width"] == pytest.approx(25)
    assert bbox["height"] == pytest.approx(15.625)

