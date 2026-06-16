from __future__ import annotations

import pytest

from canpgrid import resolve_point

LEVELS = [
    {"grid_size": [4, 4], "cell": [0, 0]},
    {"grid_size": [4, 4], "cell": [2, 1]},
    {"grid_size": [4, 4], "cell": [1, 3]},
]


def test_resolve_point_normalized_point() -> None:
    result = resolve_point(
        image_size=(1600, 1000),
        levels=LEVELS,
        point_spec={"type": "normalized_point", "value": ["1/2", "1/2"]},
    )

    assert result["point_on_original"][0] == pytest.approx(237.5)
    assert result["point_on_original"][1] == pytest.approx(117.1875)


def test_resolve_point_hybrid_point() -> None:
    result = resolve_point(
        image_size=(1600, 1000),
        levels=LEVELS,
        point_spec={
            "type": "hybrid_point",
            "base": ["1/2", "1/2"],
            "offset": [2, 3],
            "unit": "ruler_tick",
            "ruler_size": [16, 16],
        },
    )

    assert result["point_on_original"][0] == pytest.approx(240.625)
    assert result["point_on_original"][1] == pytest.approx(120.1171875)


def test_resolve_point_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        resolve_point(
            image_size=(1600, 1000),
            levels=LEVELS,
            point_spec={"type": "normalized_point", "value": ["1/3", "1/2"]},
        )


def test_resolve_point_rejects_invalid_ruler_size() -> None:
    with pytest.raises(ValueError, match="ruler_size values must be >= 1"):
        resolve_point(
            image_size=(1600, 1000),
            levels=LEVELS,
            point_spec={
                "type": "ruler_point",
                "origin": "top_left",
                "x": 1,
                "y": 1,
                "ruler_size": [0, 16],
            },
        )


def test_resolve_point_rejects_unknown_point_spec() -> None:
    with pytest.raises(ValueError, match="unknown point_spec type"):
        resolve_point(
            image_size=(1600, 1000),
            levels=LEVELS,
            point_spec={"type": "semantic_button"},
        )

