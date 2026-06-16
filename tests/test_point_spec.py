from __future__ import annotations

import pytest
from PIL import Image

from canpgrid import preview_point, resolve_point

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


def test_resolve_cell_ruler_point() -> None:
    result = resolve_point(
        image_size=(900, 2000),
        levels=[],
        point_spec={
            "type": "cell_ruler_point",
            "grid_size": [9, 20],
            "cell": [7, 1],
            "x": 5,
            "y": 4,
            "ruler_size": [10, 10],
        },
    )

    assert result["point_on_original"] == [750, 140]
    assert result["final_region_bbox_on_original"] == {
        "x1": 700,
        "y1": 100,
        "x2": 800,
        "y2": 200,
        "width": 100,
        "height": 100,
    }


def test_resolve_color_snap_point_ray(tmp_path) -> None:
    image_path = tmp_path / "color-ray.png"
    image = Image.new("RGB", (20, 20), "#ffffff")
    for y in range(5, 16):
        image.putpixel((12, y), (255, 0, 0))
    image.save(image_path)

    result = resolve_point(
        image_path,
        [],
        {
            "type": "color_snap_point",
            "base": {"type": "normalized_point", "value": [0.5, 0.5]},
            "target_color": "#ff0000",
            "tolerance": 0,
            "search": {"mode": "ray", "direction": "right", "max_distance": 5},
        },
    )

    assert result["point_on_original"] == [12, 10]
    assert result["point_resolution"]["base_point_on_original"] == [10, 10]
    assert result["point_resolution"]["matched_color"] == [255, 0, 0]
    assert result["point_resolution"]["search_mode"] == "ray"


def test_resolve_color_snap_point_nearest(tmp_path) -> None:
    image_path = tmp_path / "color-nearest.png"
    image = Image.new("RGB", (20, 20), "#ffffff")
    image.putpixel((7, 8), (0, 200, 80))
    image.save(image_path)

    result = resolve_point(
        image_path,
        [],
        {
            "type": "color_snap_point",
            "base": {"type": "normalized_point", "value": [0.25, 0.25]},
            "target_color": [0, 196, 84],
            "tolerance": 8,
            "search": {"mode": "nearest", "radius": 5},
        },
    )

    assert result["point_on_original"] == [7, 8]
    assert result["point_resolution"]["search_mode"] == "nearest"
    assert result["point_resolution"]["matched"] is True


def test_resolve_color_snap_point_requires_image_path() -> None:
    with pytest.raises(ValueError, match="requires image_path"):
        resolve_point(
            image_size=(20, 20),
            levels=[],
            point_spec={
                "type": "color_snap_point",
                "base": {"type": "normalized_point", "value": [0.5, 0.5]},
                "target_color": "#ff0000",
            },
        )


def test_resolve_color_snap_point_raises_when_color_is_missing(tmp_path) -> None:
    image_path = tmp_path / "missing-color.png"
    Image.new("RGB", (20, 20), "#ffffff").save(image_path)

    with pytest.raises(ValueError, match="did not find target_color"):
        resolve_point(
            image_path,
            [],
            {
                "type": "color_snap_point",
                "base": {"type": "normalized_point", "value": [0.5, 0.5]},
                "target_color": "#ff0000",
                "search": {"mode": "nearest", "radius": 4},
            },
        )


def test_preview_color_snap_point_marks_snapped_point(tmp_path) -> None:
    image_path = tmp_path / "preview-color-snap.png"
    image = Image.new("RGB", (30, 30), "#ffffff")
    image.putpixel((18, 15), (0, 0, 255))
    image.save(image_path)

    result = preview_point(
        image_path,
        [],
        {
            "type": "color_snap_point",
            "base": {"type": "normalized_point", "value": [0.5, 0.5]},
            "target_color": "blue",
            "tolerance": 0,
            "search": {"mode": "ray", "direction": "right", "max_distance": 5},
        },
        preview_on="original_image",
        out_dir=tmp_path,
    )

    assert result["point_on_original"] == [18, 15]
    assert result["point_resolution"]["type"] == "color_snap_point"


def test_preview_point_rejects_unknown_marker_style(tmp_path) -> None:
    image_path = tmp_path / "sample.png"

    Image.new("RGB", (320, 200), "#ffffff").save(image_path)
    with pytest.raises(ValueError, match="marker_style"):
        preview_point(
            image_path,
            [],
            {"type": "normalized_point", "value": ["1/2", "1/2"]},
            marker_style="solid_dot",
            out_dir=tmp_path,
        )
