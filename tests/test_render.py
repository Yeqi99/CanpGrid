from __future__ import annotations

from pathlib import Path

from PIL import Image

from canpgrid import create_grid_view, preview_point


def test_render_smoke_grid_ruler_hybrid(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (320, 200), "#f4f6fb").save(image_path)

    for mode in ("grid", "ruler", "hybrid"):
        result = create_grid_view(
            image_path,
            grid_size=[4, 3],
            overlay_mode=mode,
            ruler_config={"tick_x": 8, "tick_y": 8},
            out_dir=tmp_path / mode,
        )
        annotated = Path(result["annotated_image_path"])

        assert annotated.exists()
        assert annotated.suffix == ".png"
        with Image.open(annotated) as generated:
            assert generated.size == (320, 200)


def test_preview_point_current_and_both(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (320, 200), "#f4f6fb").save(image_path)
    levels = [{"grid_size": [4, 4], "cell": [1, 1]}]
    point_spec = {"type": "normalized_point", "value": ["1/2", "1/2"]}

    current = preview_point(
        image_path,
        levels,
        point_spec,
        preview_on="current_view",
        out_dir=tmp_path / "current",
    )
    current_path = Path(current["preview_image_path"])
    assert current_path.exists()
    assert current["point_on_original"] == [120, 75]
    assert "point_on_current_view" in current

    both = preview_point(
        image_path,
        levels,
        point_spec,
        preview_on="both",
        marker_style="ring_crosshair_inset",
        out_dir=tmp_path / "both",
    )
    paths = both["preview_image_paths"]
    assert Path(paths["current_view"]).exists()
    assert Path(paths["original_image"]).exists()
