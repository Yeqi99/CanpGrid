from __future__ import annotations

from pathlib import Path

from PIL import Image

from canpgrid import create_grid_view


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

