from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canpgrid import create_grid_view, preview_point, resolve_point, resolve_region, zoom_region

SAMPLE = ROOT / "examples" / "sample_images" / "sample.png"
OUT_DIR = ROOT / "outputs" / "demo"


def main() -> None:
    ensure_sample_image(SAMPLE)

    first_view = create_grid_view(
        SAMPLE,
        grid_size=[12, 7],
        overlay_mode="grid",
        out_dir=OUT_DIR,
    )

    levels_1 = [{"grid_size": [12, 7], "cell": [6, 2]}]
    second_view = zoom_region(
        SAMPLE,
        levels_1,
        next_grid_size=[8, 6],
        overlay_mode="grid",
        out_dir=OUT_DIR,
        save_cropped=True,
    )

    levels_2 = [*levels_1, {"grid_size": [8, 6], "cell": [3, 4]}]
    final_view = zoom_region(
        SAMPLE,
        levels_2,
        next_grid_size=[8, 8],
        overlay_mode="hybrid",
        ruler_config={"tick_x": 16, "tick_y": 16},
        zoom_factor=24,
        out_dir=OUT_DIR,
    )

    region = resolve_region(SAMPLE, levels_2)
    normalized_point = resolve_point(
        SAMPLE,
        levels_2,
        {"type": "normalized_point", "value": ["1/2", "1/2"]},
    )
    hybrid_point = resolve_point(
        SAMPLE,
        levels_2,
        {
            "type": "hybrid_point",
            "base": ["1/2", "1/2"],
            "offset": [2, 3],
            "unit": "ruler_tick",
            "ruler_size": [16, 16],
        },
    )
    first_preview = preview_point(
        SAMPLE,
        levels_2,
        {
            "type": "hybrid_point",
            "base": ["1/2", "1/2"],
            "offset": [2, 3],
            "unit": "ruler_tick",
            "ruler_size": [16, 16],
        },
        preview_on="both",
        marker_style="ring_crosshair_inset",
        zoom_factor=24,
        out_dir=OUT_DIR,
    )
    adjusted_preview = preview_point(
        SAMPLE,
        levels_2,
        {
            "type": "hybrid_point",
            "base": ["1/2", "1/2"],
            "offset": [1, 2],
            "unit": "ruler_tick",
            "ruler_size": [16, 16],
        },
        preview_on="both",
        marker_style="ring_crosshair_inset",
        zoom_factor=24,
        out_dir=OUT_DIR,
    )

    print(
        json.dumps(
            {
                "sample_image": str(SAMPLE),
                "first_view": first_view,
                "second_view": second_view,
                "final_view": final_view,
                "region": region,
                "normalized_point": normalized_point,
                "hybrid_point": hybrid_point,
                "first_preview": first_preview,
                "adjusted_preview": adjusted_preview,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def ensure_sample_image(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1280, 720), "#f5f7fb")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, 1280, 72), fill="#172033")
    draw.text((32, 26), "CanpGrid sample canvas", fill="#ffffff", font=font)

    panels = [
        ((80, 120, 420, 320), "#d7ecff", "A"),
        ((470, 120, 820, 500), "#e7f8df", "B"),
        ((870, 120, 1180, 300), "#ffe7cc", "C"),
        ((120, 390, 380, 610), "#eadcff", "D"),
        ((880, 360, 1160, 620), "#ffdce5", "E"),
    ]
    for box, color, label in panels:
        draw.rounded_rectangle(box, radius=8, fill=color, outline="#42526e", width=2)
        draw.text((box[0] + 16, box[1] + 16), label, fill="#172033", font=font)

    for x in range(150, 1100, 130):
        draw.ellipse((x, 650, x + 28, 678), fill="#6172f3")

    image.save(path, format="PNG")


if __name__ == "__main__":
    main()
