from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .grid import validate_grid_size
from .schemas import BBox, DetailMode, GridSize, OverlayMode, RulerConfig

DETAIL_STYLES: dict[str, dict[str, int]] = {
    "coarse": {"grid_alpha": 150, "ruler_alpha": 80, "line_width": 2},
    "medium": {"grid_alpha": 130, "ruler_alpha": 70, "line_width": 2},
    "fine": {"grid_alpha": 115, "ruler_alpha": 60, "line_width": 1},
}

DEFAULT_RULERS: dict[str, RulerConfig] = {
    "coarse": RulerConfig(tick_x=8, tick_y=8, show_minor_ticks=True, show_labels_every=1),
    "medium": RulerConfig(tick_x=16, tick_y=16, show_minor_ticks=True, show_labels_every=2),
    "fine": RulerConfig(tick_x=24, tick_y=24, show_minor_ticks=True, show_labels_every=4),
}


def load_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def save_png(image: Image.Image, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def crop_and_zoom(image: Image.Image, bbox: BBox, zoom_factor: float = 3.0) -> Image.Image:
    if zoom_factor <= 0:
        raise ValueError("zoom_factor must be > 0")
    left = max(0, min(image.width, math.floor(bbox.x1)))
    top = max(0, min(image.height, math.floor(bbox.y1)))
    right = max(0, min(image.width, math.ceil(bbox.x2)))
    bottom = max(0, min(image.height, math.ceil(bbox.y2)))
    if right <= left or bottom <= top:
        raise ValueError("resolved bbox is empty after clamping to image bounds")

    cropped = image.crop((left, top, right, bottom))
    target_size = (
        max(1, round(cropped.width * zoom_factor)),
        max(1, round(cropped.height * zoom_factor)),
    )
    return cropped.resize(target_size, Image.Resampling.LANCZOS)


def draw_grid_overlay(
    image: Image.Image,
    grid_size: GridSize,
    *,
    detail_mode: DetailMode = "medium",
    show_axis_ticks: bool = True,
    show_crosshair: bool = False,
    level_label: str | None = None,
    path_label: str | None = None,
) -> Image.Image:
    cols, rows = validate_grid_size(grid_size)
    style = _style(detail_mode)
    result = image.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    grid_color = (0, 222, 255, style["grid_alpha"])
    major_color = (255, 255, 255, min(220, style["grid_alpha"] + 50))
    line_width = style["line_width"]

    for col in range(cols + 1):
        x = _scaled_position(col, cols, image.width)
        color = major_color if col in (0, cols) else grid_color
        draw.line([(x, 0), (x, image.height)], fill=color, width=line_width)
    for row in range(rows + 1):
        y = _scaled_position(row, rows, image.height)
        color = major_color if row in (0, rows) else grid_color
        draw.line([(0, y), (image.width, y)], fill=color, width=line_width)

    if show_crosshair:
        _draw_crosshair(draw, image.size)
    if show_axis_ticks:
        _draw_grid_axis_labels(draw, image.size, cols, rows)
    _draw_view_labels(draw, image.size, level_label, path_label)
    return Image.alpha_composite(result, overlay).convert("RGB")


def draw_ruler_overlay(
    image: Image.Image,
    *,
    detail_mode: DetailMode = "medium",
    ruler_config: RulerConfig | Mapping[str, object] | None = None,
    show_axis_ticks: bool = True,
    show_crosshair: bool = False,
    level_label: str | None = None,
    path_label: str | None = None,
) -> Image.Image:
    config = _ruler_config(detail_mode, ruler_config)
    style = _style(detail_mode)
    result = image.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    ruler_color = (255, 214, 71, style["ruler_alpha"])
    major_color = (255, 255, 255, min(210, style["ruler_alpha"] + 70))

    _draw_ruler_lines(draw, image.size, config, ruler_color, major_color)
    if show_crosshair:
        _draw_crosshair(draw, image.size)
    if show_axis_ticks:
        _draw_ruler_axis_labels(draw, image.size, config)
    _draw_view_labels(draw, image.size, level_label, path_label)
    return Image.alpha_composite(result, overlay).convert("RGB")


def draw_hybrid_overlay(
    image: Image.Image,
    grid_size: GridSize,
    *,
    detail_mode: DetailMode = "medium",
    ruler_config: RulerConfig | Mapping[str, object] | None = None,
    show_axis_ticks: bool = True,
    show_crosshair: bool = False,
    level_label: str | None = None,
    path_label: str | None = None,
) -> Image.Image:
    config = _ruler_config(detail_mode, ruler_config)
    with_ruler = draw_ruler_overlay(
        image,
        detail_mode=detail_mode,
        ruler_config=config,
        show_axis_ticks=False,
        show_crosshair=False,
    )
    return draw_grid_overlay(
        with_ruler,
        grid_size,
        detail_mode=detail_mode,
        show_axis_ticks=show_axis_ticks,
        show_crosshair=show_crosshair,
        level_label=level_label,
        path_label=path_label,
    )


def draw_overlay(
    image: Image.Image,
    overlay_mode: OverlayMode,
    grid_size: GridSize,
    *,
    detail_mode: DetailMode = "medium",
    ruler_config: RulerConfig | Mapping[str, object] | None = None,
    show_axis_ticks: bool = True,
    show_crosshair: bool = False,
    level_label: str | None = None,
    path_label: str | None = None,
) -> Image.Image:
    if overlay_mode == "grid":
        return draw_grid_overlay(
            image,
            grid_size,
            detail_mode=detail_mode,
            show_axis_ticks=show_axis_ticks,
            show_crosshair=show_crosshair,
            level_label=level_label,
            path_label=path_label,
        )
    if overlay_mode == "ruler":
        return draw_ruler_overlay(
            image,
            detail_mode=detail_mode,
            ruler_config=ruler_config,
            show_axis_ticks=show_axis_ticks,
            show_crosshair=show_crosshair,
            level_label=level_label,
            path_label=path_label,
        )
    if overlay_mode == "hybrid":
        return draw_hybrid_overlay(
            image,
            grid_size,
            detail_mode=detail_mode,
            ruler_config=ruler_config,
            show_axis_ticks=show_axis_ticks,
            show_crosshair=show_crosshair,
            level_label=level_label,
            path_label=path_label,
        )
    raise ValueError("overlay_mode must be one of: grid, ruler, hybrid")


def _style(detail_mode: str) -> dict[str, int]:
    if detail_mode not in DETAIL_STYLES:
        raise ValueError("detail_mode must be one of: coarse, medium, fine")
    return DETAIL_STYLES[detail_mode]


def _ruler_config(
    detail_mode: str, ruler_config: RulerConfig | Mapping[str, object] | None
) -> RulerConfig:
    if ruler_config is None:
        return DEFAULT_RULERS[detail_mode]
    if isinstance(ruler_config, RulerConfig):
        return ruler_config
    return RulerConfig.from_mapping(ruler_config)


def _scaled_position(index: int, parts: int, length: int) -> int:
    return round(index * (length - 1) / parts)


def _draw_grid_axis_labels(
    draw: ImageDraw.ImageDraw, size: tuple[int, int], cols: int, rows: int
) -> None:
    width, height = size
    for col in range(cols + 1):
        x = _scaled_position(col, cols, width)
        _draw_label(draw, str(col), (x + 3, 3))
    for row in range(rows + 1):
        y = _scaled_position(row, rows, height)
        _draw_label(draw, str(row), (3, y + 3))


def _draw_ruler_lines(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    config: RulerConfig,
    ruler_color: tuple[int, int, int, int],
    major_color: tuple[int, int, int, int],
) -> None:
    width, height = size
    for tick in range(config.tick_x + 1):
        x = _scaled_position(tick, config.tick_x, width)
        color = major_color if tick % config.show_labels_every == 0 else ruler_color
        line_len = 18 if tick % config.show_labels_every == 0 else 10
        if config.show_minor_ticks or tick % config.show_labels_every == 0:
            draw.line([(x, 0), (x, line_len)], fill=color, width=1)
            draw.line([(x, height - line_len), (x, height)], fill=color, width=1)
            guide_color = (color[0], color[1], color[2], max(25, color[3] // 3))
            draw.line([(x, 0), (x, height)], fill=guide_color, width=1)
    for tick in range(config.tick_y + 1):
        y = _scaled_position(tick, config.tick_y, height)
        color = major_color if tick % config.show_labels_every == 0 else ruler_color
        line_len = 18 if tick % config.show_labels_every == 0 else 10
        if config.show_minor_ticks or tick % config.show_labels_every == 0:
            draw.line([(0, y), (line_len, y)], fill=color, width=1)
            draw.line([(width - line_len, y), (width, y)], fill=color, width=1)
            guide_color = (color[0], color[1], color[2], max(25, color[3] // 3))
            draw.line([(0, y), (width, y)], fill=guide_color, width=1)


def _draw_ruler_axis_labels(
    draw: ImageDraw.ImageDraw, size: tuple[int, int], config: RulerConfig
) -> None:
    width, height = size
    for tick in range(config.tick_x + 1):
        if tick % config.show_labels_every != 0:
            continue
        x = _scaled_position(tick, config.tick_x, width)
        _draw_label(draw, str(tick), (x + 3, 3))
    for tick in range(config.tick_y + 1):
        if tick % config.show_labels_every != 0:
            continue
        y = _scaled_position(tick, config.tick_y, height)
        _draw_label(draw, str(tick), (3, y + 3))


def _draw_crosshair(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    width, height = size
    x = width // 2
    y = height // 2
    draw.line([(x, 0), (x, height)], fill=(255, 90, 90, 150), width=1)
    draw.line([(0, y), (width, y)], fill=(255, 90, 90, 150), width=1)


def _draw_view_labels(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    level_label: str | None,
    path_label: str | None,
) -> None:
    width, height = size
    if width < 240 or height < 120:
        return
    labels = [label for label in (level_label, path_label) if label]
    if not labels:
        return
    y = max(4, height - 22 * len(labels) - 4)
    for label in labels:
        _draw_label(draw, label, (max(4, width - 360), y))
        y += 20


def _draw_label(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int]) -> None:
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    padding = 2
    draw.rectangle(
        (left - padding, top - padding, right + padding, bottom + padding),
        fill=(0, 0, 0, 135),
    )
    draw.text(xy, text, fill=(255, 255, 255, 235), font=font)
