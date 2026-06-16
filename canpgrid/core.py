from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .geometry import resolve_levels_to_bbox, resolve_point_in_bbox
from .grid import suggest_grid_size, validate_grid_size
from .render import (
    crop_and_zoom,
    draw_overlay,
    load_image,
    save_png,
)
from .schemas import (
    BBox,
    DetailMode,
    GridViewResult,
    Level,
    OverlayMode,
    PointResult,
    RegionResult,
    RulerConfig,
    coerce_levels,
)


def create_grid_view(
    image_path: str | Path,
    *,
    density: str = "medium",
    grid_size: Sequence[int] | None = None,
    overlay_mode: OverlayMode = "grid",
    detail_mode: DetailMode = "medium",
    ruler_config: Mapping[str, Any] | RulerConfig | None = None,
    show_axis_ticks: bool = True,
    show_crosshair: bool = False,
    out_dir: str | Path = "outputs",
) -> dict[str, Any]:
    image = load_image(image_path)
    resolved_grid_size = _grid_for_canvas(image.width, image.height, density, grid_size)
    view_id = _view_id()
    annotated = draw_overlay(
        image,
        overlay_mode,
        resolved_grid_size,
        detail_mode=detail_mode,
        ruler_config=ruler_config,
        show_axis_ticks=show_axis_ticks,
        show_crosshair=show_crosshair,
        level_label="level 0",
    )
    annotated_path = save_png(
        annotated,
        Path(out_dir) / f"{view_id}_level0_{overlay_mode}.png",
    )
    result = GridViewResult(
        view_id=view_id,
        image_width=image.width,
        image_height=image.height,
        level=0,
        grid_size=resolved_grid_size,
        bbox_on_original=BBox(0, 0, image.width, image.height),
        annotated_image_path=annotated_path,
    )
    return result.to_dict()


def zoom_region(
    image_path: str | Path,
    levels: Sequence[Mapping[str, Any] | Level],
    *,
    next_density: str = "medium",
    next_grid_size: Sequence[int] | None = None,
    overlay_mode: OverlayMode = "grid",
    detail_mode: DetailMode = "medium",
    ruler_config: Mapping[str, Any] | RulerConfig | None = None,
    zoom_factor: float = 3,
    show_axis_ticks: bool = True,
    out_dir: str | Path = "outputs",
    save_cropped: bool = False,
) -> dict[str, Any]:
    original = load_image(image_path)
    resolved_levels = coerce_levels(levels)
    bbox = resolve_levels_to_bbox((original.width, original.height), resolved_levels)
    zoomed = crop_and_zoom(original, bbox, zoom_factor=zoom_factor)
    resolved_grid_size = _grid_for_canvas(
        zoomed.width, zoomed.height, next_density, next_grid_size
    )
    view_id = _view_id()
    level_index = len(resolved_levels)
    path_label = _path_label(resolved_levels)
    annotated = draw_overlay(
        zoomed,
        overlay_mode,
        resolved_grid_size,
        detail_mode=detail_mode,
        ruler_config=ruler_config,
        show_axis_ticks=show_axis_ticks,
        level_label=f"level {level_index}",
        path_label=path_label,
    )
    out_path = Path(out_dir)
    annotated_path = save_png(
        annotated,
        out_path / f"{view_id}_level{level_index}_{overlay_mode}.png",
    )

    cropped_path = None
    if save_cropped:
        cropped_path = save_png(zoomed, out_path / f"{view_id}_level{level_index}_crop.png")

    result = GridViewResult(
        view_id=view_id,
        image_width=original.width,
        image_height=original.height,
        level=level_index,
        grid_size=resolved_grid_size,
        bbox_on_original=bbox,
        annotated_image_path=annotated_path,
        levels=resolved_levels,
        cropped_image_path=cropped_path,
    )
    return result.to_dict()


def resolve_region(
    image_path: str | Path | None = None,
    levels: Sequence[Mapping[str, Any] | Level] | None = None,
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    size = _resolve_image_size(image_path, image_size)
    bbox = resolve_levels_to_bbox(size, levels)
    return RegionResult(bbox_on_original=bbox, width=bbox.width, height=bbox.height).to_dict()


def resolve_point(
    image_path: str | Path | None = None,
    levels: Sequence[Mapping[str, Any] | Level] | None = None,
    point_spec: Mapping[str, Any] | None = None,
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if point_spec is None:
        raise ValueError("point_spec is required")
    size = _resolve_image_size(image_path, image_size)
    bbox = resolve_levels_to_bbox(size, levels)
    point = resolve_point_in_bbox(bbox, point_spec)
    return PointResult(point_on_original=point, final_region_bbox_on_original=bbox).to_dict()


def _resolve_image_size(
    image_path: str | Path | None, image_size: tuple[int, int] | None
) -> tuple[int, int]:
    if image_size is not None:
        width, height = image_size
        if width < 1 or height < 1:
            raise ValueError("image_size width and height must be >= 1")
        return width, height
    if image_path is None:
        raise ValueError("either image_path or image_size is required")
    image = load_image(image_path)
    return image.width, image.height


def _grid_for_canvas(
    width: int, height: int, density: str, grid_size: Sequence[int] | None
) -> tuple[int, int]:
    if grid_size is None:
        suggested = suggest_grid_size(width, height, density)
        return suggested[0], suggested[1]
    return validate_grid_size(grid_size)


def _view_id() -> str:
    return uuid.uuid4().hex[:12]


def _path_label(levels: Sequence[Level]) -> str | None:
    if not levels:
        return None
    parts = [
        f"{level.cell[0]},{level.cell[1]}@{level.grid_size[0]}x{level.grid_size[1]}"
        for level in levels
    ]
    return "path " + " > ".join(parts)
