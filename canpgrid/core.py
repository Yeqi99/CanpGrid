from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from math import ceil, floor
from pathlib import Path
from typing import Any

from .geometry import resolve_levels_to_bbox, resolve_point_in_bbox, resolve_point_region_in_bbox
from .grid import suggest_grid_size, validate_grid_size
from .render import (
    crop_and_zoom,
    draw_cell_ruler_overlay,
    draw_overlay,
    draw_point_preview,
    load_image,
    save_png,
)
from .schemas import (
    BBox,
    CellRulerViewResult,
    DetailMode,
    GridViewResult,
    Level,
    MarkerStyle,
    OverlayMode,
    PointResult,
    PreviewOn,
    PreviewPointResult,
    RegionResult,
    RulerConfig,
    clean_point,
    coerce_levels,
)
from .snap import resolve_color_snap


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


def create_cell_ruler_view(
    image_path: str | Path,
    levels: Sequence[Mapping[str, Any] | Level] | None = None,
    *,
    grid_size: Sequence[int],
    cell: Sequence[int],
    detail_mode: DetailMode = "medium",
    ruler_config: Mapping[str, Any] | RulerConfig | None = None,
    zoom_factor: float = 1,
    show_axis_ticks: bool = True,
    out_dir: str | Path = "outputs",
) -> dict[str, Any]:
    if zoom_factor <= 0:
        raise ValueError("zoom_factor must be > 0")
    original = load_image(image_path)
    resolved_levels = coerce_levels(levels)
    base_bbox = resolve_levels_to_bbox((original.width, original.height), resolved_levels)
    current_view = crop_and_zoom(original, base_bbox, zoom_factor=zoom_factor)
    resolved_grid_size = validate_grid_size(grid_size)
    cell_tuple = _validate_cell(cell, resolved_grid_size)
    if isinstance(ruler_config, RulerConfig):
        resolved_ruler = ruler_config
    else:
        resolved_ruler = RulerConfig.from_mapping(ruler_config)
    selected_level = Level(grid_size=resolved_grid_size, cell=cell_tuple)
    cell_bbox = resolve_levels_to_bbox(
        (original.width, original.height), (*resolved_levels, selected_level)
    )
    view_id = _view_id()
    annotated = draw_cell_ruler_overlay(
        current_view,
        resolved_grid_size,
        cell_tuple,
        detail_mode=detail_mode,
        ruler_config=resolved_ruler,
        show_axis_ticks=show_axis_ticks,
        level_label=f"level {len(resolved_levels)} cell ruler",
        path_label=_path_label((*resolved_levels, selected_level)),
    )
    annotated_path = save_png(
        annotated,
        Path(out_dir) / f"{view_id}_level{len(resolved_levels)}_cell_ruler.png",
    )
    result = CellRulerViewResult(
        view_id=view_id,
        image_width=original.width,
        image_height=original.height,
        level=len(resolved_levels),
        grid_size=resolved_grid_size,
        cell=cell_tuple,
        ruler_config=resolved_ruler,
        bbox_on_original=base_bbox,
        cell_bbox_on_original=cell_bbox,
        annotated_image_path=annotated_path,
        levels=resolved_levels,
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
    original = None
    if _point_spec_requires_image(point_spec):
        if image_path is None:
            raise ValueError("color_snap_point requires image_path because it reads pixels")
        original = load_image(image_path)
        size = (original.width, original.height)
    else:
        size = _resolve_image_size(image_path, image_size)
    bbox = resolve_levels_to_bbox(size, levels)
    point, point_region, point_resolution = _resolve_point_spec(
        bbox,
        point_spec,
        image=original,
    )
    return PointResult(
        point_on_original=point,
        final_region_bbox_on_original=point_region,
        point_resolution=point_resolution,
    ).to_dict()


def preview_point(
    image_path: str | Path,
    levels: Sequence[Mapping[str, Any] | Level],
    point_spec: Mapping[str, Any],
    *,
    preview_on: PreviewOn = "current_view",
    marker_style: MarkerStyle = "ring_crosshair",
    with_inset: bool = False,
    out_dir: str | Path = "outputs",
    zoom_factor: float = 6,
) -> dict[str, Any]:
    if preview_on not in {"current_view", "original_image", "both"}:
        raise ValueError("preview_on must be one of: current_view, original_image, both")
    if marker_style not in {"ring", "ring_crosshair", "ring_crosshair_inset"}:
        raise ValueError(
            "marker_style must be one of: ring, ring_crosshair, ring_crosshair_inset"
        )

    original = load_image(image_path)
    resolved_levels = coerce_levels(levels)
    bbox = resolve_levels_to_bbox((original.width, original.height), resolved_levels)
    point_on_original, point_region, point_resolution = _resolve_point_spec(
        bbox,
        point_spec,
        image=original,
    )
    view_id = _view_id()
    out_path = Path(out_dir)

    preview_paths: dict[str, Path] = {}
    point_on_current_view = _point_on_current_view(original, bbox, point_on_original, zoom_factor)

    if preview_on in {"current_view", "both"}:
        current_view = crop_and_zoom(original, bbox, zoom_factor=zoom_factor)
        current_preview = draw_point_preview(
            current_view,
            point_on_current_view,
            marker_style=marker_style,
            with_inset=with_inset,
        )
        preview_paths["current_view"] = save_png(
            current_preview,
            out_path / f"{view_id}_preview_current_{marker_style}.png",
        )

    if preview_on in {"original_image", "both"}:
        original_preview = draw_point_preview(
            original,
            point_on_original,
            marker_style=marker_style,
            with_inset=with_inset,
        )
        preview_paths["original_image"] = save_png(
            original_preview,
            out_path / f"{view_id}_preview_original_{marker_style}.png",
        )

    main_key = "current_view" if "current_view" in preview_paths else "original_image"
    result = PreviewPointResult(
        preview_image_path=preview_paths[main_key],
        point_on_original=point_on_original,
        point_on_current_view=point_on_current_view,
        final_region_bbox_on_original=point_region,
        preview_image_paths=preview_paths if preview_on == "both" else None,
        point_resolution=point_resolution,
    )
    return result.to_dict()


def _resolve_point_spec(
    bbox: BBox,
    point_spec: Mapping[str, Any],
    *,
    image=None,
) -> tuple[tuple[float, float], BBox, dict[str, Any] | None]:
    if point_spec.get("type") != "color_snap_point":
        return (
            resolve_point_in_bbox(bbox, point_spec),
            resolve_point_region_in_bbox(bbox, point_spec),
            None,
        )

    if image is None:
        raise ValueError("color_snap_point requires image pixels")
    base_spec = point_spec.get("base")
    if not isinstance(base_spec, Mapping) and "grid_size" in point_spec and "cell" in point_spec:
        base_spec = {
            "type": "subgrid_point",
            "grid_size": point_spec["grid_size"],
            "cell": point_spec["cell"],
            "local_point": point_spec.get("local_point", [0.5, 0.5]),
        }
    if not isinstance(base_spec, Mapping):
        raise ValueError("color_snap_point requires a base point_spec mapping")
    base_point, _base_region, base_resolution = _resolve_point_spec(
        bbox,
        base_spec,
        image=image,
    )
    snap = resolve_color_snap(image, base_point, point_spec)
    resolution = {
        **snap.metadata,
        "base_point_on_original": clean_point(base_point),
        "search_bbox_on_original": snap.search_bbox.to_dict(),
    }
    if base_resolution is not None:
        resolution["base_resolution"] = base_resolution
    return snap.point, snap.search_bbox, resolution


def _point_spec_requires_image(point_spec: Mapping[str, Any]) -> bool:
    if point_spec.get("type") == "color_snap_point":
        return True
    base = point_spec.get("base")
    return isinstance(base, Mapping) and _point_spec_requires_image(base)


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


def _validate_cell(cell: Sequence[int], grid_size: tuple[int, int]) -> tuple[int, int]:
    if isinstance(cell, (str, bytes)) or len(cell) != 2:
        raise ValueError("cell must be a two-item sequence")
    cell_x, cell_y = cell
    if not isinstance(cell_x, int) or isinstance(cell_x, bool):
        raise ValueError("cell[0] must be an integer")
    if not isinstance(cell_y, int) or isinstance(cell_y, bool):
        raise ValueError("cell[1] must be an integer")
    cols, rows = grid_size
    if cell_x < 0 or cell_x >= cols or cell_y < 0 or cell_y >= rows:
        raise ValueError("cell must be inside grid_size")
    return cell_x, cell_y


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


def _point_on_current_view(
    image, bbox: BBox, point_on_original: tuple[float, float], zoom_factor: float
) -> tuple[float, float]:
    if zoom_factor <= 0:
        raise ValueError("zoom_factor must be > 0")

    left, top, right, bottom = _int_crop_box(image.width, image.height, bbox)
    crop_width = right - left
    crop_height = bottom - top
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("resolved bbox is empty after clamping to image bounds")

    zoomed_width = max(1, round(crop_width * zoom_factor))
    zoomed_height = max(1, round(crop_height * zoom_factor))
    return (
        (point_on_original[0] - left) * (zoomed_width / crop_width),
        (point_on_original[1] - top) * (zoomed_height / crop_height),
    )


def _int_crop_box(width: int, height: int, bbox: BBox) -> tuple[int, int, int, int]:
    left = max(0, min(width, floor(bbox.x1)))
    top = max(0, min(height, floor(bbox.y1)))
    right = max(0, min(width, ceil(bbox.x2)))
    bottom = max(0, min(height, ceil(bbox.y2)))
    return left, top, right, bottom
