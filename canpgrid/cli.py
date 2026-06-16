from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from .core import (
    create_grid_view,
    zoom_region,
)
from .core import (
    preview_point as api_preview_point,
)
from .core import (
    resolve_point as api_resolve_point,
)
from .core import (
    resolve_region as api_resolve_region,
)

app = typer.Typer(
    name="canpgrid",
    help="Adaptive recursive image grid views for multimodal agents.",
    no_args_is_help=True,
)


@app.command()
def grid(
    image_path: Path = typer.Argument(..., exists=True, readable=True),
    density: str = typer.Option("medium", "--density"),
    grid_size: str | None = typer.Option(None, "--grid-size", help="Grid size like 12x7."),
    overlay_mode: str = typer.Option("grid", "--overlay-mode"),
    detail_mode: str = typer.Option("medium", "--detail-mode"),
    ruler_size: str | None = typer.Option(None, "--ruler-size", help="Ruler size like 16x16."),
    out: Path = typer.Option(Path("outputs"), "--out"),
    show_axis_ticks: bool = typer.Option(True, "--show-axis-ticks/--hide-axis-ticks"),
    show_crosshair: bool = typer.Option(False, "--show-crosshair/--hide-crosshair"),
) -> None:
    """Create an annotated guide-lined observation image."""

    result = create_grid_view(
        image_path,
        density=density,
        grid_size=_parse_size(grid_size, "grid_size") if grid_size else None,
        overlay_mode=_overlay_mode(overlay_mode),
        detail_mode=_detail_mode(detail_mode),
        ruler_config=_ruler_config(ruler_size),
        show_axis_ticks=show_axis_ticks,
        show_crosshair=show_crosshair,
        out_dir=out,
    )
    _print_json(result)


@app.command()
def zoom(
    image_path: Path = typer.Argument(..., exists=True, readable=True),
    levels: str = typer.Option(..., "--levels", help="JSON level path."),
    next_density: str = typer.Option("medium", "--next-density"),
    next_grid_size: str | None = typer.Option(
        None, "--next-grid-size", help="Grid size like 12x7."
    ),
    overlay_mode: str = typer.Option("grid", "--overlay-mode"),
    detail_mode: str = typer.Option("medium", "--detail-mode"),
    ruler_size: str | None = typer.Option(None, "--ruler-size", help="Ruler size like 16x16."),
    zoom_factor: float = typer.Option(3.0, "--zoom-factor"),
    out: Path = typer.Option(Path("outputs"), "--out"),
    show_axis_ticks: bool = typer.Option(True, "--show-axis-ticks/--hide-axis-ticks"),
    save_cropped: bool = typer.Option(False, "--save-cropped/--no-save-cropped"),
) -> None:
    """Zoom a recursive level path and create the next annotated view."""

    result = zoom_region(
        image_path,
        _parse_json(levels, "levels"),
        next_density=next_density,
        next_grid_size=_parse_size(next_grid_size, "next_grid_size") if next_grid_size else None,
        overlay_mode=_overlay_mode(overlay_mode),
        detail_mode=_detail_mode(detail_mode),
        ruler_config=_ruler_config(ruler_size),
        zoom_factor=zoom_factor,
        show_axis_ticks=show_axis_ticks,
        out_dir=out,
        save_cropped=save_cropped,
    )
    _print_json(result)


@app.command("resolve-region")
def resolve_region(
    image_path: Path = typer.Argument(..., exists=True, readable=True),
    levels: str = typer.Option(..., "--levels", help="JSON level path."),
) -> None:
    """Resolve recursive levels to a bbox on the original image."""

    result = api_resolve_region(image_path, _parse_json(levels, "levels"))
    _print_json(result)


@app.command("resolve-point")
def resolve_point(
    image_path: Path = typer.Argument(..., exists=True, readable=True),
    levels: str = typer.Option(..., "--levels", help="JSON level path."),
    point_spec: str = typer.Option(..., "--point-spec", help="JSON point_spec."),
) -> None:
    """Resolve recursive levels plus a point_spec to an original-image point."""

    result = api_resolve_point(
        image_path,
        _parse_json(levels, "levels"),
        _parse_json(point_spec, "point_spec"),
    )
    _print_json(result)


@app.command("preview-point")
def preview_point(
    image_path: Path = typer.Argument(..., exists=True, readable=True),
    levels: str = typer.Option(..., "--levels", help="JSON level path."),
    point_spec: str = typer.Option(..., "--point-spec", help="JSON point_spec."),
    preview_on: str = typer.Option("current_view", "--preview-on"),
    marker_style: str = typer.Option("ring_crosshair", "--marker-style"),
    with_inset: bool = typer.Option(False, "--with-inset/--no-inset"),
    zoom_factor: float = typer.Option(6.0, "--zoom-factor"),
    out: Path = typer.Option(Path("outputs"), "--out"),
) -> None:
    """Preview a resolved candidate focus point without executing a click."""

    result = api_preview_point(
        image_path,
        _parse_json(levels, "levels"),
        _parse_json(point_spec, "point_spec"),
        preview_on=_preview_on(preview_on),
        marker_style=_marker_style(marker_style),
        with_inset=with_inset,
        zoom_factor=zoom_factor,
        out_dir=out,
    )
    _print_json(result)


def _parse_json(value: str, field_name: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{field_name} must be valid JSON: {exc}") from exc


def _parse_size(value: str, field_name: str) -> tuple[int, int]:
    if "x" not in value:
        raise typer.BadParameter(f"{field_name} must use WIDTHxHEIGHT syntax")
    left, right = value.lower().split("x", 1)
    try:
        parsed = int(left), int(right)
    except ValueError as exc:
        raise typer.BadParameter(f"{field_name} values must be integers") from exc
    if parsed[0] < 1 or parsed[1] < 1:
        raise typer.BadParameter(f"{field_name} values must be >= 1")
    return parsed


def _ruler_config(ruler_size: str | None) -> dict[str, int | bool] | None:
    if ruler_size is None:
        return None
    tick_x, tick_y = _parse_size(ruler_size, "ruler_size")
    return {"tick_x": tick_x, "tick_y": tick_y, "show_minor_ticks": True, "show_labels_every": 2}


def _overlay_mode(value: str) -> str:
    if value not in {"grid", "ruler", "hybrid"}:
        raise typer.BadParameter("overlay_mode must be one of: grid, ruler, hybrid")
    return value


def _detail_mode(value: str) -> str:
    if value not in {"coarse", "medium", "fine"}:
        raise typer.BadParameter("detail_mode must be one of: coarse, medium, fine")
    return value


def _preview_on(value: str) -> str:
    if value not in {"current_view", "original_image", "both"}:
        raise typer.BadParameter("preview_on must be one of: current_view, original_image, both")
    return value


def _marker_style(value: str) -> str:
    if value not in {"ring", "ring_crosshair", "ring_crosshair_inset"}:
        raise typer.BadParameter(
            "marker_style must be one of: ring, ring_crosshair, ring_crosshair_inset"
        )
    return value


def _print_json(result: Any) -> None:
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
