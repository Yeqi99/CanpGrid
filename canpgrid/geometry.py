from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .grid import validate_cell, validate_grid_size
from .schemas import BBox, Level, Point, coerce_levels

ANCHORS: dict[str, tuple[float, float]] = {
    "top_left": (0.0, 0.0),
    "top": (0.5, 0.0),
    "top_right": (1.0, 0.0),
    "left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "right": (1.0, 0.5),
    "bottom_left": (0.0, 1.0),
    "bottom": (0.5, 1.0),
    "bottom_right": (1.0, 1.0),
}

ALLOWED_FRACTIONS: dict[str, float] = {
    "0": 0.0,
    "1/4": 0.25,
    "1/2": 0.5,
    "3/4": 0.75,
    "1": 1.0,
}


def resolve_levels_to_bbox(
    image_size: tuple[int, int], levels: Sequence[Mapping[str, Any] | Level] | None
) -> BBox:
    width, height = _validate_image_size(image_size)
    bbox = BBox(0.0, 0.0, float(width), float(height))

    for level in coerce_levels(levels):
        cols, rows = validate_grid_size(level.grid_size)
        cell_x, cell_y = validate_cell(level.cell, (cols, rows))
        cell_width = bbox.width / cols
        cell_height = bbox.height / rows
        bbox = BBox(
            x1=bbox.x1 + cell_x * cell_width,
            y1=bbox.y1 + cell_y * cell_height,
            x2=bbox.x1 + (cell_x + 1) * cell_width,
            y2=bbox.y1 + (cell_y + 1) * cell_height,
        )

    return bbox


def resolve_point_in_bbox(bbox: BBox, point_spec: Mapping[str, Any]) -> Point:
    if not isinstance(point_spec, Mapping):
        raise ValueError("point_spec must be a mapping")
    point_type = point_spec.get("type")
    if point_type == "normalized_point":
        fx, fy = _fraction_pair(point_spec.get("value"), "value")
        return _point_from_normalized(bbox, fx, fy)

    if point_type == "anchor_offset":
        anchor_x, anchor_y = _anchor(point_spec.get("anchor"))
        dx, dy = _fraction_pair(point_spec.get("offset"), "offset", signed=True)
        return _point_from_normalized(bbox, anchor_x + dx, anchor_y + dy)

    if point_type == "ruler_point":
        origin = point_spec.get("origin", "top_left")
        if origin != "top_left":
            raise ValueError("ruler_point currently supports origin='top_left' only")
        ruler_x, ruler_y = _ruler_size(point_spec.get("ruler_size"))
        x = _number(point_spec.get("x"), "x")
        y = _number(point_spec.get("y"), "y")
        return (bbox.x1 + bbox.width * (x / ruler_x), bbox.y1 + bbox.height * (y / ruler_y))

    if point_type == "ruler_offset":
        anchor_x, anchor_y = _anchor(point_spec.get("anchor"))
        ruler_x, ruler_y = _ruler_size(point_spec.get("ruler_size"))
        dx = _number(point_spec.get("dx"), "dx")
        dy = _number(point_spec.get("dy"), "dy")
        return (
            bbox.x1 + bbox.width * anchor_x + dx * (bbox.width / ruler_x),
            bbox.y1 + bbox.height * anchor_y + dy * (bbox.height / ruler_y),
        )

    if point_type == "hybrid_point":
        base_x, base_y = _fraction_pair(point_spec.get("base"), "base")
        offset_x, offset_y = _integer_pair(point_spec.get("offset"), "offset")
        unit = point_spec.get("unit")
        if unit != "ruler_tick":
            raise ValueError("hybrid_point unit must be 'ruler_tick'")
        ruler_x, ruler_y = _ruler_size(point_spec.get("ruler_size"))
        return (
            bbox.x1 + bbox.width * base_x + offset_x * (bbox.width / ruler_x),
            bbox.y1 + bbox.height * base_y + offset_y * (bbox.height / ruler_y),
        )

    if point_type == "subgrid_point":
        grid_size = validate_grid_size(_sequence(point_spec.get("grid_size"), "grid_size"))
        cell = validate_cell(_sequence(point_spec.get("cell"), "cell"), grid_size)
        local_x, local_y = _fraction_pair(point_spec.get("local_point"), "local_point")
        sub_width = bbox.width / grid_size[0]
        sub_height = bbox.height / grid_size[1]
        return (
            bbox.x1 + (cell[0] + local_x) * sub_width,
            bbox.y1 + (cell[1] + local_y) * sub_height,
        )

    raise ValueError(f"unknown point_spec type: {point_type!r}")


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError("image_size width and height must be >= 1")
    return width, height


def _point_from_normalized(bbox: BBox, fx: float, fy: float) -> Point:
    return (bbox.x1 + bbox.width * fx, bbox.y1 + bbox.height * fy)


def _fraction_pair(value: Any, field_name: str, *, signed: bool = False) -> tuple[float, float]:
    values = _sequence(value, field_name)
    if len(values) != 2:
        raise ValueError(f"{field_name} must contain two values")
    return (
        parse_fraction(values[0], f"{field_name}[0]", signed=signed),
        parse_fraction(values[1], f"{field_name}[1]", signed=signed),
    )


def parse_fraction(value: Any, field_name: str, *, signed: bool = False) -> float:
    if isinstance(value, str):
        sign = 1.0
        token = value
        if signed and token.startswith(("+", "-")):
            if token[0] == "-":
                sign = -1.0
            token = token[1:]
        if token in ALLOWED_FRACTIONS:
            return sign * ALLOWED_FRACTIONS[token]
        allowed = sorted(ALLOWED_FRACTIONS)
        if signed:
            raise ValueError(f"{field_name} must be one of {allowed} with optional +/- sign")
        raise ValueError(f"{field_name} must be one of {allowed}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not signed and (number < 0.0 or number > 1.0):
            raise ValueError(f"{field_name} must be between 0 and 1")
        return number

    raise ValueError(f"{field_name} must be a fraction string or number")


def _anchor(value: Any) -> tuple[float, float]:
    if not isinstance(value, str) or value not in ANCHORS:
        raise ValueError(f"anchor must be one of {sorted(ANCHORS)}")
    return ANCHORS[value]


def _ruler_size(value: Any) -> tuple[int, int]:
    values = _sequence(value, "ruler_size")
    if len(values) != 2:
        raise ValueError("ruler_size must contain two values")
    ruler_x, ruler_y = _integer_pair(values, "ruler_size")
    if ruler_x < 1 or ruler_y < 1:
        raise ValueError("ruler_size values must be >= 1")
    return ruler_x, ruler_y


def _integer_pair(value: Any, field_name: str) -> tuple[int, int]:
    values = _sequence(value, field_name)
    if len(values) != 2:
        raise ValueError(f"{field_name} must contain two values")
    first, second = values
    if not isinstance(first, int) or isinstance(first, bool):
        raise ValueError(f"{field_name}[0] must be an integer")
    if not isinstance(second, int) or isinstance(second, bool):
        raise ValueError(f"{field_name}[1] must be an integer")
    return first, second


def _number(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    return float(value)


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence")
    return value

