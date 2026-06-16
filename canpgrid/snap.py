from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Any

from PIL import Image

from .schemas import BBox, Point, clean_point

NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
}

DIRECTIONS: dict[str, tuple[int, int]] = {
    "right": (1, 0),
    "left": (-1, 0),
    "down": (0, 1),
    "up": (0, -1),
    "down_right": (1, 1),
    "down_left": (-1, 1),
    "up_right": (1, -1),
    "up_left": (-1, -1),
}


@dataclass(frozen=True)
class ColorSnapResult:
    point: Point
    search_bbox: BBox
    metadata: dict[str, Any]


def resolve_color_snap(
    image: Image.Image,
    base_point: Point,
    point_spec: Mapping[str, Any],
) -> ColorSnapResult:
    search = _mapping(point_spec.get("search", {}), "search")
    target_color = _target_color(point_spec)
    tolerance = _nonnegative_number(
        point_spec.get("tolerance", search.get("tolerance", 24)),
        "tolerance",
    )
    mode = str(search.get("mode", point_spec.get("mode", "nearest")))
    start_offset = _offset(
        search.get(
            "start_offset",
            search.get("from_offset", point_spec.get("start_offset", [0, 0])),
        ),
        "start_offset",
    )
    start = (
        round(base_point[0] + start_offset[0]),
        round(base_point[1] + start_offset[1]),
    )

    rgb_image = image.convert("RGB")
    if mode in {"ray", "scan"}:
        return _snap_ray(rgb_image, point_spec, search, start, target_color, tolerance)
    if mode == "nearest":
        return _snap_nearest(rgb_image, point_spec, search, start, target_color, tolerance)
    raise ValueError("color_snap_point search.mode must be one of: nearest, ray")


def _snap_ray(
    image: Image.Image,
    point_spec: Mapping[str, Any],
    search: Mapping[str, Any],
    start: tuple[int, int],
    target_color: tuple[int, int, int],
    tolerance: float,
) -> ColorSnapResult:
    direction = _direction(search.get("direction", point_spec.get("direction", "right")))
    max_distance = _nonnegative_int(
        search.get("max_distance", point_spec.get("max_distance", 64)),
        "max_distance",
    )
    include_start = bool(search.get("include_start", True))
    first_step = 0 if include_start else 1
    end = (
        start[0] + direction[0] * max_distance,
        start[1] + direction[1] * max_distance,
    )
    search_bbox = _line_bbox(start, end, image.size)

    for step in range(first_step, max_distance + 1):
        x = start[0] + direction[0] * step
        y = start[1] + direction[1] * step
        if not _inside(image.size, x, y):
            continue
        pixel = image.getpixel((x, y))
        if _color_distance(pixel, target_color) <= tolerance:
            return _result(
                (float(x), float(y)),
                search_bbox,
                mode="ray",
                target_color=target_color,
                matched_color=pixel,
                color_distance=_color_distance(pixel, target_color),
                fallback_used=False,
            )

    return _fallback_or_error(point_spec, start, search_bbox, "ray", target_color)


def _snap_nearest(
    image: Image.Image,
    point_spec: Mapping[str, Any],
    search: Mapping[str, Any],
    start: tuple[int, int],
    target_color: tuple[int, int, int],
    tolerance: float,
) -> ColorSnapResult:
    radius = _nonnegative_int(search.get("radius", point_spec.get("radius", 24)), "radius")
    search_bbox = _radius_bbox(start, radius, image.size)
    best: tuple[int, int, float, int] | None = None
    for y in range(int(search_bbox.y1), int(search_bbox.y2)):
        for x in range(int(search_bbox.x1), int(search_bbox.x2)):
            distance_sq = (x - start[0]) ** 2 + (y - start[1]) ** 2
            if distance_sq > radius**2:
                continue
            pixel = image.getpixel((x, y))
            color_distance = _color_distance(pixel, target_color)
            if color_distance > tolerance:
                continue
            candidate = (x, y, color_distance, distance_sq)
            if best is None or (candidate[3], candidate[2], candidate[1], candidate[0]) < (
                best[3],
                best[2],
                best[1],
                best[0],
            ):
                best = candidate

    if best is not None:
        x, y, color_distance, _distance_sq = best
        return _result(
            (float(x), float(y)),
            search_bbox,
            mode="nearest",
            target_color=target_color,
            matched_color=image.getpixel((x, y)),
            color_distance=color_distance,
            fallback_used=False,
        )

    return _fallback_or_error(point_spec, start, search_bbox, "nearest", target_color)


def _fallback_or_error(
    point_spec: Mapping[str, Any],
    start: tuple[int, int],
    search_bbox: BBox,
    mode: str,
    target_color: tuple[int, int, int],
) -> ColorSnapResult:
    fallback = point_spec.get("fallback", "error")
    if fallback == "base_point":
        return _result(
            (float(start[0]), float(start[1])),
            search_bbox,
            mode=mode,
            target_color=target_color,
            matched_color=None,
            color_distance=None,
            fallback_used=True,
        )
    raise ValueError("color_snap_point did not find target_color in the search region")


def _result(
    point: Point,
    search_bbox: BBox,
    *,
    mode: str,
    target_color: tuple[int, int, int],
    matched_color: tuple[int, int, int] | None,
    color_distance: float | None,
    fallback_used: bool,
) -> ColorSnapResult:
    metadata = {
        "type": "color_snap_point",
        "search_mode": mode,
        "target_color": list(target_color),
        "matched": matched_color is not None,
        "fallback_used": fallback_used,
        "snapped_point_on_original": clean_point(point),
    }
    if matched_color is not None:
        metadata["matched_color"] = list(matched_color)
    if color_distance is not None:
        metadata["color_distance"] = _clean_distance(color_distance)
    return ColorSnapResult(point=point, search_bbox=search_bbox, metadata=metadata)


def _color(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        token = value.strip().lower()
        if token in NAMED_COLORS:
            return NAMED_COLORS[token]
        if token.startswith("#"):
            token = token[1:]
        if len(token) != 6:
            raise ValueError("target_color string must be #RRGGBB or a known color name")
        try:
            return (int(token[0:2], 16), int(token[2:4], 16), int(token[4:6], 16))
        except ValueError as exc:
            raise ValueError("target_color string must be #RRGGBB") from exc

    if isinstance(value, Mapping):
        return (
            _byte(value.get("r", value.get("red")), "target_color.r"),
            _byte(value.get("g", value.get("green")), "target_color.g"),
            _byte(value.get("b", value.get("blue")), "target_color.b"),
        )

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        return (
            _byte(value[0], "target_color[0]"),
            _byte(value[1], "target_color[1]"),
            _byte(value[2], "target_color[2]"),
        )

    raise ValueError("target_color must be #RRGGBB, a known color name, or [r,g,b]")


def _target_color(point_spec: Mapping[str, Any]) -> tuple[int, int, int]:
    color_id = point_spec.get("target_color_id", point_spec.get("color_id"))
    if color_id is not None:
        return _palette_color(point_spec, str(color_id))
    return _color(
        point_spec.get(
            "target_color",
            point_spec.get("color", point_spec.get("rgb")),
        )
    )


def _palette_color(point_spec: Mapping[str, Any], color_id: str) -> tuple[int, int, int]:
    choices = point_spec.get(
        "color_choices",
        point_spec.get("palette", point_spec.get("colors")),
    )
    if choices is None:
        raise ValueError("target_color_id requires color_choices or palette")

    if isinstance(choices, Mapping):
        if color_id in choices:
            return _color(_color_choice_value(choices[color_id]))
        iterable = choices.values()
    elif isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        iterable = choices
    else:
        raise ValueError("color_choices must be a list or mapping")

    for choice in iterable:
        if not isinstance(choice, Mapping):
            continue
        if str(choice.get("id", choice.get("color_id", ""))) == color_id:
            return _color(_color_choice_value(choice))
    raise ValueError(f"unknown target_color_id {color_id!r}")


def _color_choice_value(choice: Any) -> Any:
    if isinstance(choice, Mapping):
        for key in ("hex", "target_color", "color", "rgb"):
            if key in choice:
                return choice[key]
        if all(key in choice for key in ("r", "g", "b")):
            return choice
    return choice


def _direction(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        token = value.strip().lower().replace("-", "_")
        if token in DIRECTIONS:
            return DIRECTIONS[token]
        raise ValueError(f"direction must be one of {sorted(DIRECTIONS)} or [dx,dy]")
    dx, dy = _offset(value, "direction")
    step_x = _sign(dx)
    step_y = _sign(dy)
    if step_x == 0 and step_y == 0:
        raise ValueError("direction cannot be [0,0]")
    return step_x, step_y


def _offset(value: Any, field_name: str) -> tuple[float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return (_number(value[0], f"{field_name}[0]"), _number(value[1], f"{field_name}[1]"))
    raise ValueError(f"{field_name} must be [x,y]")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{field_name} must be a mapping")


def _line_bbox(
    start: tuple[int, int], end: tuple[int, int], image_size: tuple[int, int]
) -> BBox:
    x1 = max(0, min(start[0], end[0]))
    y1 = max(0, min(start[1], end[1]))
    x2 = min(image_size[0], max(start[0], end[0]) + 1)
    y2 = min(image_size[1], max(start[1], end[1]) + 1)
    x2 = max(x1, x2)
    y2 = max(y1, y2)
    return BBox(float(x1), float(y1), float(x2), float(y2))


def _radius_bbox(start: tuple[int, int], radius: int, image_size: tuple[int, int]) -> BBox:
    x1 = max(0, start[0] - radius)
    y1 = max(0, start[1] - radius)
    x2 = min(image_size[0], start[0] + radius + 1)
    y2 = min(image_size[1], start[1] + radius + 1)
    return BBox(float(x1), float(y1), float(max(x1, x2)), float(max(y1, y2)))


def _inside(image_size: tuple[int, int], x: int, y: int) -> bool:
    return 0 <= x < image_size[0] and 0 <= y < image_size[1]


def _color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _byte(value: Any, field_name: str) -> int:
    number = _number(value, field_name)
    if number < 0 or number > 255 or int(number) != number:
        raise ValueError(f"{field_name} must be an integer between 0 and 255")
    return int(number)


def _nonnegative_number(value: Any, field_name: str) -> float:
    number = _number(value, field_name)
    if number < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return number


def _nonnegative_int(value: Any, field_name: str) -> int:
    number = _number(value, field_name)
    if number < 0 or int(number) != number:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(number)


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    return float(value)


def _sign(value: float) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _clean_distance(value: float) -> int | float:
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value
