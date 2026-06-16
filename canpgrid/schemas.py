from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

GridSize = tuple[int, int]
Cell = tuple[int, int]
Point = tuple[float, float]
OverlayMode = Literal["grid", "ruler", "hybrid"]
DetailMode = Literal["coarse", "medium", "fine"]


def clean_number(value: float) -> int | float:
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def clean_point(point: Point) -> list[int | float]:
    return [clean_number(point[0]), clean_number(point[1])]


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_dict(self) -> dict[str, int | float]:
        return {
            "x1": clean_number(self.x1),
            "y1": clean_number(self.y1),
            "x2": clean_number(self.x2),
            "y2": clean_number(self.y2),
            "width": clean_number(self.width),
            "height": clean_number(self.height),
        }


@dataclass(frozen=True)
class Level:
    grid_size: GridSize
    cell: Cell

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Level:
        if "grid_size" not in value:
            raise ValueError("level is missing 'grid_size'")
        if "cell" not in value:
            raise ValueError("level is missing 'cell'")
        return cls(
            grid_size=_pair_of_ints(value["grid_size"], "grid_size"),
            cell=_pair_of_ints(value["cell"], "cell"),
        )

    def to_dict(self) -> dict[str, list[int]]:
        return {"grid_size": list(self.grid_size), "cell": list(self.cell)}


@dataclass(frozen=True)
class RulerConfig:
    tick_x: int = 16
    tick_y: int = 16
    show_minor_ticks: bool = True
    show_labels_every: int = 2

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None = None) -> RulerConfig:
        if value is None:
            return cls()
        tick_x = _positive_int(value.get("tick_x", 16), "tick_x")
        tick_y = _positive_int(value.get("tick_y", 16), "tick_y")
        show_labels_every = _positive_int(value.get("show_labels_every", 2), "show_labels_every")
        return cls(
            tick_x=tick_x,
            tick_y=tick_y,
            show_minor_ticks=bool(value.get("show_minor_ticks", True)),
            show_labels_every=show_labels_every,
        )

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "tick_x": self.tick_x,
            "tick_y": self.tick_y,
            "show_minor_ticks": self.show_minor_ticks,
            "show_labels_every": self.show_labels_every,
        }


@dataclass(frozen=True)
class GridViewResult:
    view_id: str
    image_width: int
    image_height: int
    level: int
    grid_size: GridSize
    bbox_on_original: BBox
    annotated_image_path: Path
    levels: tuple[Level, ...] = ()
    cropped_image_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "view_id": self.view_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "level": self.level,
            "grid_size": list(self.grid_size),
            "bbox_on_original": self.bbox_on_original.to_dict(),
            "annotated_image_path": str(self.annotated_image_path),
        }
        if self.levels:
            data["levels"] = [level.to_dict() for level in self.levels]
        if self.cropped_image_path is not None:
            data["cropped_image_path"] = str(self.cropped_image_path)
        return data


@dataclass(frozen=True)
class RegionResult:
    bbox_on_original: BBox
    width: float
    height: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_on_original": self.bbox_on_original.to_dict(),
            "width": clean_number(self.width),
            "height": clean_number(self.height),
        }


@dataclass(frozen=True)
class PointResult:
    point_on_original: Point
    final_region_bbox_on_original: BBox

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_on_original": clean_point(self.point_on_original),
            "final_region_bbox_on_original": self.final_region_bbox_on_original.to_dict(),
        }


def coerce_levels(levels: Sequence[Mapping[str, Any] | Level] | None) -> tuple[Level, ...]:
    if levels is None:
        return ()
    coerced: list[Level] = []
    for index, level in enumerate(levels):
        if isinstance(level, Level):
            coerced.append(level)
        elif isinstance(level, Mapping):
            coerced.append(Level.from_mapping(level))
        else:
            raise ValueError(f"level {index} must be a mapping or Level")
    return tuple(coerced)


def _pair_of_ints(value: Any, field_name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-item sequence")
    first = _int(value[0], f"{field_name}[0]")
    second = _int(value[1], f"{field_name}[1]")
    return first, second


def _int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    number = _int(value, field_name)
    if number < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return number
