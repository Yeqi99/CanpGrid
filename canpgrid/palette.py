from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .schemas import BBox

DEFAULT_PALETTE_SIZE = 8


def extract_color_choices(
    image: str | Path | Image.Image,
    *,
    bbox: BBox | Mapping[str, Any] | Sequence[float] | None = None,
    palette_size: int = DEFAULT_PALETTE_SIZE,
) -> list[dict[str, Any]]:
    """Extract model-friendly color choices from an image or local crop."""
    if palette_size < 1:
        raise ValueError("palette_size must be >= 1")

    source = _load_rgb(image)
    if bbox is not None:
        source = source.crop(_coerce_bbox(bbox, source.size))

    sample = source.convert("RGB")
    sample.thumbnail((220, 220), Image.Resampling.LANCZOS)
    quantized = sample.quantize(colors=min(24, max(palette_size * 3, palette_size)))
    palette = quantized.getpalette() or []
    total = max(1, sample.width * sample.height)

    raw: list[dict[str, Any]] = []
    for count, index in quantized.getcolors(maxcolors=sample.width * sample.height) or []:
        rgb = tuple(palette[index * 3 : index * 3 + 3])
        if len(rgb) != 3:
            continue
        coverage = count / total
        raw.append(
            {
                "rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])],
                "hex": _rgb_to_hex(rgb),
                "coverage": round(coverage, 4),
                "score": _color_score(rgb, coverage),
            }
        )

    raw.sort(key=lambda item: item["score"], reverse=True)
    selected = _select_distinct(raw, palette_size, min_distance=28)

    # Keep one or two dominant background/context colors when space remains. The
    # model is told not to choose them blindly, but seeing them reduces ambiguity.
    if len(selected) < palette_size:
        selected.extend(
            _select_distinct(
                sorted(raw, key=lambda item: item["coverage"], reverse=True),
                palette_size - len(selected),
                min_distance=20,
                existing=selected,
            )
        )

    for index, item in enumerate(selected, start=1):
        item.pop("score", None)
        item["id"] = f"c{index}"
        item["name"] = _color_name(item["rgb"])
    return selected


def draw_color_choice_sheet(
    color_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    out_path: str | Path,
    *,
    labels: Mapping[str, str] | None = None,
    width: int = 1120,
) -> Path:
    """Render color choices as a compact visual multiple-choice sheet."""
    font = ImageFont.load_default()
    row_height = 74
    rows = list(color_groups.items())
    image = Image.new("RGB", (width, max(1, len(rows)) * row_height + 28), "#f6f8fb")
    draw = ImageDraw.Draw(image)
    for row_index, (group_id, colors) in enumerate(rows):
        y = 14 + row_index * row_height
        draw.rounded_rectangle((14, y, width - 14, y + row_height - 10), radius=6, fill="#ffffff")
        draw.text((26, y + 12), str(group_id)[:34], fill="#172033", font=font)
        if labels and group_id in labels:
            draw.text((26, y + 34), labels[group_id][:34], fill="#667085", font=font)
        for color_index, color in enumerate(colors):
            x = 320 + color_index * 96
            rgb = tuple(color["rgb"])
            draw.rectangle((x, y + 12, x + 38, y + 50), fill=rgb, outline="#334155")
            draw.text((x + 44, y + 14), str(color["id"]), fill="#172033", font=font)
            draw.text((x + 44, y + 32), str(color["hex"]), fill="#667085", font=font)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def compact_color_choice_prompt(
    color_groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    compact = {
        group_id: [
            {
                "id": color["id"],
                "hex": color["hex"],
                "name": color["name"],
                "coverage": color["coverage"],
            }
            for color in colors
        ]
        for group_id, colors in color_groups.items()
    }
    import json

    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def _load_rgb(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _coerce_bbox(
    bbox: BBox | Mapping[str, Any] | Sequence[float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    if isinstance(bbox, BBox):
        values = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
    elif isinstance(bbox, Mapping):
        values = (
            float(bbox["x1"]),
            float(bbox["y1"]),
            float(bbox["x2"]),
            float(bbox["y2"]),
        )
    elif isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) == 4:
        values = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    else:
        raise ValueError("bbox must be BBox, mapping, or [x1,y1,x2,y2]")

    left = max(0, min(width, round(values[0])))
    top = max(0, min(height, round(values[1])))
    right = max(0, min(width, round(values[2])))
    bottom = max(0, min(height, round(values[3])))
    if right <= left or bottom <= top:
        raise ValueError("bbox is empty")
    return left, top, right, bottom


def _select_distinct(
    candidates: Sequence[dict[str, Any]],
    limit: int,
    *,
    min_distance: float,
    existing: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    reference = list(existing or [])
    for candidate in candidates:
        if any(_color_distance(candidate["rgb"], item["rgb"]) < min_distance for item in reference):
            continue
        selected.append(dict(candidate))
        reference.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _color_score(rgb: tuple[int, int, int], coverage: float) -> float:
    red, green, blue = rgb
    brightness = (red + green + blue) / 3
    chroma = max(rgb) - min(rgb)
    dark_bonus = max(0.0, (150 - brightness) / 150)
    saturation_bonus = chroma / 255
    background_penalty = 0.45 if brightness > 238 and chroma < 18 else 0.0
    return coverage * 1.5 + dark_bonus + saturation_bonus - background_penalty


def _color_name(rgb: Sequence[int]) -> str:
    red, green, blue = rgb
    brightness = (red + green + blue) / 3
    chroma = max(rgb) - min(rgb)
    if brightness > 235 and chroma < 20:
        return "white"
    if brightness < 45 and chroma < 35:
        return "black"
    if chroma < 25:
        return "gray"
    if green > red and green > blue:
        return "green"
    if blue > red and blue > green:
        return "blue"
    if red > green and red > blue:
        return "red"
    return "mixed"


def _rgb_to_hex(rgb: Sequence[int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _color_distance(first: Sequence[int], second: Sequence[int]) -> float:
    return sum((first[index] - second[index]) ** 2 for index in range(3)) ** 0.5
