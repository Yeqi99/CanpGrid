from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canpgrid import create_grid_view
from canpgrid.evaluation import evaluate_interactions
from kimi_ui_compare import DEFAULT_BASE_URL, DEFAULT_MODEL, call_kimi

OUT_DIR = ROOT / "outputs" / "interaction_benchmark"
ASSET_DIR = OUT_DIR / "assets"
HTML_PATH = OUT_DIR / "index.html"
GRID_SIZE = [9, 20]


def main() -> None:
    global OUT_DIR, ASSET_DIR, HTML_PATH

    parser = argparse.ArgumentParser(
        description="Evaluate all-interactive-element detection against manual annotations."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--grid-size", default="9x20")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise SystemExit(
            "MOONSHOT_API_KEY is required. Interaction benchmarks always use real API calls."
        )

    OUT_DIR = Path(args.out_dir)
    ASSET_DIR = OUT_DIR / "assets"
    HTML_PATH = OUT_DIR / "index.html"
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    grid_size = parse_size(args.grid_size)
    image_path = copy_image(args.image)
    annotations = load_annotations(args.annotations)

    with Image.open(image_path) as image:
        image_size = image.size

    truth_map = draw_truth_map(image_path, annotations, ASSET_DIR / "truth_map.png")
    grid = create_grid_view(
        image_path,
        grid_size=grid_size,
        overlay_mode="grid",
        detail_mode="medium",
        out_dir=ASSET_DIR,
    )
    grid_image = Path(grid["annotated_image_path"])

    direct = run_model(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        image_size=image_size,
        images=[image_path],
        prompt=direct_prompt(image_size),
        timeout=args.timeout,
    )
    canpgrid = run_model(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        image_size=image_size,
        images=[image_path, grid_image],
        prompt=canpgrid_prompt(image_size, grid_size),
        timeout=args.timeout,
        grid_size=grid_size,
    )

    direct_eval = evaluate_interactions(annotations, direct["predictions"])
    canpgrid_eval = evaluate_interactions(annotations, canpgrid["predictions"])
    direct_map = draw_prediction_map(
        image_path,
        direct_eval,
        ASSET_DIR / "direct_prediction_map.png",
    )
    canpgrid_map = draw_prediction_map(
        image_path,
        canpgrid_eval,
        ASSET_DIR / "canpgrid_prediction_map.png",
    )

    report = {
        "model": args.model,
        "base_url": args.base_url,
        "api_status": "real_api",
        "image": {
            "path": str(image_path),
            "width": image_size[0],
            "height": image_size[1],
        },
        "annotations": annotations,
        "truth_map": str(truth_map),
        "grid_image": str(grid_image),
        "direct": {
            **direct,
            "evaluation": direct_eval,
            "prediction_map": str(direct_map),
        },
        "canpgrid": {
            **canpgrid,
            "evaluation": canpgrid_eval,
            "prediction_map": str(canpgrid_map),
        },
    }
    json_path = OUT_DIR / "interaction_benchmark.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, json_path)
    print(
        json.dumps(
            {"html_report": str(HTML_PATH), "json_report": str(json_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


def run_model(
    *,
    api_key: str,
    base_url: str,
    model: str,
    image_size: tuple[int, int],
    images: list[Path],
    prompt: str,
    timeout: int,
    grid_size: list[int] | None = None,
) -> dict[str, Any]:
    response = call_kimi(
        api_key=api_key,
        base_url=base_url,
        model=model,
        images=images,
        prompt=prompt,
        timeout=timeout,
    )
    try:
        predictions = parse_predictions(response["content"], image_size, grid_size=grid_size)
        parse_error = None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        predictions = []
        parse_error = str(exc)
    return {
        "raw": response,
        "predictions": predictions,
        "parse_error": parse_error,
        "usage": response.get("usage"),
        "prompt_chars": len(prompt),
    }


def direct_prompt(image_size: tuple[int, int]) -> str:
    width, height = image_size
    return f"""
You are looking at an app screenshot. Identify only visible interactive
clickable/tappable focus points: buttons, tabs, input fields, checkboxes,
switches, menu items, links, icon buttons, and other controls.

Return one click point per interactive control. Do not return bounding boxes.
Do not mark decorative images, status text, ordinary content text, cards, or
search suggestions unless the region itself is clearly tappable.

Image size is {width}x{height} pixels. Origin is top-left.

Return raw JSON only:
{{
  "items": [
    {{
      "label": "search",
      "role": "button",
      "click_point": {{"x": 45, "y": 45}},
      "confidence": 0.8
    }}
  ]
}}
"""


def canpgrid_prompt(image_size: tuple[int, int], grid_size: list[int]) -> str:
    width, height = image_size
    return f"""
You are given two images of the same app screenshot:
1. the original screenshot
2. a CanpGrid global grid overlay with {grid_size[0]} columns and {grid_size[1]} rows

Identify only visible interactive clickable/tappable focus points: buttons, tabs,
input fields, checkboxes, switches, menu items, links, icon buttons, and other
controls.

Return one structured click point per interactive control. Do not return bounding boxes.
Do not mark decorative images, status text, ordinary content text, cards, or
search suggestions unless the region itself is clearly tappable.

Use the CanpGrid overlay for localization. The grid has zero-indexed cells:
columns 0..{grid_size[0] - 1}, rows 0..{grid_size[1] - 1}. For each control,
return the cell containing the focus point and a normalized point inside that
cell. cell_point x=0 is the cell left edge, x=1 is the right edge, y=0 is the
top edge, y=1 is the bottom edge.

Image size is {width}x{height} pixels. Origin is top-left.

Return raw JSON only:
{{
  "items": [
    {{
      "label": "search",
      "role": "button",
      "grid_cell": {{"col": 2, "row": 1}},
      "cell_point": {{"x": 0.5, "y": 0.5}},
      "confidence": 0.8
    }}
  ]
}}
"""


def load_annotations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    annotations = data.get("annotations", data.get("targets", []))
    results = []
    for index, item in enumerate(annotations):
        x1, y1, x2, y2 = annotation_bbox(item["bbox"])
        results.append(
            {
                "id": item.get("id", f"target_{index + 1:03d}"),
                "index": item.get("index", index),
                "label": item.get("label", ""),
                "role": item.get("role", item.get("type", "button")),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "click_point": item.get(
                    "click_point",
                    {"x": round((x1 + x2) / 2), "y": round((y1 + y2) / 2)},
                ),
                "notes": item.get("notes", ""),
            }
        )
    return results


def annotation_bbox(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, dict):
        x1 = float(value.get("x1", value.get("left")))
        y1 = float(value.get("y1", value.get("top")))
        if "x2" in value or "right" in value:
            x2 = float(value.get("x2", value.get("right")))
        else:
            x2 = x1 + float(value["width"])
        if "y2" in value or "bottom" in value:
            y2 = float(value.get("y2", value.get("bottom")))
        else:
            y2 = y1 + float(value["height"])
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        x1, y1, x2, y2 = [float(part) for part in value]
    else:
        raise ValueError("annotation bbox must be mapping or [x1,y1,x2,y2]")
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def parse_predictions(
    content: str,
    image_size: tuple[int, int],
    *,
    grid_size: list[int] | None = None,
) -> list[dict[str, Any]]:
    json_text = extract_json_text(content)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        repaired = repair_common_json_key_typos(json_text)
        if repaired == json_text:
            raise
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            raise exc from None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("items", data.get("predictions", []))
    else:
        raise ValueError("model JSON must be an object or list")
    if not isinstance(items, list):
        raise ValueError("model JSON must contain an items list")
    predictions = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        parsed = normalize_prediction(item, index, image_size, grid_size=grid_size)
        if parsed is not None:
            predictions.append(parsed)
    return predictions


def normalize_prediction(
    item: dict[str, Any],
    index: int,
    image_size: tuple[int, int],
    *,
    grid_size: list[int] | None = None,
) -> dict[str, Any] | None:
    bbox = item.get("bbox")
    point = item.get("click_point", item.get("point"))
    grid_point = item.get("grid_point")
    if bbox is None and point is None and "x" in item and "y" in item:
        point = {"x": item["x"], "y": item["y"]}
    if grid_point is None and "grid_cell" in item:
        grid_point = {
            "grid_cell": item.get("grid_cell"),
            "cell_point": item.get("cell_point", item.get("within_cell", {"x": 0.5, "y": 0.5})),
        }
    if bbox is None and point is None and grid_point is None:
        return None

    parsed: dict[str, Any] = {
        "id": item.get("id", f"prediction_{index + 1:03d}"),
        "index": index,
        "label": str(item.get("label", item.get("name", ""))),
        "role": str(item.get("role", item.get("type", ""))),
        "confidence": item.get("confidence"),
    }
    if point is not None:
        parsed["click_point"] = normalize_point(point, image_size)
        parsed["point_source"] = "click_point"
    elif grid_point is not None and grid_size is not None:
        parsed["click_point"] = normalize_grid_point(grid_point, grid_size, image_size)
        parsed["point_source"] = "grid_cell_point"
    elif bbox is not None:
        parsed["click_point"] = bbox_center_point(normalize_bbox(bbox, image_size))
        parsed["point_source"] = "bbox_center_fallback"
    return parsed


def normalize_bbox(value: Any, image_size: tuple[int, int]) -> dict[str, float]:
    if isinstance(value, dict):
        x1 = float(value.get("x1", value.get("left")))
        y1 = float(value.get("y1", value.get("top")))
        if "x2" in value or "right" in value:
            x2 = float(value.get("x2", value.get("right")))
        else:
            x2 = x1 + float(value["width"])
        if "y2" in value or "bottom" in value:
            y2 = float(value.get("y2", value.get("bottom")))
        else:
            y2 = y1 + float(value["height"])
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        x1, y1, x2, y2 = [float(part) for part in value]
    else:
        raise ValueError("bbox must be mapping or [x1,y1,x2,y2]")
    width, height = image_size
    return {
        "x1": clamp(min(x1, x2), 0, width),
        "y1": clamp(min(y1, y2), 0, height),
        "x2": clamp(max(x1, x2), 0, width),
        "y2": clamp(max(y1, y2), 0, height),
    }


def bbox_center_point(bbox: dict[str, float]) -> dict[str, float]:
    return {
        "x": (bbox["x1"] + bbox["x2"]) / 2,
        "y": (bbox["y1"] + bbox["y2"]) / 2,
    }


def normalize_point(value: Any, image_size: tuple[int, int]) -> dict[str, float]:
    if isinstance(value, dict):
        x = float(value["x"])
        y = float(value["y"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        x, y = [float(part) for part in value]
    else:
        raise ValueError("click_point must be mapping or [x,y]")
    width, height = image_size
    return {"x": clamp(x, 0, width), "y": clamp(y, 0, height)}


def normalize_grid_point(
    value: Any, grid_size: list[int], image_size: tuple[int, int]
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("grid_point must be a mapping")
    cell = value.get("grid_cell", value.get("cell"))
    cell_point = value.get("cell_point", value.get("within_cell", {"x": 0.5, "y": 0.5}))
    col, row = normalize_cell(cell)
    local_x, local_y = normalize_local_point(cell_point)
    cols, rows = grid_size
    width, height = image_size
    x = (clamp(col, 0, cols - 1) + clamp(local_x, 0, 1)) * (width / cols)
    y = (clamp(row, 0, rows - 1) + clamp(local_y, 0, 1)) * (height / rows)
    return {"x": clamp(x, 0, width), "y": clamp(y, 0, height)}


def normalize_cell(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        col = float(value.get("col", value.get("column", value.get("x"))))
        row = float(value.get("row", value.get("y")))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        col, row = [float(part) for part in value]
    else:
        raise ValueError("grid_cell must be mapping or [col,row]")
    return col, row


def normalize_local_point(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        x = float(value.get("x", value.get("u")))
        y = float(value.get("y", value.get("v")))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        x, y = [float(part) for part in value]
    else:
        raise ValueError("cell_point must be mapping or [x,y]")
    return x, y


def extract_json_text(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    object_start = stripped.find("{")
    array_start = stripped.find("[")
    starts = [value for value in (object_start, array_start) if value != -1]
    if not starts:
        raise ValueError(f"response did not contain JSON: {content!r}")
    start = min(starts)
    end_char = "}" if stripped[start] == "{" else "]"
    end = stripped.rfind(end_char)
    if end == -1 or end <= start:
        raise ValueError(f"response did not contain complete JSON: {content!r}")
    return stripped[start : end + 1]


def repair_common_json_key_typos(text: str) -> str:
    result: list[str] = []
    index = 0
    length = len(text)
    in_string = False
    escape = False
    expecting_key = False

    while index < length:
        char = text[index]

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if char in "{,":
            expecting_key = True
            result.append(char)
            index += 1
            continue

        if expecting_key and char.isspace():
            result.append(char)
            index += 1
            continue

        if expecting_key and (char.isalpha() or char == "_"):
            key_start = index
            index += 1
            while index < length and (text[index].isalnum() or text[index] == "_"):
                index += 1
            key = text[key_start:index]
            if index < length and text[index] == '"':
                index += 1
            lookahead = index
            while lookahead < length and text[lookahead].isspace():
                lookahead += 1
            if lookahead < length and text[lookahead] == ":":
                result.append(f'"{key}"')
                result.append(text[index:lookahead])
                result.append(":")
                index = lookahead + 1
                expecting_key = False
                continue
            result.append(text[key_start:index])
            expecting_key = False
            continue

        if char == ":":
            expecting_key = False
        elif not char.isspace():
            expecting_key = False
        result.append(char)
        index += 1

    return "".join(result)


def draw_truth_map(image_path: Path, annotations: list[dict[str, Any]], out_path: Path) -> Path:
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for index, item in enumerate(annotations, start=1):
        bbox = item["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        draw.rounded_rectangle((x1, y1, x2, y2), radius=8, outline=(30, 150, 70, 220), width=4)
        draw_label(draw, font, str(index), x1, y1, fill=(30, 150, 70, 230))
    result = Image.alpha_composite(image, overlay).convert("RGB")
    result.save(out_path, format="PNG")
    return out_path


def draw_prediction_map(image_path: Path, evaluation: dict[str, Any], out_path: Path) -> Path:
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    colors = {
        "correct": (30, 150, 70, 230),
        "localization_error": (232, 145, 35, 230),
        "semantic_mismatch": (180, 80, 210, 230),
        "missed_interactive": (132, 145, 160, 210),
        "false_positive": (210, 65, 55, 230),
        "duplicate_prediction": (117, 80, 210, 220),
    }
    for index, row in enumerate(evaluation["rows"], start=1):
        color = colors[row["category"]]
        target = row.get("target")
        prediction = row.get("prediction")
        if prediction:
            if prediction.get("click_point"):
                point = prediction["click_point"]
                draw_crosshair(draw, point["x"], point["y"], color)
                draw_label(draw, font, str(index), point["x"] + 12, point["y"] - 12, fill=color)
        elif target and target.get("click_point"):
            point = target["click_point"]
            draw_crosshair(draw, point["x"], point["y"], color)
            draw_label(draw, font, str(index), point["x"] + 12, point["y"] - 12, fill=color)
    result = Image.alpha_composite(image, overlay).convert("RGB")
    result.save(out_path, format="PNG")
    return out_path


def draw_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: dict[str, Any],
    color: tuple[int, ...],
    width: int,
) -> None:
    draw.rounded_rectangle(
        (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]),
        radius=8,
        outline=color,
        width=width,
    )


def draw_crosshair(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, ...]) -> None:
    draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=color, width=4)
    draw.line((x - 22, y, x + 22, y), fill=color, width=2)
    draw.line((x, y - 22, x, y + 22), fill=color, width=2)


def draw_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    text: str,
    x: float,
    y: float,
    *,
    fill: tuple[int, ...],
) -> None:
    x = float(x)
    y = float(y)
    draw.rounded_rectangle((x, y, x + 32, y + 22), radius=4, fill=(0, 0, 0, 170))
    draw.text((x + 9, y + 5), text, fill=(255, 255, 255, 245), font=font)


def write_html(report: dict[str, Any], json_path: Path) -> None:
    direct_summary = report["direct"]["evaluation"]["summary"]
    grid_summary = report["canpgrid"]["evaluation"]["summary"]
    rows = render_error_rows(report["canpgrid"]["evaluation"]["rows"])
    canpgrid_map = html.escape(relative_asset(report["canpgrid"]["prediction_map"]))
    raw_direct = html.escape(report["direct"]["raw"]["content"])
    raw_grid = html.escape(report["canpgrid"]["raw"]["content"])
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CanpGrid Interaction Benchmark</title>
  <style>
    body {{
      margin: 0;
      background: #f6f8fb;
      color: #172033;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    header, main {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; letter-spacing: 0; }}
    section {{
      background: #fff;
      border: 1px solid #d8deea;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 18px;
    }}
    .muted {{ color: #667085; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    figure {{ margin: 0; }}
    img {{
      display: block;
      width: 100%;
      border: 1px solid #d8deea;
      border-radius: 6px;
      background: #fff;
    }}
    figcaption {{ color: #667085; font-size: 13px; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e9f2; padding: 8px; text-align: left; }}
    th {{ color: #667085; font-weight: 600; }}
    pre {{
      white-space: pre-wrap;
      overflow: auto;
      background: #111827;
      color: #e5e7eb;
      border-radius: 6px;
      padding: 12px;
      font-size: 12px;
    }}
    @media (max-width: 1000px) {{
      header, main {{ padding: 18px; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Interaction Benchmark</h1>
    <p class="muted">
      标注真值来自手动圈选的可点击范围。模型只输出交互点击焦点，再按几何和标签
      归类为漏识别、错识别、语义错或位置错误。预测图只画焦点点位，人工容差框
      只在真值图中展示。
    </p>
  </header>
  <main>
    <section>
      <h2>总览</h2>
      <p class="muted">JSON: {html.escape(relative_asset(json_path))}</p>
      <table>
        <thead>
          <tr><th>模式</th><th>分数</th><th>正确</th><th>语义错</th><th>漏识别</th><th>错识别</th><th>位置错</th><th>token</th></tr>
        </thead>
        <tbody>
          {render_summary_row("不用 CanpGrid", direct_summary, report["direct"]["usage"])}
          {render_summary_row("使用 CanpGrid", grid_summary, report["canpgrid"]["usage"])}
        </tbody>
      </table>
    </section>
    <section>
      <h2>图像对比</h2>
      <div class="grid">
        <figure>
          <img src="{html.escape(relative_asset(report["image"]["path"]))}" alt="source">
          <figcaption>原始截图。</figcaption>
        </figure>
        <figure>
          <img src="{html.escape(relative_asset(report["truth_map"]))}" alt="truth">
          <figcaption>人工标注的真实可点击范围。</figcaption>
        </figure>
        <figure>
          <img src="{html.escape(relative_asset(report["direct"]["prediction_map"]))}" alt="direct">
          <figcaption>不用 CanpGrid 的点击点预测与错误。</figcaption>
        </figure>
        <figure>
          <img src="{canpgrid_map}" alt="canpgrid">
          <figcaption>使用 CanpGrid 的点击点预测与错误。</figcaption>
        </figure>
      </div>
    </section>
    <section>
      <h2>CanpGrid 错误分类</h2>
      <table>
        <thead>
          <tr><th>类别</th><th>问题类型</th><th>真值</th><th>预测</th><th>说明</th></tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    <section>
      <h2>原始模型返回</h2>
      <pre>{raw_direct}</pre>
      <pre>{raw_grid}</pre>
    </section>
  </main>
</body>
</html>
"""
    HTML_PATH.write_text(document, encoding="utf-8")


def render_summary_row(label: str, summary: dict[str, Any], usage: dict[str, Any] | None) -> str:
    tokens = (usage or {}).get("total_tokens", "-")
    return f"""
<tr>
  <td>{html.escape(label)}</td>
  <td>{summary["score_0_to_100"]}</td>
  <td>{summary["correct_count"]}/{summary["target_count"]}</td>
  <td>{summary["semantic_mismatch_count"]}</td>
  <td>{summary["missed_interactive_count"]}</td>
  <td>{summary["false_positive_count"]}</td>
  <td>{summary["localization_error_count"]}</td>
  <td>{tokens}</td>
</tr>
"""


def render_error_rows(rows: list[dict[str, Any]]) -> str:
    return "\n".join(render_error_row(row) for row in rows)


def render_error_row(row: dict[str, Any]) -> str:
    target = row.get("target") or {}
    prediction = row.get("prediction") or {}
    return f"""
<tr>
  <td>{html.escape(row["category"])}</td>
  <td>{html.escape(row["problem_type"])}</td>
  <td>{html.escape(target.get("label") or target.get("id") or "-")}</td>
  <td>{html.escape(prediction.get("label") or prediction.get("id") or "-")}</td>
  <td>{html.escape(row.get("geometry", ""))}</td>
</tr>
"""


def copy_image(source: Path) -> Path:
    target = ASSET_DIR / source.name
    shutil.copyfile(source, target)
    return target


def parse_size(value: str) -> list[int]:
    left, right = value.lower().split("x", 1)
    return [int(left), int(right)]


def relative_asset(path: str | Path) -> str:
    return Path(path).resolve().relative_to(OUT_DIR.resolve()).as_posix()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


if __name__ == "__main__":
    main()
