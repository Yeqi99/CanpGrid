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
    parser.add_argument("--react-rounds", type=int, default=1)
    parser.add_argument("--object-agent", action=argparse.BooleanOptionalAction, default=True)
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
    canpgrid_react = run_react_agent(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        image_path=image_path,
        grid_image=grid_image,
        image_size=image_size,
        grid_size=grid_size,
        initial_predictions=canpgrid["predictions"],
        initial_usage=canpgrid.get("usage"),
        timeout=args.timeout,
        rounds=max(args.react_rounds, 0),
    )
    object_agent = None
    if args.object_agent:
        object_agent = run_object_agent(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            image_path=image_path,
            grid_image=grid_image,
            image_size=image_size,
            grid_size=grid_size,
            timeout=args.timeout,
            react_rounds=max(args.react_rounds, 0),
        )

    direct_eval = evaluate_interactions(annotations, direct["predictions"])
    canpgrid_eval = evaluate_interactions(annotations, canpgrid["predictions"])
    react_eval = evaluate_interactions(annotations, canpgrid_react["predictions"])
    object_eval = (
        evaluate_interactions(annotations, object_agent["predictions"]) if object_agent else None
    )
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
    react_map = draw_prediction_map(
        image_path,
        react_eval,
        ASSET_DIR / "canpgrid_react_prediction_map.png",
    )
    object_map = None
    if object_agent and object_eval:
        object_map = draw_prediction_map(
            image_path,
            object_eval,
            ASSET_DIR / "object_agent_prediction_map.png",
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
        "canpgrid_react": {
            **canpgrid_react,
            "evaluation": react_eval,
            "prediction_map": str(react_map),
        },
    }
    if object_agent and object_eval and object_map:
        report["object_agent"] = {
            **object_agent,
            "evaluation": object_eval,
            "prediction_map": str(object_map),
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


def run_react_agent(
    *,
    api_key: str,
    base_url: str,
    model: str,
    image_path: Path,
    grid_image: Path,
    image_size: tuple[int, int],
    grid_size: list[int],
    initial_predictions: list[dict[str, Any]],
    initial_usage: dict[str, Any] | None,
    timeout: int,
    rounds: int,
) -> dict[str, Any]:
    current_predictions = initial_predictions
    round_results = []
    if rounds <= 0:
        return {
            "predictions": current_predictions,
            "rounds": round_results,
            "usage": initial_usage,
            "self_check_usage": None,
            "prompt_chars": 0,
        }

    for round_index in range(1, rounds + 1):
        preview_map = draw_candidate_preview_map(
            image_path,
            current_predictions,
            ASSET_DIR / f"react_round_{round_index}_candidate_preview.png",
        )
        prompt = react_prompt(image_size, grid_size, current_predictions, round_index)
        result = run_model(
            api_key=api_key,
            base_url=base_url,
            model=model,
            image_size=image_size,
            images=[image_path, grid_image, preview_map],
            prompt=prompt,
            timeout=timeout,
            grid_size=grid_size,
        )
        result["candidate_preview_map"] = str(preview_map)
        result["round_index"] = round_index
        round_results.append(result)
        current_predictions = result["predictions"]

    self_check_usage = aggregate_usage(result.get("usage") for result in round_results)
    return {
        "predictions": current_predictions,
        "rounds": round_results,
        "usage": aggregate_usage([initial_usage, self_check_usage]),
        "self_check_usage": self_check_usage,
        "prompt_chars": sum(result.get("prompt_chars", 0) for result in round_results),
    }


def aggregate_usage(usages: Any) -> dict[str, Any] | None:
    totals: dict[str, int] = {}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals or None


def run_object_agent(
    *,
    api_key: str,
    base_url: str,
    model: str,
    image_path: Path,
    grid_image: Path,
    image_size: tuple[int, int],
    grid_size: list[int],
    timeout: int,
    react_rounds: int,
) -> dict[str, Any]:
    inventory_prompt_text = object_inventory_prompt(image_size, grid_size)
    inventory_raw = call_kimi(
        api_key=api_key,
        base_url=base_url,
        model=model,
        images=[image_path, grid_image],
        prompt=inventory_prompt_text,
        timeout=timeout,
    )
    try:
        inventory = parse_inventory(inventory_raw["content"], grid_size)
        inventory_error = None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        inventory = []
        inventory_error = str(exc)

    context_sheet = draw_object_context_sheet(
        image_path,
        inventory,
        grid_size,
        ASSET_DIR / "object_agent_multiblock_context.png",
    )
    locate = run_model(
        api_key=api_key,
        base_url=base_url,
        model=model,
        image_size=image_size,
        images=[image_path, grid_image, context_sheet],
        prompt=object_location_prompt(image_size, grid_size, inventory),
        timeout=timeout,
        grid_size=grid_size,
    )
    locate["predictions"] = dedupe_predictions_by_id(locate["predictions"])

    review = None
    final_predictions = locate["predictions"]
    if react_rounds > 0:
        preview_map = draw_candidate_preview_map(
            image_path,
            final_predictions,
            ASSET_DIR / "object_agent_candidate_preview.png",
        )
        review = run_model(
            api_key=api_key,
            base_url=base_url,
            model=model,
            image_size=image_size,
            images=[image_path, grid_image, context_sheet, preview_map],
            prompt=object_review_prompt(image_size, grid_size, inventory, final_predictions),
            timeout=timeout,
            grid_size=grid_size,
        )
        review["candidate_preview_map"] = str(preview_map)
        review["predictions"] = dedupe_predictions_by_id(review["predictions"])
        if review["predictions"]:
            final_predictions = review["predictions"]

    usage = aggregate_usage(
        [
            inventory_raw.get("usage"),
            locate.get("usage"),
            review.get("usage") if review else None,
        ]
    )
    return {
        "inventory": inventory,
        "inventory_raw": inventory_raw,
        "inventory_parse_error": inventory_error,
        "context_sheet_path": str(context_sheet),
        "locate": locate,
        "review": review,
        "predictions": final_predictions,
        "usage": usage,
        "prompt_chars": len(inventory_prompt_text)
        + locate.get("prompt_chars", 0)
        + ((review or {}).get("prompt_chars", 0)),
    }


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
Mark content cards, news rows, search suggestions, and trend items when the app
presents them as tappable entries. Do not mark decorative images, status text,
ordinary passive text, or layout chrome.

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
Mark content cards, news rows, search suggestions, and trend items when the app
presents them as tappable entries. Do not mark decorative images, status text,
ordinary passive text, or layout chrome.

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


def react_prompt(
    image_size: tuple[int, int],
    grid_size: list[int],
    predictions: list[dict[str, Any]],
    round_index: int,
) -> str:
    width, height = image_size
    candidates = format_candidate_items(predictions)
    return f"""
You are running a ReAct-style visual self-check for interactive UI localization.
You are given three images of the same app screenshot:
1. the original screenshot
2. a CanpGrid global grid overlay with {grid_size[0]} columns and {grid_size[1]} rows
3. a candidate preview image with numbered ring/crosshair markers from the previous pass

Round {round_index}: inspect the candidate markers visually, then produce the final
set of visible clickable/tappable focus points. You may confirm good candidates,
adjust inaccurate candidates, remove passive/decorative candidates, and add missing
interactive items.

Current candidate list:
{candidates}

Mark content cards, news rows, search suggestions, and trend items when the app
presents them as tappable entries. Do not mark decorative images, status text,
ordinary passive text, or layout chrome.

Prefer structured CanpGrid coordinates. The grid has zero-indexed cells:
columns 0..{grid_size[0] - 1}, rows 0..{grid_size[1] - 1}. For each final
control, return the cell containing the focus point and a normalized point inside
that cell. cell_point x=0 is the cell left edge, x=1 is the right edge, y=0 is
the top edge, y=1 is the bottom edge.

Image size is {width}x{height} pixels. Origin is top-left. This is only visual
review and localization; do not execute clicks.

Return raw JSON only:
{{
  "items": [
    {{
      "label": "search",
      "role": "button",
      "grid_cell": {{"col": 2, "row": 1}},
      "cell_point": {{"x": 0.5, "y": 0.5}},
      "source_marker": 3,
      "revision": "confirm",
      "confidence": 0.8
    }}
  ]
}}
"""


def format_candidate_items(predictions: list[dict[str, Any]]) -> str:
    items = []
    for marker, prediction in enumerate(predictions, start=1):
        point = prediction.get("click_point") or {}
        items.append(
            {
                "marker": marker,
                "label": prediction.get("label", ""),
                "role": prediction.get("role", ""),
                "x": round(float(point.get("x", 0)), 1),
                "y": round(float(point.get("y", 0)), 1),
                "source": prediction.get("point_source", ""),
            }
        )
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def object_inventory_prompt(image_size: tuple[int, int], grid_size: list[int]) -> str:
    width, height = image_size
    return f"""
You are looking at an app screenshot and a CanpGrid global grid overlay.
First identify the unique visible clickable/tappable objects. This stage is
object understanding only: do not return click coordinates.

Return each interactive object exactly once. Include buttons, tabs, input fields,
icon buttons, content cards, news rows, search suggestions, trend items, and
other tappable entries. Do not include passive status text, decorative images,
or layout chrome.

For each object, provide a stable snake_case object_id, label, role, and a rough
grid cell that contains the object or the best tap focus. The grid has
{grid_size[0]} columns and {grid_size[1]} rows, zero-indexed. Image size is
{width}x{height} pixels.

Return raw JSON only:
{{
  "objects": [
    {{
      "object_id": "search_field",
      "label": "search field",
      "role": "input",
      "rough_grid_cell": {{"col": 2, "row": 1}},
      "confidence": 0.8
    }}
  ]
}}
"""


def object_location_prompt(
    image_size: tuple[int, int],
    grid_size: list[int],
    inventory: list[dict[str, Any]],
) -> str:
    width, height = image_size
    objects = format_inventory_items(inventory)
    return f"""
You are given:
1. the original screenshot
2. the CanpGrid global grid overlay
3. a multi-block context sheet with one panel per object. Each panel shows the
   object's rough grid cell plus neighboring cells, so boundary objects stay in
   context.

Locate the click focus point for each object in this inventory. Return at most
one point per object_id. Do not create a second point for an object that is
already represented. Do not add objects outside the inventory in this stage.

Inventory:
{objects}

Use structured CanpGrid coordinates: grid_cell plus normalized cell_point.
Image size is {width}x{height} pixels. Grid cells are zero-indexed:
columns 0..{grid_size[0] - 1}, rows 0..{grid_size[1] - 1}.

Return raw JSON only:
{{
  "items": [
    {{
      "object_id": "search_field",
      "label": "search field",
      "role": "input",
      "grid_cell": {{"col": 2, "row": 1}},
      "cell_point": {{"x": 0.5, "y": 0.5}},
      "confidence": 0.8
    }}
  ]
}}
"""


def object_review_prompt(
    image_size: tuple[int, int],
    grid_size: list[int],
    inventory: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> str:
    width, height = image_size
    objects = format_inventory_items(inventory)
    candidates = format_candidate_items(predictions)
    return f"""
You are doing a final non-clicking visual self-check. You are given:
1. original screenshot
2. CanpGrid global grid overlay
3. multi-block context sheet for the object inventory
4. candidate preview image with numbered focus markers

Inventory:
{objects}

Current candidate markers:
{candidates}

For each object_id, return one final click focus point at most. Confirm good
markers, adjust inaccurate markers, and restore missing inventory objects when
you can see them. Do not return two points for the same object_id.

Use structured CanpGrid coordinates: grid_cell plus normalized cell_point.
Image size is {width}x{height} pixels. Grid cells are zero-indexed:
columns 0..{grid_size[0] - 1}, rows 0..{grid_size[1] - 1}.

Return raw JSON only:
{{
  "items": [
    {{
      "object_id": "search_field",
      "label": "search field",
      "role": "input",
      "grid_cell": {{"col": 2, "row": 1}},
      "cell_point": {{"x": 0.5, "y": 0.5}},
      "revision": "confirm",
      "confidence": 0.8
    }}
  ]
}}
"""


def format_inventory_items(inventory: list[dict[str, Any]]) -> str:
    return json.dumps(inventory, ensure_ascii=False, separators=(",", ":"))


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
        items = data.get("items", data.get("final_items", data.get("predictions", [])))
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


def parse_inventory(content: str, grid_size: list[int]) -> list[dict[str, Any]]:
    data = json.loads(repair_common_json_key_typos(extract_json_text(content)))
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("objects", data.get("items", []))
    else:
        raise ValueError("inventory JSON must be an object or list")
    if not isinstance(items, list):
        raise ValueError("inventory JSON must contain an objects list")

    inventory = []
    seen = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id", item.get("id", f"object_{index + 1:03d}")))
        if not object_id or object_id in seen:
            continue
        seen.add(object_id)
        cell_value = item.get("rough_grid_cell", item.get("grid_cell", item.get("cell")))
        try:
            col, row = normalize_cell(cell_value)
        except (TypeError, ValueError):
            col, row = 0.0, 0.0
        inventory.append(
            {
                "object_id": object_id,
                "label": str(item.get("label", item.get("name", object_id))),
                "role": str(item.get("role", item.get("type", "button"))),
                "rough_grid_cell": {
                    "col": int(clamp(round(col), 0, grid_size[0] - 1)),
                    "row": int(clamp(round(row), 0, grid_size[1] - 1)),
                },
                "confidence": item.get("confidence"),
            }
        )
    return inventory


def dedupe_predictions_by_id(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_id: dict[str, dict[str, Any]] = {}
    order = []
    for prediction in predictions:
        object_id = str(prediction.get("id", ""))
        if not object_id:
            continue
        if object_id not in best_by_id:
            best_by_id[object_id] = prediction
            order.append(object_id)
            continue
        if confidence_value(prediction) > confidence_value(best_by_id[object_id]):
            best_by_id[object_id] = prediction
    return [best_by_id[object_id] for object_id in order]


def confidence_value(prediction: dict[str, Any]) -> float:
    value = prediction.get("confidence")
    try:
        return numberish(value)
    except (TypeError, ValueError):
        return 0.0


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
        "id": item.get("id", item.get("object_id", f"prediction_{index + 1:03d}")),
        "index": index,
        "label": str(item.get("label", item.get("name", ""))),
        "role": str(item.get("role", item.get("type", ""))),
        "confidence": item.get("confidence"),
    }
    if "source_marker" in item:
        parsed["source_marker"] = item["source_marker"]
    if "revision" in item:
        parsed["revision"] = item["revision"]
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
        x = numberish(value["x"])
        y = numberish(value["y"])
    elif isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], (list, tuple)) and len(value[0]) == 2:
            x, y = [numberish(part) for part in value[0]]
        elif len(value) == 2:
            x, y = [numberish(part) for part in value]
        else:
            raise ValueError("click_point must be mapping or [x,y]")
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
        x = numberish(value.get("x", value.get("u")))
        y = numberish(value.get("y", value.get("v")))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        x, y = [numberish(part) for part in value]
    else:
        raise ValueError("cell_point must be mapping or [x,y]")
    return x, y


def numberish(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty numeric list")
        return numberish(value[0])
    return float(value)


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


def draw_candidate_preview_map(
    image_path: Path, predictions: list[dict[str, Any]], out_path: Path
) -> Path:
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    color = (30, 110, 220, 230)
    for marker, prediction in enumerate(predictions, start=1):
        point = prediction.get("click_point")
        if not point:
            continue
        draw_crosshair(draw, point["x"], point["y"], color)
        draw_label(draw, font, str(marker), point["x"] + 12, point["y"] - 12, fill=color)
    result = Image.alpha_composite(image, overlay).convert("RGB")
    result.save(out_path, format="PNG")
    return out_path


def draw_object_context_sheet(
    image_path: Path,
    inventory: list[dict[str, Any]],
    grid_size: list[int],
    out_path: Path,
    *,
    context_radius: int = 1,
    panel_size: tuple[int, int] = (360, 360),
    columns: int = 3,
) -> Path:
    source = Image.open(image_path).convert("RGB")
    font = ImageFont.load_default()
    rows = max(1, (max(1, len(inventory)) + columns - 1) // columns)
    header_height = 44
    margin = 18
    sheet = Image.new(
        "RGB",
        (
            columns * panel_size[0] + (columns + 1) * margin,
            rows * (panel_size[1] + header_height) + (rows + 1) * margin,
        ),
        "#f6f8fb",
    )
    draw = ImageDraw.Draw(sheet)

    if not inventory:
        draw.text((margin, margin), "no objects", fill="#334155", font=font)
        return save_report_png(sheet, out_path)

    for index, item in enumerate(inventory):
        col_index = index % columns
        row_index = index // columns
        panel_x = margin + col_index * (panel_size[0] + margin)
        panel_y = margin + row_index * (panel_size[1] + header_height + margin)
        cell = item["rough_grid_cell"]
        bbox = context_bbox_for_cell(
            source.size,
            grid_size,
            (cell["col"], cell["row"]),
            context_radius,
        )
        crop = source.crop(
            (
                round(bbox["x1"]),
                round(bbox["y1"]),
                round(bbox["x2"]),
                round(bbox["y2"]),
            )
        )
        crop.thumbnail(panel_size, Image.Resampling.LANCZOS)
        label = f'{index + 1}. {item["object_id"]} cell {cell["col"]},{cell["row"]}'
        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_size[0], panel_y + header_height - 6),
            radius=6,
            fill="#e8eef8",
        )
        draw.text((panel_x + 10, panel_y + 13), label, fill="#172033", font=font)
        image_x = panel_x
        image_y = panel_y + header_height
        sheet.paste(crop, (image_x, image_y))
        panel_bbox = (image_x, image_y, image_x + crop.width, image_y + crop.height)
        draw.rectangle(panel_bbox, outline="#94a3b8", width=1)
        draw_context_grid(
            draw,
            panel_bbox,
            bbox,
            source.size,
            grid_size,
            (cell["col"], cell["row"]),
        )

    return save_report_png(sheet, out_path)


def save_report_png(image: Image.Image, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")
    return out_path


def context_bbox_for_cell(
    image_size: tuple[int, int],
    grid_size: list[int],
    cell: tuple[int, int],
    radius: int,
) -> dict[str, float]:
    width, height = image_size
    cols, rows = grid_size
    min_col = int(clamp(cell[0] - radius, 0, cols - 1))
    max_col = int(clamp(cell[0] + radius, 0, cols - 1))
    min_row = int(clamp(cell[1] - radius, 0, rows - 1))
    max_row = int(clamp(cell[1] + radius, 0, rows - 1))
    return {
        "x1": min_col * (width / cols),
        "y1": min_row * (height / rows),
        "x2": (max_col + 1) * (width / cols),
        "y2": (max_row + 1) * (height / rows),
    }


def draw_context_grid(
    draw: ImageDraw.ImageDraw,
    panel_bbox: tuple[int, int, int, int],
    context_bbox: dict[str, float],
    image_size: tuple[int, int],
    grid_size: list[int],
    focus_cell: tuple[int, int],
) -> None:
    panel_x1, panel_y1, panel_x2, panel_y2 = panel_bbox
    scale_x = (panel_x2 - panel_x1) / (context_bbox["x2"] - context_bbox["x1"])
    scale_y = (panel_y2 - panel_y1) / (context_bbox["y2"] - context_bbox["y1"])
    width, height = image_size
    cols, rows = grid_size
    for col in range(cols + 1):
        x = col * (width / cols)
        if context_bbox["x1"] <= x <= context_bbox["x2"]:
            px = panel_x1 + (x - context_bbox["x1"]) * scale_x
            draw.line((px, panel_y1, px, panel_y2), fill="#38bdf8", width=1)
    for row in range(rows + 1):
        y = row * (height / rows)
        if context_bbox["y1"] <= y <= context_bbox["y2"]:
            py = panel_y1 + (y - context_bbox["y1"]) * scale_y
            draw.line((panel_x1, py, panel_x2, py), fill="#38bdf8", width=1)

    cell_x1 = focus_cell[0] * (width / cols)
    cell_y1 = focus_cell[1] * (height / rows)
    cell_x2 = (focus_cell[0] + 1) * (width / cols)
    cell_y2 = (focus_cell[1] + 1) * (height / rows)
    draw.rectangle(
        (
            panel_x1 + (cell_x1 - context_bbox["x1"]) * scale_x,
            panel_y1 + (cell_y1 - context_bbox["y1"]) * scale_y,
            panel_x1 + (cell_x2 - context_bbox["x1"]) * scale_x,
            panel_y1 + (cell_y2 - context_bbox["y1"]) * scale_y,
        ),
        outline="#f59e0b",
        width=3,
    )


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
    react = report.get("canpgrid_react")
    object_agent = report.get("object_agent")
    react_summary = react["evaluation"]["summary"] if react else None
    object_summary = object_agent["evaluation"]["summary"] if object_agent else None
    rows = render_error_rows((object_agent or react or report["canpgrid"])["evaluation"]["rows"])
    canpgrid_map = html.escape(relative_asset(report["canpgrid"]["prediction_map"]))
    react_map = html.escape(relative_asset(react["prediction_map"])) if react else ""
    react_preview = render_react_preview_figure(react)
    object_figures = render_object_agent_figures(object_agent)
    object_final_figure = render_object_agent_final_figure(object_agent)
    react_summary_row = (
        render_summary_row("CanpGrid + ReAct 自检", react_summary, react["usage"])
        if react and react_summary
        else ""
    )
    object_summary_row = (
        render_summary_row("对象清单 + 多区块 ReAct", object_summary, object_agent["usage"])
        if object_agent and object_summary
        else ""
    )
    raw_direct = html.escape(report["direct"]["raw"]["content"])
    raw_grid = html.escape(report["canpgrid"]["raw"]["content"])
    raw_react = render_react_raw(react)
    raw_object = render_object_agent_raw(object_agent)
    process_notes = render_process_notes(report)
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
      grid-template-columns: repeat(5, minmax(0, 1fr));
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
      标注真值来自手动圈选的可点击范围。模型只输出交互点击焦点，
      再按几何和标签归类为漏识别、错识别、语义错或位置错误。
      预测图只画焦点点位，人工容差框只在真值图中展示。
    </p>
  </header>
  <main>
    <section>
      <h2>总览</h2>
      <p class="muted">JSON: {html.escape(relative_asset(json_path))}</p>
      <table>
        <thead>
          <tr>
            <th>模式</th><th>分数</th><th>正确</th><th>语义错</th>
            <th>漏识别</th><th>错识别</th><th>位置错</th><th>token</th>
          </tr>
        </thead>
        <tbody>
          {render_summary_row("不用 CanpGrid", direct_summary, report["direct"]["usage"])}
          {render_summary_row("CanpGrid 单次识别", grid_summary, report["canpgrid"]["usage"])}
          {react_summary_row}
          {object_summary_row}
        </tbody>
      </table>
    </section>
    <section>
      <h2>流程说明</h2>
      {process_notes}
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
          <figcaption>CanpGrid 单次识别的点击点预测与错误。</figcaption>
        </figure>
        {react_preview}
        <figure>
          <img src="{react_map}" alt="canpgrid react">
          <figcaption>CanpGrid + ReAct 自检后的最终点击点。</figcaption>
        </figure>
        {object_figures}
        {object_final_figure}
      </div>
    </section>
    <section>
      <h2>最终错误分类</h2>
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
      {raw_react}
      {raw_object}
    </section>
  </main>
</body>
</html>
"""
    HTML_PATH.write_text(document, encoding="utf-8")


def render_react_preview_figure(react: dict[str, Any] | None) -> str:
    if not react:
        return ""
    rounds = react.get("rounds") or []
    if not rounds:
        return ""
    preview = rounds[-1].get("candidate_preview_map")
    if not preview:
        return ""
    return f"""
        <figure>
          <img src="{html.escape(relative_asset(preview))}" alt="react candidate preview">
          <figcaption>ReAct 自检前看到的候选点预览。</figcaption>
        </figure>
"""


def render_react_raw(react: dict[str, Any] | None) -> str:
    if not react:
        return ""
    blocks = []
    for result in react.get("rounds", []):
        raw = (result.get("raw") or {}).get("content", "")
        blocks.append(f"<pre>{html.escape(raw)}</pre>")
    return "\n".join(blocks)


def render_object_agent_figures(object_agent: dict[str, Any] | None) -> str:
    if not object_agent:
        return ""
    figures = []
    context = object_agent.get("context_sheet_path")
    if context:
        figures.append(
            f"""
        <figure>
          <img src="{html.escape(relative_asset(context))}" alt="object context sheet">
          <figcaption>对象清单生成的多区块上下文拼图。</figcaption>
        </figure>
"""
        )
    preview = ((object_agent.get("review") or {}).get("candidate_preview_map"))
    if preview:
        figures.append(
            f"""
        <figure>
          <img src="{html.escape(relative_asset(preview))}" alt="object candidate preview">
          <figcaption>对象定位后的候选点预览。</figcaption>
        </figure>
"""
        )
    return "\n".join(figures)


def render_object_agent_final_figure(object_agent: dict[str, Any] | None) -> str:
    if not object_agent:
        return ""
    prediction_map = object_agent.get("prediction_map")
    if not prediction_map:
        return ""
    return f"""
        <figure>
          <img src="{html.escape(relative_asset(prediction_map))}" alt="object agent">
          <figcaption>对象清单 + 多区块 ReAct 的最终点击点。</figcaption>
        </figure>
"""


def render_object_agent_raw(object_agent: dict[str, Any] | None) -> str:
    if not object_agent:
        return ""
    blocks = []
    blocks.append(
        "<pre>"
        + html.escape((object_agent.get("inventory_raw") or {}).get("content", ""))
        + "</pre>"
    )
    locate_raw = ((object_agent.get("locate") or {}).get("raw") or {}).get("content", "")
    blocks.append("<pre>" + html.escape(locate_raw) + "</pre>")
    review = object_agent.get("review")
    if review:
        blocks.append(
            "<pre>" + html.escape((review.get("raw") or {}).get("content", "")) + "</pre>"
        )
    return "\n".join(blocks)


def render_process_notes(report: dict[str, Any]) -> str:
    object_agent = report.get("object_agent")
    inventory_count = len((object_agent or {}).get("inventory") or [])
    object_note = ""
    if object_agent:
        object_note = f"""
        <tr>
          <td>对象清单 + 多区块定位</td>
          <td>
            先让模型列出唯一可点击对象，再围绕每个对象的粗略格子
            截取相邻区块。
            本次对象清单有 {inventory_count} 个对象，定位阶段按 object_id 去重，
            每个对象最多保留一个点击点。
          </td>
        </tr>
"""
    return f"""
      <table>
        <thead><tr><th>流程</th><th>含义</th></tr></thead>
        <tbody>
          <tr>
            <td>不用 CanpGrid</td>
            <td>
              模型直接看原图并输出绝对像素点，容易把语义判断和位置估计
              混在一起。
            </td>
          </tr>
          <tr>
            <td>CanpGrid 单次识别</td>
            <td>
              模型看全局网格，用 grid_cell + cell_point 表达焦点，
              CanpGrid 再换算成原图点。
            </td>
          </tr>
          <tr>
            <td>CanpGrid + ReAct 自检</td>
            <td>
              把候选点画成空心焦点预览图，模型再确认、调整、删除或
              补充点位。
            </td>
          </tr>
          {object_note}
        </tbody>
      </table>
"""


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
