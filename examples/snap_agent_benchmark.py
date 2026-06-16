from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from canpgrid import (
    compact_color_choice_prompt,
    create_grid_view,
    draw_color_choice_sheet,
    extract_color_choices,
    resolve_point,
)
from canpgrid.evaluation import evaluate_interactions
from interaction_benchmark import (
    GRID_SIZE,
    aggregate_usage,
    context_bbox_for_cell,
    dedupe_predictions_by_id,
    draw_candidate_preview_map,
    draw_object_context_sheet,
    draw_prediction_map,
    draw_truth_map,
    extract_json_text,
    load_annotations,
    normalize_prediction,
    parse_inventory,
    repair_common_json_key_typos,
)

DEFAULT_IMAGE = ROOT / "outputs" / "wechat_dom_benchmark" / "assets" / "search.png"
DEFAULT_ANNOTATIONS = (
    ROOT
    / "outputs"
    / "wechat_dom_benchmark"
    / "assets"
    / "search_targets_full_interactive.json"
)
DEFAULT_OUT_DIR = ROOT / "outputs" / "snap_agent_benchmark_search"
PALETTE_SIZE = 8


@dataclass(frozen=True)
class Provider:
    name: str
    api_key_env: str
    base_url_env: str
    model_env: str | None
    default_model: str
    disable_thinking: bool = False

    def resolve(self) -> dict[str, str]:
        return {
            "name": self.name,
            "api_key": required_env(self.api_key_env),
            "base_url": required_env(self.base_url_env),
            "model": os.environ.get(self.model_env or "", self.default_model)
            if self.model_env
            else self.default_model,
            "disable_thinking": self.disable_thinking,
        }


PROVIDERS = {
    "kimi": Provider(
        name="kimi",
        api_key_env="MOONSHOT_API_KEY",
        base_url_env="MOONSHOT_BASE_URL",
        model_env=None,
        default_model="kimi-k2.6",
        disable_thinking=True,
    ),
    "mimo": Provider(
        name="mimo",
        api_key_env="MIMO_API_KEY",
        base_url_env="MIMO_OPENAI_BASE_URL",
        model_env="MIMO_MODEL",
        default_model="mimo-v2-omni",
        disable_thinking=True,
    ),
    "hunyuan": Provider(
        name="hunyuan",
        api_key_env="TENCENT_HUNYUAN_API_KEY",
        base_url_env="TENCENT_HUNYUAN_OPENAI_BASE_URL",
        model_env="TENCENT_HUNYUAN_MODEL",
        default_model="hy3-preview",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare vision models with the CanpGrid color-snap agent flow."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--providers", default="kimi,mimo,hunyuan")
    parser.add_argument("--react-rounds", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.local")
    args = parser.parse_args()

    load_env_file(args.env_file)
    out_dir = args.out_dir
    asset_dir = out_dir / "assets"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    image_path = asset_dir / args.image.name
    shutil.copyfile(args.image, image_path)
    annotations = load_annotations(args.annotations)
    with Image.open(image_path) as image:
        image_size = image.size

    truth_map = draw_truth_map(image_path, annotations, asset_dir / "truth_map.png")
    grid = create_grid_view(
        image_path,
        grid_size=GRID_SIZE,
        overlay_mode="grid",
        detail_mode="medium",
        out_dir=asset_dir,
    )
    grid_image = Path(grid["annotated_image_path"])

    report: dict[str, Any] = {
        "image": {"path": str(image_path), "width": image_size[0], "height": image_size[1]},
        "annotations": annotations,
        "truth_map": str(truth_map),
        "grid_image": str(grid_image),
        "grid_size": GRID_SIZE,
        "providers": {},
    }
    for provider_name in [item.strip() for item in args.providers.split(",") if item.strip()]:
        provider = PROVIDERS[provider_name].resolve()
        provider_dir = asset_dir / provider_name
        provider_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        try:
            result = run_snap_agent(
                provider=provider,
                image_path=image_path,
                grid_image=grid_image,
                image_size=image_size,
                grid_size=GRID_SIZE,
                out_dir=provider_dir,
                timeout=args.timeout,
                react_rounds=max(args.react_rounds, 0),
            )
            single_eval = evaluate_interactions(annotations, result["single_pass_predictions"])
            final_eval = evaluate_interactions(annotations, result["predictions"])
            result["single_pass_evaluation"] = single_eval
            result["evaluation"] = final_eval
            result["single_pass_prediction_map"] = str(
                draw_prediction_map(
                    image_path,
                    single_eval,
                    provider_dir / "single_pass_prediction_map.png",
                )
            )
            result["prediction_map"] = str(
                draw_prediction_map(image_path, final_eval, provider_dir / "prediction_map.png")
            )
        except Exception as exc:
            result = {
                "provider": safe_provider(provider),
                "error": str(exc),
                "predictions": [],
                "usage": None,
            }
        result["elapsed_seconds"] = round(time.time() - started, 2)
        report["providers"][provider_name] = result

    json_path = out_dir / "snap_agent_benchmark.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = out_dir / "index.html"
    html_path.write_text(render_html(report, json_path), encoding="utf-8")
    print(json.dumps({"html_report": str(html_path), "json_report": str(json_path)}, indent=2))


def run_snap_agent(
    *,
    provider: dict[str, str],
    image_path: Path,
    grid_image: Path,
    image_size: tuple[int, int],
    grid_size: list[int],
    out_dir: Path,
    timeout: int,
    react_rounds: int,
) -> dict[str, Any]:
    inventory_raw = call_openai_vision(
        provider=provider,
        images=[image_path, grid_image],
        prompt=inventory_prompt(image_size, grid_size),
        timeout=timeout,
    )
    try:
        inventory = parse_inventory(inventory_raw["content"], grid_size)
        inventory_parse_error = None
    except Exception as exc:
        return {
            "provider": safe_provider(provider),
            "inventory": [],
            "inventory_raw": inventory_raw,
            "inventory_parse_error": str(exc),
            "context_sheet_path": None,
            "locate": None,
            "rounds": [],
            "single_pass_predictions": [],
            "predictions": [],
            "usage": inventory_raw.get("usage"),
        }
    context_sheet = draw_object_context_sheet(
        image_path,
        inventory,
        grid_size,
        out_dir / "multiblock_context.png",
    )
    palettes = extract_object_palettes(image_path, inventory, grid_size)
    palette_sheet = draw_palette_sheet(
        inventory,
        palettes,
        out_dir / "object_palettes.png",
    )
    locate_prompt_text = locate_prompt(image_size, grid_size, inventory, palettes)
    locate_raw = call_openai_vision(
        provider=provider,
        images=[image_path, grid_image, context_sheet, palette_sheet],
        prompt=locate_prompt_text,
        timeout=timeout,
    )
    locate_predictions, locate_errors = safe_parse_point_spec_predictions(
        locate_raw["content"], image_path, image_size, grid_size, palettes
    )
    locate_predictions = dedupe_predictions_by_id(locate_predictions)
    final_predictions = locate_predictions
    rounds = []
    for round_index in range(1, react_rounds + 1):
        preview_map = draw_candidate_preview_map(
            image_path,
            final_predictions,
            out_dir / f"review_round_{round_index}_candidate_preview.png",
        )
        review_raw = call_openai_vision(
            provider=provider,
            images=[image_path, grid_image, context_sheet, palette_sheet, preview_map],
            prompt=review_prompt(
                image_size,
                grid_size,
                inventory,
                palettes,
                final_predictions,
                round_index,
            ),
            timeout=timeout,
        )
        review_predictions, review_errors = safe_parse_point_spec_predictions(
            review_raw["content"], image_path, image_size, grid_size, palettes
        )
        review_predictions = dedupe_predictions_by_id(review_predictions)
        if review_predictions:
            final_predictions = review_predictions
        rounds.append(
            {
                "round_index": round_index,
                "raw": review_raw,
                "parse_errors": review_errors,
                "predictions": review_predictions,
                "candidate_preview_map": str(preview_map),
            }
        )

    usage = aggregate_usage(
        [
            inventory_raw.get("usage"),
            locate_raw.get("usage"),
            *[(item.get("raw") or {}).get("usage") for item in rounds],
        ]
    )
    return {
        "provider": safe_provider(provider),
        "inventory": inventory,
        "inventory_raw": inventory_raw,
        "inventory_parse_error": inventory_parse_error,
        "context_sheet_path": str(context_sheet),
        "palette_sheet_path": str(palette_sheet),
        "palettes": palettes,
        "locate": {
            "raw": locate_raw,
            "prompt_chars": len(locate_prompt_text),
            "parse_errors": locate_errors,
            "predictions": locate_predictions,
        },
        "rounds": rounds,
        "single_pass_predictions": locate_predictions,
        "predictions": final_predictions,
        "usage": usage,
    }


def inventory_prompt(image_size: tuple[int, int], grid_size: list[int]) -> str:
    width, height = image_size
    return f"""
You are identifying tappable UI objects in an app screenshot.
You are given the original screenshot and a CanpGrid global grid overlay.

First list unique visible clickable/tappable objects only. Do not return
coordinates in this stage. Include buttons, tabs, input fields, icon buttons,
content cards, search suggestions, trend/news items, and large tappable rows.
Do not include passive status text or decorative images.

Return each object exactly once with a stable snake_case object_id, label, role,
and rough_grid_cell. The grid has {grid_size[0]} columns and {grid_size[1]} rows,
zero-indexed. Image size is {width}x{height} pixels.

Return raw JSON only:
{{"objects":[{{"object_id":"search_field","label":"search field","role":"input",
"rough_grid_cell":{{"col":2,"row":1}},"confidence":0.8}}]}}
"""


def locate_prompt(
    image_size: tuple[int, int],
    grid_size: list[int],
    inventory: list[dict[str, Any]],
    palettes: dict[str, list[dict[str, Any]]],
) -> str:
    width, height = image_size
    palette_prompt = compact_palette_prompt(palettes)
    return f"""
You are locating final click focus points for known tappable UI objects.
You are given:
1. original screenshot
2. CanpGrid global grid overlay
3. a multi-block context sheet; each panel shows an object's rough cell and
   neighboring cells, so boundary objects remain visible.
4. an object color palette sheet. For each object_id it shows color choices c1,
   c2, ... extracted from that object's local context.

For each object in the inventory, return at most one final point. Do not invent
objects outside the inventory. Use a point_spec so CanpGrid can resolve it.
This benchmark is specifically testing palette-choice pixel snap. Prefer
color_snap_point whenever the object has a visible foreground, icon stroke, text
color, or button fill that matches one of its listed palette choices. Use plain
subgrid_point only for very large tappable content rows where any point inside
the row is already acceptable.

Inventory:
{json.dumps(inventory, ensure_ascii=False, separators=(",", ":"))}

Palette choices by object_id:
{palette_prompt}

Allowed point_spec forms:
1. subgrid_point:
{{"type":"subgrid_point","grid_size":{grid_size},"cell":[col,row],
"local_point":[0.5,0.5]}}

2. color_snap_point, preferred for this benchmark:
{{"type":"color_snap_point","grid_size":{grid_size},"cell":[col,row],
"local_point":[0.5,0.5],"target_color_id":"c3",
"tolerance":56,"search":{{"mode":"nearest","radius":96}},"fallback":"base_point"}}

For color_snap_point, choose the base point close to the intended click area and
choose target_color_id from that object's palette only. Prefer foreground/icon/
text colors over large white backgrounds. Use nearest for small icons or text.
Use ray only when you can state a clear direction from the base point. Use
tolerance 32-72 for antialiasing and screenshots. Do not choose white or
near-white background colors for ordinary cards, rows, or input backgrounds.
If no foreground/icon/text color is near the intended click area, use
subgrid_point instead of color_snap_point.

Image size is {width}x{height}. Origin is top-left.

Return raw JSON only:
{{"items":[{{"object_id":"search_field","label":"search field","role":"input",
"point_spec":{{"type":"subgrid_point","grid_size":{grid_size},"cell":[2,1],
"local_point":[0.5,0.5]}},"confidence":0.8}}]}}
"""


def review_prompt(
    image_size: tuple[int, int],
    grid_size: list[int],
    inventory: list[dict[str, Any]],
    palettes: dict[str, list[dict[str, Any]]],
    predictions: list[dict[str, Any]],
    round_index: int,
) -> str:
    width, height = image_size
    palette_prompt = compact_palette_prompt(palettes)
    compact_predictions = [
        {
            "object_id": item.get("id"),
            "label": item.get("label"),
            "role": item.get("role"),
            "point": item.get("click_point"),
            "point_source": item.get("point_source"),
            "point_resolution": item.get("point_resolution"),
        }
        for item in predictions
    ]
    return f"""
Round {round_index} visual self-check for click focus localization.
You are given original screenshot, global grid, multi-block object context, and
a candidate preview map with numbered ring/crosshair markers.

Review the candidate markers. Return one final point per object_id at most.
Confirm good points, adjust inaccurate points, and restore missing inventory
objects when visible. Prefer color_snap_point when one of the object's palette
choices can anchor the final point. This is observation only; do not execute
clicks.

Inventory:
{json.dumps(inventory, ensure_ascii=False, separators=(",", ":"))}

Current candidates:
{json.dumps(compact_predictions, ensure_ascii=False, separators=(",", ":"))}

Palette choices by object_id:
{palette_prompt}

Allowed point_spec forms are the same as before:
- subgrid_point
- color_snap_point with grid_size, cell, local_point, and target_color_id from
  that object's palette

Image size is {width}x{height}. Grid size is {grid_size}. Return raw JSON only:
{{"items":[{{"object_id":"search_field","label":"search field","role":"input",
"point_spec":{{"type":"subgrid_point","grid_size":{grid_size},"cell":[2,1],
"local_point":[0.5,0.5]}},"revision":"confirm","confidence":0.8}}]}}
"""


def parse_point_spec_predictions(
    content: str,
    image_path: Path,
    image_size: tuple[int, int],
    grid_size: list[int],
    palettes: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(repair_common_json_key_typos(extract_json_text(content)))
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("items", data.get("final_items", data.get("predictions", [])))
    else:
        raise ValueError("model JSON must be an object or list")
    if not isinstance(items, list):
        raise ValueError("model JSON must contain an items list")

    predictions = []
    errors = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        point_spec = item.get("point_spec")
        if isinstance(point_spec, dict):
            object_id = str(item.get("object_id", item.get("id", "")))
            point_spec = resolve_palette_point_spec(point_spec, object_id, palettes or {})
            try:
                resolved = resolve_point(image_path, [], point_spec)
            except ValueError as exc:
                errors.append(
                    {
                        "index": index,
                        "object_id": item.get("object_id", item.get("id")),
                        "error": str(exc),
                        "point_spec": point_spec,
                    }
                )
                continue
            point = resolved["point_on_original"]
            parsed = {
                "id": item.get("id", item.get("object_id", f"prediction_{index + 1:03d}")),
                "index": index,
                "label": str(item.get("label", item.get("name", ""))),
                "role": str(item.get("role", item.get("type", ""))),
                "confidence": item.get("confidence"),
                "click_point": {"x": float(point[0]), "y": float(point[1])},
                "point_source": point_spec.get("type", "point_spec"),
                "point_spec": point_spec,
                "point_resolution": resolved.get("point_resolution"),
            }
            if "revision" in item:
                parsed["revision"] = item["revision"]
            predictions.append(parsed)
            continue

        parsed = normalize_prediction(item, index, image_size, grid_size=grid_size)
        if parsed is not None:
            predictions.append(parsed)
    return predictions, errors


def safe_parse_point_spec_predictions(
    content: str,
    image_path: Path,
    image_size: tuple[int, int],
    grid_size: list[int],
    palettes: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        return parse_point_spec_predictions(
            content,
            image_path,
            image_size,
            grid_size,
            palettes,
        )
    except Exception as exc:
        return [], [{"error": str(exc), "raw_content": content}]


def extract_object_palettes(
    image_path: Path,
    inventory: list[dict[str, Any]],
    grid_size: list[int],
    *,
    context_radius: int = 1,
    palette_size: int = PALETTE_SIZE,
) -> dict[str, list[dict[str, Any]]]:
    source = Image.open(image_path).convert("RGB")
    palettes: dict[str, list[dict[str, Any]]] = {}
    for item in inventory:
        object_id = str(item["object_id"])
        cell = item["rough_grid_cell"]
        bbox = context_bbox_for_cell(
            source.size,
            grid_size,
            (cell["col"], cell["row"]),
            context_radius,
        )
        palettes[object_id] = extract_color_choices(
            source,
            bbox=bbox,
            palette_size=palette_size,
        )
    return palettes


def draw_palette_sheet(
    inventory: list[dict[str, Any]],
    palettes: dict[str, list[dict[str, Any]]],
    out_path: Path,
) -> Path:
    labels = {str(item["object_id"]): str(item.get("label", "")) for item in inventory}
    return draw_color_choice_sheet(palettes, out_path, labels=labels)


def compact_palette_prompt(palettes: dict[str, list[dict[str, Any]]]) -> str:
    return compact_color_choice_prompt(palettes)


def resolve_palette_point_spec(
    point_spec: dict[str, Any],
    object_id: str,
    palettes: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    resolved = json.loads(json.dumps(point_spec))
    if resolved.get("type") == "color_snap_point":
        if "base" not in resolved and isinstance(resolved.get("base_point"), dict):
            resolved["base"] = resolved["base_point"]
        if "base" not in resolved and "grid_size" in resolved and "cell" in resolved:
            resolved["base"] = {
                "type": "subgrid_point",
                "grid_size": resolved["grid_size"],
                "cell": resolved["cell"],
                "local_point": resolved.get("local_point", [0.5, 0.5]),
            }
        color_id = resolved.get("target_color_id", resolved.get("color_id"))
        if color_id is None and isinstance(resolved.get("target_color"), str):
            token = resolved["target_color"]
            if token.startswith("c") and token[1:].isdigit():
                color_id = token
        if color_id is not None:
            color = palette_color(palettes, object_id, str(color_id))
            resolved["target_color"] = color["hex"]
            resolved["target_color_id"] = str(color_id)
            resolved["color_choices"] = palettes.get(object_id, [])
        resolved.setdefault("fallback", "base_point")
        search = resolved.setdefault("search", {"mode": "nearest", "radius": 96})
        if isinstance(search, dict) and search.get("mode", "nearest") == "nearest":
            search.setdefault("radius", 96)
        resolved.setdefault("tolerance", 56)
    base = resolved.get("base")
    if isinstance(base, dict):
        resolved["base"] = resolve_palette_point_spec(base, object_id, palettes)
    return resolved


def palette_color(
    palettes: dict[str, list[dict[str, Any]]],
    object_id: str,
    color_id: str,
) -> dict[str, Any]:
    for color in palettes.get(object_id, []):
        if color.get("id") == color_id:
            return color
    raise ValueError(f"unknown target_color_id {color_id!r} for object_id {object_id!r}")


def call_openai_vision(
    *,
    provider: dict[str, str],
    images: list[Path],
    prompt: str,
    timeout: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}})

    payload = {
        "model": provider["model"],
        "messages": [
            {
                "role": "system",
                "content": "You are a visual UI localization agent. Return compact JSON only.",
            },
            {"role": "user", "content": content},
        ],
        "max_tokens": 4096,
    }
    if provider.get("disable_thinking"):
        payload["thinking"] = {"type": "disabled"}
    request = urllib.request.Request(
        f"{provider['base_url'].rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {provider['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{provider['name']} API request failed: {exc.code} {body}"
        ) from exc
    choice = data["choices"][0]
    return {
        "content": (choice.get("message") or {}).get("content", ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "raw_response": data,
    }


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_html(report: dict[str, Any], json_path: Path) -> str:
    provider_rows = []
    figures = []
    raw_blocks = []
    method_summary = render_method_summary(report)
    for name, result in report["providers"].items():
        provider = result.get("provider", {"name": name, "model": ""})
        if result.get("error"):
            provider_rows.append(
                f"<tr><td>{html.escape(name)}</td><td>{html.escape(provider.get('model',''))}</td>"
                f"<td colspan='10'>{html.escape(result['error'])}</td></tr>"
            )
            continue
        single = result["single_pass_evaluation"]["summary"]
        final = result["evaluation"]["summary"]
        usage = result.get("usage") or {}
        provider_rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{html.escape(provider.get('model',''))}</td>"
            f"<td>{single['score_0_to_100']}</td><td>{single['correct_count']}/"
            f"{single['target_count']}</td><td>{final['score_0_to_100']}</td>"
            f"<td>{final['correct_count']}/{final['target_count']}</td>"
            f"<td>{final['semantic_mismatch_count']}</td>"
            f"<td>{final['localization_error_count']}</td>"
            f"<td>{final['missed_interactive_count']}</td>"
            f"<td>{final['false_positive_count']}</td>"
            f"<td>{final['duplicate_prediction_count']}</td>"
            f"<td>{usage.get('total_tokens','-')}</td></tr>"
        )
        figures.append(render_provider_figures(name, result, json_path.parent))
        raw_blocks.append(render_raw_block(name, result))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CanpGrid Color Snap Agent Benchmark</title>
  <style>
    body {{ margin: 0; background: #f6f8fb; color: #172033;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }}
    header, main {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    section {{ background: white; border: 1px solid #d8deea; border-radius: 8px;
      padding: 20px; margin-bottom: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e9f2; padding: 8px; text-align: left; }}
    th {{ color: #667085; font-weight: 600; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    img {{ display: block; width: 100%; border: 1px solid #d8deea; border-radius: 6px; }}
    figcaption {{ color: #667085; font-size: 13px; margin-top: 6px; }}
    pre {{ white-space: pre-wrap; overflow: auto; background: #111827; color: #e5e7eb;
      border-radius: 6px; padding: 12px; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>Color Snap Agent Benchmark</h1>
    <p>
      对象清单、多区块上下文、point_spec / color_snap_point 定位，
      以及候选点自检。
    </p>
  </header>
  <main>
    <section>
      <h2>总览</h2>
      <p>JSON: {html.escape(relative_asset(json_path, json_path.parent))}</p>
      <table>
        <thead><tr><th>模型</th><th>Model ID</th><th>单次分</th><th>单次命中</th>
        <th>自检后分</th><th>自检后命中</th><th>语义错</th><th>位置错</th>
        <th>漏识别</th><th>错识别</th><th>重复</th><th>token</th></tr></thead>
        <tbody>{''.join(provider_rows)}</tbody>
      </table>
    </section>
    {method_summary}
    <section>
      <h2>基础图</h2>
      <div class="grid">
        <figure><img src="{html.escape(relative_asset(report['image']['path'], json_path.parent))}">
        <figcaption>原图</figcaption></figure>
        <figure><img src="{html.escape(relative_asset(report['truth_map'], json_path.parent))}">
        <figcaption>人工标注真值</figcaption></figure>
        <figure><img src="{html.escape(relative_asset(report['grid_image'], json_path.parent))}">
        <figcaption>全局 CanpGrid</figcaption></figure>
      </div>
    </section>
    {''.join(figures)}
    <section><h2>原始返回</h2>{''.join(raw_blocks)}</section>
  </main>
</body>
</html>
"""


def render_method_summary(report: dict[str, Any]) -> str:
    provider_rows = []
    best_score = None
    best_name = None
    for name, result in report["providers"].items():
        if result.get("error") or "evaluation" not in result:
            continue
        final = result["evaluation"]["summary"]
        single = result["single_pass_evaluation"]["summary"]
        final_predictions = result.get("predictions") or []
        single_predictions = result.get("single_pass_predictions") or []
        final_snaps = count_point_source(final_predictions, "color_snap_point")
        single_snaps = count_point_source(single_predictions, "color_snap_point")
        final_fallbacks = count_snap_fallbacks(final_predictions)
        parse_errors = len((result.get("locate") or {}).get("parse_errors") or [])
        score = final["score_0_to_100"]
        if best_score is None or score > best_score:
            best_score = score
            best_name = name
        provider_rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{single_snaps}</td>"
            f"<td>{final_snaps}</td><td>{final_fallbacks}</td><td>{parse_errors}</td>"
            f"<td>{single['score_0_to_100']} -> {score}</td>"
            f"<td>{final['missed_interactive_count']}</td>"
            f"<td>{final['localization_error_count']}</td></tr>"
        )

    best_text = (
        f"本轮最高为 {html.escape(str(best_name))}，{best_score} 分。"
        if best_name is not None
        else "本轮没有可用模型结果。"
    )
    return f"""
    <section>
      <h2>方法结论</h2>
      <p>
        自动主色选择题把“模型自由编颜色值”改成“模型选择 c1/c2/c3”，
        这让 JSON 和吸附流程更稳定；但它只解决最终像素微调，
        不能弥补漏掉可点击对象或选错网格区域。
        {best_text} 要冲 90+，下一步应把“对象清单”和“候选点”
        也做成选择题。
      </p>
      <table>
        <thead><tr><th>模型</th><th>单次吸附点</th><th>最终吸附点</th>
        <th>fallback</th><th>解析错</th><th>分数变化</th><th>漏识别</th>
        <th>位置错</th></tr></thead>
        <tbody>{''.join(provider_rows)}</tbody>
      </table>
    </section>
    """


def count_point_source(predictions: list[dict[str, Any]], point_source: str) -> int:
    return sum(1 for item in predictions if item.get("point_source") == point_source)


def count_snap_fallbacks(predictions: list[dict[str, Any]]) -> int:
    return sum(
        1
        for item in predictions
        if item.get("point_source") == "color_snap_point"
        and (item.get("point_resolution") or {}).get("fallback_used")
    )


def render_provider_figures(name: str, result: dict[str, Any], root: Path) -> str:
    parts = [
        ("多区块上下文", result.get("context_sheet_path")),
        ("自动提取色板", result.get("palette_sheet_path")),
        ("单次预测", result.get("single_pass_prediction_map")),
    ]
    for round_item in result.get("rounds", []):
        parts.append(
            (
                f"第 {round_item['round_index']} 轮自检预览",
                round_item.get("candidate_preview_map"),
            )
        )
    parts.append(("最终预测", result.get("prediction_map")))
    figures = []
    for caption, path in parts:
        if not path:
            continue
        figures.append(
            f"<figure><img src=\"{html.escape(relative_asset(path, root))}\">"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
    return (
        f"<section><h2>{html.escape(name)}</h2>"
        f"<div class=\"grid\">{''.join(figures)}</div></section>"
    )


def render_raw_block(name: str, result: dict[str, Any]) -> str:
    chunks = [
        ("inventory", (result.get("inventory_raw") or {}).get("content", "")),
        ("locate", ((result.get("locate") or {}).get("raw") or {}).get("content", "")),
    ]
    for round_item in result.get("rounds", []):
        chunks.append(
            (
                f"review {round_item['round_index']}",
                (round_item.get("raw") or {}).get("content", ""),
            )
        )
    body = "".join(
        f"<h3>{html.escape(name)} {html.escape(label)}</h3><pre>{html.escape(content)}</pre>"
        for label, content in chunks
    )
    return body


def relative_asset(path: str | Path, root: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def safe_provider(provider: dict[str, str]) -> dict[str, str]:
    return {
        "name": provider["name"],
        "base_url": provider["base_url"],
        "model": provider["model"],
    }


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key, value)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is required for real API benchmark calls")
    return value


if __name__ == "__main__":
    main()
