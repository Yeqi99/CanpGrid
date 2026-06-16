from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from canpgrid import create_grid_view, preview_point, resolve_point
from canpgrid.evaluation import point_in_bbox
from interaction_benchmark import extract_json_text, repair_common_json_key_typos
from snap_agent_benchmark import PROVIDERS, call_openai_vision, load_env_file, safe_provider

GRID_SIZE = [9, 20]
DEFAULT_OUT_DIR = ROOT / "outputs" / "task_action_benchmark"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate UI+instruction next-click tasks with real model calls."
    )
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--chat-image", type=Path)
    parser.add_argument("--contacts-image", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--providers", default="kimi,mimo")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.local")
    args = parser.parse_args()

    load_env_file(args.env_file)
    suite = load_task_suite(args)

    out_dir = args.out_dir
    asset_dir = out_dir / "assets"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_screens(suite, asset_dir)
    report: dict[str, Any] = {
        "api_status": "real_api",
        "grid_size": GRID_SIZE,
        "screens": prepared,
        "providers": {},
    }

    provider_names = [item.strip() for item in args.providers.split(",") if item.strip()]
    for provider_name in provider_names:
        provider = PROVIDERS[provider_name].resolve()
        provider_dir = asset_dir / provider_name
        provider_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        try:
            result = run_provider_tasks(
                provider=provider,
                screens=prepared,
                out_dir=provider_dir,
                timeout=args.timeout,
            )
        except Exception as exc:
            result = {
                "provider": safe_provider(provider),
                "error": str(exc),
                "tasks": [],
                "summary": empty_summary(),
                "usage": None,
            }
        result["elapsed_seconds"] = round(time.time() - started, 2)
        report["providers"][provider_name] = result

    json_path = out_dir / "task_action_benchmark.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = out_dir / "index.html"
    html_path.write_text(render_html(report, json_path), encoding="utf-8")
    print(json.dumps({"html_report": str(html_path), "json_report": str(json_path)}, indent=2))


def load_task_suite(args: argparse.Namespace) -> dict[str, Any]:
    if args.tasks:
        return json.loads(args.tasks.read_text(encoding="utf-8"))
    if args.chat_image and args.contacts_image:
        return sample_wechat_suite(args.chat_image, args.contacts_image)
    raise SystemExit("Provide --tasks or both --chat-image and --contacts-image")


def sample_wechat_suite(chat_image: Path, contacts_image: Path) -> dict[str, Any]:
    return {
        "name": "wechat_real_next_click_sample",
        "screens": [
            {
                "id": "wechat_chats",
                "image": str(chat_image),
                "description": "微信聊天列表",
                "targets": [
                    target("chat_search", "搜索按钮", "button", 850, 130, 950, 260),
                    target("chat_plus", "右上角加号", "button", 960, 130, 1065, 260),
                    target(
                        "chat_row_wife",
                        "亲亲老婆大人聊天行",
                        "list_item",
                        0,
                        386,
                        1080,
                        580,
                    ),
                    target("chat_row_wangxue", "王雪聊天行", "list_item", 0, 580, 1080, 775),
                    target("contacts_tab", "通讯录底部标签", "tab", 270, 2200, 540, 2400),
                ],
                "tasks": [
                    task(
                        "chat_send_wife_hello",
                        "给亲亲老婆大人发消息：你好。"
                        "只判断当前截图的下一步点击。",
                        ["chat_row_wife"],
                    ),
                    task(
                        "chat_search_wangxue",
                        "搜索王雪。只判断当前截图的下一步点击。",
                        ["chat_search"],
                    ),
                    task(
                        "chat_add_friend",
                        "添加好友。只判断当前截图的下一步点击。",
                        ["chat_plus"],
                    ),
                    task(
                        "chat_go_contacts",
                        "去通讯录找联系人 A。只判断当前截图的下一步点击。",
                        ["contacts_tab"],
                    ),
                ],
            },
            {
                "id": "wechat_contacts",
                "image": str(contacts_image),
                "description": "微信通讯录",
                "targets": [
                    target("contacts_search", "搜索按钮", "button", 850, 130, 950, 260),
                    target("contacts_plus", "右上角加号", "button", 960, 130, 1065, 260),
                    target("new_friends_row", "新的朋友", "list_item", 0, 255, 1080, 407),
                    target("contact_a_row", "联系人 A", "list_item", 0, 1645, 1080, 1790),
                    target(
                        "contact_a0_row",
                        "联系人 A0雅图晨光文具广告印刷-杨冉",
                        "list_item",
                        0,
                        1790,
                        1080,
                        1945,
                    ),
                    target("chat_tab", "微信底部标签", "tab", 0, 2200, 270, 2400),
                ],
                "tasks": [
                    task(
                        "contacts_add_friend",
                        "添加好友。只判断当前截图的下一步点击。",
                        ["contacts_plus", "new_friends_row"],
                    ),
                    task(
                        "contacts_search_a0",
                        "搜索 A0雅图晨光文具广告印刷-杨冉。"
                        "只判断当前截图的下一步点击。",
                        ["contacts_search"],
                    ),
                    task(
                        "contacts_send_a_hello",
                        "给联系人 A 发消息：你好。"
                        "只判断当前截图的下一步点击。",
                        ["contact_a_row"],
                    ),
                ],
            },
        ],
    }


def target(
    target_id: str,
    label: str,
    role: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> dict[str, Any]:
    return {
        "id": target_id,
        "label": label,
        "role": role,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1},
    }


def task(task_id: str, instruction: str, accepted_target_ids: list[str]) -> dict[str, Any]:
    return {
        "id": task_id,
        "instruction": instruction,
        "accepted_target_ids": accepted_target_ids,
    }


def prepare_screens(suite: dict[str, Any], asset_dir: Path) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for screen in suite["screens"]:
        screen_dir = asset_dir / screen["id"]
        screen_dir.mkdir(parents=True, exist_ok=True)
        source = Path(screen["image"])
        image_path = screen_dir / source.name
        shutil.copyfile(source, image_path)
        with Image.open(image_path) as image:
            image_size = image.size
        truth_map = draw_truth_map(image_path, screen["targets"], screen_dir / "truth_map.png")
        grid = create_grid_view(
            image_path,
            grid_size=GRID_SIZE,
            overlay_mode="grid",
            detail_mode="medium",
            out_dir=screen_dir,
        )
        prepared[screen["id"]] = {
            **screen,
            "image": str(image_path),
            "image_size": {"width": image_size[0], "height": image_size[1]},
            "truth_map": str(truth_map),
            "grid_image": grid["annotated_image_path"],
        }
    return prepared


def run_provider_tasks(
    *,
    provider: dict[str, str],
    screens: dict[str, Any],
    out_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    task_results = []
    usages = []
    for screen_id, screen in screens.items():
        for item in screen["tasks"]:
            task_dir = out_dir / screen_id / item["id"]
            task_dir.mkdir(parents=True, exist_ok=True)
            result = run_one_task(
                provider=provider,
                screen=screen,
                task=item,
                out_dir=task_dir,
                timeout=timeout,
            )
            task_results.append(result)
            usages.extend(
                [
                    result.get("direct", {}).get("usage"),
                    result.get("canpgrid", {}).get("usage"),
                    result.get("review", {}).get("usage"),
                ]
            )
    return {
        "provider": safe_provider(provider),
        "tasks": task_results,
        "summary": summarize_task_results(task_results),
        "usage": aggregate_usage(usages),
    }


def run_one_task(
    *,
    provider: dict[str, str],
    screen: dict[str, Any],
    task: dict[str, Any],
    out_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    image_path = Path(screen["image"])
    grid_image = Path(screen["grid_image"])
    image_size = (screen["image_size"]["width"], screen["image_size"]["height"])

    direct_prompt_text = direct_task_prompt(image_size, task)
    direct_raw = call_openai_vision(
        provider=provider,
        images=[image_path],
        prompt=direct_prompt_text,
        timeout=timeout,
    )
    direct_prediction, direct_parse_error = parse_action_prediction(
        direct_raw["content"],
        image_path=image_path,
        image_size=image_size,
    )
    direct_eval = evaluate_task_prediction(screen, task, direct_prediction, direct_parse_error)

    canpgrid_prompt_text = canpgrid_task_prompt(image_size, task)
    canpgrid_raw = call_openai_vision(
        provider=provider,
        images=[image_path, grid_image],
        prompt=canpgrid_prompt_text,
        timeout=timeout,
    )
    canpgrid_prediction, canpgrid_parse_error = parse_action_prediction(
        canpgrid_raw["content"],
        image_path=image_path,
        image_size=image_size,
    )
    canpgrid_eval = evaluate_task_prediction(
        screen,
        task,
        canpgrid_prediction,
        canpgrid_parse_error,
    )

    preview = None
    review_raw = None
    review_prediction = None
    review_parse_error = None
    review_eval = canpgrid_eval
    if canpgrid_prediction and canpgrid_prediction.get("point_spec"):
        preview = preview_point(
            image_path,
            [],
            canpgrid_prediction["point_spec"],
            preview_on="original_image",
            marker_style="ring_crosshair_inset",
            out_dir=out_dir,
        )
        review_prompt_text = review_task_prompt(image_size, task, canpgrid_prediction)
        review_raw = call_openai_vision(
            provider=provider,
            images=[image_path, grid_image, Path(preview["preview_image_path"])],
            prompt=review_prompt_text,
            timeout=timeout,
        )
        review_prediction, review_parse_error = parse_action_prediction(
            review_raw["content"],
            image_path=image_path,
            image_size=image_size,
        )
        review_eval = evaluate_task_prediction(screen, task, review_prediction, review_parse_error)

    map_path = draw_task_result_map(
        image_path,
        screen,
        task,
        {
            "direct": direct_prediction,
            "canpgrid": canpgrid_prediction,
            "review": review_prediction,
        },
        out_dir / "result_map.png",
    )
    final_eval = review_eval if review_raw else canpgrid_eval
    return {
        "screen_id": screen["id"],
        "task_id": task["id"],
        "instruction": task["instruction"],
        "accepted_target_ids": task["accepted_target_ids"],
        "direct": {
            "raw": direct_raw,
            "prediction": direct_prediction,
            "parse_error": direct_parse_error,
            "evaluation": direct_eval,
            "prompt_chars": len(direct_prompt_text),
            "usage": direct_raw.get("usage"),
        },
        "canpgrid": {
            "raw": canpgrid_raw,
            "prediction": canpgrid_prediction,
            "parse_error": canpgrid_parse_error,
            "evaluation": canpgrid_eval,
            "prompt_chars": len(canpgrid_prompt_text),
            "usage": canpgrid_raw.get("usage"),
        },
        "review": {
            "raw": review_raw,
            "prediction": review_prediction,
            "parse_error": review_parse_error,
            "evaluation": review_eval,
            "preview": preview,
            "usage": review_raw.get("usage") if review_raw else None,
        }
        if review_raw
        else None,
        "final_evaluation": final_eval,
        "result_map": str(map_path),
    }


def direct_task_prompt(image_size: tuple[int, int], task_item: dict[str, Any]) -> str:
    width, height = image_size
    return f"""
You are given one mobile app screenshot and a user task.
Choose exactly one next click/tap focus point for the current screen.

User task:
{task_item['instruction']}

If the task needs multiple future steps, return only the next click on this
current screen. Do not pretend to type unless the current screen already has the
right input focused. Do not execute anything.

Image size is {width}x{height}. Origin is top-left. Return raw JSON only:
{{
  "action": "open_search_or_open_chat_or_add_friend",
  "target_id": "short_snake_case_guess",
  "label": "visible target label",
  "role": "button_or_tab_or_list_item",
  "click_point": {{"x": 100, "y": 200}},
  "confidence": 0.8,
  "reason": "brief visual reason"
}}
"""


def canpgrid_task_prompt(image_size: tuple[int, int], task_item: dict[str, Any]) -> str:
    width, height = image_size
    return f"""
You are given two images of the same mobile app screenshot:
1. the original screenshot
2. a CanpGrid global grid overlay with {GRID_SIZE[0]} columns and {GRID_SIZE[1]} rows

User task:
{task_item['instruction']}

Choose exactly one next click/tap focus point for the current screen. If the
task needs multiple future steps, return only the next click. Do not execute
anything.

Use CanpGrid for localization. Return point_spec, not a bbox. The simplest form is:
{{"type":"subgrid_point","grid_size":{GRID_SIZE},"cell":[col,row],"local_point":[0.5,0.5]}}

Image size is {width}x{height}. Origin is top-left. Return raw JSON only:
{{
  "action": "open_search_or_open_chat_or_add_friend",
  "target_id": "short_snake_case_guess",
  "label": "visible target label",
  "role": "button_or_tab_or_list_item",
  "point_spec": {{
    "type":"subgrid_point",
    "grid_size":{GRID_SIZE},
    "cell":[0,0],
    "local_point":[0.5,0.5]
  }},
  "confidence": 0.8,
  "reason": "brief visual reason"
}}
"""


def review_task_prompt(
    image_size: tuple[int, int],
    task_item: dict[str, Any],
    prediction: dict[str, Any],
) -> str:
    width, height = image_size
    compact_prediction = {
        "target_id": prediction.get("target_id"),
        "label": prediction.get("label"),
        "role": prediction.get("role"),
        "click_point": prediction.get("click_point"),
        "point_spec": prediction.get("point_spec"),
    }
    return f"""
You are doing a non-clicking visual self-check for one UI task. You are given:
1. original screenshot
2. CanpGrid global grid overlay
3. preview image with a ring/crosshair marking the proposed next click point

User task:
{task_item['instruction']}

Current proposed point:
{json.dumps(compact_prediction, ensure_ascii=False, separators=(',', ':'))}

Confirm the point if it is the right next click. Otherwise adjust it. Return
exactly one next click point as point_spec. Do not return a bbox. Do not execute
anything.

Image size is {width}x{height}. Grid size is {GRID_SIZE}. Return raw JSON only:
{{
  "action": "confirm_or_adjust_next_click",
  "target_id": "short_snake_case_guess",
  "label": "visible target label",
  "role": "button_or_tab_or_list_item",
  "point_spec": {{
    "type":"subgrid_point",
    "grid_size":{GRID_SIZE},
    "cell":[0,0],
    "local_point":[0.5,0.5]
  }},
  "confidence": 0.8,
  "reason": "brief visual reason"
}}
"""


def parse_action_prediction(
    content: str,
    *,
    image_path: Path,
    image_size: tuple[int, int],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(repair_common_json_key_typos(extract_json_text(content)))
        if isinstance(data, list):
            data = data[0] if data else {}
        if "items" in data and isinstance(data["items"], list):
            data = data["items"][0] if data["items"] else {}
        if "item" in data and isinstance(data["item"], dict):
            data = data["item"]
        if not isinstance(data, dict):
            raise ValueError("prediction JSON must be an object")

        point_spec = data.get("point_spec")
        if not isinstance(point_spec, dict):
            point_spec = grid_point_spec_from_legacy_fields(data)

        click_point = data.get("click_point")
        if isinstance(point_spec, dict):
            resolved = resolve_point(image_path, [], point_spec)
            point = resolved["point_on_original"]
            click_point = {"x": float(point[0]), "y": float(point[1])}
        elif isinstance(click_point, dict):
            click_point = {
                "x": float(click_point["x"]),
                "y": float(click_point["y"]),
            }
            point_spec = normalized_point_spec(click_point, image_size)
        elif isinstance(data.get("bbox"), dict):
            bbox = data["bbox"]
            click_point = {
                "x": (float(bbox["x1"]) + float(bbox["x2"])) / 2,
                "y": (float(bbox["y1"]) + float(bbox["y2"])) / 2,
            }
            point_spec = normalized_point_spec(click_point, image_size)
        else:
            raise ValueError("prediction must include click_point or point_spec")

        return {
            "action": data.get("action", ""),
            "target_id": data.get("target_id", data.get("id", "")),
            "label": data.get("label", ""),
            "role": data.get("role", ""),
            "click_point": click_point,
            "point_spec": point_spec,
            "confidence": data.get("confidence"),
            "reason": data.get("reason", ""),
        }, None
    except Exception as exc:
        return None, str(exc)


def grid_point_spec_from_legacy_fields(data: dict[str, Any]) -> dict[str, Any] | None:
    grid_cell = data.get("grid_cell")
    cell_point = data.get("cell_point")
    if not isinstance(grid_cell, dict) or not isinstance(cell_point, dict):
        return None
    return {
        "type": "subgrid_point",
        "grid_size": GRID_SIZE,
        "cell": [int(grid_cell["col"]), int(grid_cell["row"])],
        "local_point": [float(cell_point["x"]), float(cell_point["y"])],
    }


def normalized_point_spec(
    click_point: dict[str, float],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    width, height = image_size
    return {
        "type": "normalized_point",
        "value": [
            max(0.0, min(1.0, click_point["x"] / width)),
            max(0.0, min(1.0, click_point["y"] / height)),
        ],
    }


def evaluate_task_prediction(
    screen: dict[str, Any],
    task_item: dict[str, Any],
    prediction: dict[str, Any] | None,
    parse_error: str | None,
) -> dict[str, Any]:
    accepted_ids = set(task_item["accepted_target_ids"])
    targets = {item["id"]: item for item in screen["targets"]}
    accepted_targets = [targets[target_id] for target_id in accepted_ids if target_id in targets]
    if parse_error or not prediction:
        return {"category": "parse_error", "score_0_to_100": 0, "parse_error": parse_error}

    point = point_tuple(prediction["click_point"])
    for target_item in accepted_targets:
        if point_in_bbox(point, bbox_tuple(target_item["bbox"])):
            return {
                "category": "correct",
                "score_0_to_100": 100,
                "matched_target_id": target_item["id"],
            }

    if prediction.get("target_id") in accepted_ids:
        return {
            "category": "localization_error",
            "score_0_to_100": 0,
            "matched_target_id": prediction.get("target_id"),
        }

    for target_item in screen["targets"]:
        if point_in_bbox(point, bbox_tuple(target_item["bbox"])):
            return {
                "category": "wrong_action",
                "score_0_to_100": 0,
                "landed_target_id": target_item["id"],
            }
    return {"category": "off_target", "score_0_to_100": 0}


def point_tuple(point: dict[str, Any]) -> tuple[float, float]:
    return float(point["x"]), float(point["y"])


def bbox_tuple(bbox: dict[str, Any]) -> tuple[float, float, float, float]:
    return float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])


def summarize_task_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    stages = ["direct", "canpgrid", "final"]
    summary: dict[str, Any] = {"task_count": len(results)}
    for stage in stages:
        evals = []
        for item in results:
            if stage == "final":
                evals.append(item["final_evaluation"])
            else:
                evals.append(item[stage]["evaluation"])
        correct = sum(1 for item in evals if item["category"] == "correct")
        summary[stage] = {
            "correct_count": correct,
            "score_0_to_100": round(100 * correct / len(evals), 1) if evals else 0,
            "parse_error_count": sum(1 for item in evals if item["category"] == "parse_error"),
            "wrong_action_count": sum(1 for item in evals if item["category"] == "wrong_action"),
            "localization_error_count": sum(
                1 for item in evals if item["category"] == "localization_error"
            ),
            "off_target_count": sum(1 for item in evals if item["category"] == "off_target"),
        }
    return summary


def empty_summary() -> dict[str, Any]:
    return {
        "task_count": 0,
        "direct": {"correct_count": 0, "score_0_to_100": 0},
        "canpgrid": {"correct_count": 0, "score_0_to_100": 0},
        "final": {"correct_count": 0, "score_0_to_100": 0},
    }


def aggregate_usage(usages: list[dict[str, Any] | None]) -> dict[str, int] | None:
    totals: dict[str, int] = {}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals or None


def draw_truth_map(image_path: Path, targets: list[dict[str, Any]], out_path: Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for target_item in targets:
        bbox = target_item["bbox"]
        box = (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])
        draw.rectangle(box, outline="#16a34a", width=4)
        draw.text((bbox["x1"] + 6, bbox["y1"] + 6), target_item["id"], fill="#16a34a", font=font)
    image.save(out_path)
    return out_path


def draw_task_result_map(
    image_path: Path,
    screen: dict[str, Any],
    task_item: dict[str, Any],
    predictions: dict[str, dict[str, Any] | None],
    out_path: Path,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    accepted = set(task_item["accepted_target_ids"])
    for target_item in screen["targets"]:
        bbox = target_item["bbox"]
        color = "#16a34a" if target_item["id"] in accepted else "#94a3b8"
        width = 5 if target_item["id"] in accepted else 2
        draw.rectangle((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]), outline=color, width=width)
        if target_item["id"] in accepted:
            draw.text((bbox["x1"] + 6, bbox["y1"] + 6), target_item["id"], fill=color, font=font)

    marker_colors = {"direct": "#2563eb", "canpgrid": "#9333ea", "review": "#dc2626"}
    for name, prediction in predictions.items():
        if not prediction:
            continue
        point = prediction["click_point"]
        color = marker_colors[name]
        x = float(point["x"])
        y = float(point["y"])
        radius = 18
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=5)
        draw.line((x - 24, y, x + 24, y), fill=color, width=3)
        draw.line((x, y - 24, x, y + 24), fill=color, width=3)
        draw.text((x + 22, y - 12), name, fill=color, font=font)
    image.save(out_path)
    return out_path


def render_html(report: dict[str, Any], json_path: Path) -> str:
    rows = []
    details = []
    for name, result in report["providers"].items():
        if result.get("error"):
            rows.append(
                f"<tr><td>{html.escape(name)}</td><td colspan='8'>"
                f"{html.escape(result['error'])}</td></tr>"
            )
            continue
        summary = result["summary"]
        usage = result.get("usage") or {}
        rows.append(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td>{summary['direct']['score_0_to_100']}</td>"
            f"<td>{summary['canpgrid']['score_0_to_100']}</td>"
            f"<td>{summary['final']['score_0_to_100']}</td>"
            f"<td>{summary['final']['correct_count']}/{summary['task_count']}</td>"
            f"<td>{summary['final']['wrong_action_count']}</td>"
            f"<td>{summary['final']['localization_error_count']}</td>"
            f"<td>{summary['final']['parse_error_count']}</td>"
            f"<td>{usage.get('total_tokens','-')}</td></tr>"
        )
        details.append(render_provider_detail(name, result, json_path.parent))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CanpGrid Task Action Benchmark</title>
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
    code {{ background: #eef2f7; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Task Action Benchmark</h1>
    <p>
      输入是 UI 截图 + 用户要求；输出只评估当前屏幕下一步点击焦点，
      不执行点击。
    </p>
  </header>
  <main>
    <section>
      <h2>总览</h2>
      <p>JSON: {html.escape(relative_asset(json_path, json_path.parent))}</p>
      <table>
        <thead><tr><th>模型</th><th>直接分</th><th>CanpGrid 分</th><th>自检后分</th>
        <th>命中</th><th>策略错</th><th>位置错</th><th>解析错</th>
        <th>token</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>读法</h2>
      <p>
        绿色框是该任务允许的下一步目标；灰框是同屏其他可点击对象。
        蓝色是直接看图，紫色是 CanpGrid，红色是 preview 自检后的最终点。
        100 分表示所有任务的最终点都落在允许目标区域内。
      </p>
    </section>
    {''.join(details)}
  </main>
</body>
</html>
"""


def render_provider_detail(name: str, result: dict[str, Any], root: Path) -> str:
    figures = []
    for item in result["tasks"]:
        final = item["final_evaluation"]
        caption = (
            f"{item['screen_id']} / {item['task_id']} / {final['category']} / "
            f"{item['instruction']}"
        )
        figures.append(
            f"<figure><img src=\"{html.escape(relative_asset(item['result_map'], root))}\">"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
    return (
        f"<section><h2>{html.escape(name)}</h2>"
        f"<div class=\"grid\">{''.join(figures)}</div></section>"
    )


def relative_asset(path: str | Path, root: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
