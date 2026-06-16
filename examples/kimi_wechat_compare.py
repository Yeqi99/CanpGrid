from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canpgrid import create_grid_view
from kimi_ui_compare import DEFAULT_BASE_URL, DEFAULT_MODEL, call_kimi

OUT_DIR = ROOT / "outputs" / "kimi_wechat_compare"
ASSET_DIR = OUT_DIR / "assets"
HTML_PATH = OUT_DIR / "index.html"
GRID_SIZE = [9, 20]

@dataclass(frozen=True)
class Target:
    target_id: str
    label: str
    instruction: str
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    source_path: Path
    targets: list[Target]


def build_scenarios(chat_list_image: Path, search_image: Path) -> list[Scenario]:
    return [
        Scenario(
            scenario_id="wechat_chat_list",
            title="微信消息列表",
            source_path=chat_list_image,
            targets=[
            Target(
                "search",
                "搜索按钮",
                "Tap the search icon in the top-right navigation bar.",
                (895, 191),
                (840, 140, 950, 245),
            ),
            Target(
                "plus",
                "加号按钮",
                "Tap the plus/add icon in the top-right navigation bar.",
                (1006, 191),
                (955, 140, 1065, 245),
            ),
            Target(
                "first_chat",
                "第一个聊天行",
                "Tap the first visible chat row named 亲亲老婆大人.",
                (640, 480),
                (170, 380, 1080, 575),
            ),
            Target(
                "contacts_tab",
                "通讯录 tab",
                "Tap the bottom tab labeled 通讯录.",
                (405, 2250),
                (300, 2175, 515, 2365),
            ),
            Target(
                "me_tab",
                "我 tab",
                "Tap the bottom tab labeled 我.",
                (945, 2250),
                (840, 2175, 1070, 2365),
            ),
            ],
        ),
        Scenario(
            scenario_id="wechat_search",
            title="微信搜索页",
            source_path=search_image,
            targets=[
            Target(
                "back",
                "返回按钮",
                "Tap the back arrow on the left side.",
                (58, 214),
                (15, 160, 105, 265),
            ),
            Target(
                "search_field",
                "搜索输入框",
                "Tap inside the search input field.",
                (575, 215),
                (105, 150, 1040, 285),
            ),
            Target(
                "camera",
                "相机按钮",
                "Tap the camera icon beside 深度思考.",
                (446, 360),
                (400, 315, 495, 410),
            ),
            Target(
                "ai_search",
                "AI搜索按钮",
                "Tap the AI搜索 action on the right.",
                (970, 360),
                (880, 315, 1065, 410),
            ),
            Target(
                "voice_button",
                "语音提问按钮",
                "Tap the large green button labeled 按住 说出你的问题.",
                (540, 1950),
                (205, 1885, 875, 2025),
            ),
            Target(
                "page_settings",
                "页面设置",
                "Tap the 页面设置 text near the bottom.",
                (540, 2260),
                (380, 2210, 700, 2325),
            ),
            ],
        ),
    ]


def main() -> None:
    global OUT_DIR, ASSET_DIR, HTML_PATH

    parser = argparse.ArgumentParser(
        description="Compare Kimi localization on real WeChat screenshots."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--chat-list-image", type=Path, required=True)
    parser.add_argument("--search-image", type=Path, required=True)
    args = parser.parse_args()

    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set MOONSHOT_API_KEY in your shell before running this script. "
            "The key is intentionally not read from code or repository files."
        )

    OUT_DIR = Path(args.out_dir)
    ASSET_DIR = OUT_DIR / "assets"
    HTML_PATH = OUT_DIR / "index.html"
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    reports = []
    for scenario in build_scenarios(args.chat_list_image, args.search_image):
        reports.append(run_scenario(scenario, api_key, args.base_url, args.model))

    output = {
        "model": args.model,
        "base_url": args.base_url,
        "grid_size": GRID_SIZE,
        "scenarios": reports,
    }
    (OUT_DIR / "kimi_wechat_comparison.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_html(output)
    print(
        json.dumps(
            {
                "html_report": str(HTML_PATH),
                "json_report": str(OUT_DIR / "kimi_wechat_comparison.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_scenario(
    scenario: Scenario, api_key: str, base_url: str, model: str
) -> dict[str, Any]:
    source_copy = ASSET_DIR / f"{scenario.scenario_id}.jpg"
    shutil.copyfile(scenario.source_path, source_copy)

    with Image.open(source_copy) as image:
        width, height = image.size

    grid_view = create_grid_view(
        source_copy,
        grid_size=GRID_SIZE,
        overlay_mode="hybrid",
        detail_mode="medium",
        ruler_config={"tick_x": 18, "tick_y": 40},
        out_dir=ASSET_DIR,
    )
    grid_path = Path(grid_view["annotated_image_path"])

    without = call_kimi(
        api_key=api_key,
        base_url=base_url,
        model=model,
        images=[source_copy],
        prompt=without_prompt(scenario, width, height),
    )
    with_canpgrid = call_kimi(
        api_key=api_key,
        base_url=base_url,
        model=model,
        images=[source_copy, grid_path],
        prompt=with_canpgrid_prompt(scenario, width, height),
    )

    without_items = parse_items(without["content"])
    with_items = parse_items(with_canpgrid["content"], image_size=(width, height))
    without_metrics = score_items(scenario.targets, without_items)
    with_metrics = score_items(scenario.targets, with_items)

    without_map = draw_prediction_map(
        source_copy,
        scenario.targets,
        without_items,
        ASSET_DIR / f"{scenario.scenario_id}_without_map.png",
        prediction_color=(224, 72, 65, 235),
    )
    with_map = draw_prediction_map(
        source_copy,
        scenario.targets,
        with_items,
        ASSET_DIR / f"{scenario.scenario_id}_with_canpgrid_map.png",
        prediction_color=(16, 143, 179, 235),
    )

    return {
        "scenario_id": scenario.scenario_id,
        "title": scenario.title,
        "source_image": relative_asset(source_copy),
        "grid_image": relative_asset(grid_path),
        "without_map": relative_asset(without_map),
        "with_canpgrid_map": relative_asset(with_map),
        "targets": [target_to_dict(target) for target in scenario.targets],
        "without_canpgrid": {
            "raw": without,
            "items": without_items,
            "metrics": without_metrics,
        },
        "with_canpgrid": {
            "raw": with_canpgrid,
            "items": with_items,
            "metrics": with_metrics,
        },
    }


def without_prompt(scenario: Scenario, width: int, height: int) -> str:
    return f"""
You are looking at a real WeChat phone screenshot.
Image size is {width}x{height} pixels. The origin is the top-left corner.
Estimate tap coordinates in original image pixels for each target.

Targets:
{target_lines(scenario.targets)}

Return raw JSON only, no markdown:
{{
  "mode": "without_canpgrid",
  "items": [
    {{"id": "search", "x": 123, "y": 456, "confidence": 0.8}}
  ]
}}
"""


def with_canpgrid_prompt(scenario: Scenario, width: int, height: int) -> str:
    return f"""
You are given two images of the same WeChat screenshot:
1. the original screenshot
2. a CanpGrid overlay image with a {GRID_SIZE[0]}x{GRID_SIZE[1]} global grid and ruler ticks

Image size is {width}x{height} pixels. The origin is the top-left corner.
Use the grid overlay to reduce coordinate scale and offset errors. Do not return
pixel coordinates yourself. Instead, return explicit CanpGrid column and row
indices plus a local point inside that cell.

Important:
- `col` means horizontal column, left to right, valid range 0..{GRID_SIZE[0] - 1}.
- `row` means vertical row, top to bottom, valid range 0..{GRID_SIZE[1] - 1}.
- Top-left cell is `col=0,row=0`.
- Bottom-right cell is `col={GRID_SIZE[0] - 1},row={GRID_SIZE[1] - 1}`.
- `local_x` and `local_y` are 0..1 positions inside that cell.

Targets:
{target_lines(scenario.targets)}

Return raw JSON only, no markdown:
{{
  "mode": "with_canpgrid",
  "items": [
    {{"id": "search", "col": 7, "row": 1, "local_x": 0.5, "local_y": 0.5, "confidence": 0.8}}
  ]
}}
"""


def target_lines(targets: list[Target]) -> str:
    return "\n".join(
        f"- {target.target_id}: {target.label}. {target.instruction}" for target in targets
    )


def parse_items(
    content: str,
    *,
    image_size: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    json_text = extract_json_text(content)
    data = json.loads(json_text)
    items = data.get("items", [])
    parsed = []
    for item in items:
        if not isinstance(item, dict):
            continue
        target_id = str(item.get("id", ""))
        if not target_id:
            continue
        try:
            x, y = item_to_point(item, image_size=image_size)
        except (KeyError, TypeError, ValueError):
            continue
        parsed.append(
            {
                "id": target_id,
                "x": x,
                "y": y,
                "confidence": item.get("confidence"),
                "cell": item.get("cell"),
                "local_point": item.get("local_point"),
            }
        )
    return parsed


def item_to_point(
    item: dict[str, Any],
    *,
    image_size: tuple[int, int] | None,
) -> tuple[float, float]:
    if "x" in item and "y" in item:
        return float(item["x"]), float(item["y"])

    if image_size is None:
        raise KeyError("x/y")

    if "col" in item and "row" in item:
        cell_x = int(item["col"])
        cell_y = int(item["row"])
        local_x = float(item.get("local_x", 0.5))
        local_y = float(item.get("local_y", 0.5))
    else:
        cell = item["cell"]
        local = item.get("local_point", [0.5, 0.5])
        if len(cell) != 2 or len(local) != 2:
            raise ValueError("cell and local_point must be 2D")
        cell_x = int(cell[0])
        cell_y = int(cell[1])
        local_x = float(local[0])
        local_y = float(local[1])

    width, height = image_size
    return (
        (cell_x + local_x) * (width / GRID_SIZE[0]),
        (cell_y + local_y) * (height / GRID_SIZE[1]),
    )


def extract_json_text(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Kimi response did not contain JSON: {content!r}")
    return stripped[start : end + 1]


def score_items(targets: list[Target], items: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {item["id"]: item for item in items}
    rows = []
    for target in targets:
        item = by_id.get(target.target_id)
        if item is None:
            rows.append(
                {
                    "id": target.target_id,
                    "label": target.label,
                    "missing": True,
                    "hit": False,
                }
            )
            continue
        x = float(item["x"])
        y = float(item["y"])
        error = math.dist((x, y), target.center)
        hit = point_in_bbox((x, y), target.bbox)
        rows.append(
            {
                "id": target.target_id,
                "label": target.label,
                "x": round(x, 1),
                "y": round(y, 1),
                "target": [round(target.center[0], 1), round(target.center[1], 1)],
                "error_px": round(error, 1),
                "hit": hit,
                "missing": False,
            }
        )

    scored = [row for row in rows if not row.get("missing")]
    errors = [float(row["error_px"]) for row in scored]
    hits = [row for row in scored if row["hit"]]
    return {
        "rows": rows,
        "returned": len(scored),
        "target_count": len(targets),
        "hit_count": len(hits),
        "hit_rate": round(len(hits) / len(targets), 3),
        "mean_error_px": round(sum(errors) / len(errors), 1) if errors else None,
        "max_error_px": round(max(errors), 1) if errors else None,
    }


def point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_prediction_map(
    source_path: Path,
    targets: list[Target],
    items: list[dict[str, Any]],
    out_path: Path,
    *,
    prediction_color: tuple[int, int, int, int],
) -> Path:
    image = Image.open(source_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    target_by_id = {target.target_id: target for target in targets}

    for target in targets:
        x1, y1, x2, y2 = target.bbox
        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=10,
            outline=(46, 160, 67, 160),
            width=4,
        )

    for index, item in enumerate(items, start=1):
        x = float(item["x"])
        y = float(item["y"])
        target = target_by_id.get(item["id"])
        hit = target is not None and point_in_bbox((x, y), target.bbox)
        color = (42, 157, 84, 240) if hit else prediction_color
        draw.ellipse((x - 18, y - 18, x + 18, y + 18), outline=color, width=5)
        draw.line((x - 32, y, x + 32, y), fill=color, width=3)
        draw.line((x, y - 32, x, y + 32), fill=color, width=3)
        draw.rounded_rectangle((x + 22, y - 18, x + 72, y + 18), radius=6, fill=(0, 0, 0, 180))
        draw.text((x + 36, y - 8), str(index), fill=(255, 255, 255, 255), font=font)

    result = Image.alpha_composite(image, overlay).convert("RGB")
    result.save(out_path, format="PNG")
    return out_path


def target_to_dict(target: Target) -> dict[str, Any]:
    return {
        "id": target.target_id,
        "label": target.label,
        "instruction": target.instruction,
        "center": list(target.center),
        "bbox": list(target.bbox),
    }


def write_html(data: dict[str, Any]) -> None:
    sections = "\n".join(render_scenario(scenario) for scenario in data["scenarios"])
    summary_rows = "\n".join(render_summary_row(scenario) for scenario in data["scenarios"])
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kimi WeChat UI Localization Compare</title>
  <style>
    body {{
      margin: 0;
      background: #f6f8fb;
      color: #172033;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    header, main {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    h3 {{ margin: 18px 0 10px; font-size: 17px; }}
    section {{
      background: white;
      border: 1px solid #d8deea;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 18px;
    }}
    .muted {{ color: #667085; }}
    .pill {{
      display: inline-block;
      border: 1px solid #d8deea;
      border-radius: 999px;
      padding: 5px 9px;
      margin: 4px 6px 0 0;
      color: #667085;
      background: #fbfcff;
      font-size: 13px;
    }}
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
    .hit {{ color: #168a45; font-weight: 650; }}
    .miss {{ color: #c24135; font-weight: 650; }}
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
    <h1>Kimi 微信真实 UI 定位对比</h1>
    <p class="muted">
      同一批目标点位，比较 Kimi 只看原图，以及 Kimi 看原图 + CanpGrid 全局网格辅助图。
      绿色框是人工标定的有效点击区域，十字是 Kimi 返回的位置。
    </p>
    <span class="pill">model {html.escape(data["model"])}</span>
    <span class="pill">grid {GRID_SIZE[0]}x{GRID_SIZE[1]}</span>
    <span class="pill">base {html.escape(data["base_url"])}</span>
  </header>
  <main>
    <section>
      <h2>总览</h2>
      <table>
        <thead>
          <tr>
            <th>截图</th><th>不用 CanpGrid</th><th>使用 CanpGrid</th><th>平均误差变化</th>
          </tr>
        </thead>
        <tbody>{summary_rows}</tbody>
      </table>
    </section>
    {sections}
  </main>
</body>
</html>
"""
    HTML_PATH.write_text(document, encoding="utf-8")


def render_summary_row(scenario: dict[str, Any]) -> str:
    without = scenario["without_canpgrid"]["metrics"]
    with_grid = scenario["with_canpgrid"]["metrics"]
    delta = None
    if without["mean_error_px"] is not None and with_grid["mean_error_px"] is not None:
        delta = round(without["mean_error_px"] - with_grid["mean_error_px"], 1)
    delta_text = "n/a" if delta is None else f"{delta:+.1f}px"
    without_text = (
        f'{without["hit_count"]}/{without["target_count"]} hit, '
        f'mean {without["mean_error_px"]}px'
    )
    with_text = (
        f'{with_grid["hit_count"]}/{with_grid["target_count"]} hit, '
        f'mean {with_grid["mean_error_px"]}px'
    )
    return f"""
<tr>
  <td>{html.escape(scenario["title"])}</td>
  <td>{without_text}</td>
  <td>{with_text}</td>
  <td>{delta_text}</td>
</tr>
"""


def render_scenario(scenario: dict[str, Any]) -> str:
    without_rows = render_metric_rows(scenario["without_canpgrid"]["metrics"])
    with_rows = render_metric_rows(scenario["with_canpgrid"]["metrics"])
    without_raw = html.escape(scenario["without_canpgrid"]["raw"]["content"])
    with_raw = html.escape(scenario["with_canpgrid"]["raw"]["content"])
    return f"""
<section>
  <h2>{html.escape(scenario["title"])}</h2>
  <div class="grid">
    <figure>
      <img src="{html.escape(scenario["source_image"])}" alt="source screenshot">
      <figcaption>原始微信截图。</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(scenario["grid_image"])}" alt="CanpGrid overlay">
      <figcaption>CanpGrid 全局网格辅助图。</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(scenario["without_map"])}" alt="without CanpGrid map">
      <figcaption>不用 CanpGrid：Kimi 直接返回的点位。</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(scenario["with_canpgrid_map"])}" alt="with CanpGrid map">
      <figcaption>使用 CanpGrid：Kimi 看网格辅助图后返回的点位。</figcaption>
    </figure>
  </div>
  <h3>不用 CanpGrid</h3>
  {without_rows}
  <h3>使用 CanpGrid</h3>
  {with_rows}
  <h3>原始返回</h3>
  <pre>{without_raw}</pre>
  <pre>{with_raw}</pre>
</section>
"""


def render_metric_rows(metrics: dict[str, Any]) -> str:
    rows = "\n".join(
        f"""
<tr>
  <td>{html.escape(row["label"])}</td>
  <td>{'missing' if row.get("missing") else f'{row["x"]:.1f}, {row["y"]:.1f}'}</td>
  <td>{'-' if row.get("missing") else f'{row["error_px"]:.1f}px'}</td>
  <td class="{'hit' if row.get("hit") else 'miss'}">{'hit' if row.get("hit") else 'miss'}</td>
</tr>
"""
        for row in metrics["rows"]
    )
    return f"""
<table>
  <thead><tr><th>目标</th><th>Kimi 点位</th><th>到参考中心误差</th><th>有效区域</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def relative_asset(path: str | Path) -> str:
    return Path(path).resolve().relative_to(OUT_DIR.resolve()).as_posix()


if __name__ == "__main__":
    main()
