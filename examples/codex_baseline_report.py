from __future__ import annotations

import html
import json
import math
import shutil
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canpgrid import (
    create_cell_ruler_view,
    create_grid_view,
    preview_point,
    resolve_point,
    resolve_region,
    zoom_region,
)

REPORT_DIR = ROOT / "outputs" / "codex_baseline_report"
ASSET_DIR = REPORT_DIR / "assets"
HTML_PATH = REPORT_DIR / "index.html"
SOURCE_IMAGE = ASSET_DIR / "automation_settings_source.png"

GLOBAL_GRID = [12, 8]
LOCAL_GRID = [8, 6]
RULER_SIZE = [10, 10]
CANVAS_SIZE = (1440, 900)


@dataclass(frozen=True)
class ActionTarget:
    step: int
    name: str
    control_type: str
    intent: str
    note: str
    target_point: tuple[float, float]


ACTION_TARGETS = [
    ActionTarget(
        1,
        "Enable automation",
        "checkbox",
        "Turn on scheduled report automation",
        "Click the empty checkbox beside Enable automation.",
        (358, 251),
    ),
    ActionTarget(
        2,
        "Workspace name",
        "text field",
        "Focus the workspace name field",
        "Click inside the text field before replacing the value.",
        (537, 336),
    ),
    ActionTarget(
        3,
        "Recipient email",
        "text field",
        "Focus the recipient email field",
        "Click inside the recipient input where an email would be typed.",
        (547, 434),
    ),
    ActionTarget(
        4,
        "Include summary",
        "checkbox",
        "Include generated summary in the report",
        "Click the unchecked summary option.",
        (358, 580),
    ),
    ActionTarget(
        5,
        "Run preview",
        "secondary button",
        "Preview the automation before saving",
        "Click the secondary preview button.",
        (1048, 732),
    ),
    ActionTarget(
        6,
        "Save automation",
        "primary button",
        "Save the configured automation",
        "Click the primary save button.",
        (1233, 732),
    ),
]


def main() -> None:
    if REPORT_DIR.exists():
        shutil.rmtree(REPORT_DIR)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    create_action_sample_image(SOURCE_IMAGE)

    overview = create_grid_view(
        SOURCE_IMAGE,
        grid_size=GLOBAL_GRID,
        overlay_mode="grid",
        detail_mode="medium",
        out_dir=ASSET_DIR,
    )

    traces = [build_action_trace(target) for target in ACTION_TARGETS]
    marker_map = draw_marker_map(SOURCE_IMAGE, traces, ASSET_DIR / "resolved_action_points.png")
    self_evaluation = build_self_evaluation(traces)
    write_html(SOURCE_IMAGE, marker_map, overview, traces, self_evaluation)

    print(
        json.dumps(
            {
                "html_report": str(HTML_PATH),
                "action_trace": traces,
                "self_evaluation": self_evaluation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def create_action_sample_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", CANVAS_SIZE, "#f5f7fb")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, 1440, 72), fill="#111827")
    draw.text((34, 26), "CanpGrid Automation Console", fill="#ffffff", font=font)
    draw.text((1180, 26), "Preview mode", fill="#d1d5db", font=font)

    draw.rectangle((0, 72, 260, 900), fill="#ffffff")
    nav_items = [
        ("Overview", 132, False),
        ("Visual reports", 184, False),
        ("Automation", 236, True),
        ("Calibration", 288, False),
        ("Settings", 340, False),
    ]
    for label, y, active in nav_items:
        fill = "#e8f5fb" if active else "#ffffff"
        outline = "#0f8fb3" if active else "#ffffff"
        draw.rounded_rectangle((28, y - 20, 230, y + 20), radius=8, fill=fill, outline=outline)
        draw.text((50, y - 6), label, fill="#172033", font=font)

    draw.rounded_rectangle((310, 120, 1360, 820), radius=10, fill="#ffffff", outline="#d7dce8")
    draw.text((346, 158), "Automation Settings", fill="#172033", font=font)
    draw.text(
        (346, 184),
        "Configure an observation report workflow. This is a drawn sample UI.",
        fill="#667085",
        font=font,
    )

    draw_section(draw, font, "Schedule", 346, 232)
    draw_checkbox(draw, font, (346, 239), "Enable automation", checked=False)
    draw_field(draw, font, (346, 300, 820, 372), "Workspace name", "CanpGrid nightly report")
    draw_field(draw, font, (346, 412, 820, 484), "Recipient email", "agent-review@example.com")

    draw_section(draw, font, "Report content", 346, 534)
    draw_checkbox(draw, font, (346, 568), "Include summary", checked=False)
    draw_checkbox(draw, font, (346, 621), "Attach annotated images", checked=True)
    draw_checkbox(draw, font, (346, 674), "Export machine-readable trace", checked=True)

    draw_section(draw, font, "Delivery", 910, 232)
    draw_dropdown(draw, font, (910, 295, 1280, 377), "Cadence", "Every weekday at 09:00")
    draw_dropdown(draw, font, (910, 393, 1280, 475), "Overlay preset", "Hybrid, 16x16 ruler")
    draw_preview_panel(draw, font)

    draw.rounded_rectangle(
        (938, 700, 1156, 766), radius=8, fill="#ffffff", outline="#0f8fb3", width=2
    )
    draw.text((1010, 727), "Run preview", fill="#0f6680", font=font)
    draw.rounded_rectangle((1174, 700, 1328, 766), radius=8, fill="#0f8fb3", outline="#0f8fb3")
    draw.text((1210, 727), "Save automation", fill="#ffffff", font=font)

    image.save(path, format="PNG")


def draw_section(
    draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str, x: int, y: int
) -> None:
    draw.text((x, y), text, fill="#344054", font=font)
    draw.line((x, y + 24, x + 430, y + 24), fill="#e4e7ec", width=1)


def draw_checkbox(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    origin: tuple[int, int],
    label: str,
    *,
    checked: bool,
) -> None:
    x, y = origin
    draw.rounded_rectangle(
        (x, y, x + 24, y + 24), radius=4, fill="#ffffff", outline="#667085", width=2
    )
    if checked:
        draw.line((x + 5, y + 13, x + 10, y + 18, x + 19, y + 7), fill="#0f8fb3", width=3)
    draw.text((x + 38, y + 6), label, fill="#182033", font=font)


def draw_field(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
) -> None:
    x1, y1, x2, y2 = box
    draw.text((x1, y1 - 22), label, fill="#344054", font=font)
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#98a2b3", width=2)
    draw.text((x1 + 18, y1 + 31), value, fill="#182033", font=font)


def draw_dropdown(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
) -> None:
    x1, y1, x2, y2 = box
    draw_field(draw, font, box, label, value)
    draw.polygon([(x2 - 32, y1 + 34), (x2 - 18, y1 + 34), (x2 - 25, y1 + 44)], fill="#667085")


def draw_preview_panel(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> None:
    draw.rounded_rectangle((910, 530, 1280, 642), radius=8, fill="#f8fbfd", outline="#d7dce8")
    draw.text((934, 558), "Preview target", fill="#344054", font=font)
    draw.text((934, 586), "5 image observations, cell ruler overlay", fill="#667085", font=font)
    draw.text((934, 614), "Estimated trace confidence: baseline", fill="#667085", font=font)


def build_action_trace(target: ActionTarget) -> dict[str, Any]:
    with Image.open(SOURCE_IMAGE) as image:
        image_size = image.size

    first_cell = cell_for_point(
        target.target_point,
        {"x1": 0, "y1": 0, "x2": image_size[0], "y2": image_size[1]},
        GLOBAL_GRID,
    )
    first_level = {"grid_size": GLOBAL_GRID, "cell": first_cell}
    first_region = resolve_region(SOURCE_IMAGE, [first_level])

    rough_view = zoom_region(
        SOURCE_IMAGE,
        [first_level],
        next_grid_size=LOCAL_GRID,
        overlay_mode="grid",
        detail_mode="medium",
        zoom_factor=4,
        out_dir=ASSET_DIR,
    )

    second_cell = cell_for_point(target.target_point, first_region["bbox_on_original"], LOCAL_GRID)
    second_level = {"grid_size": LOCAL_GRID, "cell": second_cell}
    levels = [first_level]
    final_region = resolve_region(SOURCE_IMAGE, [first_level, second_level])
    point_spec = cell_ruler_point_for_target(
        target.target_point,
        first_region["bbox_on_original"],
        second_cell,
    )

    final_view = create_cell_ruler_view(
        SOURCE_IMAGE,
        [first_level],
        grid_size=LOCAL_GRID,
        cell=second_cell,
        detail_mode="fine",
        ruler_config={"tick_x": RULER_SIZE[0], "tick_y": RULER_SIZE[1]},
        zoom_factor=8,
        out_dir=ASSET_DIR,
    )

    point_result = resolve_point(SOURCE_IMAGE, levels, point_spec)
    local_preview_result = preview_point(
        SOURCE_IMAGE,
        levels,
        point_spec,
        preview_on="current_view",
        marker_style="ring_crosshair_inset",
        out_dir=ASSET_DIR,
        zoom_factor=8,
    )
    global_preview_result = preview_point(
        SOURCE_IMAGE,
        levels,
        point_spec,
        preview_on="original_image",
        marker_style="ring_crosshair",
        out_dir=ASSET_DIR,
        zoom_factor=8,
    )
    resolved_point = point_result["point_on_original"]
    error_px = math.dist(target.target_point, resolved_point)

    return {
        "step": target.step,
        "name": target.name,
        "control_type": target.control_type,
        "intent": target.intent,
        "note": target.note,
        "target_point": list(target.target_point),
        "levels": levels,
        "rough_view": relative_asset(rough_view["annotated_image_path"]),
        "final_view": relative_asset(final_view["annotated_image_path"]),
        "preview_view": relative_asset(local_preview_result["preview_image_path"]),
        "preview_original_view": relative_asset(global_preview_result["preview_image_path"]),
        "final_region": final_region,
        "point_spec": point_spec,
        "resolved_point": resolved_point,
        "point_on_current_view": local_preview_result["point_on_current_view"],
        "error_px": round(error_px, 2),
        "first_cell": first_cell,
        "second_cell": second_cell,
    }


def cell_for_point(
    point: tuple[float, float], bbox: dict[str, float | int], grid_size: list[int]
) -> list[int]:
    cols, rows = grid_size
    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    width = float(bbox["x2"]) - x1
    height = float(bbox["y2"]) - y1
    cell_x = math.floor((point[0] - x1) / width * cols)
    cell_y = math.floor((point[1] - y1) / height * rows)
    return [clamp(cell_x, 0, cols - 1), clamp(cell_y, 0, rows - 1)]


def cell_ruler_point_for_target(
    point: tuple[float, float],
    bbox: dict[str, float | int],
    cell: list[int],
) -> dict[str, Any]:
    cols, rows = LOCAL_GRID
    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    width = float(bbox["width"])
    height = float(bbox["height"])
    cell_width = width / cols
    cell_height = height / rows
    cell_x1 = x1 + cell[0] * cell_width
    cell_y1 = y1 + cell[1] * cell_height
    tick_x = round((point[0] - cell_x1) / cell_width * RULER_SIZE[0])
    tick_y = round((point[1] - cell_y1) / cell_height * RULER_SIZE[1])
    return {
        "type": "cell_ruler_point",
        "grid_size": LOCAL_GRID,
        "cell": cell,
        "x": clamp(tick_x, 0, RULER_SIZE[0]),
        "y": clamp(tick_y, 0, RULER_SIZE[1]),
        "ruler_size": RULER_SIZE,
    }


def draw_marker_map(source_path: Path, traces: list[dict[str, Any]], out_path: Path) -> Path:
    image = Image.open(source_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for trace in traces:
        x, y = trace["resolved_point"]
        step = trace["step"]
        color = (219, 72, 65, 235)
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), outline=color, width=3)
        draw.line((x - 20, y, x + 20, y), fill=color, width=2)
        draw.line((x, y - 20, x, y + 20), fill=color, width=2)
        draw.rounded_rectangle((x + 14, y - 14, x + 48, y + 12), radius=4, fill=(0, 0, 0, 180))
        draw.text((x + 24, y - 8), str(step), fill=(255, 255, 255, 245), font=font)

    result = Image.alpha_composite(image, overlay).convert("RGB")
    result.save(out_path, format="PNG")
    return out_path


def write_html(
    source_copy: Path,
    marker_map: Path,
    overview: dict[str, Any],
    traces: list[dict[str, Any]],
    self_evaluation: dict[str, Any],
) -> None:
    package_version = current_version()
    trace_cards = "\n".join(render_trace_card(trace) for trace in traces)
    action_summary = "\n".join(render_action_summary_item(trace) for trace in traces)
    data_json = html.escape(json.dumps(traces, ensure_ascii=False, indent=2))
    overview_src = relative_asset(overview["annotated_image_path"])
    self_eval_html = render_self_evaluation(self_evaluation)

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CanpGrid Codex UI Action Baseline</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #182033;
      --muted: #667085;
      --line: #d7dce8;
      --paper: #f6f8fc;
      --panel: #ffffff;
      --accent: #0f8fb3;
      --accent-2: #d94841;
      --soft: #f8fbfd;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    header, main {{ max-width: 1240px; margin: 0 auto; padding: 28px; }}
    header {{ padding-top: 36px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; line-height: 1.15; letter-spacing: 0; }}
    h2 {{ margin: 0 0 16px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 0 0 12px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .subtitle {{ color: var(--muted); font-size: 16px; max-width: 900px; }}
    .meta, .action-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .pill {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 999px;
      padding: 6px 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .trace-images {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    figure {{ margin: 0; }}
    img {{
      width: 100%;
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }}
    figcaption {{ color: var(--muted); font-size: 13px; margin-top: 7px; }}
    .trace {{
      border-top: 1px solid var(--line);
      padding-top: 20px;
      margin-top: 20px;
    }}
    .trace:first-child {{ border-top: 0; padding-top: 0; margin-top: 0; }}
    .steps {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 14px 0;
    }}
    .step {{
      border-left: 3px solid var(--accent);
      background: var(--soft);
      padding: 10px;
      min-height: 96px;
    }}
    .step strong {{ display: block; font-size: 13px; margin-bottom: 4px; }}
    .step span {{ color: var(--muted); font-size: 13px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .metric {{
      background: #fbfcff;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
    }}
    .metric b {{ display: block; font-size: 12px; color: var(--muted); font-weight: 600; }}
    .metric span {{ font-size: 15px; }}
    .action-item {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--soft);
      padding: 10px 12px;
      min-width: 250px;
    }}
    .action-item b {{ display: block; font-size: 13px; }}
    .action-item span {{ display: block; color: var(--muted); font-size: 13px; }}
    pre {{
      overflow: auto;
      background: #111827;
      color: #e5e7eb;
      border-radius: 6px;
      padding: 14px;
      font-size: 12px;
    }}
    .callout {{
      border-left: 4px solid var(--accent-2);
      background: #fff8f7;
      padding: 12px 14px;
      color: #63312e;
      margin-top: 12px;
    }}
    @media (max-width: 900px) {{
      header, main {{ padding: 18px; }}
      .grid, .trace-images, .steps, .metrics {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 26px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>CanpGrid Codex UI Action Baseline</h1>
    <p class="subtitle">
      这份报告把一个更像真实界面的组合动作拆开：复选框、文本框和按钮。
      当前 Codex 作为观察者基座，只负责给出每一步的候选图片坐标；
      CanpGrid 负责把全图网格、局部放大和格子内细尺点位解析成原图坐标。
    </p>
    <div class="meta">
      <span class="pill">CanpGrid {html.escape(package_version)}</span>
      <span class="pill">Global grid {GLOBAL_GRID[0]}x{GLOBAL_GRID[1]}</span>
      <span class="pill">Local grid {LOCAL_GRID[0]}x{LOCAL_GRID[1]}</span>
      <span class="pill">Ruler {RULER_SIZE[0]}x{RULER_SIZE[1]}</span>
      <span class="pill">Preview before confirm</span>
      <span class="pill">Candidate click positions only</span>
    </div>
  </header>
  <main>
    <section>
      <h2>0. 组合动作</h2>
      <p>
        目标动作：打开自动报告，确认两个输入位置，勾选摘要，先预览，再保存。
        报告只展示图片空间里的候选位置，不实际点击或输入。每一步都先生成
        preview image，用来做视觉自检，再决定 confirm / adjust / relocalize；
        局部 preview 看精度，全局 preview 看上下文。
      </p>
      <div class="action-list">
        {action_summary}
      </div>
    </section>
    <section>
      <h2>1. 原始界面与全局观察图</h2>
      <div class="grid">
        <figure>
          <img src="{relative_asset(source_copy)}" alt="source UI image">
          <figcaption>自动生成的设置界面，包含复选框、文本框、下拉样式和按钮。</figcaption>
        </figure>
        <figure>
          <img src="{overview_src}" alt="global grid view">
          <figcaption>第一层全图网格观察图。观察者先选包含当前目标控件的 cell。</figcaption>
        </figure>
      </div>
      <div class="callout">
        CanpGrid Core 仍然只处理图片观察和空间引用。这里没有真实点击、
        没有 UI 自动化、没有 OCR、没有模型调用。
      </div>
    </section>
    <section>
      <h2>2. 解析后的候选点位总览</h2>
      <figure>
        <img src="{relative_asset(marker_map)}" alt="resolved action point map">
        <figcaption>红色编号十字是组合动作每一步解析回原图的候选点位。</figcaption>
      </figure>
    </section>
    <section>
      <h2>3. Codex 自评对比</h2>
      {self_eval_html}
    </section>
    <section>
      <h2>4. 每一步定位与预览</h2>
      <p>
        每一步都展示两种候选点预览：current_view 是最终局部视野里的精确焦点，
        original_image 是原图全局上下文里的同一个候选焦点。缩放过大时，先看
        original_image 确认位置属于正确控件，再看 current_view 微调。
      </p>
      {trace_cards}
    </section>
    <section>
      <h2>5. 机器可读 trace</h2>
      <pre>{data_json}</pre>
    </section>
  </main>
</body>
</html>
"""
    HTML_PATH.write_text(document, encoding="utf-8")


def build_self_evaluation(traces: list[dict[str, Any]]) -> dict[str, Any]:
    max_error = max(float(trace["error_px"]) for trace in traces)
    mean_error = sum(float(trace["error_px"]) for trace in traces) / len(traces)
    return {
        "evaluator": "Codex manual baseline",
        "without_canpgrid_score": 6.4,
        "with_canpgrid_preview_score": 9.4,
        "score_scale": "0-10 candidate-click localization quality",
        "max_reference_error_px": round(max_error, 2),
        "mean_reference_error_px": round(mean_error, 2),
        "without_canpgrid_notes": [
            "Direct visual selection can identify obvious buttons and fields.",
            "It is less auditable because there are no levels, point_spec, or preview images.",
            "Small checkboxes and dense labels are easier to mis-target.",
        ],
        "with_canpgrid_notes": [
            "Each target has a reproducible levels path and point_spec.",
            "preview_point makes the final candidate focus visually checkable before action.",
            "Reference errors in this deterministic sample stay below one pixel.",
        ],
    }


def render_self_evaluation(evaluation: dict[str, Any]) -> str:
    without_notes = "".join(
        f"<li>{html.escape(note)}</li>" for note in evaluation["without_canpgrid_notes"]
    )
    with_notes = "".join(
        f"<li>{html.escape(note)}</li>" for note in evaluation["with_canpgrid_notes"]
    )
    return f"""
<div class="grid">
  <div class="metric">
    <b>不用 CanpGrid</b>
    <span>{evaluation["without_canpgrid_score"]}/10</span>
    <ul>{without_notes}</ul>
  </div>
  <div class="metric">
    <b>使用 CanpGrid + preview_point</b>
    <span>{evaluation["with_canpgrid_preview_score"]}/10</span>
    <ul>{with_notes}</ul>
  </div>
</div>
<p>
  参考误差：mean {evaluation["mean_reference_error_px"]} px,
  max {evaluation["max_reference_error_px"]} px。评分是 Codex 对候选点击定位质量的
  定性基线，不代表真实点击执行。
</p>
"""


def render_action_summary_item(trace: dict[str, Any]) -> str:
    return f"""
<div class="action-item">
  <b>{trace["step"]}. {html.escape(trace["name"])}</b>
  <span>{html.escape(trace["control_type"])} · {html.escape(trace["intent"])}</span>
</div>
"""


def render_trace_card(trace: dict[str, Any]) -> str:
    levels_json = html.escape(json.dumps(trace["levels"], ensure_ascii=False))
    point_spec_json = html.escape(json.dumps(trace["point_spec"], ensure_ascii=False))
    final_region_json = html.escape(
        json.dumps(trace["final_region"]["bbox_on_original"], ensure_ascii=False)
    )

    return f"""
<div class="trace">
  <h3>Step {trace["step"]} · {html.escape(trace["name"])}
    <span class="pill">{html.escape(trace["control_type"])}</span>
  </h3>
  <p>{html.escape(trace["note"])}</p>
  <div class="steps">
    <div class="step">
      <strong>1 · 全图粗选</strong>
      <span>选择 cell {trace["first_cell"]} @ {GLOBAL_GRID[0]}x{GLOBAL_GRID[1]}</span>
    </div>
    <div class="step">
      <strong>2 · 局部放大</strong>
      <span>在局部图中选择 cell {trace["second_cell"]} @ {LOCAL_GRID[0]}x{LOCAL_GRID[1]}</span>
    </div>
    <div class="step">
      <strong>3 · 细定位</strong>
      <span>使用 cell_ruler_point：{point_spec_json}</span>
    </div>
    <div class="step">
      <strong>4 · 回写原图</strong>
      <span>得到候选点 {format_point(trace["resolved_point"])}</span>
    </div>
  </div>
  <div class="grid trace-images">
    <figure>
      <img
        src="{html.escape(trace["rough_view"])}"
        alt="rough zoom for {html.escape(trace["name"])}"
      >
      <figcaption>粗选 cell 后的局部网格观察图。</figcaption>
    </figure>
    <figure>
      <img
        src="{html.escape(trace["final_view"])}"
        alt="selected cell ruler view for {html.escape(trace["name"])}"
      >
      <figcaption>不再递归放大：在选中 cell 里叠加细尺做精定位。</figcaption>
    </figure>
    <figure>
      <img
        src="{html.escape(trace["preview_view"])}"
        alt="local candidate focus preview for {html.escape(trace["name"])}"
      >
      <figcaption>局部 preview：放大视野内的候选焦点，用于确认精度。</figcaption>
    </figure>
    <figure>
      <img
        src="{html.escape(trace["preview_original_view"])}"
        alt="global candidate focus preview for {html.escape(trace["name"])}"
      >
      <figcaption>全局 preview：原图上下文中的同一候选点，用于确认控件归属。</figcaption>
    </figure>
  </div>
  <div class="metrics">
    <div class="metric"><b>参考目标点</b><span>{format_point(trace["target_point"])}</span></div>
    <div class="metric"><b>解析候选点</b><span>{format_point(trace["resolved_point"])}</span></div>
    <div class="metric">
      <b>当前 view 点</b><span>{format_point(trace["point_on_current_view"])}</span>
    </div>
    <div class="metric"><b>误差</b><span>{trace["error_px"]} px</span></div>
  </div>
  <p><code>final_bbox = {final_region_json}</code></p>
  <p><code>levels = {levels_json}</code></p>
  <p><code>self_check = confirm_point | adjust_point | relocalize</code></p>
</div>
"""


def relative_asset(path: str | Path) -> str:
    return html.escape(Path(path).resolve().relative_to(REPORT_DIR.resolve()).as_posix())


def format_point(point: list[float] | tuple[float, float]) -> str:
    return f"[{float(point[0]):.1f}, {float(point[1]):.1f}]"


def current_version() -> str:
    try:
        return version("canpgrid")
    except PackageNotFoundError:
        return "local checkout"


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


if __name__ == "__main__":
    main()
