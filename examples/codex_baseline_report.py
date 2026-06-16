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

from canpgrid import create_grid_view, resolve_point, resolve_region, zoom_region

from demo import ensure_sample_image

SAMPLE = ROOT / "examples" / "sample_images" / "sample.png"
REPORT_DIR = ROOT / "outputs" / "codex_baseline_report"
ASSET_DIR = REPORT_DIR / "assets"
HTML_PATH = REPORT_DIR / "index.html"

GLOBAL_GRID = [12, 7]
LOCAL_GRID = [8, 6]
FINAL_GRID = [8, 8]
RULER_SIZE = [16, 16]


@dataclass(frozen=True)
class PartCase:
    name: str
    note: str
    target_point: tuple[float, float]


PARTS = [
    PartCase("A", "left upper blue panel", (250, 220)),
    PartCase("B", "large center green panel", (645, 310)),
    PartCase("C", "right upper orange panel", (1025, 210)),
    PartCase("D", "left lower violet panel", (250, 500)),
    PartCase("E", "right lower pink panel", (1020, 490)),
]


def main() -> None:
    ensure_sample_image(SAMPLE)
    if REPORT_DIR.exists():
        shutil.rmtree(REPORT_DIR)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    source_copy = ASSET_DIR / "source.png"
    shutil.copyfile(SAMPLE, source_copy)

    overview = create_grid_view(
        SAMPLE,
        grid_size=GLOBAL_GRID,
        overlay_mode="grid",
        detail_mode="medium",
        out_dir=ASSET_DIR,
    )

    traces = [build_part_trace(part) for part in PARTS]
    marker_map = draw_marker_map(source_copy, traces, ASSET_DIR / "resolved_points.png")
    write_html(source_copy, marker_map, overview, traces)

    print(
        json.dumps(
            {"html_report": str(HTML_PATH), "parts": traces},
            ensure_ascii=False,
            indent=2,
        )
    )


def build_part_trace(part: PartCase) -> dict[str, Any]:
    with Image.open(SAMPLE) as image:
        image_size = image.size

    first_cell = cell_for_point(
        part.target_point,
        {"x1": 0, "y1": 0, "x2": image_size[0], "y2": image_size[1]},
        GLOBAL_GRID,
    )
    first_level = {"grid_size": GLOBAL_GRID, "cell": first_cell}
    first_region = resolve_region(SAMPLE, [first_level])

    rough_view = zoom_region(
        SAMPLE,
        [first_level],
        next_grid_size=LOCAL_GRID,
        overlay_mode="grid",
        detail_mode="medium",
        zoom_factor=4,
        out_dir=ASSET_DIR,
    )

    second_cell = cell_for_point(part.target_point, first_region["bbox_on_original"], LOCAL_GRID)
    second_level = {"grid_size": LOCAL_GRID, "cell": second_cell}
    levels = [first_level, second_level]
    final_region = resolve_region(SAMPLE, levels)
    point_spec = ruler_point_for_target(part.target_point, final_region["bbox_on_original"])

    final_view = zoom_region(
        SAMPLE,
        levels,
        next_grid_size=FINAL_GRID,
        overlay_mode="hybrid",
        detail_mode="fine",
        ruler_config={"tick_x": RULER_SIZE[0], "tick_y": RULER_SIZE[1]},
        zoom_factor=20,
        out_dir=ASSET_DIR,
    )

    point_result = resolve_point(SAMPLE, levels, point_spec)
    resolved_point = point_result["point_on_original"]
    error_px = math.dist(part.target_point, resolved_point)

    return {
        "name": part.name,
        "note": part.note,
        "target_point": list(part.target_point),
        "levels": levels,
        "rough_view": relative_asset(rough_view["annotated_image_path"]),
        "final_view": relative_asset(final_view["annotated_image_path"]),
        "final_region": final_region,
        "point_spec": point_spec,
        "resolved_point": resolved_point,
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


def ruler_point_for_target(
    point: tuple[float, float], bbox: dict[str, float | int]
) -> dict[str, Any]:
    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    width = float(bbox["width"])
    height = float(bbox["height"])
    tick_x = round((point[0] - x1) / width * RULER_SIZE[0])
    tick_y = round((point[1] - y1) / height * RULER_SIZE[1])
    return {
        "type": "ruler_point",
        "origin": "top_left",
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
        name = trace["name"]
        color = (255, 55, 55, 230)
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline=color, width=3)
        draw.line((x - 18, y, x + 18, y), fill=color, width=2)
        draw.line((x, y - 18, x, y + 18), fill=color, width=2)
        draw.rectangle((x + 12, y - 11, x + 46, y + 9), fill=(0, 0, 0, 170))
        draw.text((x + 17, y - 8), name, fill=(255, 255, 255, 245), font=font)

    result = Image.alpha_composite(image, overlay).convert("RGB")
    result.save(out_path, format="PNG")
    return out_path


def write_html(
    source_copy: Path,
    marker_map: Path,
    overview: dict[str, Any],
    traces: list[dict[str, Any]],
) -> None:
    package_version = current_version()
    trace_cards = "\n".join(render_trace_card(trace) for trace in traces)
    data_json = html.escape(json.dumps(traces, ensure_ascii=False, indent=2))
    overview_src = relative_asset(overview["annotated_image_path"])

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CanpGrid Codex Baseline Report</title>
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
    header, main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    header {{ padding-top: 36px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; line-height: 1.15; letter-spacing: 0; }}
    h2 {{ margin: 0 0 16px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 0 0 12px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .subtitle {{ color: var(--muted); font-size: 16px; max-width: 820px; }}
    .meta {{
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
      background: #f8fbfd;
      padding: 10px;
      min-height: 88px;
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
    @media (max-width: 860px) {{
      header, main {{ padding: 18px; }}
      .grid, .steps, .metrics {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 26px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>CanpGrid Codex Baseline Report</h1>
    <p class="subtitle">
      这份报告用当前 Codex 作为“观察者基座”，把一张示例图片中各个视觉部件的候选点位定位过程可视化：
      全图网格、局部放大、最终 ruler/hybrid 点位，以及解析回原图的坐标。
    </p>
    <div class="meta">
      <span class="pill">CanpGrid {html.escape(package_version)}</span>
      <span class="pill">Global grid {GLOBAL_GRID[0]}x{GLOBAL_GRID[1]}</span>
      <span class="pill">Local grid {LOCAL_GRID[0]}x{LOCAL_GRID[1]}</span>
      <span class="pill">Ruler {RULER_SIZE[0]}x{RULER_SIZE[1]}</span>
      <span class="pill">No real click execution</span>
    </div>
  </header>
  <main>
    <section>
      <h2>0. 原图与全局观察图</h2>
      <div class="grid">
        <figure>
          <img src="{relative_asset(source_copy)}" alt="source image">
          <figcaption>原始示例图。部件 A-E 是人为绘制的视觉区域。</figcaption>
        </figure>
        <figure>
          <img src="{overview_src}" alt="global grid view">
          <figcaption>第一层全图网格观察图。观察者先选包含目标部件的 cell。</figcaption>
        </figure>
      </div>
      <div class="callout">
        CanpGrid Core 只处理图片空间引用。这里展示的是候选点位解析，
        不做真实点击、不识别 UI 语义、不调用模型。
      </div>
    </section>
    <section>
      <h2>1. 解析后的候选点位总览</h2>
      <figure>
        <img src="{relative_asset(marker_map)}" alt="resolved point map">
        <figcaption>红色十字是每个部件最终解析回原图的候选点位。</figcaption>
      </figure>
    </section>
    <section>
      <h2>2. 各部件定位过程</h2>
      {trace_cards}
    </section>
    <section>
      <h2>3. 机器可读 trace</h2>
      <pre>{data_json}</pre>
    </section>
  </main>
</body>
</html>
"""
    HTML_PATH.write_text(document, encoding="utf-8")


def render_trace_card(trace: dict[str, Any]) -> str:
    levels_json = html.escape(json.dumps(trace["levels"], ensure_ascii=False))
    point_spec_json = html.escape(json.dumps(trace["point_spec"], ensure_ascii=False))
    final_region_json = html.escape(
        json.dumps(trace["final_region"]["bbox_on_original"], ensure_ascii=False)
    )

    return f"""
<div class="trace">
  <h3>部件 {html.escape(trace["name"])} · {html.escape(trace["note"])}</h3>
  <div class="steps">
    <div class="step">
      <strong>Step 1 · 全图粗选</strong>
      <span>选择 cell {trace["first_cell"]} @ {GLOBAL_GRID[0]}x{GLOBAL_GRID[1]}</span>
    </div>
    <div class="step">
      <strong>Step 2 · 局部放大</strong>
      <span>在局部图中选择 cell {trace["second_cell"]} @ {LOCAL_GRID[0]}x{LOCAL_GRID[1]}</span>
    </div>
    <div class="step">
      <strong>Step 3 · 细定位</strong>
      <span>使用 ruler_point：{point_spec_json}</span>
    </div>
    <div class="step">
      <strong>Step 4 · 回写原图</strong>
      <span>得到原图候选点 {format_point(trace["resolved_point"])}</span>
    </div>
  </div>
  <div class="grid">
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
        alt="final hybrid view for {html.escape(trace["name"])}"
      >
      <figcaption>第二层 cell 后的 hybrid/ruler 精定位观察图。</figcaption>
    </figure>
  </div>
  <div class="metrics">
    <div class="metric"><b>目标参考点</b><span>{format_point(trace["target_point"])}</span></div>
    <div class="metric"><b>解析候选点</b><span>{format_point(trace["resolved_point"])}</span></div>
    <div class="metric"><b>误差</b><span>{trace["error_px"]} px</span></div>
    <div class="metric"><b>最终 bbox</b><span><code>{final_region_json}</code></span></div>
  </div>
  <p><code>levels = {levels_json}</code></p>
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
