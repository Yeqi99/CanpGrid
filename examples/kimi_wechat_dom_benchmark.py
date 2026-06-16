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

from canpgrid import create_cell_ruler_view, create_grid_view, preview_point, resolve_point
from kimi_ui_compare import DEFAULT_BASE_URL, DEFAULT_MODEL, call_kimi

OUT_DIR = ROOT / "outputs" / "wechat_dom_benchmark"
ASSET_DIR = OUT_DIR / "assets"
HTML_PATH = OUT_DIR / "index.html"
GRID_SIZE = [9, 20]
RULER_SIZE = [10, 10]
SCREEN_SIZE = (1170, 2532)


@dataclass(frozen=True)
class Target:
    target_id: str
    label: str
    instruction: str
    bbox: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    source_path: Path
    fixture_html_path: Path
    targets: list[Target]


def main() -> None:
    global OUT_DIR, ASSET_DIR, HTML_PATH

    parser = argparse.ArgumentParser(
        description="DOM-bbox grounded WeChat-like localization benchmark for Kimi."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--max-targets",
        type=int,
        default=0,
        help="Maximum targets per scenario. Use 0 for all targets.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise SystemExit(
            "MOONSHOT_API_KEY is required. This benchmark always uses real API calls "
            "and does not generate offline model-test reports."
        )

    OUT_DIR = Path(args.out_dir)
    ASSET_DIR = OUT_DIR / "assets"
    HTML_PATH = OUT_DIR / "index.html"
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios()
    scenario_reports = prepare_assets(scenarios)
    for report in scenario_reports:
        report["model_results"] = run_model_scenario(
            report["scenario"],
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            max_targets=args.max_targets,
        )

    output = {
        "model": args.model,
        "base_url": args.base_url,
        "api_status": "real_api",
        "grid_size": GRID_SIZE,
        "ruler_size": RULER_SIZE,
        "scenarios": [serialize_report(report) for report in scenario_reports],
    }
    (OUT_DIR / "wechat_dom_benchmark.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_html(output)
    print(
        json.dumps(
            {
                "html_report": str(HTML_PATH),
                "json_report": str(OUT_DIR / "wechat_dom_benchmark.json"),
                "api_status": "real_api",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_scenarios() -> list[Scenario]:
    chat_targets = [
        Target("search", "搜索按钮", "Tap the top navigation search icon.", (918, 150, 1018, 258)),
        Target("plus", "加号按钮", "Tap the top navigation plus icon.", (1034, 150, 1134, 258)),
        Target(
            "first_chat",
            "第一个聊天行",
            "Tap the first chat row named 亲亲老婆大人.",
            (170, 390, 1120, 575),
        ),
        Target(
            "contacts_tab",
            "通讯录 tab",
            "Tap the bottom contacts tab.",
            (315, 2200, 520, 2410),
        ),
        Target(
            "discover_tab",
            "发现 tab",
            "Tap the bottom discover tab.",
            (620, 2200, 790, 2410),
        ),
        Target("me_tab", "我 tab", "Tap the bottom Me tab.", (900, 2200, 1085, 2410)),
    ]
    search_targets = [
        Target("back", "返回按钮", "Tap the back arrow.", (20, 165, 110, 270)),
        Target("search_field", "搜索输入框", "Tap inside the search input.", (110, 150, 1060, 285)),
        Target("deep_think", "深度思考", "Tap the 深度思考 option.", (130, 318, 390, 420)),
        Target("camera", "相机按钮", "Tap the camera icon.", (418, 318, 514, 420)),
        Target("add", "加号按钮", "Tap the plus action.", (530, 305, 625, 420)),
        Target("ai_search", "AI搜索按钮", "Tap the AI搜索 action.", (880, 318, 1085, 420)),
        Target(
            "refresh_trends",
            "换一换",
            "Tap refresh to change search trends.",
            (880, 450, 1090, 555),
        ),
        Target(
            "trend_anthropic",
            "Anthropic实名制7月",
            "Tap the Anthropic trend/news item.",
            (45, 560, 530, 970),
        ),
        Target(
            "trend_cat",
            "奶牛猫勇敢护同伴",
            "Tap the cat trend/news item.",
            (610, 560, 1095, 970),
        ),
        Target(
            "trend_trump_fund",
            "川普3000亿重建基金",
            "Tap the Trump fund trend/news item.",
            (45, 990, 530, 1135),
        ),
        Target(
            "trend_movie",
            "火遮眼隐藏情绪",
            "Tap the movie trend/news item.",
            (610, 990, 1095, 1135),
        ),
        Target(
            "trend_ai_major",
            "AI时代选专业建议",
            "Tap the AI major trend/news item.",
            (45, 1150, 530, 1295),
        ),
        Target(
            "trend_car_oem",
            "日系车代工中国品牌",
            "Tap the car OEM trend/news item.",
            (610, 1150, 1095, 1295),
        ),
        Target(
            "voice_button",
            "语音提问按钮",
            "Tap the green voice button.",
            (205, 1888, 965, 2048),
        ),
        Target(
            "page_settings",
            "页面设置",
            "Tap the page settings link.",
            (430, 2232, 740, 2345),
        ),
    ]
    return [
        Scenario(
            "chat_list",
            "微信风格消息列表",
            ASSET_DIR / "chat_list.png",
            ASSET_DIR / "chat_list_fixture.html",
            chat_targets,
        ),
        Scenario(
            "search",
            "微信风格搜索页",
            ASSET_DIR / "search.png",
            ASSET_DIR / "search_fixture.html",
            search_targets,
        ),
    ]


def prepare_assets(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    reports = []
    for scenario in scenarios:
        if scenario.scenario_id == "chat_list":
            draw_chat_list(scenario.source_path)
        else:
            draw_search_page(scenario.source_path)
        write_fixture_html(scenario)
        write_target_manifest(scenario)
        truth_map = draw_truth_map(
            scenario.source_path,
            scenario.targets,
            ASSET_DIR / f"{scenario.scenario_id}_truth_map.png",
        )
        grid = create_grid_view(
            scenario.source_path,
            grid_size=GRID_SIZE,
            overlay_mode="grid",
            detail_mode="medium",
            out_dir=ASSET_DIR,
        )
        oracle_examples = build_oracle_examples(scenario)
        reports.append(
            {
                "scenario": scenario,
                "truth_map": truth_map,
                "grid_image": Path(grid["annotated_image_path"]),
                "oracle_examples": oracle_examples,
            }
        )
    return reports


def run_model_scenario(
    scenario: Scenario,
    *,
    api_key: str,
    base_url: str,
    model: str,
    max_targets: int,
) -> list[dict[str, Any]]:
    grid = create_grid_view(
        scenario.source_path,
        grid_size=GRID_SIZE,
        overlay_mode="grid",
        detail_mode="medium",
        out_dir=ASSET_DIR,
    )
    grid_path = Path(grid["annotated_image_path"])
    results = []
    targets = scenario.targets if max_targets <= 0 else scenario.targets[:max_targets]
    for target in targets:
        direct = run_direct_target(scenario, target, api_key, base_url, model)
        with_grid = run_canpgrid_target(
            scenario,
            target,
            grid_path,
            api_key,
            base_url,
            model,
        )
        results.append(
            {
                "target": target_to_dict(target),
                "without_canpgrid": direct,
                "with_canpgrid": with_grid,
            }
        )
    return results


def run_direct_target(
    scenario: Scenario,
    target: Target,
    api_key: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    prompt = direct_prompt(scenario, target)
    response = call_kimi(
        api_key=api_key,
        base_url=base_url,
        model=model,
        images=[scenario.source_path],
        prompt=prompt,
    )
    try:
        data = json.loads(extract_json_text(response["content"]))
        point = (float(data["x"]), float(data["y"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "error": str(exc),
            "raw": response,
            "usage": usage(response),
            "prompt_chars": len(prompt),
        }
    return {
        "ok": True,
        "point": [round(point[0], 1), round(point[1], 1)],
        "score": score_point(point, target),
        "raw": response,
        "usage": usage(response),
        "prompt_chars": len(prompt),
    }


def run_canpgrid_target(
    scenario: Scenario,
    target: Target,
    grid_path: Path,
    api_key: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    stage1_prompt_text = stage1_prompt(scenario, target)
    stage1 = call_kimi(
        api_key=api_key,
        base_url=base_url,
        model=model,
        images=[scenario.source_path, grid_path],
        prompt=stage1_prompt_text,
    )
    try:
        cell_data = json.loads(extract_json_text(stage1["content"]))
        cell = clamp_cell([int(cell_data["cell"][0]), int(cell_data["cell"][1])])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "error": str(exc),
            "stage1_raw": stage1,
            "usage": usage(stage1),
            "prompt_chars": len(stage1_prompt_text),
        }

    cell_ruler = create_cell_ruler_view(
        scenario.source_path,
        [],
        grid_size=GRID_SIZE,
        cell=cell,
        ruler_config={"tick_x": RULER_SIZE[0], "tick_y": RULER_SIZE[1]},
        out_dir=ASSET_DIR,
    )
    stage2_prompt_text = stage2_prompt(scenario, target, cell)
    stage2 = call_kimi(
        api_key=api_key,
        base_url=base_url,
        model=model,
        images=[scenario.source_path, Path(cell_ruler["annotated_image_path"])],
        prompt=stage2_prompt_text,
    )
    try:
        tick_data = json.loads(extract_json_text(stage2["content"]))
        tick_x = clamp_number(float(tick_data["x"]), 0, RULER_SIZE[0])
        tick_y = clamp_number(float(tick_data["y"]), 0, RULER_SIZE[1])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "error": str(exc),
            "cell": cell,
            "stage1_raw": stage1,
            "stage2_raw": stage2,
            "usage": add_usage(usage(stage1), usage(stage2)),
            "prompt_chars": len(stage1_prompt_text) + len(stage2_prompt_text),
        }

    point_spec = {
        "type": "cell_ruler_point",
        "grid_size": GRID_SIZE,
        "cell": cell,
        "x": tick_x,
        "y": tick_y,
        "ruler_size": RULER_SIZE,
    }
    resolved = resolve_point(scenario.source_path, [], point_spec)
    point = tuple(float(value) for value in resolved["point_on_original"])
    preview = preview_point(
        scenario.source_path,
        [],
        point_spec,
        preview_on="original_image",
        marker_style="ring_crosshair",
        out_dir=ASSET_DIR,
    )
    return {
        "ok": True,
        "cell": cell,
        "ticks": [round(tick_x, 2), round(tick_y, 2)],
        "point_spec": point_spec,
        "point": [round(point[0], 1), round(point[1], 1)],
        "score": score_point(point, target),
        "cell_ruler_image": relative_asset(cell_ruler["annotated_image_path"]),
        "preview_image": relative_asset(preview["preview_image_path"]),
        "stage1_raw": stage1,
        "stage2_raw": stage2,
        "usage": add_usage(usage(stage1), usage(stage2)),
        "prompt_chars": len(stage1_prompt_text) + len(stage2_prompt_text),
    }


def draw_chat_list(path: Path) -> None:
    image = Image.new("RGB", SCREEN_SIZE, "#f4f4f4")
    draw = ImageDraw.Draw(image)
    font_sm = font(28)
    font_md = font(38)
    font_lg = font(46)

    draw_status(draw, title="微信(74)")
    draw.text((915, 187), "⌕", fill="#111111", font=font(62))
    draw.text((1045, 183), "+", fill="#111111", font=font(64))
    draw.line((0, 270, 1170, 270), fill="#d7d7d7", width=1)
    draw.text((218, 335), "Mac 微信已登录", fill="#808080", font=font_md)
    draw.rectangle((86, 322, 142, 362), outline="#8d8d8d", width=4)

    rows = [
        ("亲亲老婆大人", "第一优先级：少糖 + 主食减量 + Quest光剑每周1...", "上午11:29"),
        ("全房通-全员性压抑（崔哥spa）", "明杰: [图片]", "上午11:05"),
        (
            "王雪 @先石（四川）科技有限公司",
            "Celia 邀请您参加腾讯会议 会议主题：张雁清...",
            "上午8:50",
        ),
        ("交易群", "石头-cnkaige（凯歌）: [动画表情]", "昨天"),
        ("Rose.", "收到", "周五"),
        ("黑马助教Lena_BEACON", "嗯", "周四"),
        ("NAU.Nov.24班级群", "咱们有没有成功申博的同学呀", "周四"),
        ("TimYa", "一些顺利啊", "6月8日"),
    ]
    y = 430
    for index, (name, message, time_text) in enumerate(rows):
        avatar_color = ["#c7ddff", "#d9d2f2", "#d8d8d8", "#ece4d7"][index % 4]
        draw.rounded_rectangle((48, y - 8, 142, y + 86), radius=12, fill=avatar_color)
        draw.text((68, y + 20), name[:1], fill="#222222", font=font_md)
        name_fill = "#111111" if index != 2 else "#d68c45"
        draw.text((220, y + 8), name, fill=name_fill, font=font_lg)
        draw.text((220, y + 72), message, fill="#a0a0a0", font=font_sm)
        draw.text((955, y + 12), time_text, fill="#9a9a9a", font=font_sm)
        draw.line((210, y + 122, 1170, y + 122), fill="#dfdfdf", width=1)
        y += 205

    draw.rectangle((0, 2168, 1170, 2532), fill="#f7f7f7")
    draw.line((0, 2168, 1170, 2168), fill="#dcdcdc", width=1)
    draw_bottom_tab(draw, 150, "微信", "74", active=True)
    draw_bottom_tab(draw, 420, "通讯录", None, active=False)
    draw_bottom_tab(draw, 720, "发现", None, active=False)
    draw_bottom_tab(draw, 1010, "我", None, active=False)
    draw.rounded_rectangle((380, 2482, 790, 2492), radius=5, fill="#111111")
    image.save(path, format="PNG")


def draw_search_page(path: Path) -> None:
    image = Image.new("RGB", SCREEN_SIZE, "#ffffff")
    draw = ImageDraw.Draw(image)
    font_sm = font(30)
    font_md = font(40)
    font_lg = font(48)

    draw_status(draw, title="")
    draw.line((72, 220, 38, 185), fill="#111111", width=5)
    draw.line((38, 185, 72, 150), fill="#111111", width=5)
    draw.rounded_rectangle((112, 148, 1062, 286), radius=18, fill="#f1f1f1")
    draw.text((150, 194), "⌕", fill="#b0b0b0", font=font(54))
    draw.text((220, 190), "搜索本地或网络结果", fill="#a9a9a9", font=font_lg)

    draw.ellipse((150, 348, 184, 382), outline="#b6b6b6", width=3)
    draw.text((205, 337), "深度思考", fill="#5b5b5b", font=font_md)
    draw.line((420, 340, 420, 392), fill="#dedede", width=2)
    draw.text((455, 333), "▣", fill="#737373", font=font(52))
    draw.text((555, 333), "+", fill="#555555", font=font(56))
    draw.text((900, 337), "AI搜索", fill="#57637a", font=font_md)

    draw.text((46, 500), "大家在搜", fill="#a5a5a5", font=font_md)
    draw.text((900, 500), "换一换 ↻", fill="#b6b6b6", font=font_sm)
    topics = [
        (60, 610, "Anthropic实名制7月...", "AI大模型", True),
        (625, 610, "奶牛猫勇敢护同伴", "萌宠", True),
        (60, 1035, "川普3000亿重建基金", "金融财经", False),
        (625, 1035, "火遮眼隐藏情绪", "电影", False),
        (60, 1195, "AI时代选专业建议", "高考", False),
        (625, 1195, "日系车代工中国品牌", "新能源头条", False),
    ]
    for x, y, title, subtitle, with_cards in topics:
        draw.text((x, y), "⌕", fill="#b6b6b6", font=font_sm)
        draw.text((x + 45, y - 6), title, fill="#686868", font=font_md)
        draw.text((x + 45, y + 56), subtitle, fill="#b1b1b1", font=font_sm)
        if with_cards:
            draw.rounded_rectangle((x + 45, y + 110, x + 225, y + 350), radius=10, fill="#161616")
            draw.rounded_rectangle(
                (x + 245, y + 110, x + 425, y + 350), radius=10, fill="#31405f"
            )
            draw.text((x + 78, y + 205), "图片", fill="#ffffff", font=font_sm)

    draw.rounded_rectangle((205, 1888, 965, 2048), radius=80, fill="#65c783")
    draw.text((325, 1942), "🎙  按住 说出你的问题", fill="#ffffff", font=font_lg)
    draw.text((495, 2280), "页面设置", fill="#59617a", font=font_md)
    draw.rounded_rectangle((380, 2482, 790, 2492), radius=5, fill="#111111")
    image.save(path, format="PNG")


def draw_status(draw: ImageDraw.ImageDraw, *, title: str) -> None:
    font_sm = font(28)
    font_md = font(44)
    draw.rectangle((0, 0, 1170, 138), fill="#f4f4f4")
    draw.text((72, 55), "12:19", fill="#111111", font=font_md)
    draw.rounded_rectangle((330, 24, 840, 126), radius=52, fill="#000000")
    draw.text((386, 55), "美团", fill="#ffdc3d", font=font_sm)
    draw.text((488, 48), "待取", fill="#ffffff", font=font_md)
    draw.text((670, 48), "去开柜", fill="#ffdc3d", font=font_md)
    if title:
        draw.text((490, 176), title, fill="#111111", font=font(46))
    draw.text((910, 50), "5G", fill="#111111", font=font_sm)
    draw.text((1020, 48), "▰▰▰", fill="#111111", font=font_sm)
    draw.rounded_rectangle((1080, 45, 1132, 83), radius=8, fill="#555555")
    draw.text((1090, 50), "75", fill="#ffffff", font=font_sm)


def draw_bottom_tab(
    draw: ImageDraw.ImageDraw, x: int, label: str, badge: str | None, *, active: bool
) -> None:
    fill = "#36c56e" if active else "#111111"
    draw.ellipse((x - 40, 2228, x + 40, 2308), fill=fill if active else "#f7f7f7")
    draw.text((x - 43, 2228), "○", fill=fill, font=font(62))
    if badge is not None:
        draw.ellipse((x + 18, 2200, x + 88, 2270), fill="#ee3e58")
        draw.text((x + 38, 2213), badge, fill="#ffffff", font=font(30))
    draw.text((x - 38, 2330), label, fill=fill, font=font(34))


def write_fixture_html(scenario: Scenario) -> None:
    target_nodes = "\n".join(
        render_target_node(target, index) for index, target in enumerate(scenario.targets, start=1)
    )
    image_name = scenario.source_path.name
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(scenario.title)} Fixture</title>
  <style>
    body {{ margin: 0; background: #eef1f5; font-family: system-ui, sans-serif; }}
    .phone {{
      position: relative;
      width: {SCREEN_SIZE[0]}px;
      height: {SCREEN_SIZE[1]}px;
      margin: 24px auto;
      background: url("{html.escape(image_name)}") center / 100% 100% no-repeat;
      box-shadow: 0 8px 30px rgba(15, 23, 42, .18);
    }}
    .target {{
      position: absolute;
      border: 2px solid rgba(37, 161, 82, .72);
      background: rgba(37, 161, 82, .05);
      color: #166534;
      font-size: 22px;
      pointer-events: none;
    }}
  </style>
</head>
<body>
  <main class="phone" aria-label="{html.escape(scenario.title)}">
    {target_nodes}
  </main>
</body>
</html>
"""
    scenario.fixture_html_path.write_text(document, encoding="utf-8")


def render_target_node(target: Target, index: int) -> str:
    x1, y1, x2, y2 = target.bbox
    return (
        f'<button class="target" data-target-id="{html.escape(target.target_id)}" '
        f'aria-label="{html.escape(target.label)}" '
        f'style="left:{x1}px;top:{y1}px;width:{x2 - x1}px;height:{y2 - y1}px">'
        f"{index}</button>"
    )


def write_target_manifest(scenario: Scenario) -> None:
    data = {
        "scenario_id": scenario.scenario_id,
        "title": scenario.title,
        "image_size": list(SCREEN_SIZE),
        "targets": [target_to_dict(target) for target in scenario.targets],
    }
    (ASSET_DIR / f"{scenario.scenario_id}_targets.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_oracle_examples(scenario: Scenario) -> list[dict[str, Any]]:
    examples = []
    for target in scenario.targets[:2]:
        cell = cell_for_point(target.center)
        point_spec = oracle_point_spec(target.center, cell)
        cell_ruler = create_cell_ruler_view(
            scenario.source_path,
            [],
            grid_size=GRID_SIZE,
            cell=cell,
            ruler_config={"tick_x": RULER_SIZE[0], "tick_y": RULER_SIZE[1]},
            out_dir=ASSET_DIR,
        )
        preview = preview_point(
            scenario.source_path,
            [],
            point_spec,
            preview_on="original_image",
            marker_style="ring_crosshair",
            out_dir=ASSET_DIR,
        )
        examples.append(
            {
                "target": target_to_dict(target),
                "cell": cell,
                "point_spec": point_spec,
                "cell_ruler_image": Path(cell_ruler["annotated_image_path"]),
                "preview_image": Path(preview["preview_image_path"]),
            }
        )
    return examples


def draw_truth_map(source_path: Path, targets: list[Target], out_path: Path) -> Path:
    image = Image.open(source_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    label_font = font(26)
    for index, target in enumerate(targets, start=1):
        x1, y1, x2, y2 = target.bbox
        draw.rounded_rectangle((x1, y1, x2, y2), radius=12, outline=(30, 150, 70, 220), width=5)
        draw.rounded_rectangle((x1 + 8, y1 + 8, x1 + 52, y1 + 44), radius=8, fill=(0, 0, 0, 170))
        draw.text((x1 + 22, y1 + 13), str(index), fill=(255, 255, 255, 245), font=label_font)
    result = Image.alpha_composite(image, overlay).convert("RGB")
    result.save(out_path, format="PNG")
    return out_path


def direct_prompt(scenario: Scenario, target: Target) -> str:
    return f"""
You are looking at one mobile UI screenshot: {scenario.title}.
Image size is {SCREEN_SIZE[0]}x{SCREEN_SIZE[1]} pixels.
Origin is top-left. Estimate the tap point for this target:

id: {target.target_id}
label: {target.label}
instruction: {target.instruction}

Return raw JSON only:
{{"id":"{target.target_id}","x":123,"y":456,"confidence":0.8}}
"""


def stage1_prompt(scenario: Scenario, target: Target) -> str:
    return f"""
You are given the original screenshot and a CanpGrid {GRID_SIZE[0]}x{GRID_SIZE[1]}
grid overlay for {scenario.title}.
Choose the single grid cell that contains the intended tap target.

Target id: {target.target_id}
Target label: {target.label}
Instruction: {target.instruction}

Cell coordinates are zero-based: [col,row].
Valid col is 0..{GRID_SIZE[0] - 1}; valid row is 0..{GRID_SIZE[1] - 1}.

Return raw JSON only:
{{"id":"{target.target_id}","cell":[0,0],"confidence":0.8}}
"""


def stage2_prompt(scenario: Scenario, target: Target, cell: list[int]) -> str:
    return f"""
You are given the original screenshot and a CanpGrid selected-cell ruler view for
{scenario.title}. The selected cell is {cell}; it has a {RULER_SIZE[0]}x{RULER_SIZE[1]}
fine ruler inside the highlighted cell.

Target id: {target.target_id}
Target label: {target.label}
Instruction: {target.instruction}

Return the fine ruler position inside the highlighted cell.
Use x/y values from 0 to {RULER_SIZE[0]} and 0 to {RULER_SIZE[1]}.

Return raw JSON only:
{{"id":"{target.target_id}","x":5,"y":5,"confidence":0.8}}
"""


def score_point(point: tuple[float, float], target: Target) -> dict[str, Any]:
    hit = point_in_bbox(point, target.bbox)
    error = math.dist(point, target.center)
    score = 10.0 if hit else max(0.0, 10.0 - error / 35)
    return {
        "hit": hit,
        "score_0_to_10": round(score, 2),
        "error_px": round(error, 1),
        "target_center": [round(target.center[0], 1), round(target.center[1], 1)],
        "target_bbox": [round(value, 1) for value in target.bbox],
    }


def serialize_report(report: dict[str, Any]) -> dict[str, Any]:
    scenario = report["scenario"]
    model_results = report["model_results"]
    return {
        "scenario_id": scenario.scenario_id,
        "title": scenario.title,
        "source_image": relative_asset(scenario.source_path),
        "fixture_html": relative_asset(scenario.fixture_html_path),
        "truth_map": relative_asset(report["truth_map"]),
        "grid_image": relative_asset(report["grid_image"]),
        "targets": [target_to_dict(target) for target in scenario.targets],
        "oracle_examples": [
            {
                "target": example["target"],
                "cell": example["cell"],
                "point_spec": example["point_spec"],
                "cell_ruler_image": relative_asset(example["cell_ruler_image"]),
                "preview_image": relative_asset(example["preview_image"]),
            }
            for example in report["oracle_examples"]
        ],
        "model_results": model_results,
        "metrics": summarize_results(model_results),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    direct = [item["without_canpgrid"] for item in results if item["without_canpgrid"].get("ok")]
    with_grid = [item["with_canpgrid"] for item in results if item["with_canpgrid"].get("ok")]
    return {
        "without_canpgrid": summarize_method(direct),
        "with_canpgrid": summarize_method(with_grid),
    }


def summarize_method(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"returned": 0, "hit_count": 0, "mean_score": None, "mean_error_px": None}
    scores = [float(item["score"]["score_0_to_10"]) for item in items]
    errors = [float(item["score"]["error_px"]) for item in items]
    tokens = [usage_total(item.get("usage")) for item in items]
    return {
        "returned": len(items),
        "hit_count": sum(1 for item in items if item["score"]["hit"]),
        "mean_score": round(sum(scores) / len(scores), 2),
        "mean_error_px": round(sum(errors) / len(errors), 1),
        "total_tokens": sum(tokens) if any(tokens) else None,
        "prompt_chars": sum(int(item.get("prompt_chars", 0)) for item in items),
    }


def write_html(data: dict[str, Any]) -> None:
    scenario_sections = "\n".join(render_scenario(scenario) for scenario in data["scenarios"])
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CanpGrid WeChat-like DOM Benchmark</title>
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
    h3 {{ margin: 18px 0 10px; font-size: 17px; letter-spacing: 0; }}
    section {{
      background: #fff;
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
    <h1>WeChat-like DOM BBox Benchmark</h1>
    <p class="muted">
      这份基准用我们自己生成的微信风格前端 fixture 和 target manifest 评分。
      命中不是靠肉眼判断，而是看模型返回点是否落进真实目标 bbox。
    </p>
    <span class="pill">api {html.escape(data["api_status"])}</span>
    <span class="pill">model {html.escape(data["model"])}</span>
    <span class="pill">grid {GRID_SIZE[0]}x{GRID_SIZE[1]}</span>
    <span class="pill">cell ruler {RULER_SIZE[0]}x{RULER_SIZE[1]}</span>
  </header>
  <main>{scenario_sections}</main>
</body>
</html>
"""
    HTML_PATH.write_text(document, encoding="utf-8")


def render_scenario(scenario: dict[str, Any]) -> str:
    examples = "\n".join(render_oracle_example(example) for example in scenario["oracle_examples"])
    model_results = render_model_results(scenario)
    metrics = scenario.get("metrics")
    fixture_html = html.escape(scenario["fixture_html"])
    metric_html = "<p class=\"muted\">没有模型结果；请检查真实 API 调用是否返回结果。</p>"
    if metrics:
        metric_html = f"<pre>{html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))}</pre>"
    return f"""
<section>
  <h2>{html.escape(scenario["title"])}</h2>
  <p class="muted">
    Fixture HTML: <a href="{fixture_html}">{fixture_html}</a>
  </p>
  <div class="grid">
    <figure>
      <img src="{html.escape(scenario["source_image"])}" alt="source fixture">
      <figcaption>生成的微信风格界面。</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(scenario["truth_map"])}" alt="truth map">
      <figcaption>绿色框是 target manifest 里的真实可点区域。</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(scenario["grid_image"])}" alt="global grid">
      <figcaption>CanpGrid 全局网格，用于第一步选择 cell。</figcaption>
    </figure>
  </div>
  <h3>Oracle cell-ruler 示例</h3>
  <div class="grid">{examples}</div>
  <h3>模型指标</h3>
  {metric_html}
  {model_results}
</section>
"""


def render_oracle_example(example: dict[str, Any]) -> str:
    label = example["target"]["label"]
    return f"""
<figure>
  <img src="{html.escape(example["cell_ruler_image"])}" alt="cell ruler example">
  <figcaption>{html.escape(label)}：先选 cell {example["cell"]}，再用细尺选点。</figcaption>
</figure>
<figure>
  <img src="{html.escape(example["preview_image"])}" alt="preview example">
  <figcaption>{html.escape(label)}：候选焦点预览，仍然不执行点击。</figcaption>
</figure>
"""


def render_model_results(scenario: dict[str, Any]) -> str:
    results = scenario.get("model_results", [])
    if not results:
        return ""
    rows = "\n".join(render_result_row(item) for item in results)
    return f"""
<table>
  <thead>
    <tr>
      <th>目标</th><th>不用 CanpGrid</th><th>使用 CanpGrid</th>
      <th>token 对比</th><th>过程</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
"""


def render_result_row(item: dict[str, Any]) -> str:
    target = item["target"]
    direct = item["without_canpgrid"]
    with_grid = item["with_canpgrid"]
    direct_text = method_cell(direct)
    with_text = method_cell(with_grid)
    direct_tokens = usage_total(direct.get("usage")) if direct.get("ok") else 0
    with_tokens = usage_total(with_grid.get("usage")) if with_grid.get("ok") else 0
    process = "-"
    if with_grid.get("ok"):
        process = f"cell {with_grid['cell']} ticks {with_grid['ticks']}"
    return f"""
<tr>
  <td>{html.escape(target["label"])}</td>
  <td>{direct_text}</td>
  <td>{with_text}</td>
  <td>{direct_tokens} / {with_tokens}</td>
  <td>{html.escape(process)}</td>
</tr>
"""


def method_cell(method: dict[str, Any]) -> str:
    if not method.get("ok"):
        return '<span class="miss">parse fail</span>'
    score = method["score"]
    klass = "hit" if score["hit"] else "miss"
    point = method["point"]
    return (
        f'<span class="{klass}">{html.escape("hit" if score["hit"] else "miss")}</span> '
        f'{point}, score {score["score_0_to_10"]}, err {score["error_px"]}px'
    )


def target_to_dict(target: Target) -> dict[str, Any]:
    return {
        "id": target.target_id,
        "label": target.label,
        "instruction": target.instruction,
        "center": [round(target.center[0], 1), round(target.center[1], 1)],
        "bbox": [round(value, 1) for value in target.bbox],
    }


def extract_json_text(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"response did not contain JSON: {content!r}")
    return stripped[start : end + 1]


def cell_for_point(point: tuple[float, float]) -> list[int]:
    col = math.floor(point[0] / SCREEN_SIZE[0] * GRID_SIZE[0])
    row = math.floor(point[1] / SCREEN_SIZE[1] * GRID_SIZE[1])
    return clamp_cell([col, row])


def oracle_point_spec(point: tuple[float, float], cell: list[int]) -> dict[str, Any]:
    cell_width = SCREEN_SIZE[0] / GRID_SIZE[0]
    cell_height = SCREEN_SIZE[1] / GRID_SIZE[1]
    cell_x1 = cell[0] * cell_width
    cell_y1 = cell[1] * cell_height
    x = round((point[0] - cell_x1) / cell_width * RULER_SIZE[0], 2)
    y = round((point[1] - cell_y1) / cell_height * RULER_SIZE[1], 2)
    return {
        "type": "cell_ruler_point",
        "grid_size": GRID_SIZE,
        "cell": cell,
        "x": clamp_number(x, 0, RULER_SIZE[0]),
        "y": clamp_number(y, 0, RULER_SIZE[1]),
        "ruler_size": RULER_SIZE,
    }


def clamp_cell(cell: list[int]) -> list[int]:
    return [clamp_int(cell[0], 0, GRID_SIZE[0] - 1), clamp_int(cell[1], 0, GRID_SIZE[1] - 1)]


def point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def usage(response: dict[str, Any]) -> dict[str, int]:
    raw = response.get("usage") or {}
    return {
        "prompt_tokens": int(raw.get("prompt_tokens") or 0),
        "completion_tokens": int(raw.get("completion_tokens") or 0),
        "total_tokens": int(raw.get("total_tokens") or 0),
    }


def add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        "prompt_tokens": left.get("prompt_tokens", 0) + right.get("prompt_tokens", 0),
        "completion_tokens": left.get("completion_tokens", 0) + right.get("completion_tokens", 0),
        "total_tokens": left.get("total_tokens", 0) + right.get("total_tokens", 0),
    }


def usage_total(value: dict[str, int] | None) -> int:
    if not value:
        return 0
    return int(value.get("total_tokens", 0))


def relative_asset(path: str | Path) -> str:
    return Path(path).resolve().relative_to(OUT_DIR.resolve()).as_posix()


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def clamp_number(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
