# CanpGrid

面向多模态智能体的自适应递归图像网格观察工具。

CanpGrid 是一个渐进式图片观察辅助工具。它可以为任意图片生成带辅助线的观察图，支持递归放大局部区域，并把结构化空间引用解析回原图中的区域或点。

CanpGrid Core 不执行点击，不做 UI 自动化，不做目标检测，不做 OCR，不调用模型，也不记录 UI 经验库。它只是视觉观察和空间引用层。

## 功能

- 自适应图片网格
- 带辅助线的缩放观察图
- 递归区域观察
- grid、ruler、hybrid 三种辅助线模式
- 网格交点坐标系
- levels 路径解析为原图 bbox
- point_spec 解析为原图 point
- 候选焦点 preview 图
- Python API
- CLI JSON 输出
- 为未来 Calibration 预留设计

## 基本概念

```text
原图
-> 带辅助线的网格观察图
-> 选择 cell 并放大
-> 带辅助线的局部观察图
-> 解析 region / point
```

智能体观察的是辅助线图像，并用结构化 `levels` 或 `point_spec` 描述区域和焦点。CanpGrid 再把这些描述解析回原图坐标。

## 安装

```bash
python -m pip install -e ".[dev]"
```

需要 Python 3.10 或更高版本。

## Python API 示例

```python
from canpgrid import (
    create_cell_ruler_view,
    create_grid_view,
    preview_point,
    resolve_point,
    resolve_region,
    zoom_region,
)

view = create_grid_view(
    "examples/sample.png",
    grid_size=[12, 7],
    overlay_mode="grid",
    out_dir="outputs",
)

levels = [{"grid_size": [12, 7], "cell": [6, 2]}]

zoomed = zoom_region(
    "examples/sample.png",
    levels,
    next_grid_size=[8, 6],
    overlay_mode="hybrid",
    ruler_config={"tick_x": 16, "tick_y": 16},
    out_dir="outputs",
)

region = resolve_region("examples/sample.png", levels)

cell_ruler = create_cell_ruler_view(
    "examples/sample.png",
    levels,
    grid_size=[8, 6],
    cell=[3, 4],
    ruler_config={"tick_x": 10, "tick_y": 10},
    zoom_factor=4,
    out_dir="outputs",
)

point = resolve_point(
    "examples/sample.png",
    levels,
    {
        "type": "cell_ruler_point",
        "grid_size": [8, 6],
        "cell": [3, 4],
        "x": 3,
        "y": 6,
        "ruler_size": [10, 10],
    },
)

preview = preview_point(
    "examples/sample.png",
    levels,
    {
        "type": "cell_ruler_point",
        "grid_size": [8, 6],
        "cell": [3, 4],
        "x": 3,
        "y": 6,
        "ruler_size": [10, 10],
    },
    preview_on="both",
    marker_style="ring_crosshair_inset",
    out_dir="outputs",
)
```

`create_grid_view`、`zoom_region` 和 `create_cell_ruler_view` 默认都会生成
`annotated_image_path`。原图 bbox 是配套元数据，不是主要产物。
`create_cell_ruler_view` 会高亮一个已选 grid cell，并在这个 cell 内画更细
的尺子；模型可以回答“cell `[3, 4]` 的横 `3`、竖 `6`”，不用为了选点再多
做一次递归放大。
`preview_point` 会生成不执行点击的候选焦点预览图，让智能体先看一眼这个点是否正确。
当局部图缩放很大时，建议使用 `preview_on="both"`：current_view 看精度，
original_image 看全局控件归属。

## CLI 示例

```bash
canpgrid grid examples/sample.png --density medium --out outputs/
canpgrid grid examples/sample.png --grid-size 12x7 --out outputs/
canpgrid grid examples/sample.png --overlay-mode ruler --detail-mode fine --ruler-size 16x16 --out outputs/
canpgrid zoom examples/sample.png --levels '[{"grid_size":[12,7],"cell":[6,2]}]' --out outputs/
canpgrid zoom examples/sample.png --levels '[{"grid_size":[12,7],"cell":[6,2]}]' --overlay-mode hybrid --ruler-size 16x16 --out outputs/
canpgrid cell-ruler examples/sample.png --levels '[{"grid_size":[12,7],"cell":[6,2]}]' --grid-size 8x6 --cell 3x4 --ruler-size 10x10 --zoom-factor 4 --out outputs/
canpgrid resolve-region examples/sample.png --levels '[{"grid_size":[12,7],"cell":[6,2]},{"grid_size":[8,6],"cell":[3,4]}]'
canpgrid resolve-point examples/sample.png --levels '[{"grid_size":[12,7],"cell":[6,2]}]' --point-spec '{"type":"cell_ruler_point","grid_size":[8,6],"cell":[3,4],"x":3,"y":6,"ruler_size":[10,10]}'
canpgrid preview-point examples/sample.png --levels '[{"grid_size":[12,7],"cell":[6,2]}]' --point-spec '{"type":"normalized_point","value":["1/2","1/2"]}' --preview-on both --marker-style ring_crosshair_inset
```

所有 CLI 命令都输出 JSON，方便智能体调用。

## 协议概要

`grid_size = [cols, rows]` 表示横向列数和纵向行数。交点从 `[0, 0]` 到 `[cols, rows]`，cell 用左上角交点坐标表示。

递归路径使用 `levels`：

```json
[
  {"grid_size": [12, 7], "cell": [6, 2]},
  {"grid_size": [8, 6], "cell": [3, 4]}
]
```

每一层 cell 都只相对于当前局部画布有效。

支持的 `point_spec` 包括：

- `normalized_point`
- `anchor_offset`
- `ruler_point`
- `ruler_offset`
- `hybrid_point`
- `cell_ruler_point`
- `subgrid_point`

更多内容见 `docs/`。

## Calibration 方向

Calibration 不属于 Core。未来可以比较不同模型在 none/grid/ruler/hybrid 辅助线下的定位准确率，计算相对无辅助线 baseline 的 `overlay_gain`，并生成 Model Visual Profile。

## Demo 和测试

```bash
python examples/demo.py
python examples/codex_baseline_report.py
MOONSHOT_API_KEY=... python examples/kimi_wechat_dom_benchmark.py
pytest
ruff check .
```

Demo 输出保存在 `outputs/demo/`。
Codex baseline 观察报告输出在 `outputs/codex_baseline_report/index.html`，
用于查看一个包含复选框、文本框和按钮的小型 UI 组合动作如何从全图网格、
局部放大一路解析到候选点位。
微信风格 DOM 基准输出在 `outputs/wechat_dom_benchmark/index.html`，包含
可复现 fixture HTML、真实 target bbox、全局网格、选中格子细尺和 preview。
这个脚本永远使用真实 Kimi API 调用；如果环境里没有 `MOONSHOT_API_KEY`，
会直接失败，不生成离线模型测试报告。它会比较“直接给坐标”和“两步
CanpGrid 细尺流程”，并记录 token 消耗。

## 数据制造工作台

打开 `tools/annotation_workbench.html`，可以把任意 app 截图拖进工作台，
手动圈出可点击区域，填写 label/role，然后导出
`canpgrid.interactions.v1` JSON 标注。

用真实模型 API 评测这份标注：

```bash
MOONSHOT_API_KEY=... python examples/interaction_benchmark.py \
  --image path/to/screenshot.png \
  --annotations path/to/annotations.canpgrid.json
```

评测脚本会让模型自由识别全部可交互元素，再和你的标注对比，把错误分成：
漏识别、错识别、语义错、位置错误、重复识别和正确识别。详见
`docs/interaction-dataset.md`。

## 许可证

MIT License

## 品牌

Part of the CANPAI open agent infrastructure.
