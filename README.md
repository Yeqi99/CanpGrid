# CanpGrid

Adaptive Recursive Image Grid for Multimodal Agents

CanpGrid is a progressive image observation tool for multimodal agents. It
generates guide-lined zoom views, supports recursive region inspection, and
resolves structured spatial references back to original image regions or points.

CanpGrid does not perform clicking, UI automation, object detection, OCR, task
execution, or UI memory. It is only a visual observation and spatial referencing
layer.

## Features

- Adaptive image grids
- Guide-lined zoom views
- Recursive region inspection
- Grid, ruler, and hybrid overlays
- Grid intersection coordinate system
- Region path to original bbox
- Flexible point_spec protocol
- Candidate point preview images
- CLI and Python API
- Agent-friendly JSON outputs
- Calibration-ready design

## Concept

```text
Original image
-> guide-lined grid view
-> zoom selected cell
-> guide-lined local view
-> resolve region / resolve point
```

The model observes annotated images and describes regions using structured grid
paths. CanpGrid maps those paths back to original-image coordinates.

## Installation

```bash
python -m pip install -e ".[dev]"
```

CanpGrid requires Python 3.10 or newer.

## Python API

```python
from canpgrid import create_grid_view, preview_point, resolve_point, resolve_region, zoom_region

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

point = resolve_point(
    "examples/sample.png",
    levels,
    {
        "type": "hybrid_point",
        "base": ["1/2", "1/2"],
        "offset": [2, 3],
        "unit": "ruler_tick",
        "ruler_size": [16, 16],
    },
)

preview = preview_point(
    "examples/sample.png",
    levels,
    {
        "type": "hybrid_point",
        "base": ["1/2", "1/2"],
        "offset": [2, 3],
        "unit": "ruler_tick",
        "ruler_size": [16, 16],
    },
    preview_on="both",
    marker_style="ring_crosshair_inset",
    out_dir="outputs",
)
```

`create_grid_view` and `zoom_region` always return an `annotated_image_path`.
The bbox metadata is companion data, not the main output.
`preview_point` creates a non-executing focus preview so an agent can inspect a
candidate point before confirming or adjusting it. Use `preview_on="both"` when
the local view is highly zoomed: the current-view preview checks precision, and
the original-image preview keeps the global UI context visible.

## CLI

Create a grid observation view:

```bash
canpgrid grid examples/sample.png --density medium --out outputs/
```

Use an explicit grid:

```bash
canpgrid grid examples/sample.png --grid-size 12x7 --out outputs/
```

Use a ruler overlay:

```bash
canpgrid grid examples/sample.png --overlay-mode ruler --detail-mode fine --ruler-size 16x16 --out outputs/
```

Zoom a selected region:

```bash
canpgrid zoom examples/sample.png --levels '[{"grid_size":[12,7],"cell":[6,2]}]' --out outputs/
```

Resolve a region:

```bash
canpgrid resolve-region examples/sample.png --levels '[{"grid_size":[12,7],"cell":[6,2]},{"grid_size":[8,6],"cell":[3,4]}]'
```

Resolve a point:

```bash
canpgrid resolve-point examples/sample.png --levels '[{"grid_size":[12,7],"cell":[6,2]}]' --point-spec '{"type":"normalized_point","value":["1/2","1/2"]}'
```

Preview a candidate point:

```bash
canpgrid preview-point examples/sample.png --levels '[{"grid_size":[12,7],"cell":[6,2]}]' --point-spec '{"type":"normalized_point","value":["1/2","1/2"]}' --preview-on both --marker-style ring_crosshair_inset
```

All CLI commands emit JSON.

## Protocol Summary

CanpGrid uses grid intersections. A `grid_size` of `[12, 7]` means 12 columns
and 7 rows. Intersections run from `[0, 0]` to `[12, 7]`; cells run from
`[0, 0]` to `[11, 6]` and are addressed by their top-left intersection.

Recursive zoom paths are represented as `levels`:

```json
{
  "levels": [
    {"grid_size": [12, 7], "cell": [6, 2]},
    {"grid_size": [8, 6], "cell": [3, 4]}
  ]
}
```

Each level is relative to the local canvas produced by the previous level.

Point specs include:

- `normalized_point`
- `anchor_offset`
- `ruler_point`
- `ruler_offset`
- `hybrid_point`
- `subgrid_point`

See [docs/protocol.md](docs/protocol.md) and
[docs/point-spec.md](docs/point-spec.md) for details. See
[docs/preview-point.md](docs/preview-point.md) for point preview and self-check.

## Calibration Potential

CanpGrid Core does not call models. Future CanpGrid Calibration work can compare
model localization accuracy across no overlay, grid, ruler, and hybrid modes,
then produce a Model Visual Profile for each model. It can also compute
`overlay_gain` against a no-guide baseline.

## Demo

```bash
python examples/demo.py
```

Demo outputs are saved to `outputs/demo/`.

## Version Observation Report

```bash
python examples/codex_baseline_report.py
```

This generates `outputs/codex_baseline_report/index.html`, an HTML report that
shows a Codex-baseline localization trace for a small UI action scenario with
checkboxes, text fields, and buttons. It is a progress artifact for release
review; it identifies candidate image-space positions only and does not execute
real clicks or add UI automation to Core.

## Tests

```bash
pytest
ruff check .
```

## License

MIT License

## Branding

Part of the CANPAI open agent infrastructure.
