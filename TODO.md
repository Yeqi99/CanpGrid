# CanpGrid Implementation Plan

CanpGrid Core is implemented first. Calibration is reserved as documentation and
future project structure only.

## Stage 1: Open-source package foundation

- Create standard Python package metadata in `pyproject.toml`.
- Add MIT license, `.gitignore`, GitHub Actions CI, README files, and docs.
- Keep the package dependency set small: Pillow, Typer, Pytest, Ruff.

## Stage 2: Core geometry protocol

- Implement density-based adaptive grid suggestions in `canpgrid/grid.py`.
- Implement recursive level-to-bbox resolution in `canpgrid/geometry.py`.
- Implement point specs: `normalized_point`, `anchor_offset`, `ruler_point`,
  `ruler_offset`, `hybrid_point`, and `subgrid_point`.
- Keep all geometry independent from rendering and image IO.

## Stage 3: Overlay rendering

- Implement image loading, crop/resize helpers, and PNG output.
- Implement `grid`, `ruler`, and `hybrid` guide overlays.
- Draw edge ticks and labels without placing large labels in each cell.

## Stage 4: Public API and CLI

- Implement `create_grid_view`, `zoom_region`, `resolve_region`, and
  `resolve_point` in `canpgrid/core.py`.
- Implement JSON-oriented Typer commands in `canpgrid/cli.py`.
- Ensure path inputs accept `str` and `Path`.

## Stage 5: Demo, docs, and tests

- Add an auto-generating demo image in `examples/demo.py`.
- Add English and Chinese READMEs plus protocol docs.
- Add tests for grid suggestion, geometry, point specs, validation, and render
  smoke coverage.
- Verify with editable install, `pytest`, `canpgrid --help`, and the demo.

## Stage 6: Repository publishing

- Commit the initial release.
- Create or attach a public GitHub repository if credentials are available.
- Push `main` and set repository description/topics where possible.

