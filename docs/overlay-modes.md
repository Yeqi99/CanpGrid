# Overlay Modes

## grid

Use `grid` for coarse observation and recursive zooming.

- Shows grid lines.
- Shows edge x/y ticks.
- Does not place large labels in cell centers.

## ruler

Use `ruler` for final localization.

- Shows fine ruler ticks.
- Supports configurable tick counts.
- Works with `ruler_point` and `ruler_offset`.

## hybrid

Use `hybrid` when an agent needs both region structure and fine point offsets.

- Keeps the grid feel.
- Adds fine ruler lines.
- Works naturally with `hybrid_point`.

## selected-cell ruler

Use `create_cell_ruler_view` or `canpgrid cell-ruler` when an agent wants to
select a point inside a visible cell instead of spending another turn zooming.

- Keeps the full current view visible.
- Highlights one selected cell.
- Draws a finer ruler only inside that cell.
- Works with `cell_ruler_point`.

This mode is useful for dense UI surfaces where the target is already
recognizable but a plain cell center would be too imprecise.

## Detail modes

`detail_mode` can be `coarse`, `medium`, or `fine`. It changes the density and
visual weight of guide elements while keeping output PNGs agent-friendly.
