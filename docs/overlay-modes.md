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

## Detail modes

`detail_mode` can be `coarse`, `medium`, or `fine`. It changes the density and
visual weight of guide elements while keeping output PNGs agent-friendly.

