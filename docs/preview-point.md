# preview_point

`preview_point` generates a visual self-check image for a resolved candidate
focus point.

It is still part of CanpGrid Core's image observation boundary. It does not
execute clicks, does not type text, does not call a model, and does not perform
UI automation.

## API

```python
from canpgrid import preview_point

preview = preview_point(
    image_path,
    levels,
    point_spec,
    preview_on="current_view",
    marker_style="ring_crosshair",
    with_inset=False,
    out_dir="outputs",
)
```

## Inputs

- `image_path`: original image path.
- `levels`: recursive region path.
- `point_spec`: final focus point description.
- `preview_on`: `current_view`, `original_image`, or `both`.
- `marker_style`: `ring`, `ring_crosshair`, or `ring_crosshair_inset`.
- `with_inset`: add a local magnified inset.
- `out_dir`: preview output directory.

## Preview targets

`preview_on` controls where the candidate point is drawn:

- `current_view`: mark the point on the final cropped/zoomed observation view.
  This is best for checking fine placement.
- `original_image`: mark the same point on the full original image. This is best
  for checking global context when the local view is zoomed in too far.
- `both`: generate both images. This is the recommended self-check mode for UI
  work because it answers two different questions: "is the focus precise?" and
  "is it on the intended control?".

When the original-image preview is meant to preserve global context, prefer
`marker_style="ring_crosshair"` without an inset. Use `ring_crosshair_inset`
mainly on local/current views where the magnified patch helps inspect small
details.

## Output

```json
{
  "preview_image_path": "outputs/example_preview_current_ring_crosshair.png",
  "point_on_original": [640, 360],
  "point_on_current_view": [96, 112],
  "final_region_bbox_on_original": {
    "x1": 620,
    "y1": 340,
    "x2": 660,
    "y2": 380,
    "width": 40,
    "height": 40
  }
}
```

When `preview_on="both"`, the result also includes `preview_image_paths`:

```json
{
  "preview_image_paths": {
    "current_view": "outputs/example_preview_current_ring_crosshair.png",
    "original_image": "outputs/example_preview_original_ring_crosshair.png"
  }
}
```

## Marker styles

### ring

An adaptive hollow ring. The center remains transparent so image content is not
covered.

### ring_crosshair

A hollow ring plus a small center crosshair. This is the default because it is
easy to inspect without using a large solid dot.

### ring_crosshair_inset

The same marker on the main preview image plus a small magnified inset placed
away from the candidate point when possible.

## Self-check flow

After seeing the preview image, an agent can choose:

- `confirm_point`: the previewed point is correct.
- `adjust_point`: revise `point_spec`, often by changing a small ruler offset,
  then call `preview_point` again.
- `relocalize`: discard the final point and choose new `levels` or a new
  localization path.

`adjust_point` does not need a separate tool function. The model can update the
existing `point_spec`, for example by changing a `hybrid_point.offset`, then
generate a new preview image for visual review.
