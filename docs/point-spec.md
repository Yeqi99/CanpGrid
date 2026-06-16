# point_spec

CanpGrid supports multiple point descriptions so an agent can choose the most
natural form for the current overlay.

## normalized_point

```json
{
  "type": "normalized_point",
  "value": ["1/2", "1/2"]
}
```

The value is a relative point inside the final region. Supported fraction
strings are `0`, `1/4`, `1/2`, `3/4`, and `1`.

## anchor_offset

```json
{
  "type": "anchor_offset",
  "anchor": "center",
  "offset": ["+1/4", "-1/4"]
}
```

Anchors include `top_left`, `top`, `top_right`, `left`, `center`, `right`,
`bottom_left`, `bottom`, and `bottom_right`.

## ruler_point

```json
{
  "type": "ruler_point",
  "origin": "top_left",
  "x": 12,
  "y": 7,
  "ruler_size": [16, 16]
}
```

`x` and `y` are ruler units inside the final region, not original-image pixels.

## cell_ruler_point

```json
{
  "type": "cell_ruler_point",
  "grid_size": [9, 20],
  "cell": [7, 1],
  "x": 3,
  "y": 6,
  "ruler_size": [10, 10]
}
```

This is for selection, not zoom. The model first chooses a visible grid cell,
then gives a fine ruler position inside that cell. CanpGrid maps the cell and
ruler ticks back to original-image pixels.

Use this when the target is already visible enough and the model should not
spend another turn zooming into the cell.

For result metadata, the effective final region is the selected cell bbox on
the original image.

## ruler_offset

```json
{
  "type": "ruler_offset",
  "anchor": "center",
  "dx": 3,
  "dy": -2,
  "ruler_size": [16, 16]
}
```

The offset is measured in ruler ticks from the anchor.

## hybrid_point

```json
{
  "type": "hybrid_point",
  "base": ["1/2", "1/2"],
  "offset": [2, 3],
  "unit": "ruler_tick",
  "ruler_size": [16, 16]
}
```

The base is a normalized point. The offset is an integer ruler-tick offset.
Negative offsets are supported.

## subgrid_point

```json
{
  "type": "subgrid_point",
  "grid_size": [8, 8],
  "cell": [5, 3],
  "local_point": ["1/2", "1/2"]
}
```

This describes a micro-grid inside the final resolved region.

## color_snap_point

```json
{
  "type": "color_snap_point",
  "base": {
    "type": "cell_ruler_point",
    "grid_size": [9, 20],
    "cell": [7, 1],
    "x": 5,
    "y": 4,
    "ruler_size": [10, 10]
  },
  "target_color": "#34c759",
  "tolerance": 24,
  "search": {
    "mode": "ray",
    "direction": "right",
    "max_distance": 40
  }
}
```

`color_snap_point` is a pixel-level snap after the model has already described
a coarse candidate point. CanpGrid first resolves `base`, then searches original
image pixels for `target_color`.

Use it when the model can identify the right local object but cannot place the
final point accurately enough by ruler ticks alone.

Supported `target_color` forms:

- `"#RRGGBB"`
- `[r, g, b]`
- `{"r": 52, "g": 199, "b": 89}`
- common names such as `blue`, `red`, `green`, `black`, or `white`

`tolerance` is an RGB distance, useful for antialiasing and screenshot
compression.

Supported search modes:

- `nearest`: find the nearest matching pixel within `radius`.
- `ray`: scan from the base point in `direction` and return the first matching
  pixel within `max_distance`.

`search.start_offset` can move the scan start in original-image pixels before
searching. By default, a miss raises an error so the agent can relocalize. Set
`fallback` to `base_point` only when a non-snapped coarse point is acceptable.

The result includes `point_resolution` metadata with the base point, snapped
point, matched color, search mode, and search bbox.
