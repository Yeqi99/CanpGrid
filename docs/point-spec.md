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

