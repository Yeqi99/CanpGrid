# Coordinate Protocol

## Grid intersections

The top-left intersection of the current canvas is `[0, 0]`.

`x` increases to the right. `y` increases downward.

`grid_size` is `[cols, rows]`.

For `grid_size = [12, 7]`:

- Intersections range from `[0, 0]` to `[12, 7]`.
- Cells range from `[0, 0]` to `[11, 6]`.
- A cell is addressed by its top-left intersection.

Example:

```json
{"grid_size": [12, 7], "cell": [6, 2]}
```

This selects the cell whose top-left intersection is `[6, 2]`.

## Recursive levels

Each selected cell is cropped from the current local canvas. The crop becomes a
new local canvas whose coordinate system starts again at `[0, 0]`.

```json
[
  {"grid_size": [12, 7], "cell": [6, 2]},
  {"grid_size": [8, 6], "cell": [3, 4]}
]
```

The second level is relative to the first level's crop, not the original image.

## Main output

Every `create_grid_view` and `zoom_region` call creates a new annotated image.

The annotated image is the main artifact for the next observation turn. The
original-image bbox is companion metadata.

