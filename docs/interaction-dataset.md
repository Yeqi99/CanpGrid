# Interaction Dataset Workbench

CanpGrid can use manually annotated screenshots as evaluation data for
interactive-element localization.

## Annotation workflow

Open the static workbench:

```text
tools/annotation_workbench.html
```

Then:

1. Upload an app screenshot.
2. Drag rectangles around clickable or tappable regions.
3. Set each region's label and role.
4. Export the JSON manifest.

The workbench stores coordinates in original-image pixels.

## Manifest shape

```json
{
  "schema_version": "canpgrid.interactions.v1",
  "image": {
    "name": "screenshot.png",
    "width": 1170,
    "height": 2532
  },
  "annotations": [
    {
      "id": "target_001",
      "label": "search",
      "role": "button",
      "required": true,
      "bbox": {
        "x1": 918,
        "y1": 150,
        "x2": 1018,
        "y2": 258,
        "width": 100,
        "height": 108
      },
      "click_point": {
        "x": 968,
        "y": 204
      }
    }
  ]
}
```

## Real API benchmark

Run:

```bash
MOONSHOT_API_KEY="..." python examples/interaction_benchmark.py \
  --image path/to/screenshot.png \
  --annotations path/to/annotations.canpgrid.json
```

The benchmark always uses real model calls. It compares:

- direct screenshot observation
- screenshot plus CanpGrid global grid observation

It asks the model to identify all visible interactive elements, not just a
target list.

## Error categories

- `correct`: the predicted click point or bbox lands inside the annotated
  clickable region.
- `missed_interactive`: an annotated interactive region was not found. This is
  a recall/understanding problem.
- `false_positive`: the model predicted an interactive region that is not in
  the annotation manifest. This is a precision/understanding problem.
- `semantic_mismatch`: the model's predicted region overlaps a real target, but
  the label is for another control. This is an understanding problem.
- `localization_error`: the label matches a real target, but the predicted
  position misses the annotated bbox. This is a position problem.
- `duplicate_prediction`: the model predicted the same annotated region more
  than once. This is a precision problem.

The output report is written to:

```text
outputs/interaction_benchmark/index.html
```
