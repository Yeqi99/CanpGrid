# Concept

CanpGrid is a visual observation layer for multimodal agents.

It turns an image into a guide-lined observation view. An agent can inspect the
view, select a grid cell, receive a new guide-lined local view, and continue
until the relevant region is clear enough. The final structured path can then be
resolved back to the original image.

Core responsibilities:

- Create annotated image views.
- Suggest adaptive grid sizes.
- Draw grid, ruler, and hybrid overlays.
- Resolve recursive region paths to original-image bounding boxes.
- Resolve final point specs to original-image points.

Core non-goals:

- No clicking.
- No UI automation.
- No OCR.
- No object detection.
- No task execution.
- No model calls.
- No UI memory or success feedback loop.

