# Calibration

CanpGrid Calibration is not part of CanpGrid Core.

Core only creates annotated observation images and resolves spatial references.
It does not call models, score model behavior, execute tasks, or maintain UI
state.

Future Calibration work can evaluate model localization under different visual
observation configurations:

- `none`: no guide overlay baseline
- `grid`: coarse recursive region selection
- `ruler`: fine local coordinate selection
- `hybrid`: grid plus ruler for region and point descriptions
- `cell_ruler`: selected cell plus fine ruler for point selection without
  another zoom step

Calibration can measure whether overlays improve spatial accuracy compared with
the no-overlay baseline:

```text
overlay_gain = score_with_overlay - score_without_overlay
```

A future Model Visual Profile could include:

- Best overlay mode by task family.
- Preferred grid density by image size.
- Preferred ruler tick count.
- Accuracy and token tradeoff for one-step direct coordinates vs two-step
  `cell_ruler_point`.
- Error distribution by region size.
- Overlay gain compared with baseline.
