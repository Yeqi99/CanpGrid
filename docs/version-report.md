# Version Observation Report

Every release should include a generated observation report that shows what
CanpGrid can currently do from an agent-facing perspective.

The baseline report uses Codex as the manual observer and demonstrates the
spatial localization flow on one generated UI action image:

```text
generated UI image
-> global grid view
-> selected rough cell
-> local zoom view
-> selected local cell
-> hybrid/ruler final view
-> resolved candidate point on the original image
```

Run it with:

```bash
python examples/codex_baseline_report.py
```

The HTML output is written to:

```text
outputs/codex_baseline_report/index.html
```

This report is intentionally outside CanpGrid Core. It uses a drawn UI screen
with checkboxes, text fields, and buttons so progress is easier to inspect, but
it does not perform real clicks, does not call a model, does not identify UI
semantics programmatically, and does not execute tasks. It only visualizes how
image-space references become candidate points on the original image.

For each future version, regenerate this report after tests pass and inspect:

- Whether each stage creates an annotated image.
- Whether the chosen `levels` are easy to audit.
- Whether final point specs are readable.
- Whether resolved candidate points land where expected.
- Whether a multi-step UI-like action is easy to audit.
- Whether the report makes version progress clear to a human reviewer.
