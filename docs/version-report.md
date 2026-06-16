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
-> selected-cell ruler final view
-> resolved candidate point on the original image
-> preview image for confirm / adjust / relocalize
```

Run it with:

```bash
python examples/codex_baseline_report.py
```

The HTML output is written to:

```text
outputs/codex_baseline_report/index.html
```

## Optional Kimi comparison

To compare a Kimi vision model against the same generated UI, set the API key in
your shell and run:

```bash
export MOONSHOT_API_KEY="..."
python examples/kimi_ui_compare.py
```

The script uses Kimi's OpenAI-compatible API endpoint
`https://api.moonshot.cn/v1`, defaults to the multimodal `kimi-k2.6` model, and
disables model thinking for compact visible JSON output. It sends one
source-only prompt and one CanpGrid-preview prompt, then stores raw model
responses in:

```text
outputs/codex_baseline_report/kimi_comparison.json
```

Never commit API keys to this repository.

## DOM-bbox grounded UI benchmark

For model comparisons that need less subjective scoring than real screenshots,
use the WeChat-like DOM benchmark:

```bash
MOONSHOT_API_KEY="..." python examples/kimi_wechat_dom_benchmark.py
```

The HTML output is written to:

```text
outputs/wechat_dom_benchmark/index.html
```

This benchmark generates fixture HTML, image fixtures, exact target bbox
manifests, global CanpGrid views, selected-cell ruler examples, and preview
images. It always runs real model comparisons and scores a returned point as a
hit only when it lands inside the target bbox. If `MOONSHOT_API_KEY` is missing,
the script fails instead of generating an offline model-test report. It also
records visible API token usage when the provider returns usage metadata.

This report is intentionally outside CanpGrid Core. It uses a drawn UI screen
with checkboxes, text fields, and buttons so progress is easier to inspect, but
it does not perform real clicks, does not call a model, does not identify UI
semantics programmatically, and does not execute tasks. It only visualizes how
image-space references become candidate points on the original image.

For each future version, regenerate this report after tests pass and inspect:

- Whether each stage creates an annotated image.
- Whether the chosen `levels` are easy to audit.
- Whether final point specs are readable.
- Whether selected-cell ruler views make "cell + fine tick" choices readable.
- Whether preview images make the final focus easy to confirm.
- Whether local and original-image previews are both available when zoom hides context.
- Whether resolved candidate points land where expected.
- Whether a multi-step UI-like action is easy to audit.
- Whether model comparisons use a manifest or DOM bbox, not visual guesswork.
- Whether the report makes version progress clear to a human reviewer.
