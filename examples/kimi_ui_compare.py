from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codex_baseline_report import HTML_PATH, REPORT_DIR, main as generate_report

DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MODEL = "kimi-k2.6"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Kimi UI localization with/without CanpGrid."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out", default=str(REPORT_DIR / "kimi_comparison.json"))
    args = parser.parse_args()

    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set MOONSHOT_API_KEY in your shell before running this script. "
            "The key is intentionally not read from code or repository files."
        )

    generate_report()
    source = REPORT_DIR / "assets" / "automation_settings_source.png"
    canpgrid_map = REPORT_DIR / "assets" / "resolved_action_points.png"

    without = call_kimi(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        images=[source],
        prompt=without_canpgrid_prompt(),
    )
    with_canpgrid = call_kimi(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        images=[source, canpgrid_map],
        prompt=with_canpgrid_prompt(),
    )

    output = {
        "model": args.model,
        "base_url": args.base_url,
        "report_html": str(HTML_PATH),
        "without_canpgrid": without,
        "with_canpgrid": with_canpgrid,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"kimi_comparison": str(out_path)}, ensure_ascii=False, indent=2))


def call_kimi(
    *,
    api_key: str,
    base_url: str,
    model: str,
    images: list[Path],
    prompt: str,
) -> str:
    content: list[dict[str, Any]] = []
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}})
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a visual localization evaluator. Return compact JSON only.",
            },
            {"role": "user", "content": content},
        ],
        "max_tokens": 2048,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kimi API request failed: {exc.code} {body}") from exc
    return data["choices"][0]["message"]["content"]


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def without_canpgrid_prompt() -> str:
    return """
The image is a drawn UI. Estimate candidate click coordinates in original image pixels
for this 6-step action:
1 Enable automation checkbox
2 Workspace name text field
3 Recipient email text field
4 Include summary checkbox
5 Run preview button
6 Save automation button

Return JSON only:
{
  "mode": "without_canpgrid",
  "score_0_to_10": number,
  "actions": [{"step": 1, "name": "...", "x": number, "y": number, "confidence": number}]
}
"""


def with_canpgrid_prompt() -> str:
    return """
You are given the original UI image and a CanpGrid candidate-point preview map.
The red numbered ring/crosshair markers are candidate focus positions for the same
6-step action. Evaluate whether the marked positions are usable.

Return JSON only:
{
  "mode": "with_canpgrid",
  "score_0_to_10": number,
  "confirmed_steps": [1,2,3],
  "needs_adjustment": [{"step": 1, "reason": "..."}]
}
"""


if __name__ == "__main__":
    main()
