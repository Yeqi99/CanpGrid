from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from interaction_benchmark import parse_predictions  # noqa: E402


def test_parse_predictions_uses_bbox_center_as_point_fallback() -> None:
    predictions = parse_predictions(
        """
        {
          "items": [
            {
              "label": "search",
              "role": "button",
              "bbox": {"x1": 10, "y1": 20, "x2": 50, "y2": 80}
            }
          ]
        }
        """,
        (100, 100),
    )

    assert predictions == [
        {
            "id": "prediction_001",
            "index": 0,
            "label": "search",
            "role": "button",
            "confidence": None,
            "click_point": {"x": 30.0, "y": 50.0},
            "point_source": "bbox_center_fallback",
        }
    ]
    assert "bbox" not in predictions[0]


def test_parse_predictions_keeps_explicit_click_point() -> None:
    predictions = parse_predictions(
        '{"items":[{"label":"send","role":"button","click_point":{"x":12,"y":34}}]}',
        (100, 100),
    )

    assert predictions[0]["click_point"] == {"x": 12.0, "y": 34.0}
    assert predictions[0]["point_source"] == "click_point"


def test_parse_predictions_repairs_common_unquoted_point_key_typo() -> None:
    predictions = parse_predictions(
        """
        {
          "items": [
            {
              "label": "attach",
              "role": "button",
              "click_point": {"x": 40, y": 60}
            },
            {
              "label": "send",
              "role": "button",
              "click_point": {"x": 70, y: 80}
            }
          ]
        }
        """,
        (100, 100),
    )

    assert [item["click_point"] for item in predictions] == [
        {"x": 40.0, "y": 60.0},
        {"x": 70.0, "y": 80.0},
    ]


def test_parse_predictions_resolves_canpgrid_cell_point() -> None:
    predictions = parse_predictions(
        """
        {
          "items": [
            {
              "label": "voice",
              "role": "button",
              "grid_cell": {"col": 4, "row": 15},
              "cell_point": {"x": 0.5, "y": 0.25}
            }
          ]
        }
        """,
        (900, 2000),
        grid_size=[9, 20],
    )

    assert predictions[0]["click_point"] == {"x": 450.0, "y": 1525.0}
    assert predictions[0]["point_source"] == "grid_cell_point"
