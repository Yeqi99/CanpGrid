from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from interaction_benchmark import (  # noqa: E402
    dedupe_predictions_by_id,
    parse_inventory,
    parse_predictions,
)


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


def test_parse_predictions_accepts_react_final_items_metadata() -> None:
    predictions = parse_predictions(
        """
        {
          "final_items": [
            {
              "label": "settings",
              "role": "button",
              "grid_cell": {"col": 1, "row": 2},
              "cell_point": {"x": 0.25, "y": 0.75},
              "source_marker": 4,
              "revision": "adjust"
            }
          ]
        }
        """,
        (400, 400),
        grid_size=[4, 4],
    )

    assert predictions[0]["click_point"] == {"x": 125.0, "y": 275.0}
    assert predictions[0]["source_marker"] == 4
    assert predictions[0]["revision"] == "adjust"


def test_parse_predictions_accepts_duplicate_coordinate_arrays() -> None:
    predictions = parse_predictions(
        """
        {
          "items": [
            {
              "label": "back",
              "role": "button",
              "click_point": {"x": [48, 72], "y": [72, 72]}
            }
          ]
        }
        """,
        (100, 100),
    )

    assert predictions[0]["click_point"] == {"x": 48.0, "y": 72.0}


def test_parse_predictions_accepts_single_nested_coordinate_pair() -> None:
    predictions = parse_predictions(
        """
        {
          "items": [
            {
              "label": "AI search",
              "role": "button",
              "click_point": [[820, 140]]
            }
          ]
        }
        """,
        (900, 2000),
    )

    assert predictions[0]["click_point"] == {"x": 820.0, "y": 140.0}


def test_parse_predictions_uses_object_id_as_prediction_id() -> None:
    predictions = parse_predictions(
        """
        {
          "items": [
            {
              "object_id": "search_field",
              "label": "search",
              "role": "input",
              "click_point": {"x": 20, "y": 30}
            }
          ]
        }
        """,
        (100, 100),
    )

    assert predictions[0]["id"] == "search_field"


def test_dedupe_predictions_by_id_keeps_highest_confidence() -> None:
    predictions = dedupe_predictions_by_id(
        [
            {"id": "search", "confidence": 0.2, "click_point": {"x": 1, "y": 1}},
            {"id": "search", "confidence": 0.9, "click_point": {"x": 2, "y": 2}},
            {"id": "back", "confidence": 0.7, "click_point": {"x": 3, "y": 3}},
        ]
    )

    assert [item["id"] for item in predictions] == ["search", "back"]
    assert predictions[0]["click_point"] == {"x": 2, "y": 2}


def test_parse_inventory_normalizes_unique_objects() -> None:
    inventory = parse_inventory(
        """
        {
          "objects": [
            {
              "object_id": "search_field",
              "label": "search",
              "role": "input",
              "rough_grid_cell": {"col": 2, "row": 1}
            },
            {
              "object_id": "search_field",
              "label": "duplicate",
              "rough_grid_cell": {"col": 3, "row": 1}
            }
          ]
        }
        """,
        [9, 20],
    )

    assert inventory == [
        {
            "object_id": "search_field",
            "label": "search",
            "role": "input",
            "rough_grid_cell": {"col": 2, "row": 1},
            "confidence": None,
        }
    ]
