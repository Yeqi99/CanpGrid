from __future__ import annotations

from canpgrid.evaluation import evaluate_interactions


def test_evaluate_interactions_classifies_correct_prediction() -> None:
    result = evaluate_interactions(
        [{"id": "search", "label": "search", "role": "button", "bbox": [90, 20, 130, 60]}],
        [{"label": "search", "role": "button", "click_point": [110, 40]}],
    )

    assert result["summary"]["correct_count"] == 1
    assert result["summary"]["missed_interactive_count"] == 0
    assert result["rows"][0]["category"] == "correct"


def test_evaluate_interactions_uses_stable_id_as_semantic_alias() -> None:
    result = evaluate_interactions(
        [{"id": "page_settings", "label": "页面设置", "role": "button", "bbox": [90, 20, 190, 60]}],
        [{"label": "page settings", "role": "button", "click_point": [120, 40]}],
    )

    assert result["summary"]["correct_count"] == 1
    assert result["rows"][0]["category"] == "correct"


def test_evaluate_interactions_classifies_localization_error() -> None:
    result = evaluate_interactions(
        [{"id": "search", "label": "search button", "role": "button", "bbox": [90, 20, 130, 60]}],
        [{"label": "search button", "role": "button", "click_point": [400, 300]}],
    )

    assert result["summary"]["correct_count"] == 0
    assert result["summary"]["localization_error_count"] == 1
    assert result["summary"]["recognition_recall"] == 1
    assert result["rows"][0]["problem_type"] == "position_problem"


def test_evaluate_interactions_classifies_semantic_mismatch() -> None:
    result = evaluate_interactions(
        [{"id": "search", "label": "search input", "role": "input", "bbox": [90, 20, 300, 60]}],
        [{"label": "deep thinking", "role": "button", "click_point": [120, 40]}],
    )

    assert result["summary"]["correct_count"] == 0
    assert result["summary"]["semantic_mismatch_count"] == 1
    assert result["rows"][0]["category"] == "semantic_mismatch"
    assert result["rows"][0]["problem_type"] == "understanding_semantic"


def test_evaluate_interactions_classifies_missed_and_false_positive() -> None:
    result = evaluate_interactions(
        [
            {"id": "search", "label": "search", "role": "button", "bbox": [90, 20, 130, 60]},
            {"id": "save", "label": "save", "role": "button", "bbox": [200, 20, 260, 60]},
        ],
        [{"label": "profile", "role": "button", "click_point": [500, 300]}],
    )

    assert result["summary"]["missed_interactive_count"] == 2
    assert result["summary"]["false_positive_count"] == 1
    assert {row["category"] for row in result["rows"]} == {
        "missed_interactive",
        "false_positive",
    }
