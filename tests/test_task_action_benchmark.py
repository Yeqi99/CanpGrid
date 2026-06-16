from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from examples.task_action_benchmark import (
    evaluate_task_prediction,
    parse_action_prediction,
    sample_wechat_suite,
    target,
    task,
)


def test_sample_wechat_suite_has_task_targets() -> None:
    suite = sample_wechat_suite(Path("chat.jpg"), Path("contacts.jpg"))

    assert len(suite["screens"]) == 2
    assert suite["screens"][0]["tasks"][0]["accepted_target_ids"] == ["chat_row_wife"]
    assert suite["screens"][1]["tasks"][0]["accepted_target_ids"] == [
        "contacts_plus",
        "new_friends_row",
    ]


def test_parse_action_prediction_resolves_point_spec(tmp_path) -> None:
    image_path = tmp_path / "screen.png"
    Image.new("RGB", (900, 2000), "#ffffff").save(image_path)
    content = json.dumps(
        {
            "target_id": "search",
            "label": "search",
            "role": "button",
            "point_spec": {
                "type": "subgrid_point",
                "grid_size": [9, 20],
                "cell": [7, 1],
                "local_point": [0.5, 0.5],
            },
        }
    )

    prediction, error = parse_action_prediction(
        content,
        image_path=image_path,
        image_size=(900, 2000),
    )

    assert error is None
    assert prediction is not None
    assert prediction["click_point"] == {"x": 750.0, "y": 150.0}


def test_evaluate_task_prediction_classifies_wrong_action_and_localization() -> None:
    screen = {
        "targets": [
            target("search", "搜索", "button", 100, 100, 200, 200),
            target("plus", "加号", "button", 300, 100, 400, 200),
        ]
    }
    task_item = task("add", "添加好友", ["plus"])

    wrong_action = evaluate_task_prediction(
        screen,
        task_item,
        {"target_id": "search", "click_point": {"x": 150, "y": 150}},
        None,
    )
    localization = evaluate_task_prediction(
        screen,
        task_item,
        {"target_id": "plus", "click_point": {"x": 450, "y": 150}},
        None,
    )

    assert wrong_action["category"] == "wrong_action"
    assert localization["category"] == "localization_error"
